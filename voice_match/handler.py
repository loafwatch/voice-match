"""Wyoming event handler: receives audio from Home Assistant, forwards it upstream.

Milestone 1 is a pure pass-through. The audio is buffered in full (the upstream
faster-whisper server only transcribes once it receives AudioStop anyway), then
replayed to the upstream speech-to-text server, and every event it sends back is
relayed to the caller. Buffering gives later stages (trimming, speaker
recognition) the whole utterance to work with.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import List, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info
from wyoming.server import AsyncEventHandler

from .config import Settings

_LOGGER = logging.getLogger(__name__)

# Upstream events that are part of a transcription response.
_RELAY_TYPES = {"transcript-start", "transcript-chunk", "transcript-stop"}


async def fetch_upstream_info(settings: Settings) -> Info:
    """Ask the upstream server what it offers."""
    async with AsyncClient.from_uri(settings.upstream_uri) as client:
        await client.write_event(Describe().event())
        while True:
            event = await asyncio.wait_for(client.read_event(), settings.describe_timeout)
            if event is None:
                raise ConnectionError("Upstream closed the connection during describe")
            if Info.is_type(event.type):
                return Info.from_event(event)


def proxy_info(upstream: Info) -> Info:
    """Present the upstream ASR program under our own name, models unchanged."""
    programs = []
    for program in upstream.asr:
        programs.append(
            replace(
                program,
                name="voice-match",
                description=f"Voice Match proxy for {program.name}",
                attribution=Attribution(
                    name="voice-match", url="https://github.com/loafwatch/voice-match"
                ),
            )
        )
    return Info(asr=programs)


class ProxyEventHandler(AsyncEventHandler):
    """One instance per connection from Home Assistant."""

    def __init__(self, settings: Settings, info_cache: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.info_cache = info_cache
        self.transcribe: Optional[Transcribe] = None
        self.audio_start: Optional[AudioStart] = None
        self.chunks: List[AudioChunk] = []
        self.audio_started_at = 0.0

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self._describe()
            return True

        if Transcribe.is_type(event.type):
            self.transcribe = Transcribe.from_event(event)
            return True

        if AudioStart.is_type(event.type):
            self.audio_start = AudioStart.from_event(event)
            self.chunks = []
            self.audio_started_at = time.monotonic()
            return True

        if AudioChunk.is_type(event.type):
            self.chunks.append(AudioChunk.from_event(event))
            return True

        if AudioStop.is_type(event.type):
            await self._transcribe(AudioStop.from_event(event))
            self.transcribe = None
            self.audio_start = None
            self.chunks = []
            return True

        _LOGGER.debug("Ignoring event type %s", event.type)
        return True

    async def _describe(self) -> None:
        try:
            info = await fetch_upstream_info(self.settings)
            self.info_cache["info"] = info
        except Exception as err:  # noqa: BLE001 - fall back to last known info
            info = self.info_cache.get("info")
            if info is None:
                _LOGGER.error("Upstream describe failed and nothing cached: %s", err)
                info = Info()
            else:
                _LOGGER.warning("Upstream describe failed, using cached info: %s", err)
        await self.write_event(proxy_info(info).event())

    async def _transcribe(self, stop: AudioStop) -> None:
        received_at = time.monotonic()
        if self.audio_start is None:
            await self.write_event(Error(text="AudioStop without AudioStart").event())
            return

        audio_bytes = sum(len(c.audio) for c in self.chunks)
        seconds = audio_bytes / max(
            1, self.audio_start.rate * self.audio_start.width * self.audio_start.channels
        )

        try:
            upstream_seconds, sent_transcript = await asyncio.wait_for(
                self._forward(stop), self.settings.upstream_timeout
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.exception("Upstream transcription failed")
            await self.write_event(Error(text=f"voice-match upstream error: {err}").event())
            return

        total = time.monotonic() - received_at
        _LOGGER.info(
            "audio=%.2fs total=%.0fms upstream=%.0fms proxy_overhead=%.0fms text=%r",
            seconds,
            total * 1000,
            upstream_seconds * 1000,
            (total - upstream_seconds) * 1000,
            sent_transcript if self.settings.log_transcripts else "(hidden)",
        )

    async def _forward(self, stop: AudioStop) -> tuple[float, str]:
        """Replay buffered audio upstream; relay its response. Returns timing + text."""
        assert self.audio_start is not None
        async with AsyncClient.from_uri(self.settings.upstream_uri) as client:
            await client.write_event((self.transcribe or Transcribe()).event())
            await client.write_event(self.audio_start.event())
            for chunk in self.chunks:
                await client.write_event(chunk.event())
            await client.write_event(stop.event())
            # Upstream only starts work at AudioStop, so time from here.
            sent_at = time.monotonic()

            while True:
                event = await client.read_event()
                if event is None:
                    raise ConnectionError("Upstream closed before sending a transcript")
                if Transcript.is_type(event.type):
                    upstream_seconds = time.monotonic() - sent_at
                    await self.write_event(event)
                    return upstream_seconds, Transcript.from_event(event).text
                if event.type in _RELAY_TYPES:
                    await self.write_event(event)
                elif Error.is_type(event.type):
                    await self.write_event(event)
                    raise RuntimeError(Error.from_event(event).text)
