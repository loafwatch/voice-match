"""Wyoming event handler: receives audio from Home Assistant, forwards it upstream.

The audio is buffered in full (the upstream faster-whisper server only
transcribes once it receives AudioStop anyway). While it streams in, Silero VAD
scores it; at the end the non-command parts are trimmed off, the rest is
replayed to the upstream speech-to-text server, and every event it sends back is
relayed to the caller.
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

from . import debug_audio
from .config import Settings
from .trim import decide
from .vad import SAMPLE_RATE, SileroVad, VadStream

_LOGGER = logging.getLogger(__name__)

# Bytes per chunk when replaying trimmed audio upstream (~64 ms at 16 kHz).
_REPLAY_CHUNK = 2048

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

    def __init__(
        self, settings: Settings, info_cache: dict, vad: Optional[SileroVad], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.info_cache = info_cache
        self.vad = vad
        self.vad_stream: Optional[VadStream] = None
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
            self.vad_stream = None
            if self.vad is not None and self.settings.trim.mode != "off":
                fmt = (self.audio_start.rate, self.audio_start.width, self.audio_start.channels)
                if fmt == (SAMPLE_RATE, 2, 1):
                    self.vad_stream = self.vad.stream()
                else:
                    _LOGGER.warning("Audio format %s not 16 kHz/16-bit/mono; not trimming", fmt)
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.chunks.append(chunk)
            if self.vad_stream is not None:
                try:
                    self.vad_stream.feed(chunk.audio)
                except Exception:  # noqa: BLE001 - never lose a command over VAD
                    _LOGGER.exception("VAD failed; not trimming this command")
                    self.vad_stream = None
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

        start = self.audio_start
        bytes_per_second = max(1, start.rate * start.width * start.channels)
        audio = b"".join(c.audio for c in self.chunks)
        seconds = len(audio) / bytes_per_second

        kept = audio
        trim_note = "off"
        vad_ms = 0.0
        if self.vad_stream is not None:
            try:
                probs = self.vad_stream.finish()
                vad_ms = self.vad_stream.compute_seconds * 1000
                frame_bytes = start.width * start.channels
                result = decide(probs, len(audio) // frame_bytes, self.settings.trim)
                kept = audio[result.start * frame_bytes : result.end * frame_bytes]
                trim_note = (
                    f"{result.reason} kept={result.start / start.rate:.2f}-"
                    f"{result.end / start.rate:.2f}s speech_segments={len(result.segments)}"
                )
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Trimming failed; sending the whole recording")
                kept, trim_note = audio, "failed, kept all"
        self.vad_stream = None

        if self.settings.debug_audio_dir:
            debug_audio.save(
                self.settings.debug_audio_dir, self.settings.debug_audio_keep,
                start.rate, start.width, start.channels, audio, kept,
            )

        try:
            upstream_seconds, sent_transcript = await asyncio.wait_for(
                self._forward(stop, kept), self.settings.upstream_timeout
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.exception("Upstream transcription failed")
            await self.write_event(Error(text=f"voice-match upstream error: {err}").event())
            return

        total = time.monotonic() - received_at
        _LOGGER.info(
            "audio=%.2fs sent=%.2fs trim=[%s] vad=%.0fms total=%.0fms upstream=%.0fms "
            "proxy_overhead=%.0fms text=%r",
            seconds,
            len(kept) / bytes_per_second,
            trim_note,
            vad_ms,
            total * 1000,
            upstream_seconds * 1000,
            (total - upstream_seconds) * 1000,
            sent_transcript if self.settings.log_transcripts else "(hidden)",
        )

    async def _forward(self, stop: AudioStop, audio: bytes) -> tuple[float, str]:
        """Replay audio upstream; relay its response. Returns timing + text."""
        start = self.audio_start
        assert start is not None
        async with AsyncClient.from_uri(self.settings.upstream_uri) as client:
            await client.write_event((self.transcribe or Transcribe()).event())
            await client.write_event(start.event())
            for i in range(0, len(audio), _REPLAY_CHUNK):
                await client.write_event(
                    AudioChunk(
                        rate=start.rate, width=start.width, channels=start.channels,
                        audio=audio[i : i + _REPLAY_CHUNK],
                    ).event()
                )
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
