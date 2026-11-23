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

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info
from wyoming.server import AsyncEventHandler

from . import debug_audio, ha_notify
from .config import Settings
from .trim import command_ended, decide
from .speaker import SpeakerModel, SpeakerTracker, Voiceprints
from .vad import SAMPLE_RATE, SileroVad, VadStream


class SpeakerContext:
    """Shared, loaded once: the speaker model and the enrolled voiceprints."""

    def __init__(
        self, model: SpeakerModel, prints: dict, voiceprints: Optional["Voiceprints"] = None
    ) -> None:
        self.model = model
        self.prints = prints
        self.voiceprints = voiceprints  # needed for enrolment

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
        self,
        settings: Settings,
        info_cache: dict,
        vad: Optional[SileroVad],
        *args,
        speaker: Optional[SpeakerContext] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.info_cache = info_cache
        self.vad = vad
        self.speaker = speaker
        self.tracker: Optional[SpeakerTracker] = None
        self.samples = np.zeros(0, dtype=np.float32)
        self.vad_stream: Optional[VadStream] = None
        self.transcribe: Optional[Transcribe] = None
        self.audio_start: Optional[AudioStart] = None
        self.chunks: List[AudioChunk] = []
        self.audio_started_at = 0.0
        self.ended_early = False

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
            self.ended_early = False
            self.vad_stream = None
            if self.vad is not None and self.settings.trim.mode != "off":
                fmt = (self.audio_start.rate, self.audio_start.width, self.audio_start.channels)
                if fmt == (SAMPLE_RATE, 2, 1):
                    self.vad_stream = self.vad.stream()
                    self.samples = np.zeros(0, dtype=np.float32)
                    self.tracker = None
                    if self.speaker is not None and self.speaker.prints:
                        self.tracker = SpeakerTracker(
                            self.speaker.model, self.speaker.prints, self.settings.speaker
                        )
                else:
                    _LOGGER.warning("Audio format %s not 16 kHz/16-bit/mono; not trimming", fmt)
            return True

        if AudioChunk.is_type(event.type):
            if self.ended_early:
                return True  # already answered; HA will stop sending shortly
            chunk = AudioChunk.from_event(event)
            self.chunks.append(chunk)
            if self.vad_stream is not None:
                try:
                    self.vad_stream.feed(chunk.audio)
                except Exception:  # noqa: BLE001 - never lose a command over VAD
                    _LOGGER.exception("VAD failed; not trimming this command")
                    self.vad_stream = None
            if self.tracker is not None and self.vad_stream is not None:
                try:
                    pcm = np.frombuffer(chunk.audio, dtype="<i2").astype(np.float32) / 32768.0
                    self.samples = np.concatenate([self.samples, pcm])
                    self.tracker.update(self.samples, self.vad_stream.probs)
                except Exception:  # noqa: BLE001 - fall back to VAD-only
                    _LOGGER.exception("Speaker matching failed; using VAD only for this command")
                    self.tracker = None
            if self.settings.early_end and self._command_over():
                # The command is over even if the room isn't quiet (e.g. TV):
                # answer now. Home Assistant stops recording when the
                # transcript arrives.
                self.ended_early = True
                await self._transcribe(AudioStop())
            return True

        if AudioStop.is_type(event.type):
            if not self.ended_early:
                await self._transcribe(AudioStop.from_event(event))
            self.transcribe = None
            self.audio_start = None
            self.chunks = []
            self.ended_early = False
            return True

        _LOGGER.debug("Ignoring event type %s", event.type)
        return True

    def _command_over(self) -> bool:
        if self.vad_stream is None:
            return False
        vad_over = command_ended(self.vad_stream.probs, self.settings.trim)
        if self.tracker is None:
            return vad_over
        # With voiceprints: a household voice spoke and then stopped (even if a
        # TV carries on), or VAD heard silence after a household voice. If no
        # household voice has spoken, leave it to Home Assistant to end the
        # recording - it may still be coming, or be a guest in a quiet room.
        if self.tracker.ended():
            return True
        return vad_over and bool(self.tracker.first_run())

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
        vad_kept = audio
        trim_note = "off"
        vad_ms = 0.0
        speaker_note = ""
        who: Optional[str] = None
        who_score: Optional[float] = None
        frame_bytes = start.width * start.channels
        if self.vad_stream is not None:
            try:
                probs = self.vad_stream.finish()
                vad_ms = self.vad_stream.compute_seconds * 1000
                result = decide(probs, len(audio) // frame_bytes, self.settings.trim)
                vad_kept = kept = audio[result.start * frame_bytes : result.end * frame_bytes]
                trim_note = (
                    f"{result.reason} kept={result.start / start.rate:.2f}-"
                    f"{result.end / start.rate:.2f}s speech_segments={len(result.segments)}"
                )
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Trimming failed; sending the whole recording")
                kept, trim_note = audio, "failed, kept all"
        if self.tracker is not None:
            found = self.tracker.result()
            spk_ms = self.tracker.compute_seconds * 1000
            if found is not None:
                s_start, s_end, name, score = found
                s_end = min(len(audio) // frame_bytes, s_end)
                kept = audio[s_start * frame_bytes : s_end * frame_bytes]
                trim_note = (
                    f"speaker kept={s_start / start.rate:.2f}-{s_end / start.rate:.2f}s"
                )
                speaker_note = f" speaker={name} score={score:.2f} spk={spk_ms:.0f}ms"
                who, who_score = name, score
            else:
                who = "unknown"
                best = max((w.score for w in self.tracker.windows), default=0.0)
                speaker_note = f" speaker=unknown best_score={best:.2f} spk={spk_ms:.0f}ms"
        self.vad_stream = None
        self.tracker = None

        if (
            self.settings.enroll_name
            and vad_kept
            and self.speaker is not None
            and self.speaker.voiceprints is not None
        ):
            # Only the embedding (numbers) is stored, never the audio.
            try:
                pcm = np.frombuffer(vad_kept, dtype="<i2").astype(np.float32) / 32768.0
                if self.speaker.voiceprints.add(self.settings.enroll_name, pcm):
                    speaker_note += " enrolment_saved=yes"
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Could not save enrolment embedding")

        if self.settings.debug_audio_dir:
            debug_audio.save(
                self.settings.debug_audio_dir, self.settings.debug_audio_keep,
                start.rate, start.width, start.channels, audio, kept,
            )

        # Tell Home Assistant who spoke while faster-whisper works, so the
        # sensor is up to date before the transcript reaches the conversation.
        notify: Optional[asyncio.Task] = None
        if who is not None and self.settings.ha_webhook_url:
            notify = asyncio.create_task(
                ha_notify.send_speaker(self.settings.ha_webhook_url, who, who_score)
            )

        try:
            upstream_seconds, sent_transcript = await asyncio.wait_for(
                self._forward(stop, kept, notify), self.settings.upstream_timeout
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.exception("Upstream transcription failed")
            await self.write_event(Error(text=f"voice-match upstream error: {err}").event())
            return

        total = time.monotonic() - received_at
        _LOGGER.info(
            "audio=%.2fs sent=%.2fs ended_early=%s trim=[%s]%s vad=%.0fms total=%.0fms "
            "upstream=%.0fms proxy_overhead=%.0fms text=%r",
            seconds,
            len(kept) / bytes_per_second,
            self.ended_early,
            trim_note,
            speaker_note,
            vad_ms,
            total * 1000,
            upstream_seconds * 1000,
            (total - upstream_seconds) * 1000,
            sent_transcript if self.settings.log_transcripts else "(hidden)",
        )

    async def _forward(
        self, stop: AudioStop, audio: bytes, before_transcript: Optional[asyncio.Task] = None
    ) -> tuple[float, str]:
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
                    if before_transcript is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(before_transcript), 0.5)
                        except asyncio.TimeoutError:
                            pass  # don't hold up the answer for a slow HA update
                    await self.write_event(event)
                    return upstream_seconds, Transcript.from_event(event).text
                if event.type in _RELAY_TYPES:
                    await self.write_event(event)
                elif Error.is_type(event.type):
                    await self.write_event(event)
                    raise RuntimeError(Error.from_event(event).text)
