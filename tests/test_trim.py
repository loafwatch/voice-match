"""Trimming: decision logic, the real Silero model, and end-to-end via the proxy."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncTcpServer

from voice_match.config import Settings
from voice_match.handler import ProxyEventHandler
from voice_match.trim import TrimSettings, decide, speech_segments
from voice_match.vad import DEFAULT_MODEL, FRAME, FRAME_MS

FPS = 1000 / FRAME_MS  # frames per second (31.25)


def probs_from(spans, total_s):
    """Speech probability 0.9 inside (start, end) second spans, else 0.05."""
    n = int(total_s * FPS)
    p = [0.05] * n
    for a, b in spans:
        for i in range(int(a * FPS), int(b * FPS)):
            p[i] = 0.9
    return p, n * FRAME


def kept_seconds(r):
    return r.start / 16000, r.end / 16000


# ---------- decision logic ----------

def test_edges_cuts_leading_and_trailing_silence():
    p, n = probs_from([(0.5, 1.5)], 3.0)
    start, end = kept_seconds(decide(p, n, TrimSettings(mode="edges")))
    assert 0.25 <= start <= 0.35  # 200 ms pad before
    assert 1.75 <= end <= 1.85  # 300 ms pad after


def test_first_drops_speech_after_long_gap():
    p, n = probs_from([(0.2, 1.5), (2.5, 6.0)], 7.0)  # command, 1 s gap, TV
    start, end = kept_seconds(decide(p, n, TrimSettings(mode="first")))
    assert end < 2.0


def test_first_keeps_short_pause_inside_command():
    p, n = probs_from([(0.2, 1.2), (1.8, 2.6)], 4.0)  # 0.6 s pause
    _, end = kept_seconds(decide(p, n, TrimSettings(mode="first")))
    assert end > 2.6


def test_edges_keeps_everything_between_first_and_last_speech():
    p, n = probs_from([(0.2, 1.5), (2.5, 6.0)], 7.0)
    _, end = kept_seconds(decide(p, n, TrimSettings(mode="edges")))
    assert end > 6.0


def test_no_speech_keeps_all():
    p, n = probs_from([], 3.0)
    r = decide(p, n, TrimSettings())
    assert (r.start, r.end) == (0, n)


def test_off_keeps_all():
    p, n = probs_from([(1.0, 1.5)], 3.0)
    r = decide(p, n, TrimSettings(mode="off"))
    assert (r.start, r.end) == (0, n)


def test_blips_shorter_than_min_speech_are_ignored():
    p, n = probs_from([(1.0, 1.05)], 3.0)  # 50 ms click
    assert speech_segments(p, TrimSettings()) == []


# ---------- real model ----------

needs_model = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="run scripts/fetch_models.py")


@needs_model
def test_silero_sees_no_speech_in_silence_or_tone():
    from voice_match.vad import SileroVad

    vad = SileroVad()
    t = np.arange(16000 * 2) / 16000
    for signal in (np.zeros_like(t), 0.3 * np.sin(2 * np.pi * 440 * t)):
        stream = vad.stream()
        pcm = (signal * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 2048):  # odd chunk sizes vs 1024-byte frames
            stream.feed(pcm[i : i + 2048])
        probs = stream.finish()
        assert len(probs) == int(np.ceil(len(t) / FRAME))
        assert max(probs) < 0.5


# ---------- end to end through the proxy, with a loudness-based fake VAD ----------

class LoudnessVad:
    """Stands in for Silero: a frame is 'speech' if it is loud."""

    def stream(self):
        return _LoudStream()


class _LoudStream:
    def __init__(self):
        self.buf = b""
        self.probs = []
        self.compute_seconds = 0.0

    def feed(self, pcm):
        self.buf += pcm
        while len(self.buf) >= FRAME * 2:
            frame = np.frombuffer(self.buf[: FRAME * 2], "<i2")
            self.buf = self.buf[FRAME * 2 :]
            self.probs.append(0.9 if np.abs(frame).mean() > 3000 else 0.05)

    def finish(self):
        return self.probs


class CountingUpstream(AsyncEventHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n = 0

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            self.n += len(AudioChunk.from_event(event).audio)
        elif AudioStop.is_type(event.type):
            await self.write_event(Transcript(text=str(self.n)).event())
            return False
        return True


async def test_proxy_sends_only_the_command(unused_tcp_port_factory):
    up_port, proxy_port = unused_tcp_port_factory(), unused_tcp_port_factory()
    upstream = AsyncTcpServer("127.0.0.1", up_port)
    await upstream.start(CountingUpstream)
    settings = Settings(
        uri="", upstream_uri=f"tcp://127.0.0.1:{up_port}", upstream_timeout=5,
        describe_timeout=1, log_transcripts=True, debug=True, trim=TrimSettings(mode="first"),
    )
    proxy = AsyncTcpServer("127.0.0.1", proxy_port)
    await proxy.start(partial(ProxyEventHandler, settings, {}, LoudnessVad()))

    def seg(seconds, loud):
        n = int(16000 * seconds)
        return (np.full(n, 8000 if loud else 0, "<i2")).tobytes()

    # 0.5 s quiet, 1 s "command", 1.2 s quiet, 3 s "TV"
    audio = seg(0.5, False) + seg(1.0, True) + seg(1.2, False) + seg(3.0, True)
    try:
        async with AsyncTcpClient("127.0.0.1", proxy_port) as client:
            await client.write_event(Transcribe().event())
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            for i in range(0, len(audio), 1024):
                await client.write_event(
                    AudioChunk(rate=16000, width=2, channels=1, audio=audio[i : i + 1024]).event()
                )
            await client.write_event(AudioStop().event())
            while True:
                event = await client.read_event()
                if Transcript.is_type(event.type):
                    break
        sent_seconds = int(Transcript.from_event(event).text) / 32000
        assert 1.4 <= sent_seconds <= 1.6  # command + 200 ms before + 300 ms after
    finally:
        await proxy.stop()
        await upstream.stop()
