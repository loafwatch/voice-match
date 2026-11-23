"""Guided enrolment sessions and the control API."""

from __future__ import annotations

import asyncio
import json
from functools import partial

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncEventHandler, AsyncTcpServer

from voice_match.config import Settings
from voice_match.control import ControlServer
from voice_match.enrolment import Enrolment
from voice_match.speaker import Voiceprints
from voice_match.trim import TrimSettings

SR = 16000


class SeededModel:
    """Different random embedding per distinct audio level; 'like' maps a level to a fixed vector."""

    def __init__(self):
        self.like = {}

    def embed(self, samples):
        key = round(float(np.abs(samples).mean()), 3)
        if key in self.like:
            return self.like[key]
        rng = np.random.default_rng(int(key * 1e6))
        e = rng.normal(size=256)
        return e / np.linalg.norm(e)


def speech(level=0.1, seconds=3.0):
    return np.full(int(SR * seconds), level, np.float32)


@pytest.fixture
def vp(tmp_path):
    model = SeededModel()
    v = Voiceprints(model, tmp_path)
    v.add("speaker_a", speech(0.5), stamp="a1")
    v.build()
    return v


async def test_full_session_builds_a_new_voice(vp):
    pushed = []

    async def notify(status):
        pushed.append(status)

    e = Enrolment(vp, notify)
    await e.start("speaker_b", total=3)
    assert await e.add_sample(speech(0.1, 0.8)) == "voice enrolment retry"  # too short
    assert await e.add_sample(speech(0.11)) == "voice enrolment sample 1 of 3"
    assert await e.add_sample(speech(0.12)) == "voice enrolment sample 2 of 3"
    assert await e.add_sample(speech(0.13)) == "voice enrolment complete speaker_b"
    assert not e.active
    assert set(vp.prints) == {"speaker_a", "speaker_b"}  # live, no restart
    assert pushed[-1]["state"] == "idle" and pushed[-1]["message"] == "complete speaker_b"
    assert [v["name"] for v in pushed[-1]["voices"]] == ["speaker_a", "speaker_b"]


async def test_sample_that_sounds_like_someone_else_is_rejected(vp):
    vp.model.like[0.2] = vp.prints["speaker_a"]  # this "voice" is speaker_a
    e = Enrolment(vp)
    await e.start("speaker_b", total=3)
    assert await e.add_sample(speech(0.2)) == "voice enrolment other speaker_a"
    assert e.session.done == 0


async def test_replace_keeps_only_new_samples_and_add_keeps_old(vp):
    e = Enrolment(vp)
    files = lambda: sorted(p.name for p in (vp.emb_dir / "speaker_a").glob("*.npz"))
    assert files() == ["a1.npz"]
    await e.start("speaker_a", total=3, mode="add")
    for lvl in (0.31, 0.32, 0.33):
        await e.add_sample(speech(lvl))
    assert len(files()) == 4
    await e.start("speaker_a", total=3, mode="replace")
    for lvl in (0.41, 0.42, 0.43):
        await e.add_sample(speech(lvl))
    assert len(files()) == 3 and "a1.npz" not in files()


async def test_cancel_discards_only_this_sessions_samples(vp):
    e = Enrolment(vp)
    await e.start("speaker_a", total=5, mode="add")
    await e.add_sample(speech(0.21))
    await e.cancel()
    assert sorted(p.name for p in (vp.emb_dir / "speaker_a").glob("*.npz")) == ["a1.npz"]


async def test_delete_voice_removes_only_that_person(vp, tmp_path):
    vp.add("speaker_b", speech(0.6), stamp="b1")
    vp.build()
    (tmp_path / "unrelated.txt").write_text("keep")
    e = Enrolment(vp)
    await e.delete_voice("speaker_b")
    assert set(vp.prints) == {"speaker_a"}
    assert not (vp.emb_dir / "speaker_b").exists()
    assert not (vp.print_dir / "speaker_b.npy").exists()
    assert (vp.print_dir / "speaker_a.npy").exists()
    assert (vp.emb_dir / "speaker_a" / "a1.npz").exists()
    assert (tmp_path / "unrelated.txt").exists()


async def _http(port, method, path, body=None, token=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(data)}\r\n"
    if token:
        head += f"X-Voice-Match-Token: {token}\r\n"
    writer.write(head.encode() + b"\r\n" + data)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    status = int(raw.split(b" ", 2)[1])
    return status, json.loads(raw.split(b"\r\n\r\n", 1)[1])


async def test_control_api_requires_token_and_validates_names(vp, unused_tcp_port):
    token = "x" * 32
    server = ControlServer(Enrolment(vp), token, host="127.0.0.1", port=unused_tcp_port)
    await server.start()
    try:
        assert (await _http(unused_tcp_port, "GET", "/status"))[0] == 401
        assert (await _http(unused_tcp_port, "GET", "/status", token="wrong" * 8))[0] == 401
        st, body = await _http(unused_tcp_port, "POST", "/enrol/start", {"name": "../etc"}, token)
        assert st == 400
        st, body = await _http(unused_tcp_port, "POST", "/enrol/start", {"name": "Speaker_B", "total": 5}, token)
        assert st == 200 and body["state"] == "active" and body["name"] == "speaker_b"
        st, body = await _http(unused_tcp_port, "POST", "/enrol/cancel", {}, token)
        assert body["state"] == "idle"
    finally:
        await server.stop()


class MustNotBeCalled(AsyncEventHandler):
    async def handle_event(self, event):
        raise AssertionError("speech-to-text must not run during enrolment")


async def test_enrolment_sample_skips_speech_to_text(vp, unused_tcp_port_factory):
    from test_trim import LoudnessVad
    from voice_match.handler import ProxyEventHandler, SpeakerContext

    up_port, proxy_port = unused_tcp_port_factory(), unused_tcp_port_factory()
    upstream = AsyncTcpServer("127.0.0.1", up_port)
    await upstream.start(MustNotBeCalled)
    e = Enrolment(vp)
    await e.start("speaker_b", total=3)
    settings = Settings(
        uri="", upstream_uri=f"tcp://127.0.0.1:{up_port}", upstream_timeout=5,
        describe_timeout=1, log_transcripts=False, debug=False, trim=TrimSettings(mode="first"),
    )
    ctx = SpeakerContext(vp.model, vp.prints, vp, e)
    proxy = AsyncTcpServer("127.0.0.1", proxy_port)
    await proxy.start(partial(ProxyEventHandler, settings, {}, LoudnessVad(), speaker=ctx))
    audio = np.concatenate([np.zeros(4000, "<i2"), np.full(SR * 3, 8000, "<i2"), np.zeros(SR, "<i2")]).tobytes()
    try:
        async with AsyncTcpClient("127.0.0.1", proxy_port) as client:
            await client.write_event(Transcribe().event())
            await client.write_event(AudioStart(rate=SR, width=2, channels=1).event())
            for i in range(0, len(audio), 1024):
                await client.write_event(AudioChunk(rate=SR, width=2, channels=1, audio=audio[i : i + 1024]).event())
            await client.write_event(AudioStop().event())
            event = await asyncio.wait_for(client.read_event(), 3)
        assert Transcript.from_event(event).text == "voice enrolment sample 1 of 3"
        assert not list(vp.data_dir.rglob("*.wav"))
    finally:
        await proxy.stop()
        await upstream.stop()
