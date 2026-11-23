"""End-to-end test: client -> voice-match -> fake upstream STT server."""

from __future__ import annotations

import asyncio
from functools import partial

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncTcpServer

from voice_match.config import Settings
from voice_match.handler import ProxyEventHandler

RATE, WIDTH, CHANNELS = 16000, 2, 1

UPSTREAM_INFO = Info(
    asr=[
        AsrProgram(
            name="fake-whisper",
            attribution=Attribution(name="test", url="https://example.com"),
            installed=True,
            description="fake",
            version="1",
            models=[
                AsrModel(
                    name="tiny",
                    attribution=Attribution(name="test", url="https://example.com"),
                    installed=True,
                    description=None,
                    version=None,
                    languages=["en"],
                )
            ],
            supports_transcript_streaming=True,
        )
    ]
)


class FakeUpstream(AsyncEventHandler):
    """Replies with the number of audio bytes and the requested language."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bytes = 0
        self.language = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(UPSTREAM_INFO.event())
        elif Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language
        elif AudioChunk.is_type(event.type):
            self.bytes += len(AudioChunk.from_event(event).audio)
        elif AudioStop.is_type(event.type):
            await self.write_event(Event(type="transcript-start"))
            await self.write_event(Transcript(text=f"{self.bytes} {self.language}").event())
            return False
        return True


async def _start(server, handler):
    await server.start(handler)
    return server


@pytest.fixture
async def servers(unused_tcp_port_factory):
    up_port, proxy_port = unused_tcp_port_factory(), unused_tcp_port_factory()
    upstream = await _start(AsyncTcpServer("127.0.0.1", up_port), FakeUpstream)
    settings = Settings(
        uri=f"tcp://127.0.0.1:{proxy_port}",
        upstream_uri=f"tcp://127.0.0.1:{up_port}",
        upstream_timeout=5,
        describe_timeout=2,
        log_transcripts=True,
        debug=True,
    )
    proxy = await _start(
        AsyncTcpServer("127.0.0.1", proxy_port), partial(ProxyEventHandler, settings, {}, None)
    )
    yield proxy_port
    await proxy.stop()
    await upstream.stop()


async def _read_until(client, type_):
    events = []
    while True:
        event = await asyncio.wait_for(client.read_event(), 5)
        assert event is not None
        events.append(event)
        if event.type == type_:
            return events


async def test_describe_renames_program(servers):
    async with AsyncTcpClient("127.0.0.1", servers) as client:
        await client.write_event(Describe().event())
        events = await _read_until(client, "info")
    info = Info.from_event(events[-1])
    assert info.asr[0].name == "voice-match"
    assert info.asr[0].models[0].languages == ["en"]
    assert info.asr[0].supports_transcript_streaming is True


async def test_audio_passes_through_unchanged(servers):
    chunk = b"\x01\x00" * 1600  # 0.1 s
    async with AsyncTcpClient("127.0.0.1", servers) as client:
        await client.write_event(Transcribe(language="en").event())
        await client.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event())
        for _ in range(20):
            await client.write_event(
                AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=chunk).event()
            )
        await client.write_event(AudioStop().event())
        events = await _read_until(client, "transcript")
    assert [e.type for e in events] == ["transcript-start", "transcript"]
    assert Transcript.from_event(events[-1]).text == f"{len(chunk) * 20} en"


async def test_upstream_down_returns_error(unused_tcp_port_factory):
    proxy_port, dead_port = unused_tcp_port_factory(), unused_tcp_port_factory()
    settings = Settings(
        uri="", upstream_uri=f"tcp://127.0.0.1:{dead_port}", upstream_timeout=2,
        describe_timeout=1, log_transcripts=False, debug=False,
    )
    proxy = await _start(
        AsyncTcpServer("127.0.0.1", proxy_port), partial(ProxyEventHandler, settings, {}, None)
    )
    try:
        async with AsyncTcpClient("127.0.0.1", proxy_port) as client:
            await client.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event())
            await client.write_event(AudioStop().event())
            events = await _read_until(client, "error")
        assert "upstream" in events[-1].data["text"]
    finally:
        await proxy.stop()
