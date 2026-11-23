"""The speaker of each command is posted to the Home Assistant webhook."""

from __future__ import annotations

import asyncio
import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.server import AsyncTcpServer

from test_speaker import B, FakeModel
from test_trim import CountingUpstream, LoudnessVad
from voice_match.config import Settings
from voice_match.handler import ProxyEventHandler, SpeakerContext
from voice_match.trim import TrimSettings


def start_webhook_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, json.loads(body)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


async def test_speaker_is_posted_before_transcript(unused_tcp_port_factory):
    http, received = start_webhook_server()
    up_port, proxy_port = unused_tcp_port_factory(), unused_tcp_port_factory()
    upstream = AsyncTcpServer("127.0.0.1", up_port)
    await upstream.start(CountingUpstream)
    settings = Settings(
        uri="", upstream_uri=f"tcp://127.0.0.1:{up_port}", upstream_timeout=5,
        describe_timeout=1, log_transcripts=False, debug=False, trim=TrimSettings(mode="first"),
        ha_webhook_url=f"http://127.0.0.1:{http.server_port}/api/webhook/example",
    )
    ctx = SpeakerContext(FakeModel(), {"speaker_b": B})
    proxy = AsyncTcpServer("127.0.0.1", proxy_port)
    await proxy.start(partial(ProxyEventHandler, settings, {}, LoudnessVad(), speaker=ctx))

    def seg(seconds, loud):
        return np.full(int(16000 * seconds), 8000 if loud else 0, "<i2").tobytes()

    audio = seg(0.3, False) + seg(1.5, True) + seg(1.5, False)
    try:
        async with AsyncTcpClient("127.0.0.1", proxy_port) as client:
            await client.write_event(Transcribe().event())
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
            for i in range(0, len(audio), 1024):
                await client.write_event(
                    AudioChunk(rate=16000, width=2, channels=1, audio=audio[i : i + 1024]).event()
                )
            await client.write_event(AudioStop().event())
            event = await asyncio.wait_for(client.read_event(), 3)
            assert Transcript.is_type(event.type)
        # posted before the transcript was returned
        assert received == [("/api/webhook/example", {"speaker": "speaker_b", "score": 1.0})]
    finally:
        await proxy.stop()
        await upstream.stop()
        http.shutdown()


async def test_unreachable_home_assistant_does_not_break_voice():
    from voice_match.ha_notify import send_speaker

    await send_speaker("http://127.0.0.1:9/api/webhook/x", "unknown", None, timeout=0.5)
