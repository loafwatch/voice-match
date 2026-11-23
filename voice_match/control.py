"""Small HTTP control API for enrolment, called by Home Assistant (rest_command).

Every request needs the header `X-Voice-Match-Token: <VM_CONTROL_TOKEN>`.
Endpoints (JSON in, JSON out):

  GET  /status
  POST /enrol/start   {"name": "speaker_a", "total": 5, "mode": "replace"|"add"}
  POST /enrol/cancel
  POST /voices/delete {"name": "speaker_a"}
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re

from .enrolment import Enrolment

_LOGGER = logging.getLogger(__name__)
NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
MAX_BODY = 4096


class ControlServer:
    def __init__(self, enrolment: Enrolment, token: str, host: str = "0.0.0.0", port: int = 10311) -> None:
        self.enrolment = enrolment
        self.token = token
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        _LOGGER.info("Control API listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, payload = 400, {"error": "bad request"}
        try:
            line = (await asyncio.wait_for(reader.readline(), 5)).decode("latin-1").strip()
            method, path, _ = line.split(" ", 2)
            headers = {}
            while True:
                h = (await asyncio.wait_for(reader.readline(), 5)).decode("latin-1").strip()
                if not h:
                    break
                k, _, v = h.partition(":")
                headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length", "0") or 0)
            if length > MAX_BODY:
                raise ValueError("body too large")
            body = await asyncio.wait_for(reader.readexactly(length), 5) if length else b""
            if not hmac.compare_digest(headers.get("x-voice-match-token", ""), self.token):
                status, payload = 401, {"error": "unauthorised"}
            else:
                data = json.loads(body) if body else {}
                status, payload = await self._route(method, path.split("?")[0], data)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Control request failed: %s", err)
        out = json.dumps(payload).encode()
        writer.write(
            f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(out)}\r\nConnection: close\r\n\r\n".encode()
            + out
        )
        try:
            await writer.drain()
        finally:
            writer.close()

    async def _route(self, method: str, path: str, data: dict) -> tuple[int, dict]:
        e = self.enrolment
        if method == "GET" and path == "/status":
            return 200, e.status()
        if method != "POST":
            return 405, {"error": "method not allowed"}
        if path == "/enrol/start":
            name = str(data.get("name", "")).strip().lower()
            if not NAME_RE.match(name):
                return 400, {"error": "name: lower-case letters, digits, _ or - only"}
            return 200, await e.start(name, int(data.get("total", 5)), str(data.get("mode", "replace")))
        if path == "/enrol/cancel":
            return 200, await e.cancel()
        if path == "/voices/delete":
            name = str(data.get("name", "")).strip().lower()
            if not NAME_RE.match(name):
                return 400, {"error": "bad name"}
            return 200, await e.delete_voice(name)
        return 404, {"error": "not found"}
