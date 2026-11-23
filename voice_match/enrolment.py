"""Guided enrolment sessions, started from Home Assistant (e.g. a panel button).

While a session is active, each command spoken at the satellite is treated as
an enrolment sample: its speaker embedding is stored (never the audio) and
voice-match answers with a short status transcript instead of sending the
audio to speech-to-text, so the assistant replies "got it, 2 of 5" rather
than trying to act on the sentence.

Status transcripts (handled by a Home Assistant sentence automation):
  voice enrolment sample <done> of <total>
  voice enrolment complete <name>
  voice enrolment retry               (too little speech)
  voice enrolment other <name>        (sounded like someone already enrolled)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from .speaker import Voiceprints
from .vad import SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

MIN_SPEECH_SECONDS = 1.5
OTHER_VOICE_THRESHOLD = 0.72  # same person on a new sentence scores ~0.8-0.9; similar voices ~0.65


@dataclass
class Session:
    name: str
    total: int
    mode: str  # "replace" or "add"
    deadline: float
    done: int = 0
    files: list[Path] = field(default_factory=list)


class Enrolment:
    def __init__(
        self,
        voiceprints: Voiceprints,
        notify: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        self.voiceprints = voiceprints
        self.notify = notify
        self.session: Optional[Session] = None
        self.last_message = ""

    # ----- state -----

    @property
    def active(self) -> bool:
        if self.session and time.monotonic() > self.session.deadline:
            _LOGGER.info("Enrolment timed out")
            self._discard()
            self.session = None
            self.last_message = "timed out"
        return self.session is not None

    def voices(self) -> list[dict]:
        out = []
        for name in sorted(self.voiceprints.prints):
            stats = self.voiceprints.stats(name)
            out.append({"name": name, "recordings": stats.get("recordings"), "built": stats.get("built")})
        return out

    def status(self) -> dict:
        s = self.session if self.active else None
        return {
            "state": "active" if s else "idle",
            "name": s.name if s else "",
            "done": s.done if s else 0,
            "total": s.total if s else 0,
            "message": self.last_message,
            "voices": self.voices(),
        }

    async def _push(self) -> None:
        if self.notify:
            await self.notify(self.status())

    # ----- control -----

    async def start(self, name: str, total: int = 5, mode: str = "replace", timeout: float = 300) -> dict:
        if self.active:
            self._discard()
        self.session = Session(
            name=name, total=max(3, min(int(total), 12)), mode=mode if mode in ("replace", "add") else "replace",
            deadline=time.monotonic() + timeout,
        )
        self.last_message = "started"
        _LOGGER.info("Enrolment started (%d samples, %s)", self.session.total, self.session.mode)
        await self._push()
        return self.status()

    async def cancel(self) -> dict:
        if self.session:
            self._discard()
            self.session = None
            self.last_message = "cancelled"
            _LOGGER.info("Enrolment cancelled; samples from this session discarded")
        await self._push()
        return self.status()

    async def delete_voice(self, name: str) -> dict:
        removed = self.voiceprints.delete(name)
        self.last_message = f"deleted {name}" if removed else f"no voice called {name}"
        await self._push()
        return self.status()

    def _discard(self) -> None:
        if self.session:
            for f in self.session.files:
                f.unlink(missing_ok=True)

    # ----- samples -----

    async def add_sample(self, audio: np.ndarray) -> str:
        """Store one sample; returns the status transcript."""
        s = self.session
        if s is None:
            return "voice enrolment retry"
        s.deadline = time.monotonic() + 300
        if len(audio) < MIN_SPEECH_SECONDS * SAMPLE_RATE:
            self.last_message = "too short"
            await self._push()
            return "voice enrolment retry"

        emb = await asyncio.to_thread(self.voiceprints.model.embed, audio)
        others = {n: float(emb @ p) for n, p in self.voiceprints.prints.items() if n != s.name}
        if others:
            other, score = max(others.items(), key=lambda kv: kv[1])
            if score >= OTHER_VOICE_THRESHOLD:
                self.last_message = f"sounded like {other}"
                await self._push()
                return f"voice enrolment other {other}"

        path = self.voiceprints.save_embedding(s.name, emb, len(audio) / SAMPLE_RATE)
        s.files.append(path)
        s.done += 1
        if s.done < s.total:
            self.last_message = f"sample {s.done} of {s.total}"
            await self._push()
            return f"voice enrolment sample {s.done} of {s.total}"

        # Finished: replace the old samples (unless adding), rebuild, go live.
        if s.mode == "replace":
            keep = {f.name for f in s.files}
            for old in (self.voiceprints.emb_dir / s.name).glob("*.npz"):
                if old.name not in keep:
                    old.unlink(missing_ok=True)
        await asyncio.to_thread(self.voiceprints.build)
        name = s.name
        self.session = None
        self.last_message = f"complete {name}"
        _LOGGER.info("Enrolment complete; voiceprints now: %d", len(self.voiceprints.prints))
        await self._push()
        return f"voice enrolment complete {name}"
