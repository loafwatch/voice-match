"""Optional: save each command's audio before and after trimming, for tuning.

Off unless VM_DEBUG_AUDIO_DIR is set. Files stay on the server, and only the
most recent `keep` commands are kept. Never commit these files.
"""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def save(directory: str, keep: int, rate: int, width: int, channels: int,
         original: bytes, trimmed: bytes) -> None:
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        for suffix, data in (("in", original), ("out", trimmed)):
            with wave.open(str(path / f"{stamp}_{suffix}.wav"), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(width)
                wav.setframerate(rate)
                wav.writeframes(data)
        files = sorted(path.glob("*_in.wav"))
        for old in files[: max(0, len(files) - keep)]:
            old.unlink(missing_ok=True)
            old.with_name(old.name.replace("_in.wav", "_out.wav")).unlink(missing_ok=True)
    except OSError as err:
        _LOGGER.warning("Could not save debug audio: %s", err)
