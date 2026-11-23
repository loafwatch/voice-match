#!/usr/bin/env python3
"""Fetch the Silero VAD ONNX model (MIT) from its pinned PyPI release.

Downloads the silero-vad wheel without dependencies, checks its SHA-256 and
extracts only the ONNX model (and its licence) into voice_match/models/.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

VERSION = "6.3"
WHEEL_SHA256 = "4ac390372ab796e13051684f3f61bb87e2ee49ed947626285f4d55a19e6a3b90"
DEST = Path(__file__).resolve().parent.parent / "voice_match" / "models"


def main() -> None:
    if (DEST / "silero_vad.onnx").exists():
        print("silero_vad.onnx already present")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--quiet",
             f"silero-vad=={VERSION}", "-d", tmp],
            check=True,
        )
        wheel = next(Path(tmp).glob("silero_vad-*.whl"))
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest != WHEEL_SHA256:
            sys.exit(f"Checksum mismatch for {wheel.name}: {digest}")
        with zipfile.ZipFile(wheel) as zf:
            (DEST / "silero_vad.onnx").write_bytes(zf.read("silero_vad/data/silero_vad.onnx"))
            lic = [n for n in zf.namelist() if n.endswith("LICENSE")]
            if lic:
                (DEST / "SILERO_VAD_LICENSE").write_bytes(zf.read(lic[0]))
    print(f"Fetched Silero VAD {VERSION} into {DEST}")


if __name__ == "__main__":
    main()
