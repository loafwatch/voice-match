#!/usr/bin/env python3
"""Fetch the models voice-match uses, pinned by SHA-256.

- Silero VAD (MIT): from the silero-vad wheel on PyPI (only the ONNX model
  and its licence are kept).
- WeSpeaker ResNet34-LM speaker embedding model, trained on VoxCeleb
  (CC BY 4.0): the ONNX export published with sherpa-onnx's releases.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "6.3"
WHEEL_SHA256 = "4ac390372ab796e13051684f3f61bb87e2ee49ed947626285f4d55a19e6a3b90"
DEST = Path(__file__).resolve().parent.parent / "voice_match" / "models"

SPEAKER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)
SPEAKER_SHA256 = "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"
SPEAKER_NOTICE = """WeSpeaker ResNet34-LM speaker embedding model
Trained on VoxCeleb by the WeSpeaker project (https://github.com/wenet-e2e/wespeaker).
Model licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
following the VoxCeleb dataset licence (https://mm.kaist.ac.kr/datasets/voxceleb/).
ONNX file as published by sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx).
Used unmodified.
"""


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    fetch_silero()
    fetch_speaker()


def fetch_speaker() -> None:
    target = DEST / "wespeaker_resnet34_lm.onnx"
    if target.exists():
        print("speaker model already present")
        return
    with urllib.request.urlopen(SPEAKER_URL, timeout=120) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SPEAKER_SHA256:
        sys.exit(f"Checksum mismatch for speaker model: {digest}")
    target.write_bytes(data)
    (DEST / "WESPEAKER_NOTICE").write_text(SPEAKER_NOTICE)
    print("Fetched WeSpeaker ResNet34-LM")


def fetch_silero() -> None:
    if (DEST / "silero_vad.onnx").exists():
        print("silero_vad.onnx already present")
        return
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
