"""Silero VAD (MIT) run directly with onnxruntime - no PyTorch.

The model scores 32 ms frames (512 samples at 16 kHz) with a probability of
speech. Each frame is fed with the previous 64 samples as context and a
recurrent state, exactly as Silero's own ONNX wrapper does.

Audio is scored as it streams in, so by the time the command ends almost all
of the work is already done.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime

SAMPLE_RATE = 16000
FRAME = 512  # samples per frame (32 ms)
CONTEXT = 64  # samples of context prepended to each frame
FRAME_MS = FRAME * 1000 / SAMPLE_RATE

DEFAULT_MODEL = Path(__file__).parent / "models" / "silero_vad.onnx"


class SileroVad:
    """Loads the model once; hands out a stream per utterance."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        # Privacy: never let onnxruntime emit telemetry events.
        if hasattr(onnxruntime, "disable_telemetry_events"):
            onnxruntime.disable_telemetry_events()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def stream(self) -> "VadStream":
        return VadStream(self)

    def run_frame(self, frame: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        out, state = self.session.run(
            None, {"input": frame[np.newaxis, :], "state": state, "sr": self._sr}
        )
        return float(out[0][0]), state


class VadStream:
    """Scores 16 kHz mono 16-bit PCM incrementally."""

    def __init__(self, vad: SileroVad) -> None:
        self.vad = vad
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.probs: list[float] = []
        self.compute_seconds = 0.0

    def feed(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        self._drain()

    def finish(self) -> list[float]:
        if len(self.pending):
            pad = np.zeros(FRAME - len(self.pending), dtype=np.float32)
            self.pending = np.concatenate([self.pending, pad])
            self._drain()
        return self.probs

    def _drain(self) -> None:
        started = time.perf_counter()
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            prob, self.state = self.vad.run_frame(
                np.concatenate([self.context, frame]), self.state
            )
            self.context = frame[-CONTEXT:]
            self.probs.append(prob)
        self.compute_seconds += time.perf_counter() - started
