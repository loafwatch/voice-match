"""Speaker recognition: voiceprints and per-window matching.

Model: WeSpeaker ResNet34-LM trained on VoxCeleb (CC BY 4.0, see
models/WESPEAKER_NOTICE), run with onnxruntime on the CPU. Input is 80-band
Kaldi filterbank features (25 ms / 10 ms, no dither, mean-normalised);
output is a 256-number embedding. Two embeddings from the same voice point in
a similar direction (cosine similarity near 1); different voices score low.

During a command, a 1 s window is scored every 250 ms against each enrolled
voiceprint. That finds the household member's speech even when a TV is
talking the whole time.
"""

from __future__ import annotations

import json
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime

from .vad import FRAME, SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = Path(__file__).parent / "models" / "wespeaker_resnet34_lm.onnx"


@dataclass(frozen=True)
class SpeakerSettings:
    threshold: float = 0.45  # cosine similarity needed to count as a match
    window_ms: float = 1000
    hop_ms: float = 250
    gap_ms: float = 1000  # no match for this long after a match = command over
    min_speech: float = 0.3  # fraction of a window VAD must call speech to score it
    pad_ms: float = 250  # extra audio kept either side of the matched windows


class SpeakerModel:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        if hasattr(onnxruntime, "disable_telemetry_events"):
            onnxruntime.disable_telemetry_events()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = onnxruntime.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._fbank_opts = knf.FbankOptions()
        self._fbank_opts.frame_opts.dither = 0
        self._fbank_opts.frame_opts.samp_freq = SAMPLE_RATE
        self._fbank_opts.frame_opts.frame_length_ms = 25
        self._fbank_opts.frame_opts.frame_shift_ms = 10
        self._fbank_opts.mel_opts.num_bins = 80
        self._fbank_opts.energy_floor = 0.0

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Float samples in [-1, 1] -> unit-length 256-d embedding."""
        fb = knf.OnlineFbank(self._fbank_opts)
        fb.accept_waveform(SAMPLE_RATE, (samples * 32768.0).astype(np.float32))
        fb.input_finished()
        feats = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)
        feats -= feats.mean(axis=0)
        emb = self.session.run(None, {"feats": feats[np.newaxis]})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) != (SAMPLE_RATE, 2, 1):
            raise ValueError(f"{path.name}: need 16 kHz, 16-bit, mono")
        return np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


class Voiceprints:
    """Enrolled voices: data_dir/enrol/<name>/*.wav -> data_dir/voiceprints/<name>.npy"""

    def __init__(self, model: SpeakerModel, data_dir: str | Path) -> None:
        self.model = model
        self.enrol_dir = Path(data_dir) / "enrol"
        self.print_dir = Path(data_dir) / "voiceprints"
        self.prints: dict[str, np.ndarray] = {}

    def build(self) -> None:
        """(Re)build voiceprints from enrolment recordings, then load them."""
        self.print_dir.mkdir(parents=True, exist_ok=True)
        if self.enrol_dir.is_dir():
            for person in sorted(p for p in self.enrol_dir.iterdir() if p.is_dir()):
                wavs = sorted(person.glob("*.wav"))
                if not wavs:
                    continue
                target = self.print_dir / f"{person.name}.npy"
                newest = max(w.stat().st_mtime for w in wavs)
                if target.exists() and target.stat().st_mtime >= newest:
                    continue
                self._enrol(person.name, wavs, target)
        for npy in sorted(self.print_dir.glob("*.npy")):
            self.prints[npy.stem] = np.load(npy)
        _LOGGER.info("Voiceprints loaded: %d", len(self.prints))

    def _enrol(self, name: str, wavs: list[Path], target: Path) -> None:
        embs, seconds = [], 0.0
        per_file: list[np.ndarray] = []
        for wav in wavs:
            try:
                audio = read_wav(wav)
            except (OSError, ValueError, EOFError) as err:
                _LOGGER.warning("Skipping %s: %s", wav.name, err)
                continue
            if len(audio) < SAMPLE_RATE // 2:
                continue
            seconds += len(audio) / SAMPLE_RATE
            e = self.model.embed(audio)
            per_file.append(e)
            embs.append(e)
        if not embs:
            _LOGGER.warning("No usable enrolment audio for one voice")
            return
        vp = np.mean(embs, axis=0)
        vp /= np.linalg.norm(vp)
        np.save(target, vp)
        sims = [float(e @ vp) for e in per_file]
        stats = {
            "recordings": len(per_file),
            "seconds": round(seconds, 1),
            "self_similarity_min": round(min(sims), 3),
            "self_similarity_mean": round(float(np.mean(sims)), 3),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        target.with_suffix(".json").write_text(json.dumps(stats, indent=2))
        _LOGGER.info("Built voiceprint %s: %s", name, stats)


@dataclass
class Window:
    end: int  # sample index where the window ends
    speaker: Optional[str]
    score: float


class SpeakerTracker:
    """Scores sliding windows of one utterance as it streams in."""

    def __init__(self, model: SpeakerModel, prints: dict[str, np.ndarray], s: SpeakerSettings) -> None:
        self.model = model
        self.names = list(prints)
        self.matrix = np.stack([prints[n] for n in self.names])
        self.s = s
        self.window = int(s.window_ms / 1000 * SAMPLE_RATE)
        self.hop = int(s.hop_ms / 1000 * SAMPLE_RATE)
        self.next_end = self.window
        self.windows: list[Window] = []
        self.compute_seconds = 0.0

    def update(self, audio: np.ndarray, vad_probs: list[float]) -> None:
        """Score every window whose end has arrived. `audio` = all samples so far."""
        while len(audio) >= self.next_end:
            start, end = self.next_end - self.window, self.next_end
            self.next_end += self.hop
            frames = vad_probs[start // FRAME : end // FRAME]
            if not frames or np.mean(np.array(frames) >= 0.5) < self.s.min_speech:
                self.windows.append(Window(end, None, 0.0))
                continue
            t = time.perf_counter()
            scores = self.matrix @ self.model.embed(audio[start:end])
            self.compute_seconds += time.perf_counter() - t
            best = int(np.argmax(scores))
            self.windows.append(Window(end, self.names[best], float(scores[best])))

    # ----- decisions -----

    def _matches(self) -> list[Window]:
        return [w for w in self.windows if w.speaker and w.score >= self.s.threshold]

    def first_run(self) -> list[Window]:
        """Matching windows of the first command (later matches after a long gap dropped)."""
        matches = self._matches()
        if not matches:
            return []
        gap = int(self.s.gap_ms / 1000 * SAMPLE_RATE)
        run = [matches[0]]
        for w in matches[1:]:
            if w.end - run[-1].end > gap:
                break
            run.append(w)
        return run

    def ended(self) -> bool:
        """A household voice spoke and has not been heard for gap_ms."""
        run = self.first_run()
        if not run or not self.windows:
            return False
        return self.windows[-1].end - run[-1].end > int(self.s.gap_ms / 1000 * SAMPLE_RATE)

    def result(self) -> Optional[tuple[int, int, str, float]]:
        """(start sample, end sample, speaker, mean score) of the command, or None."""
        run = self.first_run()
        if not run:
            return None
        votes: dict[str, list[float]] = {}
        for w in run:
            votes.setdefault(w.speaker, []).append(w.score)
        speaker = max(votes, key=lambda n: (len(votes[n]), np.mean(votes[n])))
        pad = int(self.s.pad_ms / 1000 * SAMPLE_RATE)
        start = max(0, run[0].end - self.window - pad)
        return start, run[-1].end + pad, speaker, float(np.mean(votes[speaker]))
