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


class Voiceprints:
    """Enrolled voices, stored as numbers only - never as audio.

    data_dir/embeddings/<name>/*.npz   one 256-number embedding per enrolment
                                       command (plus its length in seconds)
    data_dir/voiceprints/<name>.npy    the person's voiceprint (average)
    data_dir/voiceprints/<name>.json   quality stats

    An embedding describes how a voice sounds; it cannot be turned back into
    speech. Enrolment audio is converted as soon as it is captured and never
    written to disk. Any legacy enrolment recordings (data_dir/enrol/<name>/
    *.wav) are converted at start-up and then deleted - only those .wav
    files, and only after their embedding has been saved and read back.
    """

    def __init__(self, model: SpeakerModel, data_dir: str | Path) -> None:
        self.model = model
        self.data_dir = Path(data_dir)
        self.legacy_dir = self.data_dir / "enrol"
        self.emb_dir = self.data_dir / "embeddings"
        self.print_dir = self.data_dir / "voiceprints"
        self.prints: dict[str, np.ndarray] = {}

    # ----- enrolment -----

    def add(self, name: str, audio: np.ndarray, stamp: Optional[str] = None) -> Optional[Path]:
        """Embed one enrolment command's speech and store only the embedding."""
        if len(audio) < SAMPLE_RATE // 2:
            return None
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        path = self.emb_dir / name / f"{stamp}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embedding=self.model.embed(audio), seconds=len(audio) / SAMPLE_RATE)
        return path

    def convert_legacy_recordings(self) -> int:
        """Turn legacy enrolment .wav files into embeddings, then delete the .wav files."""
        if not self.legacy_dir.is_dir():
            return 0
        deleted = 0
        for person in sorted(p for p in self.legacy_dir.iterdir() if p.is_dir()):
            for wav in sorted(person.glob("*.wav")):
                if not wav.is_file() or wav.is_symlink():
                    continue
                try:
                    audio = read_wav(wav)
                except (OSError, ValueError, EOFError, wave.Error) as err:
                    _LOGGER.warning("Not converting %s (left in place): %s", wav.name, err)
                    continue
                saved = self.add(person.name, audio, stamp=wav.stem)
                if saved is None:
                    _LOGGER.warning("Not converting %s (too short; left in place)", wav.name)
                    continue
                try:
                    check = np.load(saved)["embedding"]
                    ok = check.shape == (256,) and np.isfinite(check).all()
                except (OSError, ValueError, KeyError):
                    ok = False
                if not ok:
                    _LOGGER.warning("Embedding for %s did not verify; recording kept", wav.name)
                    continue
                wav.unlink()
                deleted += 1
        if deleted:
            _LOGGER.info("Converted and deleted %d enrolment recording(s); no audio is kept", deleted)
        return deleted

    # ----- voiceprints -----

    def build(self) -> None:
        """Convert any legacy recordings, (re)build voiceprints, then load them."""
        self.convert_legacy_recordings()
        self.print_dir.mkdir(parents=True, exist_ok=True)
        if self.emb_dir.is_dir():
            for person in sorted(p for p in self.emb_dir.iterdir() if p.is_dir()):
                files = sorted(person.glob("*.npz"))
                if not files:
                    continue
                target = self.print_dir / f"{person.name}.npy"
                newest = max(f.stat().st_mtime for f in files)
                if target.exists() and target.stat().st_mtime >= newest:
                    continue
                self._build_one(person.name, files, target)
        self.prints.clear()
        for npy in sorted(self.print_dir.glob("*.npy")):
            self.prints[npy.stem] = np.load(npy)
        _LOGGER.info("Voiceprints loaded: %d", len(self.prints))

    def _build_one(self, name: str, files: list[Path], target: Path) -> None:
        embs, seconds = [], 0.0
        for f in files:
            try:
                with np.load(f) as data:
                    embs.append(data["embedding"])
                    seconds += float(data["seconds"])
            except (OSError, ValueError, KeyError) as err:
                _LOGGER.warning("Skipping embedding %s: %s", f.name, err)
        if not embs:
            return
        vp = np.mean(embs, axis=0)
        vp /= np.linalg.norm(vp)
        np.save(target, vp)
        sims = [float(e @ vp) for e in embs]
        stats = {
            "recordings": len(embs),
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
