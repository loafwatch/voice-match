"""Voiceprints are stored as numbers only; legacy recordings are converted then deleted."""

from __future__ import annotations

import wave

import numpy as np

from voice_match.speaker import Voiceprints

SR = 16000


class FakeModel:
    def embed(self, samples):
        rng = np.random.default_rng(int(abs(samples).sum() * 1000) % 2**32)
        e = rng.normal(size=256)
        return e / np.linalg.norm(e)


def write_wav(path, seconds=1.0, amp=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.full(int(SR * seconds), amp, "<i2")).tobytes())


def test_legacy_recordings_converted_then_only_they_are_deleted(tmp_path):
    enrol = tmp_path / "enrol" / "speaker_a"
    write_wav(enrol / "20260101-000001.wav")
    write_wav(enrol / "20260101-000002.wav", amp=2000)
    (enrol / "notes.txt").write_text("keep me")
    (enrol / "keep.json").write_text("{}")
    (enrol / "broken.wav").write_bytes(b"not a wav")  # unreadable: must be kept
    write_wav(enrol / "short.wav", seconds=0.2)  # too short to use: must be kept
    write_wav(tmp_path / "elsewhere.wav")  # not in an enrolment folder: must be kept

    vp = Voiceprints(FakeModel(), tmp_path)
    vp.build()

    remaining = sorted(p.name for p in enrol.iterdir())
    assert remaining == ["broken.wav", "keep.json", "notes.txt", "short.wav"]
    assert (tmp_path / "elsewhere.wav").exists()
    assert enrol.is_dir()  # folder itself left alone
    saved = sorted(p.name for p in (tmp_path / "embeddings" / "speaker_a").iterdir())
    assert saved == ["20260101-000001.npz", "20260101-000002.npz"]
    assert "speaker_a" in vp.prints
    assert vp.prints["speaker_a"].shape == (256,)


def test_add_stores_no_audio(tmp_path):
    vp = Voiceprints(FakeModel(), tmp_path)
    path = vp.add("speaker_b", np.full(SR, 0.1, np.float32), stamp="x")
    with np.load(path) as data:
        assert set(data.files) == {"embedding", "seconds"}
        assert data["embedding"].shape == (256,)
    assert not list(tmp_path.rglob("*.wav"))


def test_voiceprint_rebuilds_when_new_embeddings_arrive(tmp_path):
    vp = Voiceprints(FakeModel(), tmp_path)
    vp.add("speaker_a", np.full(SR, 0.1, np.float32), stamp="1")
    vp.build()
    first = vp.prints["speaker_a"].copy()
    import os, time

    later = vp.add("speaker_a", np.full(SR, 0.3, np.float32), stamp="2")
    t = time.time() + 5
    os.utime(later, (t, t))
    vp.build()
    assert not np.allclose(first, vp.prints["speaker_a"])
