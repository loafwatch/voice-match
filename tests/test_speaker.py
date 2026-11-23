"""Speaker tracking logic (with a fake embedding model) and the real model."""

from __future__ import annotations

import numpy as np
import pytest

from voice_match.speaker import DEFAULT_MODEL, SpeakerSettings, SpeakerTracker
from voice_match.vad import FRAME

SR = 16000
A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])
TV = np.array([0.0, 0.0, 1.0])


class FakeModel:
    """Embedding = which 'voice' fills most of the window, encoded in the samples."""

    def embed(self, samples):
        codes = np.round(samples * 10).astype(int)
        counts = [np.sum(codes == k) for k in (1, 2, 3)]
        return [A, B, TV][int(np.argmax(counts))]


def audio_for(spans, total_s):
    """spans: (start_s, end_s, voice 1|2|3); TV (3) everywhere else."""
    x = np.full(int(total_s * SR), 0.3, np.float32)
    for a, b, v in spans:
        x[int(a * SR) : int(b * SR)] = v / 10
    return x


def track(x, prints=None, step=1024):
    prints = prints or {"a": A, "b": B}
    t = SpeakerTracker(FakeModel(), prints, SpeakerSettings())
    probs = [0.9] * (len(x) // FRAME + 1)  # all speech (TV never stops)
    for end in range(step, len(x) + step, step):
        t.update(x[:end], probs)
    return t


def test_finds_command_inside_continuous_tv():
    t = track(audio_for([(2.0, 4.0, 1)], 8.0))
    start, end, who, _ = t.result()
    assert who == "a"
    assert 1.0 <= start / SR <= 2.0  # includes 250 ms padding
    assert 4.0 <= end / SR <= 5.0


def test_ends_once_voice_stops_even_with_tv():
    x = audio_for([(2.0, 4.0, 1)], 8.0)
    t = track(x[: int(4.6 * SR)])
    assert not t.ended()  # 0.6 s after: maybe just a pause
    t = track(x[: int(5.8 * SR)])  # > 1 s since the voice was last matched
    assert t.ended()


def test_identifies_second_person():
    _, _, who, _ = track(audio_for([(1.0, 3.0, 2)], 6.0)).result()
    assert who == "b"


def test_tv_only_finds_nobody():
    t = track(audio_for([], 6.0))
    assert t.result() is None
    assert not t.ended()


def test_silent_windows_are_not_scored():
    t = SpeakerTracker(FakeModel(), {"a": A}, SpeakerSettings())
    x = audio_for([(0.0, 3.0, 1)], 3.0)
    t.update(x, [0.05] * (len(x) // FRAME + 1))  # VAD: no speech
    assert all(w.speaker is None for w in t.windows)


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="run scripts/fetch_models.py")
def test_real_model_embedding_shape_and_norm():
    from voice_match.speaker import SpeakerModel

    rng = np.random.default_rng(0)
    e = SpeakerModel().embed((rng.normal(0, 0.1, SR)).astype(np.float32))
    assert e.shape == (256,)
    assert abs(np.linalg.norm(e) - 1) < 1e-4
