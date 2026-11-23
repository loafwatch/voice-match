"""Decide which part of an utterance to keep, from per-frame speech probabilities.

Modes:
  off    keep everything
  edges  cut non-speech before the first and after the last speech
  first  keep only the first burst of speech: speech after a silence longer
         than `gap_ms` (typically the TV carrying on after the command) is cut

If no speech is found the audio is kept whole, so nothing is ever thrown away
just because the VAD missed it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vad import FRAME, FRAME_MS


@dataclass(frozen=True)
class TrimSettings:
    mode: str = "first"
    threshold: float = 0.5
    min_speech_ms: float = 120
    min_silence_ms: float = 250
    gap_ms: float = 800
    pad_before_ms: float = 200
    pad_after_ms: float = 300


@dataclass(frozen=True)
class TrimResult:
    start: int  # first sample kept
    end: int  # one past the last sample kept
    segments: list[tuple[int, int]]  # speech segments, in samples
    reason: str


def speech_segments(probs: list[float], s: TrimSettings) -> list[tuple[int, int]]:
    """Frames -> speech segments (in frames), with hysteresis and merging."""
    on, off = s.threshold, max(s.threshold - 0.15, 0.01)
    raw: list[tuple[int, int]] = []
    start = None
    for i, p in enumerate(probs):
        if start is None and p >= on:
            start = i
        elif start is not None and p < off:
            raw.append((start, i))
            start = None
    if start is not None:
        raw.append((start, len(probs)))

    min_silence = s.min_silence_ms / FRAME_MS
    merged: list[tuple[int, int]] = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] < min_silence:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    min_speech = s.min_speech_ms / FRAME_MS
    return [seg for seg in merged if seg[1] - seg[0] >= min_speech]


def decide(probs: list[float], total_samples: int, s: TrimSettings) -> TrimResult:
    if s.mode == "off":
        return TrimResult(0, total_samples, [], "off")

    segs = speech_segments(probs, s)
    seg_samples = [(a * FRAME, min(b * FRAME, total_samples)) for a, b in segs]
    if not segs:
        return TrimResult(0, total_samples, [], "no speech found, kept all")

    keep = segs
    reason = "edges"
    if s.mode == "first":
        gap = s.gap_ms / FRAME_MS
        keep = [segs[0]]
        for seg in segs[1:]:
            if seg[0] - keep[-1][1] > gap:
                break
            keep.append(seg)
        reason = f"first ({len(keep)}/{len(segs)} segments)"

    pad_before = int(s.pad_before_ms / 1000 * 16000)
    pad_after = int(s.pad_after_ms / 1000 * 16000)
    start = max(0, keep[0][0] * FRAME - pad_before)
    end = min(total_samples, keep[-1][1] * FRAME + pad_after)
    return TrimResult(start, end, seg_samples, reason)


def command_ended(probs: list[float], s: TrimSettings) -> bool:
    """True once the first burst of speech is followed by a silence > gap_ms.

    Only in "first" mode: anything said after that gap would be trimmed off
    anyway, so there is no point waiting for the room to go quiet.
    """
    if s.mode != "first":
        return False
    segs = speech_segments(probs, s)
    if not segs:
        return False
    trailing = len(probs) - segs[-1][1]
    return trailing > s.gap_ms / FRAME_MS
