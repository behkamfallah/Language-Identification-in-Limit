"""Adversarial text and enumeration constructions."""

from __future__ import annotations

from .common import AdversaryMode
from .languages import TargetSpec


def adversary_stream(target: TargetSpec, horizon: int, mode: AdversaryMode) -> list[int]:
    """Produce a distinct sample stream from the target language."""
    if horizon <= 0:
        return []

    seen: set[int] = set()
    stream: list[int] = []

    def emit(candidate: int) -> None:
        if candidate >= 1 and candidate not in seen:
            stream.append(candidate)
            seen.add(candidate)

    if mode == "noise_then_progression":
        # Front-load the finite noise set so the progression structure is hidden
        # for a few steps.
        for value in sorted(target.noise):
            if len(stream) >= horizon:
                return stream
            emit(value)

        step = 0
        while len(stream) < horizon:
            emit(target.a + target.b * step)
            step += 1
        return stream

    if mode == "delay_by_sparse_subset":
        # Show a very sparse arithmetic subsequence first, then reveal the noise,
        # and only later fill in the full progression.
        sparse_stride = 4 * target.b
        step = 0
        while len(stream) < horizon // 2:
            emit(target.a + sparse_stride * step)
            step += 1

        for value in sorted(target.noise):
            if len(stream) >= horizon:
                return stream
            emit(value)

        step = 0
        while len(stream) < horizon:
            emit(target.a + target.b * step)
            step += 1
        return stream

    raise ValueError(f"unknown adversary mode: {mode}")
