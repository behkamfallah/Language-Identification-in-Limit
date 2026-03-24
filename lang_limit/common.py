"""Shared constants, types, and small helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal

Language = Callable[[int], bool]
AdversaryMode = Literal["noise_then_progression", "delay_by_sparse_subset"]
OrderingStrategy = Literal["adversarial", "complexity", "random"]

SUPPORTED_MODES: tuple[AdversaryMode, ...] = (
    "noise_then_progression",
    "delay_by_sparse_subset",
)
SUPPORTED_ORDERINGS: tuple[OrderingStrategy, ...] = (
    "adversarial",
    "complexity",
    "random",
)

START = "^"
END = "$"
BIN_DIGITS = "01"


def integer_code_length(value: int) -> int:
    """Return a simple universal-code-style bit length for a positive integer."""
    if value < 0:
        raise ValueError("value must be non-negative")
    return max(1, value.bit_length())


def bits_for_max_value(max_value: int) -> int:
    """Return the fixed binary width for values in ``[1, max_value]``."""
    if max_value < 1:
        raise ValueError("max_value must be positive")
    return max(1, max_value.bit_length())


def encode_int_bin(value: int, bits: int) -> str:
    """Encode an integer in zero-padded binary with fixed width."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if bits < 1:
        raise ValueError("bits must be positive")
    return format(value, "b").zfill(bits)


def normalize_distribution(dist: Mapping[str, float]) -> dict[str, float]:
    """Normalize a non-negative mapping into a probability distribution."""
    total = sum(dist.values())
    if total <= 0:
        size = len(dist)
        if size == 0:
            return {}
        return {key: 1.0 / size for key in dist}
    return {key: value / total for key, value in dist.items()}


def kl_tv_over_shared_support(
    p_dist: Mapping[str, float],
    q_dist: Mapping[str, float],
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Compute KL divergence and total variation over a shared support."""
    if p_dist.keys() != q_dist.keys():
        raise ValueError("p_dist and q_dist must have identical supports")

    kl = 0.0
    tv = 0.0
    for key in p_dist:
        p = max(p_dist[key], eps)
        q = max(q_dist[key], eps)
        kl += p * math.log(p / q)
        tv += abs(p_dist[key] - q_dist[key])
    return kl, 0.5 * tv
