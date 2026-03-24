"""Metrics and post-processing helpers for experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .common import Language


@dataclass(frozen=True)
class GenerationTrace:
    """Per-step validity and diversity traces for a generator run."""

    outputs: tuple[int, ...]
    valid_flags: tuple[bool, ...]
    cumulative_valid_rate: tuple[float, ...]
    cumulative_unique_outputs: tuple[int, ...]


@dataclass(frozen=True)
class GenerationMetrics:
    """Generation validity summary over a finite horizon."""

    valid_rate: float
    last_fail_t: int
    t_star_est: float
    unique_outputs: int
    early_unique_outputs: int


@dataclass(frozen=True)
class DiscriminatorMetrics:
    """Classifier two-sample summary."""

    accuracy: float | None
    support_max: int
    sample_size: int


@dataclass(frozen=True)
class TextBigramMetrics:
    """Bigram language-model summary for the observed text."""

    kl: float
    tv: float
    support_size: int


def build_generation_trace(
    stream: Sequence[int],
    outputs: Sequence[int],
    target: Language,
) -> GenerationTrace:
    """Build per-step validity and diversity traces."""
    if len(stream) != len(outputs):
        raise ValueError("stream and outputs must have the same length")

    seen: set[int] = set()
    generated_seen: set[int] = set()
    valid_flags: list[bool] = []
    cumulative_valid_rate: list[float] = []
    cumulative_unique_outputs: list[int] = []

    valid_count = 0
    for step, (observed, output) in enumerate(zip(stream, outputs), start=1):
        # By design, an output is valid only if it belongs to the target and is
        # still unseen after incorporating the current observation.
        seen.add(observed)
        is_valid = target(output) and output not in seen
        valid_flags.append(is_valid)
        valid_count += int(is_valid)
        cumulative_valid_rate.append(valid_count / step)

        # We track diversity over the raw generator outputs, not just valid ones,
        # because the paper-facing argument is partly about exploratory breadth.
        generated_seen.add(output)
        cumulative_unique_outputs.append(len(generated_seen))

    return GenerationTrace(
        outputs=tuple(outputs),
        valid_flags=tuple(valid_flags),
        cumulative_valid_rate=tuple(cumulative_valid_rate),
        cumulative_unique_outputs=tuple(cumulative_unique_outputs),
    )


def summarize_generation(trace: GenerationTrace, early_window: int = 20) -> GenerationMetrics:
    """Collapse a generation trace into headline summary metrics."""
    valid_flags = trace.valid_flags
    last_fail = max((index for index, ok in enumerate(valid_flags, start=1) if not ok), default=0)
    t_star_est = (last_fail + 1) if last_fail < len(valid_flags) else float("inf")
    # The "early diversity" number is useful for contrasting KM-style collapse
    # against more exploratory heuristics.
    early_unique_outputs = trace.cumulative_unique_outputs[min(early_window, len(trace.outputs)) - 1] if trace.outputs else 0

    return GenerationMetrics(
        valid_rate=trace.cumulative_valid_rate[-1] if trace.cumulative_valid_rate else 0.0,
        last_fail_t=last_fail,
        t_star_est=float(t_star_est),
        unique_outputs=trace.cumulative_unique_outputs[-1] if trace.cumulative_unique_outputs else 0,
        early_unique_outputs=early_unique_outputs,
    )


def evaluate_generation(
    stream: Sequence[int],
    outputs: Sequence[int],
    target: Language,
    early_window: int = 20,
) -> GenerationMetrics:
    """Score validity and diversity of generated outputs."""
    return summarize_generation(build_generation_trace(stream, outputs, target), early_window=early_window)


def collect_valid_outputs_after_burn(
    stream: Sequence[int],
    outputs: Sequence[int],
    target: Language,
    burn: int,
) -> list[int]:
    """Collect valid generated outputs from the second half of the run."""
    if len(stream) != len(outputs):
        raise ValueError("stream and outputs must have the same length")

    seen = set(stream[:burn])
    valid_outputs: list[int] = []

    for index in range(burn, len(stream)):
        # The discriminator is meant to judge late-stage behavior, so we only
        # keep valid outputs after a configurable burn-in period.
        seen.add(stream[index])
        output = outputs[index]
        if target(output) and output not in seen:
            valid_outputs.append(output)

    return valid_outputs
