"""Generation strategies used in the experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .languages import NamedLanguage, is_consistent, smallest_unseen_member


def first_consistent_generator_step(
    t: int,
    samples: set[int],
    languages: Sequence[NamedLanguage],
) -> tuple[int, int]:
    """Pick the first consistent language and emit its next unseen member."""
    # This baseline intentionally commits to the earliest consistent language,
    # which means it gets trapped forever by large supersets such as the universe.
    for index, language in enumerate(languages[: min(t, len(languages))], start=1):
        if is_consistent(samples, language):
            return smallest_unseen_member(language, samples), index
    return 1, 0


def two_largest_diff_generator_step(samples: set[int]) -> int:
    """Extrapolate an arithmetic step from the two largest observed samples."""
    if len(samples) < 2:
        return 1

    second_largest, largest = sorted(samples)[-2:]
    # This heuristic reads the last observed gap as if it were the true step size.
    # It is intentionally simple and can overfit badly when the stream is adversarial.
    step = max(1, largest - second_largest)
    multiplier = 2
    while True:
        candidate = second_largest + multiplier * step
        if candidate not in samples:
            return candidate
        multiplier += 1


def _count_members_up_to(language: NamedLanguage, support_max: int) -> int:
    """Count candidate members inside the finite support window ``[1, support_max]``."""
    return sum(1 for value in range(1, support_max + 1) if language(value))


def mdl_generator_step(
    samples: set[int],
    languages: Sequence[NamedLanguage],
    support_max: int,
) -> tuple[int, int, float]:
    """Choose the consistent language with minimum two-part MDL score."""
    best_output = 1
    best_index = 0
    best_score = float("inf")
    best_tiebreak = (float("inf"), float("inf"), float("inf"))

    # Unlike the KM construction, the MDL baseline scores the whole finite
    # candidate set available in the experiment rather than the prefix L_1..L_t.
    for index, language in enumerate(languages, start=1):
        if not is_consistent(samples, language):
            continue

        support_size = _count_members_up_to(language, support_max)
        if support_size <= 0:
            continue

        # Model cost prefers simpler hypotheses; data cost prefers hypotheses
        # that concentrate mass on the observed positives inside the same window.
        model_bits = float(language.description_bits)
        data_bits = len(samples) * math.log2(support_size)
        score = model_bits + data_bits
        tiebreak = (score, model_bits, float(support_size))
        if tiebreak < best_tiebreak:
            best_output = smallest_unseen_member(language, samples)
            best_index = index
            best_score = score
            best_tiebreak = tiebreak

    return best_output, best_index, best_score


@dataclass
class KMState:
    """Running state for the KM-style generator."""

    mt: int = 0


@dataclass
class MDLState:
    """Running state for the MDL baseline."""

    support_floor: int = 64
    support_multiplier: int = 2

    def support_max(self, samples: set[int]) -> int:
        """Return the current finite support window used for scoring."""
        if not samples:
            return self.support_floor
        return max(self.support_floor, self.support_multiplier * max(samples))


def km_generate_step(
    t: int,
    observed: int,
    samples: set[int],
    languages: Sequence[NamedLanguage],
    state: KMState,
) -> tuple[int, int, int]:
    """Run one KM-style generation step using membership queries."""
    # KM only reasons over the candidate prefix L_1..L_t available by time t.
    candidate_languages = list(languages[: min(t, len(languages))])
    t_eff = len(candidate_languages)

    if not candidate_languages:
        state.mt = max(state.mt, observed)
        return 1, 0, state.mt

    m0 = max(state.mt, observed)

    consistent = [False] * (t_eff + 1)  # 1-indexed
    for index, language in enumerate(candidate_languages, start=1):
        consistent[index] = is_consistent(samples, language)

    if not any(consistent[1:]):
        state.mt = m0
        return 1, 0, state.mt

    # Cache prefix membership tables so the critical-language test can reuse them.
    membership = [[False] * (m0 + 1) for _ in range(t_eff + 1)]
    for index, language in enumerate(candidate_languages, start=1):
        for value in range(1, m0 + 1):
            membership[index][value] = language(value)

    m = m0
    while True:
        m += 1
        # The algorithm grows a shared prefix length m until some critical
        # language exposes a fresh unseen element inside that prefix.
        for index, language in enumerate(candidate_languages, start=1):
            membership[index].append(language(m))

        critical_indices: list[int] = []
        for n in range(1, t_eff + 1):
            if not consistent[n]:
                continue

            row_n = membership[n]
            is_critical = True
            for index in range(1, n + 1):
                if not consistent[index]:
                    continue

                row_i = membership[index]
                # A critical language stays inside every earlier consistent language on [1..m].
                if any(row_n[value] and not row_i[value] for value in range(1, m + 1)):
                    is_critical = False
                    break

            if is_critical:
                critical_indices.append(n)

        # We follow the KM convention of choosing the highest-indexed critical
        # language, then emitting its first unseen member inside the current
        # prefix window.
        chosen_nt = max(critical_indices)
        row_nt = membership[chosen_nt]
        for value in range(1, m + 1):
            if row_nt[value] and value not in samples:
                state.mt = m
                return value, chosen_nt, state.mt


class StepGenerator(Protocol):
    """Stateful generator interface used by the runner."""

    name: str

    def step(self, t: int, observed: int, samples: set[int]) -> int:
        """Emit the next generated value."""


@dataclass
class FirstConsistentGenerator:
    """Thin wrapper around the first-consistent baseline."""

    languages: Sequence[NamedLanguage]
    name: str = "first_consistent"

    def step(self, t: int, observed: int, samples: set[int]) -> int:
        output, _ = first_consistent_generator_step(t, samples, self.languages)
        return output


@dataclass(frozen=True)
class TwoLargestDiffGenerator:
    """Structure heuristic based on the last two maxima."""

    name: str = "two_largest"

    def step(self, t: int, observed: int, samples: set[int]) -> int:
        return two_largest_diff_generator_step(samples)


@dataclass
class KMGenerator:
    """Thin wrapper around the KM-style critical-language algorithm."""

    languages: Sequence[NamedLanguage]
    state: KMState = field(default_factory=KMState)
    name: str = "km"

    def step(self, t: int, observed: int, samples: set[int]) -> int:
        output, _, _ = km_generate_step(t, observed, samples, self.languages, self.state)
        return output


@dataclass
class MDLGenerator:
    """Two-part-code baseline over the candidate language list."""

    languages: Sequence[NamedLanguage]
    state: MDLState = field(default_factory=MDLState)
    name: str = "mdl"

    def step(self, t: int, observed: int, samples: set[int]) -> int:
        support_max = self.state.support_max(samples)
        output, _, _ = mdl_generator_step(samples, self.languages, support_max=support_max)
        return output


def make_default_generators(languages: Sequence[NamedLanguage]) -> tuple[StepGenerator, ...]:
    """Return the standard generator suite in report order."""
    return (
        FirstConsistentGenerator(languages),
        MDLGenerator(languages),
        TwoLargestDiffGenerator(),
        KMGenerator(languages),
    )


def run_generator(generator: StepGenerator, stream: Sequence[int]) -> list[int]:
    """Run a stateful generator over an observed sample stream."""
    samples: set[int] = set()
    outputs: list[int] = []

    for t, observed in enumerate(stream, start=1):
        # The generators act after seeing the current adversary token, so
        # validity is always judged against the sample set up through time t.
        samples.add(observed)
        outputs.append(generator.step(t, observed, samples))

    return outputs
