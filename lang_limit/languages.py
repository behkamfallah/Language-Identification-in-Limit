"""Language definitions and candidate-language construction."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .common import Language, OrderingStrategy, integer_code_length

DEFAULT_NOISE = frozenset({2, 4, 8})
UNSTRUCTURED_TYPE_COST = 32
PROGRESSION_TYPE_COST = 3
PROGRESSION_WITH_NOISE_TYPE_COST = 4
DIVISIBILITY_TYPE_COST = 5
PADDING_PENALTY = 32


@dataclass(frozen=True)
class TargetSpec:
    """Definition of a target language of the form P_{a,b} union noise."""

    a: int = 50
    b: int = 7
    noise: frozenset[int] = DEFAULT_NOISE

    def __post_init__(self) -> None:
        if self.a < 1:
            raise ValueError("a must be positive")
        if self.b <= 0:
            raise ValueError("b must be positive")

    def build_language(self) -> Language:
        """Materialize the target language predicate."""
        return make_progression_with_noise(self.a, self.b, set(self.noise))


@dataclass(frozen=True)
class NamedLanguage:
    """A predicate plus a readable label for debugging and reporting."""

    name: str
    predicate: Language
    description_bits: int = 1

    def __call__(self, value: int) -> bool:
        return self.predicate(value)


def make_universe() -> Language:
    """Return the universe of positive integers."""
    return lambda value: value >= 1


def make_arith_progression(a: int, b: int) -> Language:
    """Return the arithmetic progression P_{a,b} = {a + b*k : k >= 0}."""
    if a < 1:
        raise ValueError("a must be positive")
    if b <= 0:
        raise ValueError("b must be positive")

    return lambda value: value >= a and (value - a) % b == 0


def make_progression_with_noise(a: int, b: int, noise: set[int]) -> Language:
    """Return P_{a,b} union a finite noise set."""
    progression = make_arith_progression(a, b)
    sanitized_noise = frozenset(value for value in noise if value >= 1)
    return lambda value: value in sanitized_noise or progression(value)


def is_consistent(samples: set[int], language: Language) -> bool:
    """Return True when all observed samples belong to ``language``."""
    return all(language(sample) for sample in samples)


def smallest_unseen_member(language: Language, seen: set[int], start: int = 1) -> int:
    """Return the smallest positive member of ``language`` that is not in ``seen``."""
    candidate = max(1, start)
    while True:
        if candidate not in seen and language(candidate):
            return candidate
        candidate += 1


def _noise_code_length(noise: set[int]) -> int:
    """Return a simple bit cost for encoding a finite noise set."""
    return sum(integer_code_length(value) + 1 for value in sorted(noise))


def _progression_with_noise_description_bits(a: int, b: int, noise: set[int]) -> int:
    """Return a simple description-length proxy for P_{a,b} union noise."""
    return (
        PROGRESSION_WITH_NOISE_TYPE_COST
        + integer_code_length(a)
        + integer_code_length(b)
        + _noise_code_length(noise)
    )


def _progression_description_bits(a: int, b: int) -> int:
    """Return a simple description-length proxy for a pure progression P_{a,b}."""
    return PROGRESSION_TYPE_COST + integer_code_length(a) + integer_code_length(b)


def _divisibility_description_bits(modulus: int) -> int:
    """Return a description-length proxy for a divisibility predicate."""
    return DIVISIBILITY_TYPE_COST + integer_code_length(modulus)


def order_candidate_languages(
    languages: list[NamedLanguage],
    ordering: OrderingStrategy,
    seed: int = 0,
) -> list[NamedLanguage]:
    """Return candidate languages in the requested index order."""
    if ordering == "adversarial":
        return list(languages)

    if ordering == "complexity":
        # Name provides a stable tie-break when two candidates have the same cost.
        return sorted(languages, key=lambda language: (language.description_bits, language.name))

    if ordering == "random":
        rng = random.Random(seed)
        ordered = list(languages)
        rng.shuffle(ordered)
        return ordered

    raise ValueError(f"unknown ordering strategy: {ordering}")


def make_candidate_languages(
    target: TargetSpec,
    true_language: Language,
    count: int,
    seed: int = 0,
) -> list[NamedLanguage]:
    """Build a readable list of candidate languages for the experiment."""
    if count <= 0:
        return []

    rng = random.Random(seed)
    noise = set(target.noise)
    # This extra noise makes the main superset trap vary with the run seed.
    extra_noise = set(rng.sample(range(1, 40), k=6))
    # The early positions are intentionally structured:
    # - L1 is the superset trap for naive identification/generation.
    # - L5 is the true target language.
    # - the middle entries provide a few stable distractors.
    languages: list[NamedLanguage] = [
        NamedLanguage("L1_universe", make_universe(), description_bits=UNSTRUCTURED_TYPE_COST),
        NamedLanguage(
            "L2_target_plus_extra_noise",
            make_progression_with_noise(target.a, target.b, noise | extra_noise),
            description_bits=_progression_with_noise_description_bits(
                target.a,
                target.b,
                noise | extra_noise,
            ),
        ),
        NamedLanguage(
            "L3_progression_2_mod_3",
            make_arith_progression(2, 3),
            description_bits=_progression_description_bits(2, 3),
        ),
        NamedLanguage(
            "L4_multiples_of_5",
            lambda value: value >= 1 and value % 5 == 0,
            description_bits=_divisibility_description_bits(5),
        ),
        NamedLanguage(
            "L5_true_target",
            true_language,
            description_bits=_progression_with_noise_description_bits(target.a, target.b, noise),
        ),
    ]

    # The runtime behavior is a five-candidate core followed by padding.
    # This subset block remains in the file but does not contribute candidates.
    for k in range(1, 30):
        if len(languages) >= count:
            return languages[:count]
            languages.append(
            NamedLanguage(
                f"L_subset_stride_{target.b * (2 ** k)}",
                make_progression_with_noise(target.a, target.b * (2**k), noise),
                description_bits=_progression_with_noise_description_bits(
                    target.a,
                    target.b * (2**k),
                    noise,
                ),
            )
        )

    # Random padding keeps the candidate family countable-looking without
    # changing the intended early ordering of the hand-crafted languages.
    while len(languages) < count:
        a_r = rng.randint(1, 60)
        b_r = rng.randint(2, 30)
        noise_r = set(rng.sample(range(1, 80), k=rng.randint(0, 3)))
        languages.append(
            NamedLanguage(
                f"L_random_{len(languages) + 1}",
                make_progression_with_noise(a_r, b_r, noise_r),
                description_bits=_progression_with_noise_description_bits(a_r, b_r, noise_r)
                + PADDING_PENALTY,
            )
        )

    return languages[:count]
