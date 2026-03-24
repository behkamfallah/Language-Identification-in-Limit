"""Experiment orchestration and report formatting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .adversaries import adversary_stream
from .common import AdversaryMode, Language, OrderingStrategy, SUPPORTED_MODES, SUPPORTED_ORDERINGS
from .generators import make_default_generators, run_generator
from .languages import TargetSpec, make_candidate_languages, order_candidate_languages
from .metrics import DiscriminatorMetrics, GenerationMetrics, GenerationTrace, TextBigramMetrics, build_generation_trace, collect_valid_outputs_after_burn, summarize_generation
from .models import bigram_metrics_on_text, discriminator_accuracy_bin, sample_uniform_language


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one demo run."""

    seed: int = 0
    horizon: int = 120
    mode: AdversaryMode = "delay_by_sparse_subset"
    ordering: OrderingStrategy = "adversarial"
    target_max: int = 5000
    early_window: int = 20
    discriminator_sample_cap: int = 60
    discriminator_min_sample: int = 30
    target: TargetSpec = field(default_factory=TargetSpec)


@dataclass(frozen=True)
class GeneratorOutcome:
    """Metrics for one generator on one experiment configuration."""

    name: str
    generation: GenerationMetrics
    discriminator: DiscriminatorMetrics
    trace: GenerationTrace


@dataclass(frozen=True)
class DemoResult:
    """Complete report data for one demo run."""

    config: ExperimentConfig
    stream: tuple[int, ...]
    generator_outcomes: tuple[GeneratorOutcome, ...]
    text_bigram: TextBigramMetrics


def evaluate_discriminator(
    valid_outputs: list[int],
    target_language: Language,
    config: ExperimentConfig,
) -> DiscriminatorMetrics:
    """Compare generated samples to target samples with a simple discriminator."""
    if not valid_outputs:
        return DiscriminatorMetrics(accuracy=None, support_max=config.target_max, sample_size=0)

    # The discriminator should compare on a shared finite support window, so we
    # clip both the generated and reference samples to the same max value.
    support_max = min(max(valid_outputs), config.target_max)
    clipped_outputs = [value for value in valid_outputs if value <= support_max]
    clipped_outputs = clipped_outputs[: config.discriminator_sample_cap]
    sample_size = len(clipped_outputs)

    if sample_size < config.discriminator_min_sample:
        return DiscriminatorMetrics(accuracy=None, support_max=support_max, sample_size=sample_size)

    real_samples = sample_uniform_language(target_language, support_max, sample_size, seed=config.seed + 1)
    accuracy = discriminator_accuracy_bin(real_samples, clipped_outputs, support_max, seed=config.seed + 2)
    return DiscriminatorMetrics(accuracy=accuracy, support_max=support_max, sample_size=sample_size)


def run_demo(config: ExperimentConfig) -> DemoResult:
    """Run one modular experiment and return structured results."""
    target_language = config.target.build_language()
    candidate_languages = make_candidate_languages(
        target=config.target,
        true_language=target_language,
        count=config.horizon,
        seed=config.seed,
    )
    candidate_languages = order_candidate_languages(candidate_languages, config.ordering, seed=config.seed)
    stream = adversary_stream(config.target, config.horizon, config.mode)

    outcomes: list[GeneratorOutcome] = []
    # Late-stage metrics ignore the first half of the run as a cold-start phase.
    burn = config.horizon // 2
    for generator in make_default_generators(candidate_languages):
        # Each generator sees the same stream so the resulting traces are
        # directly comparable in the report and plots.
        outputs = run_generator(generator, stream)
        trace = build_generation_trace(stream, outputs, target_language)
        generation = summarize_generation(trace, early_window=config.early_window)
        # The discriminator only judges outputs produced after the burn point.
        valid_outputs = collect_valid_outputs_after_burn(stream, outputs, target_language, burn=burn)
        discriminator = evaluate_discriminator(valid_outputs, target_language, config)
        outcomes.append(
            GeneratorOutcome(
                name=generator.name,
                generation=generation,
                discriminator=discriminator,
                trace=trace,
            )
        )

    text_bigram = bigram_metrics_on_text(stream, target_language, support_max=config.target_max)
    return DemoResult(
        config=config,
        stream=tuple(stream),
        generator_outcomes=tuple(outcomes),
        text_bigram=text_bigram,
    )


def format_demo_report(result: DemoResult) -> str:
    """Render a human-readable text report for one demo run."""
    config = result.config
    target = config.target
    lines = [
        f"=== Demo: mode={config.mode}, ordering={config.ordering}, seed={config.seed}, T={config.horizon} ===",
        f"Target: K = P_{{{target.a},{target.b}}} union V, V={sorted(target.noise)}",
        "Candidate list contains a five-language hand-crafted core plus random padding hypotheses.",
        "",
    ]

    # The text report mirrors the key quantities used in the figures so the CLI
    # output is still useful when plots are not being inspected directly.
    for outcome in result.generator_outcomes:
        generation = outcome.generation
        discriminator = outcome.discriminator
        accuracy = discriminator.accuracy
        accuracy_text = f"{accuracy:.3f}" if accuracy is not None and not math.isnan(accuracy) else "NA"
        t_star = generation.t_star_est
        t_star_text = str(int(t_star)) if math.isfinite(t_star) else "inf"
        lines.append(
            f"{outcome.name:>16}: "
            f"valid_rate={generation.valid_rate:.3f} "
            f"last_fail={generation.last_fail_t} "
            f"t_star={t_star_text} "
            f"unique={generation.unique_outputs} "
            f"early_unique={generation.early_unique_outputs} "
            f"disc_acc={accuracy_text} "
            f"(disc_M={discriminator.support_max}, n={discriminator.sample_size})"
        )

    lines.extend(
        [
            "",
            (
                f"Bigram LM on text (binary encodings, M={config.target_max}): "
                f"KL={result.text_bigram.kl:.3f}, "
                f"TV={result.text_bigram.tv:.3f}, "
                f"|K intersection [1..M]|={result.text_bigram.support_size}"
            ),
        ]
    )
    return "\n".join(lines)


def run_default_demos(
    seed: int = 0,
    horizon: int = 120,
    target_max: int = 5000,
) -> tuple[DemoResult, ...]:
    """Run the two standard demo modes."""
    return tuple(
        run_demo(
            ExperimentConfig(
                seed=seed,
                horizon=horizon,
                mode=mode,
                ordering=ordering,
                target_max=target_max,
            )
        )
        for ordering in SUPPORTED_ORDERINGS
        for mode in SUPPORTED_MODES
    )
