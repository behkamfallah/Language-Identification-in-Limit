"""Plot generation for language-limit experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .learned import LearnedModelResult
from .report import DemoResult

GENERATOR_ORDER = ("first_consistent", "mdl", "km", "two_largest")
GENERATOR_LABELS = {
    "first_consistent": "First Consistent",
    "mdl": "MDL",
    "km": "KM",
    "two_largest": "Two Largest",
}
GENERATOR_COLORS = {
    "first_consistent": "#C44E52",
    "mdl": "#8172B2",
    "km": "#4C72B0",
    "two_largest": "#55A868",
}
MODE_LABELS = {
    "noise_then_progression": "Noise Then Progression",
    "delay_by_sparse_subset": "Delay by Sparse Subset",
}
ORDERING_LABELS = {
    "adversarial": "Adversarial",
    "complexity": "Complexity Ordered",
    "random": "Random",
}


def _configure_style() -> None:
    # Use a warm paper-like palette so the figures feel deliberate and readable
    # in slides or a draft rather than like notebook defaults.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F8F6F1",
            "axes.facecolor": "#FFFDF8",
            "axes.edgecolor": "#D9D4C7",
            "axes.labelcolor": "#2E2A25",
            "xtick.color": "#2E2A25",
            "ytick.color": "#2E2A25",
            "text.color": "#2E2A25",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.facecolor": "#FFFDF8",
            "legend.edgecolor": "#D9D4C7",
        }
    )


def _outcome_map(result: DemoResult):
    return {outcome.name: outcome for outcome in result.generator_outcomes}


def _result_label(result: DemoResult) -> str:
    mode_label = MODE_LABELS[result.config.mode]
    ordering_label = ORDERING_LABELS[result.config.ordering]
    return f"{mode_label}\n{ordering_label}\nseed={result.config.seed}"


def _file_safe_result_label(result: DemoResult) -> str:
    return f"mode-{result.config.mode}__ordering-{result.config.ordering}__seed-{result.config.seed}"


def _time_steps(length: int) -> list[int]:
    return list(range(1, length + 1))


def _finite_t_star(value: float, horizon: int) -> int:
    return int(value) if math.isfinite(value) else horizon + 1


def _bar_offsets(
    x_positions: list[int],
    series_index: int,
    num_series: int,
    bar_width: float,
) -> list[float]:
    """Return centered grouped-bar offsets for one series."""
    center = (num_series - 1) / 2
    return [x + (series_index - center) * bar_width for x in x_positions]


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_convergence_curve(result: DemoResult, output_dir: Path) -> Path:
    """Plot cumulative valid rate over time for all generators."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    steps = _time_steps(len(result.stream))
    outcomes = _outcome_map(result)

    for name in GENERATOR_ORDER:
        trace = outcomes[name].trace
        ax.plot(
            steps,
            trace.cumulative_valid_rate,
            label=GENERATOR_LABELS[name],
            color=GENERATOR_COLORS[name],
            linewidth=2.6,
        )

    ax.set_title(
        f"Convergence Curve: {MODE_LABELS[result.config.mode]} / "
        f"{ORDERING_LABELS[result.config.ordering]} (seed={result.config.seed})"
    )
    ax.set_xlabel("Time step t")
    ax.set_ylabel("Cumulative valid rate up to t")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    return _save_figure(fig, output_dir / f"convergence_curve__{_file_safe_result_label(result)}.png")


def plot_instantaneous_validity(result: DemoResult, output_dir: Path) -> Path:
    """Plot step-by-step validity for the KM and heuristic generators."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    steps = _time_steps(len(result.stream))
    outcomes = _outcome_map(result)

    for name, offset in (("km", 0.06), ("mdl", 0.0), ("two_largest", -0.06)):
        trace = outcomes[name].trace
        # Small vertical offsets keep the two binary traces visible when both
        # generators agree at the same time step.
        y_values = [int(flag) + offset for flag in trace.valid_flags]
        ax.scatter(
            steps,
            y_values,
            label=GENERATOR_LABELS[name],
            color=GENERATOR_COLORS[name],
            s=24,
            alpha=0.85,
        )
        ax.step(
            steps,
            [int(flag) for flag in trace.valid_flags],
            where="mid",
            color=GENERATOR_COLORS[name],
            alpha=0.35,
            linewidth=1.4,
        )

    ax.set_title(
        f"Instantaneous Validity: {MODE_LABELS[result.config.mode]} / "
        f"{ORDERING_LABELS[result.config.ordering]} (seed={result.config.seed})"
    )
    ax.set_xlabel("Time step t")
    ax.set_ylabel("Valid output")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc="lower right")
    return _save_figure(fig, output_dir / f"instantaneous_validity__{_file_safe_result_label(result)}.png")


def plot_diversity_vs_time(result: DemoResult, output_dir: Path) -> Path:
    """Plot cumulative unique outputs over time."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    steps = _time_steps(len(result.stream))
    outcomes = _outcome_map(result)

    for name in ("km", "mdl", "two_largest"):
        trace = outcomes[name].trace
        ax.plot(
            steps,
            trace.cumulative_unique_outputs,
            label=GENERATOR_LABELS[name],
            color=GENERATOR_COLORS[name],
            linewidth=2.6,
        )

    ax.set_title(
        f"Diversity vs Time: {MODE_LABELS[result.config.mode]} / "
        f"{ORDERING_LABELS[result.config.ordering]} (seed={result.config.seed})"
    )
    ax.set_xlabel("Time step t")
    ax.set_ylabel("Unique outputs so far")
    ax.legend(loc="upper left")
    return _save_figure(fig, output_dir / f"diversity_vs_time__{_file_safe_result_label(result)}.png")


def plot_time_to_convergence(results: list[DemoResult], output_dir: Path) -> Path:
    """Plot a grouped bar chart of time-to-convergence across runs."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x_positions = list(range(len(results)))
    bar_width = 0.22

    for offset_index, name in enumerate(GENERATOR_ORDER):
        # Infinite convergence times are capped visually at T+1 and labeled
        # explicitly so the bar chart remains finite and readable.
        values = [_finite_t_star(_outcome_map(result)[name].generation.t_star_est, result.config.horizon) for result in results]
        offsets = _bar_offsets(x_positions, offset_index, len(GENERATOR_ORDER), bar_width)
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=GENERATOR_LABELS[name],
            color=GENERATOR_COLORS[name],
        )
        for bar, result in zip(bars, results):
            t_star = _outcome_map(result)[name].generation.t_star_est
            if not math.isfinite(t_star):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    "inf",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=90,
                )

    ax.set_title("Time to Convergence")
    ax.set_xlabel("Run")
    ax.set_ylabel("Estimated t_star (inf capped at T+1)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_result_label(result) for result in results])
    ax.legend(loc="upper right")
    return _save_figure(fig, output_dir / "time_to_convergence.png")


def plot_mode_comparison(results: list[DemoResult], output_dir: Path) -> Path:
    """Compare valid rate and convergence time across mode/ordering settings."""
    _configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # Group by both mode and ordering so fair-indexing sweeps are not averaged
    # together into a misleading single "mode" number.
    grouped: dict[tuple[str, str], dict[str, list[DemoResult]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for outcome in result.generator_outcomes:
            grouped[(result.config.mode, result.config.ordering)][outcome.name].append(result)

    group_keys = [key for key in grouped if key[0] in MODE_LABELS and key[1] in ORDERING_LABELS]
    group_keys.sort(key=lambda key: (key[1], key[0]))
    x_positions = list(range(len(group_keys)))
    bar_width = 0.22

    for axis, metric_name in zip(axes, ("valid_rate", "t_star_est")):
        for offset_index, name in enumerate(GENERATOR_ORDER):
            values: list[float] = []
            for group_key in group_keys:
                runs = grouped[group_key][name]
                if metric_name == "valid_rate":
                    values.append(mean(_outcome_map(result)[name].generation.valid_rate for result in runs))
                else:
                    values.append(
                        mean(
                            _finite_t_star(_outcome_map(result)[name].generation.t_star_est, result.config.horizon)
                            for result in runs
                        )
                    )
            offsets = _bar_offsets(x_positions, offset_index, len(GENERATOR_ORDER), bar_width)
            axis.bar(
                offsets,
                values,
                width=bar_width,
                label=GENERATOR_LABELS[name],
                color=GENERATOR_COLORS[name],
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [f"{MODE_LABELS[mode]}\n{ORDERING_LABELS[ordering]}" for mode, ordering in group_keys],
            rotation=10,
        )
        if metric_name == "valid_rate":
            axis.set_title("Mode and Ordering: Valid Rate")
            axis.set_ylabel("Average valid rate")
            axis.set_ylim(0.0, 1.05)
        else:
            axis.set_title("Mode and Ordering: Time to Convergence")
            axis.set_ylabel("Average t_star (inf capped at T+1)")

    axes[1].legend(loc="upper right")
    return _save_figure(fig, output_dir / "mode_comparison.png")


def plot_ordering_comparison(results: list[DemoResult], output_dir: Path) -> Path:
    """Compare generators across ordering strategies after averaging over modes/seeds."""
    _configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    grouped: dict[str, dict[str, list[DemoResult]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for outcome in result.generator_outcomes:
            grouped[result.config.ordering][outcome.name].append(result)

    orderings = [ordering for ordering in ORDERING_LABELS if ordering in grouped]
    x_positions = list(range(len(orderings)))
    bar_width = 0.22

    for axis, metric_name in zip(axes, ("valid_rate", "t_star_est")):
        for offset_index, name in enumerate(GENERATOR_ORDER):
            values: list[float] = []
            for ordering in orderings:
                runs = grouped[ordering][name]
                if metric_name == "valid_rate":
                    values.append(mean(_outcome_map(result)[name].generation.valid_rate for result in runs))
                else:
                    values.append(
                        mean(
                            _finite_t_star(_outcome_map(result)[name].generation.t_star_est, result.config.horizon)
                            for result in runs
                        )
                    )
            offsets = _bar_offsets(x_positions, offset_index, len(GENERATOR_ORDER), bar_width)
            axis.bar(
                offsets,
                values,
                width=bar_width,
                label=GENERATOR_LABELS[name],
                color=GENERATOR_COLORS[name],
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels([ORDERING_LABELS[ordering] for ordering in orderings], rotation=10)
        if metric_name == "valid_rate":
            axis.set_title("Ordering Comparison: Valid Rate")
            axis.set_ylabel("Average valid rate")
            axis.set_ylim(0.0, 1.05)
        else:
            axis.set_title("Ordering Comparison: Time to Convergence")
            axis.set_ylabel("Average t_star (inf capped at T+1)")

    axes[1].legend(loc="upper right")
    return _save_figure(fig, output_dir / "ordering_comparison.png")


def plot_distributional_gap(results: list[DemoResult], output_dir: Path) -> Path:
    """Plot KL and TV across runs for the observed text."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x_positions = list(range(len(results)))
    bar_width = 0.35

    kl_values = [result.text_bigram.kl for result in results]
    tv_values = [result.text_bigram.tv for result in results]

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        kl_values,
        width=bar_width,
        label="KL divergence",
        color="#DD8452",
    )
    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        tv_values,
        width=bar_width,
        label="TV distance",
        color="#937860",
    )

    ax.set_title("Distributional vs Structural Gap")
    ax.set_xlabel("Run")
    ax.set_ylabel("Distance")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_result_label(result) for result in results])
    ax.legend(loc="upper right")
    return _save_figure(fig, output_dir / "distributional_gap.png")


def plot_learned_model_bridge(results: list[LearnedModelResult], output_dir: Path) -> Path:
    """Plot validity-versus-breadth diagnostics for the learned bridge model."""
    _configure_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    grouped: dict[str, list[LearnedModelResult]] = defaultdict(list)
    for result in results:
        grouped[result.mode].append(result)

    modes = [mode for mode in MODE_LABELS if mode in grouped]
    x_positions = list(range(len(modes)))
    metric_values = {
        "Valid Rate": [mean(result.valid_rate for result in grouped[mode]) for mode in modes],
        "Novelty Rate": [mean(result.novelty_rate for result in grouped[mode]) for mode in modes],
        "Coverage": [mean(result.coverage_ratio for result in grouped[mode]) for mode in modes],
        "Distinct Rollouts": [mean(result.distinct_rollout_ratio for result in grouped[mode]) for mode in modes],
    }
    metric_colors = {
        "Valid Rate": "#4C72B0",
        "Novelty Rate": "#55A868",
        "Coverage": "#C44E52",
        "Distinct Rollouts": "#8172B2",
    }
    bar_width = 0.18

    for metric_index, (metric_name, values) in enumerate(metric_values.items()):
        ax.bar(
            _bar_offsets(x_positions, metric_index, len(metric_values), bar_width),
            values,
            width=bar_width,
            label=metric_name,
            color=metric_colors[metric_name],
        )

    ax.set_title("Learned Autoregressive Bridge: Validity vs Breadth")
    ax.set_xlabel("Run")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODE_LABELS[mode] for mode in modes])
    ax.legend(loc="upper right")
    return _save_figure(fig, output_dir / "learned_model_bridge.png")


def generate_all_plots(
    results: list[DemoResult],
    output_dir: str | Path,
    learned_results: list[LearnedModelResult] | None = None,
) -> list[Path]:
    """Generate the full figure set for a batch of demo results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    # Per-run plots come first, then the cross-run summaries.
    for result in results:
        saved_paths.append(plot_convergence_curve(result, output_path))
        saved_paths.append(plot_instantaneous_validity(result, output_path))
        saved_paths.append(plot_diversity_vs_time(result, output_path))

    if results:
        saved_paths.append(plot_time_to_convergence(results, output_path))
        saved_paths.append(plot_mode_comparison(results, output_path))
        saved_paths.append(plot_ordering_comparison(results, output_path))
        saved_paths.append(plot_distributional_gap(results, output_path))

    if learned_results:
        saved_paths.append(plot_learned_model_bridge(learned_results, output_path))

    return saved_paths
