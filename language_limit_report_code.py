"""CLI entry point for the modular language-limit report experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lang_limit.common import SUPPORTED_MODES, SUPPORTED_ORDERINGS
from lang_limit.learned import format_learned_model_report, run_learned_model_experiment
from lang_limit.plots import generate_all_plots
from lang_limit.report import ExperimentConfig, format_demo_report, run_demo


def _json_ready(value):
    """Recursively convert run outputs into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value)]
    return value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Random seed for candidate-language padding. Can be passed multiple times.",
    )
    parser.add_argument("--steps", type=int, default=120, help="Number of observed adversary samples.")
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        action="append",
        help="Adversary mode to run. Defaults to all supported modes.",
    )
    parser.add_argument(
        "--ordering",
        choices=SUPPORTED_ORDERINGS,
        action="append",
        help="Candidate-language ordering to run. Defaults to all supported orderings.",
    )
    parser.add_argument(
        "--target-max",
        type=int,
        default=5000,
        help="Maximum support value used by the text and discriminator proxies.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory where generated figures will be written.",
    )
    parser.add_argument(
        "--skip-learned",
        action="store_true",
        help="Skip the learned autoregressive bridge experiment.",
    )
    parser.add_argument(
        "--results-json",
        help="Optional path for a machine-readable JSON export of the run results.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one or more modular demos and print their reports."""
    args = parse_args()
    seeds = args.seed or [0]
    modes = args.mode or list(SUPPORTED_MODES)
    orderings = args.ordering or list(SUPPORTED_ORDERINGS)
    results = []
    learned_results = []

    # We materialize every requested run first so the summary plots can compare
    # modes, orderings, and seeds in one pass at the end.
    for index, seed in enumerate(seeds):
        for ordering in orderings:
            for mode in modes:
                if results:
                    print()
                config = ExperimentConfig(
                    seed=seed,
                    horizon=args.steps,
                    mode=mode,
                    ordering=ordering,
                    target_max=args.target_max,
                )
                result = run_demo(config)
                results.append(result)
                print(format_demo_report(result))

    if not args.skip_learned:
        seen_mode_seed: set[tuple[int, str]] = set()
        for result in results:
            key = (result.config.seed, result.config.mode)
            # The learned bridge ignores candidate ordering, so dedupe by seed/mode.
            if key in seen_mode_seed:
                continue
            seen_mode_seed.add(key)
            if learned_results or results:
                print()
            target_language = result.config.target.build_language()
            try:
                learned_result = run_learned_model_experiment(
                    mode=result.config.mode,
                    seed=result.config.seed,
                    stream=list(result.stream),
                    target=target_language,
                    support_max=result.config.target_max,
                )
            except RuntimeError as error:
                print(f"learned_gru skipped: {error}")
                break
            learned_results.append(learned_result)
            print(format_learned_model_report(learned_result))

    # Plot generation happens once after all runs finish so the batch-level
    # figures can aggregate across the collected results.
    saved_plots = generate_all_plots(results, Path(args.plots_dir), learned_results=learned_results)
    if args.results_json:
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "demo_results": [asdict(result) for result in results],
            "learned_results": [asdict(result) for result in learned_results],
            "plots": [str(path) for path in saved_plots],
        }
        output_path.write_text(json.dumps(_json_ready(payload), indent=2))

    print()
    print(f"Saved {len(saved_plots)} plot(s) to {Path(args.plots_dir).resolve()}")


if __name__ == "__main__":
    main()
