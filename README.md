# Language Limit Experiments

Reproducible toy experiments on the gap between identification in the limit and generation in the limit, centered on one practical question:

- when can a generator become behaviorally correct without learning the underlying structure?

The code now includes:

- adversarial, complexity-ordered, and random candidate-language indexings
- a `first_consistent` consistency ablation
- an implemented MDL baseline over the finite candidate set
- the KM-style critical-language generator
- a simple structure-exploiting heuristic
- a small learned autoregressive GRU bridge that uses no candidate-language oracle
- plotting and JSON export for paper-ready summaries

## Requirements

- Python 3.10+
- `matplotlib` for figures
- `torch` for the learned autoregressive bridge experiment

The symbolic experiments do not require `torch`. If PyTorch is unavailable, you can still run the main suite with `--skip-learned`.

## Quick Start

From the project root:

```bash
python3 language_limit_report_code.py
```

By default this runs:

- both adversary modes
- all three ordering strategies
- the four symbolic generators
- the learned GRU bridge
- plot generation into [plots](/Users/behkamfallah/Documents/lang-id/plots)

## Useful Commands

Run one adversary mode:

```bash
python3 language_limit_report_code.py --mode noise_then_progression
```

Run one ordering only:

```bash
python3 language_limit_report_code.py --ordering complexity
```

Run multiple seeds:

```bash
python3 language_limit_report_code.py --seed 0 --seed 1 --seed 2
```

Skip the learned-model bridge:

```bash
python3 language_limit_report_code.py --skip-learned
```

Export machine-readable results:

```bash
python3 language_limit_report_code.py --results-json results/full_suite.json
```

Use a different plot directory:

```bash
python3 language_limit_report_code.py --plots-dir figures
```

## Current Candidate Construction

The current implementation in [lang_limit/languages.py](/Users/behkamfallah/Documents/lang-id/lang_limit/languages.py) builds a finite candidate list of size `count`, where the report code sets `count = horizon` and the default horizon is `120`.

In the current implementation, the hand-crafted core contains exactly five candidates in fixed raw order:

- `L1_universe`
- `L2_target_plus_extra_noise`
- `L3_progression_2_mod_3`
- `L4_multiples_of_5`
- `L5_true_target`

All remaining slots are filled by random progression-with-noise padding sampled from the run seed. With the default horizon `T=120`, this yields `120` total candidates, of which `115` are random padding candidates.

The implementation does not filter duplicates, so randomly generated padding candidates may coincide with the true target or with other existing hypotheses.

## What the Main Results Mean

The current project is organized around three comparisons.

1. `Ordering sensitivity`
   The `first_consistent` ablation and the KM-style generator depend strongly on how the candidate list is indexed. Under adversarial ordering, `first_consistent` fails completely; under complexity ordering it succeeds quickly.

2. `MDL as a fairer baseline`
   The MDL baseline searches the full finite candidate set and scores consistent hypotheses by model cost plus data cost. It is intentionally stronger than `first_consistent`, and in the current toy family it is largely insensitive to indexing.

3. `Validity versus breadth`
   High validity does not imply broad generation. The symbolic generators and the learned GRU bridge both show versions of this tension: it is possible to generate mostly valid outputs while exploring only a narrow part of the target language.

## Figures

Per-run figures:

- `convergence_curve__...png`
- `instantaneous_validity__...png`
- `diversity_vs_time__...png`

Cross-run summaries:

- `time_to_convergence.png`
- `mode_comparison.png`
- `ordering_comparison.png`
- `distributional_gap.png`
- `learned_model_bridge.png`

The most important figures for the paper-facing story are:

- `ordering_comparison.png`
  This makes the fairness issue explicit: adversarial indexing can destroy naive consistency methods, while complexity ordering rescues them.

- `diversity_vs_time__mode-delay_by_sparse_subset__ordering-adversarial__seed-0.png`
  This is the clearest symbolic illustration of validity without breadth.

- `learned_model_bridge.png`
  This is the non-oracle bridge result: the tiny GRU achieves near-perfect valid continuation but very low coverage and rollout diversity.

## Project Layout

- [language_limit_report_code.py](/Users/behkamfallah/Documents/lang-id/language_limit_report_code.py)
  CLI entry point. Runs experiments, prints reports, writes plots, and can export JSON.

- [lang_limit/common.py](/Users/behkamfallah/Documents/lang-id/lang_limit/common.py)
  Shared types, constants, and small helper functions.

- [lang_limit/languages.py](/Users/behkamfallah/Documents/lang-id/lang_limit/languages.py)
  Target language definitions, candidate-language construction, and ordering strategies.

- [lang_limit/adversaries.py](/Users/behkamfallah/Documents/lang-id/lang_limit/adversaries.py)
  Positive-only text constructions.

- [lang_limit/generators.py](/Users/behkamfallah/Documents/lang-id/lang_limit/generators.py)
  `first_consistent`, `mdl`, `two_largest`, and `km`.

- [lang_limit/metrics.py](/Users/behkamfallah/Documents/lang-id/lang_limit/metrics.py)
  Trace building, convergence proxies, and diversity summaries.

- [lang_limit/models.py](/Users/behkamfallah/Documents/lang-id/lang_limit/models.py)
  Bigram LM and discriminator diagnostics.

- [lang_limit/learned.py](/Users/behkamfallah/Documents/lang-id/lang_limit/learned.py)
  Tiny GRU bridge experiment over difference-token continuations.

- [lang_limit/report.py](/Users/behkamfallah/Documents/lang-id/lang_limit/report.py)
  Structured orchestration for symbolic experiments.

- [lang_limit/plots.py](/Users/behkamfallah/Documents/lang-id/lang_limit/plots.py)
  All figure generation.

## Reproducibility

Validation:

```bash
python3 -m py_compile language_limit_report_code.py lang_limit/*.py
```

Example full-suite command used for the paper-facing summaries:

```bash
python3 language_limit_report_code.py \
  --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
  --results-json results/full_suite.json
```

This writes:

- a console report
- PNG figures in [plots](/Users/behkamfallah/Documents/lang-id/plots)
- a machine-readable export in [results/full_suite.json](/Users/behkamfallah/Documents/lang-id/results/full_suite.json)

## Notes on Scope

- The arithmetic-progression family `K = P_{a,b} union V` is a pedagogical toy model, not a realistic benchmark for modern language modeling.
- The KM generator still uses candidate-language membership queries, as in the original formal model.
- The learned GRU bridge is included specifically to show that the validity-versus-breadth tension also appears in a non-oracle autoregressive setting.
