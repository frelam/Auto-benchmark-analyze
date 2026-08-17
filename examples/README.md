# Examples

This directory shows the two input forms, the two modes, and the generated
output. Everything here runs against a JSON scores file — **no GPU and no
inference endpoint required**.

## Run it

```bash
bash scripts/install.sh          # once (CPU-safe install: eval + charts)
bash examples/run_example.sh     # runs `run` and lists the output
```

or manually:

```bash
benchmark-diagnosis --config examples/run-from-scores.yaml run
```

The same run, as pure CLI arguments (input form #2 — no config file):

```bash
benchmark-diagnosis run --model llama-3-8b --scores examples/scores.json \
  --arch dense --params 8 --release-date 2024-04-01
```

## The two input forms

Every command takes its settings **either from a YAML config or from CLI
arguments** — never both at once. CLI arguments override the config's `run:`
section. The three profiles show the three model sources:

| Profile | Model source | What happens |
|---|---|---|
| `run-from-scores.yaml` | `scores` (JSON file) | evaluation skipped; scores read directly |
| `run-endpoint.yaml` | `endpoint` (service IP) | the OpenAI-compatible harness path is selected automatically |
| `run-full-llm.yaml` | `weights` (HF id) | the tool auto-deploys with vLLM, then evaluates |

The model source is auto-derived: exactly one of `--model-path` (weights),
`--base-url` (endpoint), or `--scores` (scores) must be provided — pass two and
the tool refuses with a clear error. For a weights run, `--model` is optional
and auto-derived from `--model-path` (the basename / last segment of the HF id).

## The three modes

`run.mode` (or `--mode`) selects how far the analysis goes:

| Mode | Produces |
|---|---|
| `benchmark` | evaluation only — writes `scores.json` (feed back with `--scores`); no diagnosis |
| `analyze` | evaluation + analysis only (no recommendations) |
| `full` | evaluation + analysis + optimization recommendations |

## Advisor auto-selection

The recommendation engine first attributes the missing capabilities
(benchmark correlation + bad cases), then scores the **tool-maintained
experience base** — concrete datasets, hyperparameter knobs, expected effects,
and accumulated outcome deltas — against that deficit. `recommendation.advisor_mode`
is `auto` by default:

- an analyst LLM is configured (`llm.model` set) → `llm_rules`: the LLM
  **re-ranks** the experience-base candidates (grounded; it never invents
  datasets or knobs);
- otherwise → `rules`: the deterministic experience-base engine.

`llm_rules` with no `llm.model` configured fails fast **before** evaluation, so
a misconfigured advisor never wastes a run.

## What the output looks like

Running the scores example writes everything under `examples/output/`:

```
examples/output/
├── report.md              # the full diagnosis report (Markdown)
├── metrics.json           # the same content as machine-readable JSON
└── figures/
    ├── fig_model_scores.png    # per-benchmark bar chart (portfolio members highlighted)
    ├── fig_model_clusters.png  # per-cluster verdict bar (green/red + percentile/z)
    └── fig_model_gaps.png      # quantified gap per cluster (how far below expectation)
```

The committed `output/` files are a snapshot produced by actually running the
example — regenerate them any time with `bash examples/run_example.sh`
(asset version ids and timestamps will differ; the structure will not).

`report.md` shows, per capability cluster, the weighted score, percentile,
z-score, verdict, the scored benchmarks, a **quantified gap** (how far below the
model's expectation curve each cluster sits), a **missing-capability profile**
(which capabilities the benchmark correlation + bad cases point to), and — in
`full` mode — recommended **datasets to add**, **hyperparameter adjustments**,
an **expected effect**, and a **reasoning chain** (deficit → intervention →
historical outcome → linked rule) with validation experiments. The figures are
linked inline at the bottom of the report.
