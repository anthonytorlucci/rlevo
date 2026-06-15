# Visualisation and recording

> **Status:** stub — prose and `{{#include}}` anchors coming in a follow-up PR.

**Why this exists.** The two-product model (ADR-0013) is non-obvious: a live
TUI that shows metrics only, and a post-run HTML report that replays environment
state. Contributors must not collapse the two products back into one or add viz
dependencies to production crates.

**Key source of truth.** ADR-0013, `rules.md §10`.

## The two-product model (ADR-0013)

| Product | What it shows | Where it lives |
|---------|--------------|----------------|
| Live TUI | Metrics only (return, best fitness, diversity) — **no env panel** | `rlevo-benchmarks` behind `tui` feature |
| Post-run HTML report | `EpisodeRecord` replay — env state per step, per-family rendering | `rlevo-benchmarks` behind `report` feature + `rlevo-benchmarks-report-client` |

## Production crate isolation (ADR-0007)

Neither `rlevo-benchmarks` nor any viz crate is a `[dependencies]` entry in any
production crate. The viz layer wraps production types; it does not constrain
them. `Visualize` is **not** a supertrait of `Environment`.

## Feature gates

```toml
# In rlevo-examples/Cargo.toml (the correct place for heavy examples):
viz-tui = ["rlevo-benchmarks/tui"]
viz-report = ["rlevo-benchmarks/report", "rlevo-environments/record"]
```

## `EpisodeRecord` schema v6 (ADR-0014)

Key fields: `format_version`, `run_id`, `seed`, `env_family`,
`kind: Training|Evaluation`, `frames`, `metrics`, `checkpoints`. Do not add
fields without bumping `FORMAT_VERSION` and authoring a schema-change ADR.

## Accessibility rule

Colour is never the only signal. Pair every hue with a hue-redundant cue (glyph
shape, line style, modifier). This applies to the TUI, the HTML report, plots,
and palettes.

## Outline

1. Wrapping an env in `TuiEnvTap` — feature gate, handle lifecycle.
2. Wrapping an env in `RecordingTap` — `RecordWriter`, `RecordingConfig`, `EnvFamily`.
3. Adding per-family rendering — implementing the `RecordedEnvFamily` trait.
4. The report client — how it reads `EpisodeRecord` and renders frames.
5. Adding a new canonical metric to `rlevo-metrics-registry` (ADR-0015).
