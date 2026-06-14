# Seeing what happened

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The problem.** You ran a 30k-step training job. Did it work? *Why?* You need
to watch it live and inspect it afterwards.

**Learning goal.** The two observation products (ADR-0013):

1. A **live metrics-only TUI** during training — per-episode reward, best
   fitness, population diversity.
2. A **post-run record + static HTML report** for replay — `EpisodeRecord`
   schema v6 (ADR-0014).

## The new seams

- `TuiEnvTap<E, D, SD, AD>` — transparent `Environment` wrapper emitting
  per-episode events to a `TuiHandle`
  (`rlevo-benchmarks::env_wrappers::tui_env_tap`).
- `RecordingTap<E, ...>` — writes every `reset` / `step` to a `RecordSink`;
  `RecordWriter::open(...)`, `RecordingConfig`, `EnvFamily`
  (`rlevo-benchmarks::record`).
- `EpisodeRecord` schema v6 fields: `format_version`, `run_id`, `seed`,
  `env_family`, `kind: Training|Evaluation`, `frames`, `metrics`,
  `population_samples`.

## Accessibility note

The report and TUI pair colour with hue-redundant signals (glyph shape, line
style) — never colour alone. This is a first-class design constraint in this
project; the contributor book covers it in depth.

## Outline

1. Wrapping an environment in `TuiEnvTap` — one-line change, live metrics.
2. Wrapping in `RecordingTap` — writing an `EpisodeRecord` to disk.
3. Re-opening a run with `RecordedRun::open(dir)` and inspecting the schema.
4. Launching the HTML report client.
5. Make it yours — record a Ch1 EA run vs a Ch2 RL run and compare.

## Example

```bash
cargo run -p rlevo-examples --example ch09_recording --features viz-report
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch09_recording.rs}} -->
