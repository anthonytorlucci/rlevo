# rlevo-metrics-registry

![Alt Text](rlevo-logo.png)

Single source of truth for the canonical training-metric registry shared across
the `rlevo` workspace (ADR-0015).

## Status

**Stable.** A tiny, `#![no_std]`, dependency-free leaf crate. Its only job is to
define — once — the set of metric field names the rest of the workspace agrees
on, together with each metric's paradigm, sampling cadence, and display title.

## Why this crate exists

Two consumers need the *same* metric vocabulary:

- **`rlevo-benchmarks`** — the live-TUI capture layer and the on-disk recording
  layer classify `tracing` field names against the table to decide what to
  sparkline and what to persist.
- **`rlevo-benchmarks-report-client`** — the WASM report client groups panels
  (RL vs EO), decides per-metric smoothing, and renders human-readable titles.

The report client compiles to `wasm32` and **cannot** depend on
`rlevo-benchmarks`, which transitively pulls in `burn` → `rand` → `getrandom`.
Before ADR-0015 the metric list was a flat `&[&str]` hand-copied into the client
with no compile-time guard, and metric *semantics* (RL-vs-EO grouping,
per-update vs per-generation cadence, titles) were re-derived in three
disconnected hardcoded places.

This crate replaces all of that with one typed [`MetricDescriptor`] table. It is
`#![no_std]` with no dependencies, so it builds everywhere and lets both
consumers share a single definition instead of mirroring it.

## Public surface

| Item | Purpose |
|---|---|
| `MetricDescriptor` | Typed row: `name`, `kind`, `cadence`, `title`, `unit` |
| `MetricKind` | `Rl` / `Eo` / `Shared` — drives report panel grouping |
| `Cadence` | `PerUpdate` / `PerGeneration` / `PerEpisode` — drives report smoothing |
| `CANONICAL_METRICS` | The ordered descriptor table (RL → shared → EO) |
| `descriptor(name)` | Look up a descriptor by field name, `None` if non-canonical |
| `is_canonical_metric(name)` | `true` if the name is recognised |
| `title_for(name)` | Human-readable title, falling back to the raw name |
| `is_per_generation(name)` | `true` for per-generation metrics (plotted raw, no rolling mean) |

`MetricDescriptor::name` is the exact `tracing` field name the recorder matches
against — it is the wire contract between the algorithm crates and the recorder.

## The registry

`CANONICAL_METRICS` holds 32 descriptors, ordered for a stable report layout:

- **RL training stats (per update)** — `policy_loss`, `value_loss`, `loss`,
  `entropy`, `approx_kl`, `clip_frac`, plus the v6 additions
  `explained_variance`, `old_approx_kl`, the per-iteration episode-return
  stats (`episode_return_{mean,std,min,max}`, `episode_length_mean`),
  `env_steps_sampled`, `steps_per_sec`, and `learning_rate`.
- **Per-episode terminal triple (shared)** — `episode_return`,
  `episode_length`, `episode_wall_clock_secs`.
- **DQN family** — `td_loss`, `q_values`.
- **SAC family** — `qf1_loss`, `qf2_loss`, `actor_loss`, `alpha`, `alpha_loss`.
- **Schedules** — `clip_range`, `n_updates`.
- **EO training stats (per generation)** — `best_fitness`, `mean_fitness`,
  `worst_fitness`, `best_fitness_ever`.

A few rows (`alpha_loss`, `clip_range`) are *reserved*: no producer emits them
yet, so the corresponding report panel simply does not appear until one does
(absent metrics are skipped, never rendered empty). They are kept in the table
so the v6 metric set stays complete.

## Extending the registry

1. Add one [`MetricDescriptor`] row to `CANONICAL_METRICS` (use the `d` / `du`
   const constructors for the unit-less / unit-carrying cases).
2. Emit a matching `tracing::info!` field from the algorithm.

That single edit makes the field visible in the live-TUI sparklines, the on-disk
recording stream, and the report's RL/EO panel grouping — no client-side change
required.

## Testing

```bash
cargo test -p rlevo-metrics-registry
```

The suite guards the table's invariants: every descriptor name is recognised,
names are unique, the fitness metrics are `Eo` / `PerGeneration`, the RL
diagnostics are `Rl`, the terminal triple is `Shared` / `PerEpisode`, and
`title_for` falls back to the raw name for undescribed metrics.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or [MIT License](../../LICENSE-MIT) at your option.
