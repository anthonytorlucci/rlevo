# rlevo-examples

Application-tier examples for `rlevo`: visualisation, recording, reporting, and benchmarking harness demos. Lightweight library examples live in `crates/rlevo/examples/` instead — see ADR 0012 for the boundary rule.

## When to use this crate

Any example that imports `rlevo-benchmarks` (harness, suites, recording, reporting) belongs here. If your example only uses the five library sub-crates (`rlevo-core`, `rlevo-environments`, `rlevo-evolution`, `rlevo-reinforcement-learning`, `rlevo-hybrid`), add it to `crates/rlevo/examples/` instead.

## Running examples

This crate is excluded from `default-members` — address it explicitly with `-p rlevo-examples`:

```bash
cargo run -p rlevo-examples --example <name> --features <flags>
```

### Benchmarking harness

No feature flags required.

```bash
cargo run -p rlevo-examples --example tabular_bandit
cargo run -p rlevo-examples --example ga_rastrigin
```

### Live TUI (`--features viz-tui`)

```bash
cargo run -p rlevo-examples --example tui_ppo_cartpole --features viz-tui
```

### Recording (`--features viz-tui,viz-record`)

```bash
cargo run -p rlevo-examples --example record_ppo_cartpole    --features viz-tui,viz-record
cargo run -p rlevo-examples --example record_sphere_landscape --features viz-record
cargo run -p rlevo-examples --example record_grids            --features viz-tui,viz-record
cargo run -p rlevo-examples --example record_toy_text         --features viz-tui,viz-record
```

### Report (`--features viz-record,viz-report`)

The `*_with_client` variants stream data to the report server. Start the client first:

```bash
cargo run -p rlevo-benchmarks-report-client
```

Then in a second terminal:

```bash
cargo run -p rlevo-examples --example report_ppo_cartpole_with_client      --features viz-record,viz-report
cargo run -p rlevo-examples --example report_sphere_landscape_with_client  --features viz-record,viz-report
cargo run -p rlevo-examples --example report_grids_with_client             --features viz-record,viz-report
cargo run -p rlevo-examples --example report_toy_text_with_client          --features viz-record,viz-report
```

### Locomotion (`--features locomotion,...`)

```bash
cargo run -p rlevo-examples --example record_inverted_pendulum             --features locomotion,viz-record
cargo run -p rlevo-examples --example report_inverted_pendulum_with_client --features locomotion,viz-record,viz-report
```

### Box2D (`--features box2d,...`)

```bash
cargo run -p rlevo-examples --example record_lunar_lander             --features box2d,viz-record
cargo run -p rlevo-examples --example report_lunar_lander_with_client --features box2d,viz-record,viz-report
```

## Example layout

```
examples/
  common/         shared helpers (ppo_cartpole model/config)
  harness/        benchmarking harness demos (tabular_bandit, ga_rastrigin)
  rl/             PPO on CartPole — TUI, record, report
  evolution/      sphere landscape — record, report, with EA run
  grids/          grid environments — record, report
  toy_text/       toy-text environments — record, report
  locomotion/     inverted pendulum — record, report
  box2d/          lunar lander — record, report
```

## Features

| Feature | Enables |
|---|---|
| `viz-tui` | Live ratatui TUI dashboard (`rlevo-benchmarks/tui`) |
| `viz-record` | On-disk episode recording (`rlevo-benchmarks/record`, `rlevo-environments/record`) |
| `viz-report` | Static-HTML report emitter (`rlevo-benchmarks/report`) |
| `locomotion` | Rapier3D locomotion environments |
| `box2d` | Rapier2D Box2D-style environments |
