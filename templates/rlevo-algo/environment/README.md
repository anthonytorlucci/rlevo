# {{project-name}}

An environment of your own, run through
[`rlevo`](https://github.com/anthonytorlucci/rlevo)'s evaluation harness by a
random agent and by the shipped DQN, trained on it.

```bash
cargo run --release
```

prints one row per agent as a Markdown table: the mean return and episode
length over the evaluation episodes that follow training, and how many
trials each row averages. `cargo run` without `--release` gives the same
table more slowly. `cargo test` runs the environment's tests.

## What to change

- **`src/env.rs`**: `MyEnv`, a corridor that works as it stands, with its
  observation and action types. Replace it with your task and keep its
  shape. The file's top comment says how an episode ends, what a step after
  the end must do, and where the environment's randomness comes from.
- **`src/main.rs`**: `config`, the one configuration every row runs under;
  the time limit that cuts long episodes; and the rows.

The observation implements `HostRow`, the numbers a learner reads, so a
shipped agent such as `DqnAgent` can train on it. An environment run only
by agents of your own does not need it.
