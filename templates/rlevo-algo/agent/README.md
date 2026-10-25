# {{project-name}}

An agent of your own, trained and measured on CartPole through
[`rlevo`](https://github.com/anthonytorlucci/rlevo)'s evaluation harness,
beside a random agent and the shipped DQN.

```bash
cargo run --release
```

prints one row per agent as a Markdown table: the mean return and episode
length over the evaluation episodes that follow training, and how many
trials each row averages. `cargo run` without `--release` gives the same
table more slowly. `cargo test` runs the agent's tests.

## What to change

- **`src/agent.rs`**: `MyAgent`, a linear Q-learner that works as it stands.
  Replace it with your algorithm and keep the `Agent` impl's signatures. The
  file's top comment says what the harness calls, in what order, and what
  a `Terminated` step means against a `Truncated` one.
- **`src/main.rs`**: `config`, the one configuration every row runs under
  (the training budget, the evaluation episodes, the seeds); the
  environment; and the rows.

`MyAgent` is there to show the hooks at work, not to do well: its row moves
a long way with the seed. One trial is one seed. Raise
`num_trials_per_env` before reading anything into a difference between two
rows.
