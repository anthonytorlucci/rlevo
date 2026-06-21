# Contributing

`rlevo` is open to contributions. The most valuable contributions right now are:

- **New environments** — anything that implements `Environment` and fills a gap in
  the benchmark suite (locomotion, combinatorial optimisation, real-world domains).
- **RL algorithms** — PPO and SAC are the most requested; Double DQN and Dueling
  DQN are smaller scope.
- **Benchmarks and comparisons** — rigorous experiments against Python baselines
  (see [Research Directions](02-research-directions.md)).
- **Bug reports** — especially convergence failures or results that diverge from
  published baselines.

## Where to start

The **contributor book** (`docs/contributor-book/`) covers:
- workspace architecture and crate boundaries,
- the `seed_stream` / host-RNG convention,
- how to add a new `Strategy`,
- how to add a new `Environment`,
- testing philosophy and CI setup.

Read it before opening a PR that touches `rlevo::evo` or
`rlevo::rl`.

## Issue tracker

Open issues, feature requests, and the development roadmap are tracked on
[GitHub](https://github.com/anthonytorlucci/rlevo/issues). If you are planning a
non-trivial contribution, open an issue first to discuss the design — the ADR
history in the contributor book records decisions that are easy to accidentally
re-litigate without that context.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
