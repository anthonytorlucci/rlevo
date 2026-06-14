# Part II — Guided Tour

This part tells a story. It starts with the simplest interesting problem —
minimising a mathematical function — and builds incrementally toward a
gradient-based RL agent learning to balance a pole. Each section introduces one
new concept, shows it in working `rlevo` code, and links to the relevant
foundation material and appendix entries for readers who want to go deeper.

You can run every example as-is:

```bash
cargo run -p rlevo-examples --example <name>
```

## The narrative arc

| Section | New concept introduced |
| ------- | ---------------------- |
| [Optimising a Function](01-optimizing-a-function.md) | `Landscape`, `Strategy`, the GA, the harness |
| [The Ask/Tell Contract](02-ask-tell-in-depth.md) | How `ask`/`tell` works inside `rlevo`; the `seed_stream`; writing a manual loop |
| [Classic Control: CartPole with DQN](03-classic-control.md) | `Environment`, `Action`, `Observation`, DQN, experience replay |
| [Bring Your Own Environment](04-extending-the-environment.md) | Implementing the `Environment` trait for a custom domain |

Each section builds on the previous one. The ask/tell section is the bridge:
it opens up the harness internals before you encounter them again in the RL
context, where the loop is more complex.
