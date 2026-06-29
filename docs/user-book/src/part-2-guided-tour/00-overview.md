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
| [Optimising a Function](10-optimizing-a-function.md) | `Landscape`, `Strategy`, the GA, the harness |
| [Classic Control: CartPole with DQN](20-classic-control.md) | `Environment`, `Action`, `Observation`, DQN, experience replay |
| [Bring Your Own Environment](99-extending-the-environment.md) | Implementing the `Environment` trait for a custom domain |

Each section builds on the previous one and is centred on code you can run. We
keep the main text on the example and push the deeper mechanics into
supplementary material you can reach for when you want it — the harness's
[ask/tell contract](../appendix-d-suppl/ask-tell-contract.md) and the
[fitness-landscape theory](../appendix-d-suppl/fitness-landscape.md) behind the
benchmark problems both live in [Appendix D](../appendix-d-suppl/index.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
