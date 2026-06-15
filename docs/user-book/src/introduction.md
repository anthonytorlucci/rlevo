# Introduction

`rlevo` is a library for **evolutionary deep reinforcement learning** in Rust,
built on the [Burn](https://burn.dev) tensor framework. It gives you two ways to
search for good behaviour — **evolution** (populations that mutate and select)
and **gradient-based reinforcement learning** (agents that learn from reward) -
*and lets you mix them*.

This book serves two kinds of reader simultaneously:

- **Practitioners** who want to get something running fast: follow Part II, skim
  the theory boxes, run the examples.
- **Students and researchers** who want to understand *why* each algorithm works:
  start with Part I, use Part II to see the concepts in code, and reach for the
  Appendices when you want derivations and pseudocode.

Neither path requires you to read the book cover to cover. Cross-references point
from the guided tour back to the foundations, and from the foundations forward to
where those ideas appear in `rlevo` code.

## The shape of every problem

Almost everything in `rlevo` is one of two seams:

- An **`Environment`** is the *world*: it has a state, it accepts an action, and
  it hands back an observation and a reward. Balancing a pole, walking a robot,
  or — in the degenerate case — evaluating a mathematical function: all
  environments.
- A **`Strategy`** is the *searcher*: it proposes candidate solutions and
  updates itself based on how well they scored. A genetic algorithm, an
  evolution strategy, an estimation-of-distribution model: all strategies.

Between them sits a **fitness** (for evolution) or a **reward** (for RL) — the
single number that says "this was good." The whole library is machinery for
proposing candidates, scoring them against an environment, and proposing better
ones.

```text
        proposes candidates              scores them
  Strategy  ───────────────►  (you)  ───────────────►  Environment
     ▲                                                      │
     └──────────────  updates from fitness  ◄──────────────┘
```

Notice that *you* sit in the middle. `rlevo` deliberately does **not** own the
loop. The `Strategy` *asks* you for the next batch of candidates and you *tell*
it how they scored — the **ask/tell** contract. Part II makes this concrete.

## How this book is organised

| Part | What you'll find |
| ---- | ---------------- |
| I — Foundations | RL and EA concepts in `rlevo` |
| II — Guided Tour | A narrative walkthrough: function optimisation → the ask/tell contract → classic control with RL |
| III — Open Problems | Honest gaps, active research directions, and how to contribute |
| Appendices | Algorithm derivations, pseudocode, math notation, and a full bibliography |

## Before you start

You'll need a recent Rust toolchain. The examples in this book live in the
`rlevo-examples` crate:

```bash
cargo run -p rlevo-examples --example <name>
```

Each section names the example it's built from. The code in the book *is* that
example — compiled and tested in CI — so what you read is what runs.

You do **not** need to know Burn to start. Tensors appear only when a section
needs them, and are kept at arm's length until then.

---

*Co-Authored-By: Anthropic Claude Sonnet 4.6*
*Reviewed-By: (Human) Anthony Torlucci*
