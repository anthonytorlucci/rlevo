# Hello, `rlevo`!

Welcome to `rlevo` and thank you for interest. `rlevo` is a library for 
**evolutionary deep reinforcement learning** wrriten in Rust,
built on the [Burn](https://burn.dev) deep learning framework. It gives you
two ways to search for good behaviour — **evolutionary computation (EC)** 
(populations that mutate  and select) and **gradient-based reinforcement 
learning (RL)** (agents that learn  from reward) - *and lets you mix them*.

### What this guide is

In this user guide, we will illustrate how `rlevo` defines and implements the 
fundamental concepts in reinforcement learning and evolutionary computation. 
We will discuss how `rlevo` takes advantage of Rust's type safety and const 
generics to encode the observation space rank, action space dimensionality, and 
network layer widths as type-level constants. `rlevo` was designed from the 
ground up for large, high-dimensional problems. A misconfigured 
environment-to-network interface becomes a compiler error rather than a silent 
shape broadcast or a runtime panic during a training run. That's the 
qualitative difference — you shift an entire class of bugs from "discovered 
during execution" to "impossible to express."

We don't expect users to be experts in deep learning, reinforcement learning, 
or evolutionary computation, but we do expect some familiarity. You 
will find links to the [docs.rs](https://docs.rs/rlevo/latest/rlevo/) and 
references listed in the bibliography for further reading. 

### What this guide is not

This guide is not a dissertation. We aim to provide users and researchers with 
enough information to build their own solutions for their own unique problem. 

Also, this is not an argument for Rust over Python. We (the original author) 
chose Rust for its type-safety, performance, and genuine enjoyment. We were 
already familiar with Burn, saw a gap in RL and EC, and set out to fill it. If 
you also enjoy building in Rust, we hope you enjoy building with `rlevo`.

So let's jump in!

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

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
