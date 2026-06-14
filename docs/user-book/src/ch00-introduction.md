# Introduction

`rlevo` is a library for **evolutionary deep reinforcement learning** in Rust,
built on the [Burn](https://burn.dev) tensor framework. It gives you two ways to
search for good behaviour — **evolution** (populations that mutate and select)
and **gradient-based reinforcement learning** (agents that learn from reward) —
and lets you mix them.

This book is for **researchers**. You bring a problem from your own field; the
book shows you how to express it in `rlevo` and turn the crank. We won't derive
the algorithms — we'll *use* them, starting from a problem so small it fits in
your head and ending with a custom environment of your own.

## The shape of every problem

Almost everything in `rlevo` is one of two seams. Learn these two words and the
rest of the book is just variations:

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
it how they scored — the **ask/tell** contract. That means you can evaluate
candidates however you like: in parallel, on a cluster, against a simulator you
already own, or against a real experiment. Chapter 1 makes this concrete.

## What you'll build

| Chapter | You'll make rlevo… |
| ------- | ------------------ |
| 1 | minimise a function with a genetic algorithm |
| 2 | balance a pole with a Deep Q-Network |
| 3 | wrap your own problem as an `Environment` |
| 4 | evolve the weights of a neural network instead of training them |
| 5 | bolt local refinement onto any strategy (memetic algorithms) |
| 6 | beat a deceptive landscape with estimation-of-distribution algorithms |
| 7 | let evolution choose the network architecture |
| 8 | combine evolutionary search with gradient learning |
| 9 | watch a run live and replay it afterwards as an HTML report |

## Before you start

You'll need a recent Rust toolchain. `rlevo` is a Cargo workspace; the examples
in this book live in the `rlevo-examples` crate and you can run any of them with:

```bash
cargo run -p rlevo-examples --example <name>
```

Each chapter names the example it's built from. The code in the book *is* that
example — pulled in verbatim, compiled and tested in CI — so what you read is
what runs.

You do **not** need to know Burn to start. We introduce tensors only when a
chapter needs them (Chapter 2), and we keep them at arm's length.

Turn the page and let's optimise something.
