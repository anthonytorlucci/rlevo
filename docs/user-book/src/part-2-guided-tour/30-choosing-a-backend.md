# Choosing a backend: CPU vs GPU

Every tensor in `rlevo` is generic over a **backend** — the type parameter `B`
you have seen threaded through `Strategy<B>`, `Tensor<B, 2>`, and the
neuroevolution phenotypes. That generality is inherited from
[Burn](https://burn.dev), and it buys you something concrete: you write a run
*once* and choose, at the type level, whether its arithmetic executes on the CPU
(`Flex`) or the GPU (`Wgpu`). No `#[cfg]`, no duplicated code path, no rewrite.

The tempting assumption is that the GPU is simply the faster choice and you
should reach for it by default. It is not, and you should not. Whether the GPU
helps depends entirely on the *shape* of the computation — and `rlevo`'s two
headline workloads sit at opposite ends of that spectrum. This chapter shows you
where each backend wins, why, and how to switch between them with a single type
argument.

## The two regimes

A backend's job is to run tensor operations. A GPU runs a *few large* operations
extraordinarily fast, because it amortises a fixed per-operation cost — kernel
dispatch, queue submission, and host↔device synchronisation — across thousands
of parallel lanes. A CPU backend like `Flex` has no such fixed cost per op: an
operation is a direct, in-process `gemm` call. So the question is never "CPU or
GPU?" in the abstract; it is "does each operation do *enough* work to pay off the
GPU's overhead?"

That reframing sorts `rlevo`'s workloads cleanly:

- **Single-environment RL rollout — CPU wins.** A PPO-on-CartPole rollout (the
  [Classic Control](20-classic-control.md) chapter's shape) steps one
  environment at a time through a tiny `4 → 64 → 64` network. Every step is a
  handful of small operations, each of which reads a scalar back to the host to
  choose an action — a host↔device sync that stalls the GPU queue. There is no
  batch to amortise anything. On this workload the GPU is roughly **70× slower**
  than `Flex`, which is exactly why the bundled `ppo_cartpole` scaffolding is
  pinned to `Autodiff<Flex>`.
- **Population neuroevolution — GPU wins.** Evolving the weights of a network
  flips every one of those properties. A whole *population* of \\(P\\) candidate
  networks is one `(P, …)` tensor (the [parameter
  bridge](../part-1-foundations/neuroevolution/41-param-bridge.md)), and scoring
  it is a single batched matmul with **no autodiff tape** and **no per-step
  sync**. That is precisely the large-batched-arithmetic case the GPU was built
  for.

The rest of this chapter makes the second claim concrete and measurable.

## One generic run, two backends

The example lives at
[`backend_sweep_neuroevolution`](https://github.com/anthonytorlucci/rlevo/blob/main/crates/rlevo-examples/examples/evolution/backend_sweep_neuroevolution.rs).
Its fitness function interprets each genome row as the weights of a small
`in → hidden → 1` MLP, runs the **entire population** over a fixed input batch in
one shot, and scores each candidate by mean-squared error against a fixed
nonlinear target. The heavy lifting is two batched matmuls:

```rust,ignore
{{#rustdoc_include ../../../../crates/rlevo-examples/examples/evolution/backend_sweep_neuroevolution.rs:forward}}
```

The shapes tell the story. With a population of \\(P\\), a sample batch of
\\(N\\), input width \\(I\\), and hidden width \\(H\\), the first matmul is
\\(P \times N \times I \times H\\) fused multiply-adds — a single dense
operation whose size grows with the population. This is deliberately *not* the
bundled [`FromLandscape`](https://docs.rs/rlevo-evolution) adapter, which pulls
every genome row to the host and evaluates on the CPU: that adapter is correct,
but it is backend-agnostic precisely because the compute never touches the
device. To make the backend choice *matter*, the arithmetic has to stay on
\\(B\\)'s device — so we implement
[`BatchFitnessFn`](https://docs.rs/rlevo-evolution) directly and keep everything
on-device.

Everything above the fitness — the genetic algorithm, the
[`EvolutionaryHarness`](https://docs.rs/rlevo-evolution), the ask/tell loop — is
the same machinery you met in [Optimising a Function](10-optimizing-a-function.md).
None of it names a concrete backend. So the entire run is one function generic
over `B: Backend`, and the sweep instantiates it twice:

```rust,ignore
{{#rustdoc_include ../../../../crates/rlevo-examples/examples/evolution/backend_sweep_neuroevolution.rs:sweep}}
```

`run::<Flex>` and `run::<Wgpu>` are the *same code*. The CPU and GPU runs differ
by a single type argument — that is Burn's backend genericity, and `rlevo`'s, in
one line.

## What you should see

Run it in release mode — the CPU backend leans on optimised `gemm`, so a debug
build would flatter the GPU unfairly:

```bash
cargo run --release -p rlevo-examples --example backend_sweep_neuroevolution
```

On an Apple M2 (`in = 16`, `hidden = 128`, `samples = 512`, `gens = 20`):

```text
pop_size     Flex (CPU)      Wgpu (GPU)     speedup
      64        0.970 s         0.187 s        5.18x
     512        8.209 s         0.479 s       17.14x
    4096       64.829 s         3.000 s       21.61x
```

Two things to read off this table. First, the GPU wins across the *whole* range
— even at \\(P = 64\\), because a 512-sample batched matmul is already large
enough to pay off the dispatch overhead. Second, and more important, the GPU's
margin **widens** with the population: as \\(P\\) grows the CPU cost scales
linearly with the arithmetic, while the GPU absorbs more of it in parallel. The
absolute numbers are machine-specific; the *shape* of the result — a rising
speedup curve — is the lesson.

There is no CPU-favourable crossover inside this range. That regime lives at the
opposite extreme, in the single-environment PPO rollout, where `Flex` is ~70×
faster. The two examples bracket the trade-off:

> **The rule of thumb.** Sequential-and-tiny favours the CPU; batched-and-large
> favours the GPU. Population evaluation and vectorised inference are
> batched-and-large by construction, which is why neuroevolution is `rlevo`'s
> natural home for the GPU.

## Why the difference is so stark

It is worth being precise about the two forces at work, because they generalise
beyond this example.

**Kernel dispatch and sync overhead.** Each GPU operation must be packaged into a
command buffer, submitted to the device queue, and — whenever a result is read
back — synchronised against the host. For the tiny sequential PPO ops this
fixed cost dwarfs the arithmetic. For a `(P, N, I) · (P, I, H)` matmul the same
fixed cost is paid once over billions of multiply-adds, so it vanishes into the
noise.

**No autodiff tape.** Gradient-based training wraps its backend in
`Autodiff<B>`, which records a backward graph on every forward op — extra
allocation and bookkeeping the GPU must also synchronise around. Weight-only
neuroevolution needs *no gradients*: the evolution stack is generic over
`B: Backend`, **not** `AutodiffBackend`, so a population forward pass is pure
inference with none of that overhead. The genetic algorithm gets its search
signal from fitness comparisons, not from `.backward()`. That is a structural
advantage of neuroevolution on the GPU, not an incidental one.

## A note on logging

If you have run the GPU path elsewhere and seen a burst of adapter-selection and
autotune log lines that the CPU path never produced, that is not your
imagination — but it is also not the backend being noisier for its own sake.
Those records come from `cubecl`/`wgpu` via the plain `log` crate, and they only
surface when a `tracing` subscriber with a `LogTracer` bridge is installed to
forward them. This example installs **no** subscriber, so the GPU run is as quiet
as the CPU run. If you want GPU compute *and* quiet logs in your own binary,
install a `tracing` subscriber with an `EnvFilter` that drops the `cubecl_wgpu`
and `cubecl_runtime` targets below `warn`.

## Make it yours

The fastest way to feel the crossover is to move the workload toward the
CPU-favourable end and watch the speedup collapse:

- **Shrink the per-evaluation work.** Drop `hidden` to `8` and `samples` to `16`.
  Now each matmul is small, the GPU's dispatch overhead stops being amortised,
  and at `pop_size = 64` the CPU can pull *ahead* — the crossover the PPO example
  demonstrates at its extreme, reproduced here in miniature.
- **Grow the population.** Push `POP_SIZES` past `4096` and the GPU margin keeps
  widening until you saturate device memory.
- **Switch the whole run to the GPU permanently.** Change one type alias — every
  `run::<Flex>` becomes `run::<Wgpu>` — and nothing else moves. That is the
  point.

## Where this is going

You now have the practical half of `rlevo`'s backend story: choose `Flex` for
sequential RL rollouts and small problems, reach for `Wgpu` when you are
evaluating populations or batches large enough to keep the device busy. The
conceptual half — *why* weights-as-a-flat-genome makes population inference a
single batched matmul in the first place — is developed in
[The Parameter Bridge](../part-1-foundations/neuroevolution/41-param-bridge.md),
and the broader case for pairing gradient-free evolution with gradient-based RL
is made in [Why Combine Them?](../part-1-foundations/50-why-combine.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
