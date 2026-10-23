---
project: rlevo
status: active
type: decision
date: 2026-10-23
tags: [adr, decision, architecture, burn, burn-0.22, rlevo-core, rlevo-evolution, rlevo-reinforcement-learning, rlevo-hybrid, autodiff, devices, breaking-change]
---

# ADR 0093: The backend is a runtime device (Burn 0.22)

## Status

**Accepted (2026-10-23).**

Moves the workspace from Burn 0.21.0 to 0.22.1 (cubecl 0.10 to 0.11.3).
Supersedes no ADR, and reverses one clause of another (below). Nineteen ADRs spell Burn 0.21's backend parameter and
are read as respelled: `Tensor<B, D>` is `Tensor<D>`, `Linear<B>` is
`Linear`, `Strategy<B>` is `Strategy`, `B::Device` is `Device`, and a
`B: AutodiffBackend` bound is autodiff as device state (Decision 2). They are
[0002](0002-collapse-evolution-traits-into-rlevo-evolution.md),
[0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md),
[0016](0016-memetic-wrapper-and-local-search-seam.md),
[0017](0017-probability-model-trait-and-eda-strategy.md),
[0020](0020-synthetic-pixel-over-grid-env.md),
[0022](0022-tensorgenome-gat-population-storage.md),
[0025](0025-stateful-policy-rollout-contract.md),
[0028](0028-tensor-batch-conversion-seam.md),
[0030](0030-permutation-tensorgenome-and-population-nonempty-invariant.md),
[0034](0034-fitness-hygiene-chokepoint-convention.md),
[0038](0038-continuous-action-components-const.md),
[0050](0050-replay-strategy-seam.md),
[0052](0052-hostrow-supertrait-splits-layout-from-backend.md),
[0054](0054-policy-head-construction-is-fallible.md),
[0055](0055-config-invariant-enforcement-allocation.md),
[0069](0069-sanitized-fitness-is-reduced-in-f64.md),
[0080](0080-library-crates-select-no-burn-backend.md),
[0086](0086-the-fitness-sanitisers-are-public.md) and
[0089](0089-the-contract-crate-does-not-depend-on-burn.md).
None rests on the parameter in a way 0.22 reverses. Three carry more than a
spelling:

- **0080.** Its decision stands: no library crate selects a backend. Its
  statement that the RL crate needs no `autodiff` feature is false on 0.22,
  where `optim` brings `autodiff` (Decision 5).
- **0052.** The doubled `TensorConvertible<R, B> + TensorConvertible<R,
  B::InnerBackend>` bound that motivated it is gone with the parameter. The
  decision stands for ADR 0089's reason.
- **0025.** Its guarantee that a rollout policy is not an `AutodiffBackend`
  type becomes the runtime contract of Decision 4. Its signature was already
  superseded by ADR 0091.

Two ADRs keep their decisions while their context changes (Measured, *NaN
semantics*):
[0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md)'s
`clamp_preserving_nan` stays, though `clamp` no longer rescues `NaN` on wgpu;
[0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md)'s
observation guards stay, though `relu` no longer erases `NaN` on Flex.

One clause of [0058](0058-target-update-type-unifies-cadence-and-tau.md) is
reversed. It returned `f64` from `TargetUpdate::tau` and
`TargetUpdate::fires_at` because the `soft_update` methods took `f64`. Those
methods are gone (Decision 3), so both return `f32`, the type
`polyak_update` takes, and no agent casts the value back.

## Context

Burn 0.22 removes the backend type parameter from tensors and modules. A
runtime `Device` says where they run, and autodiff is a property of the
device rather than a backend type:

| Burn 0.21 | Burn 0.22 |
|---|---|
| `Tensor<B, D>`, `Tensor<B, D, Int>` | `Tensor<D>`, `Tensor<D, Int>` |
| `Linear<B>`, a module `M<B>`, `Module<B>` | `Linear`, `M`, `Module` |
| `B::Device`, `type B = Flex` | `burn::tensor::Device`, built by `Device::flex()` or `Device::wgpu(..)` |
| `Autodiff<B>` as the backend; `B: AutodiffBackend` | `device.autodiff()`; `module.train()` / `module.valid()` return the same type |
| `AutodiffModule::InnerModule`, `B::InnerBackend` | gone: a `valid()` snapshot is the module's own type |
| `OptimizerAdaptor<Adam, M, B>` | `ModuleOptimizer` |
| `B::seed(&device, s)` | `device.seed(s)` |
| `into_scalar().elem::<T>()`, `TensorData::into_vec` | `into_scalar::<T>()`, `try_into_vec` |

Every rlevo signature that named `B` changes: 17 public traits (eight in
`rlevo-evolution`, eight in `rlevo-reinforcement-learning`, and
`rlevo-core`'s `TensorConvertible`), the RL agents, the evolution strategies
and the hybrid fitness adapter. The model
traits were also built on the type-level split. `DqnModel`, `C51Model`,
`QrDqnModel`, `DeterministicPolicy` and `ContinuousQ` each had a
`forward_inner` over `Self::InnerModule` for the target network and a
`soft_update` returning one. `SquashedGaussianPolicy` had
`forward_sample_inner`, and `PpoPolicy` had `deterministic_env_row_inner`.
On 0.22 there is no inner type for them to name.

The change touched 167 `.rs` files. A scripted rewrite of 19 rules produced
+5,355 / −8,033 lines; the hand share is +1,470 / −1,432 across 102 files,
documentation and CI included.

## Decision

1. **No backend parameter anywhere.** Every tensor, module, trait and type
   in the workspace drops it. A device is a `burn::tensor::Device` value.
   Library code takes the device it is given and never constructs one, so
   ADR 0080 holds as before: the leaf picks the backend, through the
   umbrella's `flex` and `wgpu` features or a crate's dev-dependencies.

2. **An RL agent switches autodiff on itself.** `new(net, config, device)`
   accepts a network built on any device and trains on `device.autodiff()`,
   after `net.train()`. It is `train()`, not `fork`: `Device::flex() ==
   Device::flex().autodiff()` holds, so `fork` onto the autodiff form of the
   same device is a no-op, and the network stays untrainable (measured). The
   target path (target networks, the bootstrap inputs, SAC's actor
   snapshot, every `act_*_with` inference path) runs on
   `device.without_autodiff()`.

3. **A model trait asks for `forward` only.** The target network is a
   `valid()` snapshot, which is the model's own type, so `forward` serves
   both paths. Every `soft_update` implementation was the same
   `polyak_update(&active.valid(), target, tau)` call, which the agent now
   makes itself. `forward_inner`, `soft_update` and `forward_sample_inner`
   are removed. `PpoPolicy::deterministic_env_row_inner(inner, obs)` becomes
   the method `deterministic_env_row(&self, obs)`, called on a `valid()`
   snapshot. `TargetUpdate::tau` and `fires_at` return `f32`, the type
   `polyak_update` takes.

4. **A test replaces the type guarantee.** On 0.21 the target network's type
   kept it off the autodiff graph. Now each off-policy agent with a target
   network (DQN, C51, QR-DQN, DDPG, TD3, SAC) has a
   `test_*_target_path_is_never_tracked`. It asserts that every target
   parameter is on a plain device, does not require a gradient and is
   untracked, after construction and after a Polyak update, and that the
   trained networks are on an autodiff device. The assertions are the
   test-only `params_are_plain` and `params_train` in
   `rlevo-reinforcement-learning`'s `algorithms::shared`. The tests fail under
   each mutant tried: a target built with `clone()` instead of `valid()`, a
   Polyak update blended from the training network, and SAC's actor
   snapshot cloned.

5. **The RL crate turns on Burn's `optim`.** 0.22 puts the optimizers
   behind `optim`, which brings `autodiff`. Neither selects a backend, so
   the RL crate resolves `burn` to `[autodiff, burn-optim, optim, std]` and
   every other library crate to `[std]`. The `libraries-select-no-backend`
   CI job checks exactly these sets.

6. **Tests, benches, examples and doc examples name their device.**
   `Device::flex()`, `Device::flex().autodiff()` or a named adapter, never
   `Device::default()`. On 0.22 the default device is whichever backend the
   build compiles first, which in a test build that compiles wgpu is a wgpu
   device, and a wgpu device aborts on a runner with no GPU adapter.
   `tools/no-default-device.sh`, run in CI, rejects `Device::default()` and a
   `device` binding initialised with `Default::default()`. A bare
   `Default::default()` passed to a device parameter escapes a grep; clippy's
   `default_trait_access` names it as `Device::default()`. That census found
   21 such sites, all now `Device::flex()`.

7. **A function that ran once per backend takes its device as a value.**
   Dropping the type argument turns `run::<Flex>()` and `run::<Wgpu>()`
   into two identical calls that still compile, and a body that built its
   device from the type then runs both on one device. Four sites did this
   (a staging bench, the learn-step bench, an evolution parity test and the
   backend-sweep example). Each now takes a `Device`, and the benches select
   theirs with a closed `BenchDevice` enum.

8. **Checkpoints are burnpack, and rlevo writes none yet.** Burn 0.22 saves
   a module with `save_file` / `try_load_file` and an optimizer with
   `ModuleOptimizer::save` / `load`, as separate records, gated on `std`
   alone; the facade's `store` feature adds only SafeTensors and PyTorch
   interop. A round trip of a DQN agent's networks and optimizer state was
   exact (measured). No rlevo crate saves a module today. The `Module`
   derive no longer generates a record type per module, so the four
   `*Record` / `*RecordItem` pairs the RL crate's public policy heads
   carried are gone; every module's record is `burn::store::ModuleRecord`.

9. **rlevo does not enable `burn::rl`.** Burn 0.22's opt-in `rl` feature has
   its own `Environment` (untyped `State` / `Action`, `const MAX_STEPS`,
   `step` returning `{ next_state, reward: f64, done, truncated }`, an
   infallible `reset`), `Policy`, `TransitionBuffer` and `PolicyLearner`.
   rlevo's `Environment<OR, AR>` has typed ranks, a `Snapshot` whose
   `EpisodeStatus` separates termination from truncation, fallible `reset` and
   `step`, and a finite `f32` reward. The two are unrelated, and an author
   reading both should not expect one to implement the other.

## Measured

Apple M2 Pro (12 CPU cores, 19 GPU cores, Metal 4), 32 GiB, macOS, rustc
1.99.0; wgpu on its default adapter, the M2 Pro GPU through Metal. Each
comparison runs the same commit on Burn 0.21 and on 0.22.

**Known answers.** A fingerprint per algorithm family: a seeded run with
every metrics row and the trained weights hashed bit for bit, on Flex.

| Run | 0.21 to 0.22 |
|---|---|
| DQN, C51, QR-DQN (`CartPole`, 2,500 steps); DDPG, TD3 (2,000 steps); PPO discrete and continuous; CMA-ES (Rastrigin-D10, 300 generations) | bit-identical |
| SAC (2,000 steps) | identical for 25 episodes, then the last bit of one critic loss |
| PPG (`CartPole`, 2,048 steps) | differs from the first update |

- **SAC.** Burn's `softplus` changed from `log(exp(βx) + 1) / β` to
  `log1p(exp(βx)) / β`, with the identity above a threshold of 20. Of fifteen
  ops probed (forward and gradient over 4,096 inputs), it is the only one
  whose output moved; its gradient did not. SAC's squash Jacobian uses it.
- **PPG.** Burn initialises a parameter lazily, from Flex's process-global
  RNG, the first time it is read. On 0.21 PPG's auxiliary value head, which
  no rollout reads, drew its weights after the value network did; on 0.22
  `train()` at construction draws every parameter in field order. Reading
  the policy's weights at construction on 0.21 reproduces the 0.22 run bit
  for bit.

**NaN semantics.** Probed on both versions:

| | 0.21 Flex | 0.21 wgpu | 0.22 Flex and wgpu |
|---|---|---|---|
| `relu(NaN)` | `0.0` | `NaN` | `NaN` |
| `argmax([1, NaN, 3, 2])` | `1` | `2` | `1` |
| `clamp(NaN, -10, 10)` | `NaN` | `-10` | `NaN` |
| ReLU actor on an all-`NaN` observation | `0.0847` | `NaN` | `NaN` |
| `floor(NaN).int()` | `0` | `i32::MIN` | `0` (Flex), `i32::MIN` (wgpu) |

On 0.22 a non-finite observation reaches a continuous agent's action as
`NaN` on both backends, so the action fails `is_valid`. A discrete agent
still picks an in-domain index, the first `NaN`'s, or `0` from an all-`NaN`
row. ADR 0067's guards stay: they name the cause where the action shows only
the effect, and they remain the one observer on a backend whose `relu`
erases `NaN`. ADR 0066's helper stays, because `clamp`'s `NaN` behaviour is
undocumented and Vulkan, CUDA and the `cubecl-cpp` paths are unmeasured.
C51's index clamp stays load-bearing, because a `NaN`-to-integer conversion
still differs between the backends.

**Cones.** Unique packages in `cargo tree -e normal`, less one; how many are
`burn*` after the slash:

| Crate | 0.21 | 0.22 |
|---|---|---|
| `rlevo-core` (with `burn`) | 104 / 8 | 114 / 12 |
| `rlevo-reinforcement-learning` | 108 / 8 | 120 / 14 |
| `rlevo-evolution` | 114 / 8 | 123 / 12 |
| `rlevo-hybrid` | 115 / 8 | 124 / 12 |
| `rlevo` (default) | 209 / 11 | 195 / 16 |
| `rlevo --features wgpu` | 255 / 15 | 403 / 20 |

The outside-in probes give the same walk results on both versions. A
consumer of `rlevo-core` alone builds 16 packages, none from Burn, either
way. `rlevo-core` with `rlevo-reinforcement-learning` goes from 109
packages (8 Burn) to 121 (14), and `rlevo` at default features from 210
(11) to 196 (16).

The wgpu cone grows because Burn's default wgpu configuration enables
cubecl's autotune persistence, an on-disk cache backed by `turso`, an
embedded SQLite, with its crypto and ICU dependencies. The `wgpu` feature
keeps `burn/default`, because dropping it also drops autotune and fusion.

**Flex runtime.** Criterion medians: one full DQN update from
`learn_step_bench`'s Flex groups (`host_row` staging), and evolution's
`operators`. The small-MLP and DE rows were re-run with the versions in the
opposite order and agree within 2%.

| Benchmark | 0.21 | 0.22 | 0.22 / 0.21 |
|---|---|---|---|
| learn step, pixel conv net, batch 64 / 256 | 21.0 / 80.7 ms | 8.7 / 28.9 ms | 0.42 / 0.36 |
| learn step, wide MLP, batch 64 / 256 | 3.95 / 6.00 ms | 4.26 / 6.36 ms | 1.08 / 1.06 |
| learn step, small MLP, batch 64 / 256 | 123 / 249 µs | 148 / 274 µs | 1.20 / 1.10 |
| DE generation, population 64 / 1,024 | 11.5 / 120 µs | 12.9 / 138 µs | 1.12 / 1.15 |
| tournament selection, population 64 / 1,024 | 1.3 / 14.7 µs | 1.4 / 14.8 µs | 1.07 / 1.01 |

Convolution is two to three times faster. Small dense work is 6 to 30%
slower, most where a step is many small operations: the small MLP with
per-row `roundtrip` staging is 1.28 to 1.29 times slower. That fits a fixed
cost per operation; it was not profiled.

## Consequences

- Every public signature that named a backend parameter changes, and the
  model traits lose their `_inner` and `soft_update` methods. An
  implementation keeps its `forward`.
- An earlier ADR is read with the parameter dropped; `CLAUDE.md` says how.
- A wgpu result from 0.21 does not carry over: NaN semantics, last-bit
  kernel numerics and the seeded initialisation layout all changed on wgpu.
  No wgpu answer is pinned in the workspace.
- The target network's isolation from the autodiff graph is a tested
  property rather than a typed one, so an agent added later needs its own
  target-path test.

## Alternatives considered

- **Keep `forward_inner` and `soft_update` under a runtime contract.**
  Rejected: with a `valid()` snapshot of the model's own type, `forward_inner`
  is `forward` under another name, and nothing overrides `soft_update`. A
  hook kept for an override nobody writes is surface without a consumer
  (ADR 0081).
- **The caller switches autodiff on.** Rejected: every caller would have to
  know that an RL agent needs `device.autodiff()` and `net.train()`, and a
  caller who passes a plain network gets one that does not train, with no
  error until `backward`.
- **Land the migration per crate, with two Burns in the tree between pull
  requests.** Rejected: possible along two groups that share no Burn type
  (core and RL; evolution and hybrid), but the umbrella, `rlevo-test-support`
  and `rlevo-examples` cross both groups and would carry two Burns, and the
  transitional names would be public surface whose only future is deletion.

## References

- Burn 0.22 release notes and its migration guide, `burn-book/src/migrating-to-0.22.md`.
- ADR [0080](0080-library-crates-select-no-burn-backend.md): library crates
  select no backend.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md):
  every public item needs an implementor and a consumer.
- ADRs [0066](0066-clamp-nan-behavior-is-backend-specific-pin-with-is-nan.md)
  and [0067](0067-non-finite-observations-are-dropped-at-replay-ingestion.md):
  the NaN guards whose context changed.
