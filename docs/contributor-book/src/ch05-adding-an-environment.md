# Adding an environment

> **Status:** stub — prose and `{{#include}}` anchors coming in a follow-up PR.

**Why this exists.** Environments are the primary extension point for domain
researchers. The checklist here is the minimum a new environment must satisfy
before merging.

**Key source of truth.** [`docs/rules.md`](https://github.com/anthonytorlucci/rlevo/blob/main/docs/rules.md) §10, ADR-0011.

## Trait checklist

- [ ] `State<SR>` implemented for the state type.
- [ ] `Observation<R>` implemented for the observation type.
- [ ] `Action<AR>` implemented for the action type.
- [ ] `SnapshotBase` used (or a custom `Snapshot` implemented).
- [ ] `Environment<R, SR, AR>` with `reset` and `step` returning
      `Result<SnapshotType, EnvironmentError>`.
- [ ] `ConstructableEnv::new(render: bool)` factory (ADR-0011).
- [ ] `HostRow<R>` + `TensorConvertible<R, B>` for the observation (required
      for neural paths; `TensorConvertible` requires `HostRow` as a
      supertrait).
- [ ] Unit tests: happy path, invalid action, terminal state, `render = false`.
- [ ] `shape()` length matches const generic rank — verified in a test.

## `ConstructableEnv` (ADR-0011)

`fn new(render: bool)` was removed from `Environment` in ADR-0011. Use the
standalone `ConstructableEnv` factory trait instead. Do not add a `new` method
directly to `Environment`.

## Module placement

New environments go in `crates/rlevo-environments/src/` under the appropriate
subdirectory:

- `classic/` — standard control tasks (CartPole, MountainCar, …)
- `games/` — discrete game environments
- `landscapes/` — static optimisation landscapes (Sphere, Rastrigin, …)
- `locomotion/` — continuous physics environments (behind a feature flag)

## Outline

1. Step-by-step walk-through using `KArmedBandit` as the reference.
2. The const-generic trap — how to get `R`, `SR`, `AR` wrong and what the
   compiler error looks like.
3. `HostRow` / `TensorConvertible` — host-row layout, the device round-trip
   built on top of it, and precision caveats.
4. Test harness — the `MockEnvironment` pattern from `rlevo-core`.
5. Feature-gating optional backends (`box2d`, `locomotion`).
