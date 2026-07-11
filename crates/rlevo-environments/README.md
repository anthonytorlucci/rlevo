# rlevo-environments

Standard benchmark environments and landscapes for the `rlevo` workspace.

This crate provides a collection of reinforcement learning environments ranging from classic tabular problems through continuous-control physics simulations. All environments implement the `rlevo-core` `Environment` trait — a common `reset` / `step` interface that makes them drop-in compatible with every algorithm in `rlevo-reinforcement-learning` and `rlevo-evolution`.

---

## Environments

### Classic Control

Ports of the canonical Gymnasium control tasks implemented in pure Rust.

| Environment | Module | Observation | Action | Notes |
|---|---|---|---|---|
| `CartPole` | `classic::CartPole` | 4-D continuous | Discrete(2) | Physics match Gymnasium `CartPole-v1` exactly |
| `Acrobot` | `classic::Acrobot` | 6-D continuous | Discrete(3) | Sutton two-link dynamics |
| `MountainCar` | `classic::MountainCar` | 2-D continuous | Discrete(3) | Sparse reward; needs exploration |
| `MountainCarContinuous` | `classic::MountainCarContinuous` | 2-D continuous | Continuous(1) | Dense reward variant |
| `Pendulum` | `classic::Pendulum` | 3-D continuous | Continuous(1) | Underactuated swing-up |

---

### Toy Text

Tabular MDPs for baseline algorithm validation. Every state is fully observable.

| Environment | Module | State space | Action |
|---|---|---|---|
| `FrozenLake` | `toy_text::FrozenLake` | 16 or 64 discrete | Discrete(4) |
| `CliffWalking` | `toy_text::CliffWalking` | 48 discrete | Discrete(4) |
| `Taxi` | `toy_text::Taxi` | 500 discrete | Discrete(6) |
| `Blackjack` | `toy_text::Blackjack` | 32×11×2 discrete | Discrete(2) |

---

### Gridworlds

Twelve partially observable grid environments inspired by [Farama Minigrid](https://minigrid.farama.org). All share an egocentric 7×7 view and a 7-action discrete space. Physics are implemented once in `grids::core` and reused across every variant.

Eleven of the twelve encode that view as the shared 3-channel `GridObservation` (`[type, color, state]` per cell). **`GoToDoorEnv` is the one exception**: it emits a bespoke 4-channel `GoToDoorObservation`, whose extra channel carries the episode's mission (ADR 0043 — see the row below).

| Environment | Grid size | Observation | Key mechanic |
|---|---|---|---|
| `EmptyEnv` | 6×6 | 7×7×3 | Reach the goal |
| `DoorKeyEnv` | 8×8 | 7×7×3 | Pick up key, unlock door, reach goal |
| `LavaGapEnv` | 7×7 | 7×7×3 | Navigate a gap in a lava wall |
| `FourRoomsEnv` | 19×19 | 7×7×3 | Long-horizon exploration |
| `UnlockEnv` | 6×6 | 7×7×3 | Single lock/key |
| `UnlockPickupEnv` | 7×7 | 7×7×3 | Unlock then pick up object |
| `MemoryEnv` | 13×13 (min 11, odd) | 7×7×3 | POMDP recall: a green Key-or-Ball cue is **sampled each episode**, then must be matched by *type* at a fork from which the cue is not observable |
| `MultiRoomEnv` | variable | 7×7×3 | Chain of rooms |
| `CrossingEnv` | 11×11 | 7×7×3 | Navigate obstacles |
| `DistShiftEnv` | 9×… | 7×7×3 | Adaption to distribution shift |
| `DynamicObstaclesEnv` | 6×6 | 7×7×3 | Moving obstacles |
| `GoToDoorEnv` | 6×6 | **7×7×4** | Instruction-conditioned: the four door colours **and the target are re-sampled each episode**; the mission colour rides in observation channel 3, broadcast to every cell |

`MemoryEnv`'s minimum size of 11 is not arbitrary. rlevo's egocentric view applies no occlusion, so only *distance* can keep the cue out of sight from the fork; `size >= 11` is exactly the distance required (Invariant M, ADR 0043). Lowering it re-opens the bug that made the task solvable without memory.

Be precise about what that buys, though. The cue is **not** hidden everywhere — because the view reaches 6 cells backward from the cue's fixed column (`x = 1`), an agent standing anywhere in the corridor at `x <= 7` can turn to face West and re-read it. That leak zone is fixed by the view geometry and does not depend on `size`. What Invariant M guarantees is that the cue is unobservable **at the fork**, and that no single observation ever contains the cue *and* a fork object (they are more than a view-width apart) — which is what makes a memoryless policy unable to beat chance.

Since the leak zone is fixed while the corridor grows, the **recall horizon scales with `size`**: the cue-free corridor cells are exactly `x ∈ [8, size - 3]`. At the floor of `size = 11` that is the single gap cell (≈1–2 steps of retention); at the **default `size = 13`** it is three cells; at `17`, seven. The default is deliberately *not* the minimum: `11` is the smallest **correct** size, not a memory benchmark, and shipping it would ship the weakest recall task the layout supports. Use the default (`13`, `max_steps = 845`) for memory research, or `17` for a long-horizon variant; drop to `11` only when step budget matters more than horizon.

---

### Box2D-style Physics

Continuous-control environments powered by [Rapier2D](https://rapier.rs). Enabled by the `box2d` feature (default on).

| Environment | Module | Observation | Action |
|---|---|---|---|
| `BipedalWalker` | `box2d::BipedalWalker` | 24-D continuous | Continuous(4) |
| `LunarLanderDiscrete` | `box2d::lunar_lander` | 8-D continuous | Discrete(4) |
| `LunarLanderContinuous` | `box2d::lunar_lander` | 8-D continuous | Continuous(2) |
| `CarRacing` | `box2d::CarRacing` | 96×96 pixel | Continuous(3) |

---

### Locomotion

MuJoCo-style locomotion environments in pure Rust via [Rapier3D](https://rapier.rs). Enabled by the `locomotion` feature (default on).

| Environment | Module | Notes |
|---|---|---|
| `InvertedPendulum` | `locomotion::InvertedPendulum` | 1-D balance |
| `InvertedDoublePendulum` | `locomotion::InvertedDoublePendulum` | Harder balance; sparser failure |
| `Reacher` | `locomotion::Reacher` | 2-DOF arm reaching |
| `Swimmer` | `locomotion::Swimmer` | 3-link swimmer |

---

### Games (planned for v0.2)

`Chess` and `ConnectFour` are planned for a future release. Stub modules exist
in-source (`src/games/chess/` and `src/games/connect_four.rs`) but do not yet
implement the `Environment` trait and are hidden from the public API docs.

---

### Optimization Landscapes

Single-objective fitness functions for evaluating evolutionary algorithms. Each
is a stateless N-D evaluator; the global optimum is a *minimum* by convention.
The functions are grouped into three tiers by intended use.

**Tier 1 — scalable n-D.** Arbitrary-dimension functions for sweeping
performance against problem size.

| Function | Module | Dim | Notes |
|---|---|---|---|
| Sphere | `landscapes::sphere` | n-D | Convex, unimodal; trivial baseline |
| Rastrigin | `landscapes::rastrigin` | n-D | Highly multimodal; regular cosine lattice |
| Ackley | `landscapes::ackley` | n-D | Multimodal; near-flat outer region, one deep basin |
| Griewank | `landscapes::griewank` | n-D | Dense lattice of minima; paradoxically easier at high n |
| Michalewicz | `landscapes::michalewicz` | n-D | Steep ridges, near-flat plateaus; n!-scaling minima |
| Penalized No.1 | `landscapes::penalized1` | n-D | Sinusoidal lattice with quartic boundary penalties |
| Rosenbrock | `landscapes::rosenbrock` | n-D (n≥2) | Smooth curved "banana" valley; near-singular Hessian |
| Schwefel | `landscapes::schwefel` | n-D | Deceptive; optimum far from centre near the domain edge |
| Concatenated Trap | `landscapes::concatenated_trap` | binary (n·k) | Deceptive, decomposable; strong all-zeros trap |

**Tier 2 — classical 2-D.** Well-known low-dimensional surfaces with
characterised optima, useful for visualisation and surrogate-model tests.

| Function | Module | Dim | Notes |
|---|---|---|---|
| Branin RCOS | `landscapes::branin` | 2-D | Three equal, non-symmetric global minima; smooth |
| Bukin No.6 | `landscapes::bukin6` | 2-D | Knife-edge parabolic ridge; non-smooth |
| Cross-in-Tray | `landscapes::cross_in_tray` | 2-D | Four equal minima; V-kinks along the axes |
| Easom | `landscapes::easom` | 2-D | Needle-in-haystack; flat except a tiny basin at (π, π) |
| Goldstein–Price | `landscapes::goldstein_price` | 2-D | Six-order-of-magnitude range; f* = 3 |
| Himmelblau | `landscapes::himmelblau` | 2-D | Four equal minima; classic niching test |
| Six-Hump Camel | `landscapes::six_hump_camel` | 2-D | Two global minima among six humps |

**Tier 3 — stress tests.** Pathological surfaces that probe a specific failure
mode (non-smoothness, deception, vanishing optimal volume).

| Function | Module | Dim | Notes |
|---|---|---|---|
| Alpine No.1 | `landscapes::alpine1` | n-D | Non-smooth; ~eight kinks per axis stall gradients |
| Deb No.1 | `landscapes::deb1` | n-D (n≤2) | 10ⁿ equal optima; diversity / non-uniqueness test |
| Eggholder | `landscapes::eggholder` | n-D (n≥2) | Deceptive; optimum pinned to the domain boundary |
| Lunacek bi-Rastrigin | `landscapes::lunacek_bi_rastrigin` | n-D (n≥2) | Competing wide/narrow funnels plus Rastrigin oscillation |
| Needle-Eye | `landscapes::needle_eye` | n-D | Piecewise-constant; astronomically small optimal region |
| Modified Rosenbrock | `landscapes::rosenbrock_flat` | n-D (n≥2) | Bent knife-edge; flat, non-differentiable ridge |
| Trefethen | `landscapes::trefethen` | 2-D | Five incommensurate frequencies; no periodic lattice |

---

## Quick Start

```rust,no_run
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{CartPole, CartPoleConfig, CartPoleAction};
use rlevo_environments::wrappers::TimeLimit;

let env = CartPole::with_config(CartPoleConfig::default());
let mut env = TimeLimit::new(env, 500);

let mut snap = env.reset().expect("reset");
while !snap.is_done() {
    snap = env.step(CartPoleAction::Right).expect("step");
}
```

Gridworld environments use the shared `GridAction` type:

```rust,no_run
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::grids::{EmptyEnv, EmptyConfig};
use rlevo_environments::grids::core::GridAction;

let mut env = EmptyEnv::with_config(EmptyConfig::default(), false);
let mut snap = env.reset().expect("reset");
while !snap.is_done() {
    snap = env.step(GridAction::Forward).expect("step");
}
```

---

## Cargo Features

| Feature | Default | Description |
|---|---|---|
| `box2d` | yes | Box2D-style physics environments via `rapier2d` |
| `locomotion` | yes | Locomotion environments via `rapier3d` + `nalgebra` |
| `bench` | no | `BenchAdapter` and preset `Suite` factories for the `rlevo-benchmarks` harness; keeps the base dep cone lean when off |
| `record` | no | `RecordedEnvFamily` impls tying each env to its recording family (implies `bench`) |

Disable physics environments to shrink compile time:

```toml
[dependencies]
rlevo-environments = { path = "…", default-features = false }
```

---

## Design

### Configuration Builder

Every environment ships with a `XyzConfig` struct and a `XyzConfigBuilder` with a fluent API. Defaults match the reference Gymnasium implementation where one exists.

```rust,no_run
use rlevo_environments::classic::{CartPole, CartPoleConfig};

let env = CartPole::with_config(
    CartPoleConfig {
        seed: 42,
        ..CartPoleConfig::default()
    }
);
```

### Reproducibility

Stochastic environments seed a **persistent** RNG once at construction from `config.seed` and let it advance across resets (ADR 0029). A fixed seed therefore reproduces a fixed *sequence* of episodes, not one episode over and over — successive `reset()` calls sample independent episodes, which is what makes an env like `MemoryEnv` or `GoToDoorEnv` a learning problem at all. Pass different seeds across parallel workers to get independent rollouts.

To replay one specific episode bit-for-bit, use the inherent `reset_with_seed(seed)` on the environments that expose it (`MemoryEnv`, `GoToDoorEnv`).

### Wrappers

`wrappers::TimeLimit<E>` wraps any `Environment` and injects a truncation signal after a configurable number of steps. The underlying environment's `Terminated` status is preserved separately from the wrapper's `Truncated` — algorithms that distinguish the two (PPO, SAC) can act on both signals correctly.

### Episode Status

`Snapshot::status()` returns one of three variants:

- `Running` — episode continues
- `Terminated` — natural episode end (goal reached, fell over, game over)
- `Truncated` — wall-clock time limit exceeded (`TimeLimit` wrapper)

### Const Generics

State and observation dimensions are encoded as const generics (`State<D>`, `Observation<D>`). Mismatched dimensions produce a compile-time error rather than a runtime panic.

---

## Testing

```bash
# Unit tests (all environments)
cargo test -p rlevo-environments

# Gridworld solvability integration tests
# (scripted optimal policies verify physics correctness)
cargo test -p rlevo-environments --test grids_solvable
```

---

## References

- G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, "OpenAI Gym," arXiv preprint arXiv:1606.01540, Jun. 2016. https://arxiv.org/abs/1606.01540
- M. Chevalier-Boisvert, B. Dai, M. Towers, R. de Lazcano, L. Willems, S. Lahlou, S. Pal, P. S. Castro, and J. Terry, "Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks," arXiv preprint arXiv:2306.13831, Jun. 2023. https://arxiv.org/abs/2306.13831
- A. G. Barto, R. S. Sutton, and C. W. Anderson, "Neuronlike adaptive elements that can solve difficult learning control problems," IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13, no. 5, pp. 834–846, Sep./Oct. 1983. doi: 10.1109/TSMC.1983.6313077.
- D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. P. Lillicrap, K. Simonyan, and D. Hassabis, “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm,” arXiv preprint arXiv:1712.01815, Dec. 2017. https://arxiv.org/abs/1712.01815
- J. Leike, M. Martic, V. Krakovna, P. A. Ortega, T. Everitt, A. Lefrancq, L. Orseau, and S. Legg, "AI safety gridworlds," arXiv preprint arXiv:1711.09883, Nov. 2017. https://arxiv.org/abs/1711.09883
- Rapier physics engine — https://rapier.rs
- Burn framework — https://burn.dev

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
