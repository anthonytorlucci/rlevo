# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking changes

- **`rlevo-evolution` switches to a maximise-native convention** (ADR 0023) —
  `ObjectiveSense { Minimize, Maximize }` is introduced; `BatchFitnessFn`
  gains a required `sense()` method (no default) and `CoupledFitness` (used by
  `coevolution`) later gains the same requirement for parity (ADR 0035,
  resolves #160). `Landscape::sense()` defaults to `Minimize`. A cost
  objective now declares its sense once instead of every caller hand-negating
  fitness. `rlevo-benchmarks`' record schema bumps **v6 → v7**:
  `RunManifest` gains `objective_sense` (absent ⇒ `Maximize`).
- **`Probability` and `NonNegativeRate` newtypes replace bare `f32` rate
  fields** (ADR 0031) across `rlevo-evolution` config and operator signatures
  — `GaConfig`/`BinaryGaConfig`/`CgpConfig`/`GepConfig`, `GaCrossover`,
  `WritebackPolicy` fields now take these validated types instead of `f32`.
- **NEAT's `NodeId`/`InnovationId`/`SpeciesId` become opaque newtypes**
  (ADR 0032), no longer interchangeable with `u64` or each other; construct
  with `::new`, read with `.get()`.
- **`rlevo-evolution` state/params/genome structs lose their public fields**
  (issue #141 sweep) — `StrategyMetrics`, `CmaEsState`/`CmsaEsState`,
  `EsState`, `GepState`, `MemeticState`, the metaheuristic states
  (Abc/Bat/Cuckoo/Firefly/Gwo/Woa), the EDA states (CompactGenetic/
  UnivariateBernoulli/UnivariateGaussian), `HillClimbingParams`/
  `SimulatedAnnealingParams`, `NasGenome`/`NasParams`, `Species`/
  `TopologyGenome`, and `function_set::Symbol`'s inner id are now private (or
  `pub(crate)` for in-place-mutated NEAT types) behind accessors and
  validating constructors.
- **`Population::new` and related constructors now return
  `Result<Self, ConfigError>`** (ADR 0030), rejecting a zero-row/zero-column
  tensor instead of yielding an inhabitable-but-invalid population.
- **`ContinuousAction` gains a required `const COMPONENTS: usize`** (ADR
  0038, resolves #100) — the previous default `random()` sampled `Self::RANK`
  values instead of the flattened component count, so any multi-component
  rank-1 action (e.g. BipedalWalker's 4-dim action) panicked unconditionally.
  All 11 impls now declare `COMPONENTS` explicitly; no default is provided so
  the gap is a compile error, not a runtime one.
- **box2d `State` structs (bipedal_walker, car_racing, lunar_lander) are now
  encapsulated** (ADR 0039, resolves #117) — fields that used to be `pub`
  aliases over external Rapier handles are now `pub(crate)` behind
  `#[must_use]` accessors, and `is_valid()` now genuinely validates handle
  liveness, finiteness, and structural invariants instead of rubber-stamping
  `true`. `CarRacingState::current_tile` becomes `Option<usize>`.
- **`LunarLander`'s terminal reward is now overwritten, not accumulated**
  (resolves #122) — crash/out-of-bounds is a flat −100 and landing is a flat
  +100 (matching Gymnasium), replacing that step's shaping delta and control
  cost. Previously a hard crash could net a positive reward via Rapier's
  stiff-contact shaping spike; recorded LunarLander benchmark numbers will
  shift. `Running`/`Truncated` rewards are unchanged.
- **`SnapshotBase` gains `metadata: Option<SnapshotMetadata>`** (ADR 0042,
  resolves #128) — `SnapshotBase<R, ObservationType, RewardType>` now carries
  an optional `SnapshotMetadata` field and a fluent `#[must_use]
  with_metadata(self, SnapshotMetadata) -> Self` builder; `Snapshot::metadata()`
  is overridden on `SnapshotBase` to return it instead of the inherited `None`
  default. `running`/`terminated`/`truncated` now construct with `metadata:
  None`; attach metadata with a `.with_metadata(...)` tail. The two bespoke
  hand-rolled `impl Snapshot<1>` types collapse to type aliases over
  `SnapshotBase` — `LocomotionSnapshot<O>` (`rlevo-environments::locomotion::common`)
  and `LunarLanderSnapshot` (`rlevo-environments::box2d::lunar_lander::snapshot`) —
  so their type names are unaffected, but their constructors' metadata
  arguments move to the `.with_metadata()` tail (no `#[deprecated]` shim: a
  constructor cannot be deprecated-and-retained on a type that is now a
  foreign alias). This unblocks `TimeLimit` composition for all six
  previously-locked-out environments (4 locomotion + `LunarLanderDiscrete` /
  `LunarLanderContinuous`).

- **`MemoryEnv` and `GoToDoorEnv` config and observation surfaces change**
  (ADR 0043, resolves #109) — both environments claimed properties they did not
  have, and fixing them removes the config fields that caused the defect.
  - `MemoryConfig::swap_fork` is **removed**, and `MemoryConfig::new` changes
    arity: `new(size, max_steps, seed)` (was `new(max_steps, seed, swap_fork)`).
    `size` is a new field — odd and `>= 11`, rejected by `Validate` otherwise.
    Defaults are `size = 13`, `max_steps = 845` (`5 * size²`), `seed = 0`.
    The default sits deliberately **above** the minimum: `11` is the smallest
    size at which the cue is unobservable from the fork (Invariant M), but it is
    also the size at which the cue-free corridor run collapses to a single cell,
    so it is the *weakest* recall task the layout supports. `13` gives a
    three-cell cue-free run for ~40% more step budget.
    A `swap_fork=…` key in a `MemoryConfig` config string is now an error, not
    a silently ignored no-op.
  - `GoToDoorConfig::target_color` is **removed**, and `GoToDoorConfig::new`
    changes arity: `new(size, max_steps, seed)` (was
    `new(size, max_steps, seed, target_color)`). The target is sampled per
    episode. `target_color=…` / `color=…` config-string keys are now errors.
  - **`GoToDoorEnv`'s observation and snapshot types change.** It no longer
    emits the shared `GridObservation` / `GridSnapshot`; it emits
    `GoToDoorObservation` (`[7, 7, 4]`) and `GoToDoorSnapshot`. Rank is still
    `3`, so `Environment<3, 3, 1>` is unchanged, but any code naming its
    `ObservationType` / `SnapshotType`, or feeding a `7×7×3` model, must be
    updated. This is the grid family's only 4-channel observation.
  - Both configs pinned a quantity the environment is supposed to sample every
    episode; determinism for tests is served by the new `reset_with_seed` (ADR
    0029) instead, which exercises the real sampling environment.

- **`ContextualBanditObservation` closes its construction surface** (resolves
  #124) — the `pub context: usize` field is **private**; read it with the new
  `context()` accessor and build one with the fallible
  `ContextualBanditObservation::<C>::new(context) -> Result<Self, StateError>`,
  which rejects `context >= C` with `StateError::InvalidData`. The public field
  let a caller construct an out-of-range context that then panicked with an
  index-out-of-bounds inside `TensorConvertible::write_host_row`'s one-hot
  encoder — a panic on user-supplied data, which `docs/rules.md` §4 forbids.
  `Deserialize` is now hand-written and validates through the same constructor,
  so the identical hole is closed on the serde path (the wire format is
  byte-identical — a single `context` field — so existing persisted
  observations still load; an out-of-range one now errors instead of panicking
  later). The **`Default` derive is removed**: it yielded `context: 0`, which is
  out of range at `C == 0` and was the only construction path that skipped
  validation. `context < C` is now an invariant no public API can break.
- **`EnvironmentError` is now `#[non_exhaustive]` and gains a
  `StepAfterEpisodeEnd { status: EpisodeStatus }` variant** (ADR 0044, resolves
  #105). Downstream code can no longer `match` on `EnvironmentError`
  exhaustively — add a `_` arm. Calling `step()` after a snapshot whose
  `is_done()` is `true` now returns `Err(StepAfterEpisodeEnd { .. })` instead
  of silently continuing; the carried `status` says whether the episode ended
  by intrinsic MDP termination (`Terminated`) or wrapper-imposed truncation
  (`Truncated`). Any rollout loop that already breaks or resets on `is_done()`
  is unaffected — every loop in this workspace already did. A loop that stepped
  past termination was corrupting its own trajectory and now fails loudly;
  call `reset()` to start a new episode. So far only the `toy_text` family and
  the `TimeLimit` wrapper enforce this — the remaining environments are tracked
  in #289.
- **Every n-dimensional landscape constructor is now fallible and its `dim` is
  private** (resolves #110) — `Sphere::new(dim)` and its 14 siblings return
  `Result<Self, ConfigError>` instead of `Self`, rejecting a degenerate `dim`
  at construction via the ADR 0026 `config::nonzero` / `config::at_least`
  helpers, exactly as ADR 0030 did for `Population`. Ten landscapes require
  `dim >= 1` (`Sphere`, `Rastrigin`, `Ackley`, `Griewank`, `Schwefel`,
  `Alpine1`, `Deb1`, `Needle`, `Michalewicz`, `Penalized1`); four require
  `dim >= 2` because their sum runs over adjacent coordinate pairs and is empty
  at `n = 1` (`Rosenbrock`, `RosenbrockFlat`, `Eggholder`,
  `LunacekBiRastrigin`); `ConcatenatedTrap::new` requires both `num_blocks` and
  `block_size` to be non-zero and their product to not overflow `usize`.
  Migration: `Sphere::new(d)` becomes `Sphere::new(d).expect("dim >= 1")` at a
  setup boundary (`main`, a test), or `?` where a `Result` is already threaded.
  The `dim` field is no longer `pub` — read it through the new
  `#[must_use] pub const fn dim(&self) -> usize`, and construct through `new`
  rather than a struct literal. `new` is also no longer `const`; no `const` or
  `static` landscape item existed in the workspace, so nothing else moves. No
  persisted data is affected — landscapes carry no serialized form.

### `rlevo-core`

**Added**

- `Observable<OR>` projection trait decoupling observation tensor order from
  state rank, for modality-changing POMDPs such as a pixel-over-compact-state
  environment (ADR 0019); `Environment<R, SR, AR>` permits `R != SR`.
- `config::Validate` trait with `ConfigError` and check helpers
  (`positive`/`in_range`/`ordered`/`distinct`/`nonzero`/`at_least`) as the
  shared, fail-fast hyperparameter-validation convention (ADR 0026); adopted
  by config structs across `rlevo-evolution`, `rlevo-reinforcement-learning`,
  and `rlevo-hybrid`.
- `Bounds<f32>` — validated-by-construction inclusive `[lo, hi]` range newtype
  (rejects `lo > hi` and NaN) replacing raw `(f32, f32)` pairs across
  range-shaped config/state fields (ADR 0027).
- `stack_to_tensor` host-row batch-conversion seam: `TensorConvertible` now
  derives `to_tensor` from a `row_shape`/`write_host_row` primitive so a
  batch uploads as one `Tensor::from_data` instead of per-item transfers +
  `cat` (ADR 0028); migrated across ~27 impls.
- `Probability` ([0,1]) and `NonNegativeRate` (finite, ≥0) validated newtypes
  (ADR 0031).
- Public `splitmix64` mixer, promoted from a duplicated private copy (ADR
  0033).
- `EnvironmentError::Config(#[from] ConfigError)` variant (ADR 0040) — gives
  reset-time config-domain failures (e.g. invalid terrain roughness) one
  shared, structured error channel instead of a panic.
- `EnvironmentError::StepAfterEpisodeEnd { status }` variant (ADR 0044) — a
  structured channel for a *sequencing* fault, kept distinct from
  `InvalidAction` because the action is legal and only the call order is
  wrong. `Environment::step`'s rustdoc now states the post-terminal contract
  normatively, with a migration note disclosing which environments do not yet
  enforce it (#289).

**Fixed**

- Environments re-seeded their RNG from `config.seed` inside `reset()`,
  replaying bit-identical episode noise instead of drawing fresh randomness;
  swept across all 11 stochastic environments so the persistent RNG stream
  now advances across resets (seeding happens once, at construction). New
  inherent `reset_with_seed` gives deterministic single-episode replay (ADR
  0029). Successive episodes now differ where they previously repeated.

**Changed**

- `SnapshotBase` struct gains a `pub metadata: Option<SnapshotMetadata>` field
  and a `with_metadata` builder; `Snapshot::metadata()` is now overridden on
  `SnapshotBase` (ADR 0042).

### `rlevo-environments`

**Added**

- `PixelGridEnv` — first production consumer of `Observable<OR>`, projecting a
  compact rank-1 grid latent into a rank-3 `[20, 20, 3]` RGB image (ADR 0020).
- `SantaFeAntEnv` — canonical GP/POMDP benchmark: artificial-ant trail
  following with a one-bit `food_ahead` percept on a 32×32 toroidal grid, plus
  a structured render path and optional `AsciiRenderable` debug helper.
- Three-tier benchmark landscape function suite (unimodal / multimodal /
  deceptive) for evolutionary-algorithm evaluation.
- `TensorConvertible` impls for `BipedalWalkerObservation` and
  `CarRacingObservation`, previously missing/self-contradicting and blocking
  DRL usage of both environments entirely (resolves #116).
- `GoToDoorObservation` (`[7, 7, 4]`), `GoToDoorSnapshot`, and the consts
  `MISSION_CHANNEL`, `GO_TO_DOOR_OBS_CHANNELS`, `DOOR_COUNT` — re-exported from
  `grids` alongside `GoToDoorEnv` (ADR 0043).
- `MemoryEnv::reset_with_seed`, `MemoryEnv::cue`, `MemoryEnv::size`, and
  `GoToDoorEnv::reset_with_seed`, `GoToDoorEnv::doors` — the accessors a
  scripted oracle or a replay needs now that both envs sample per episode.
- `episode::EpisodeGuard` — the reusable post-terminal guard env authors hold
  on their struct (ADR 0044). It stores an `EpisodeStatus`, never a
  `done: bool`, so termination keeps a single source of truth (`docs/rules.md`
  §10). Call `check()?` as the first statement of `step()`, `record()` the
  emitted status on a single exit path, and `reset()` it once a reset has
  actually succeeded.
- **`RecordedEnvFamily` now covers every built-in environment** (resolves #126)
  — `bench::family` previously carried impls for only six envs, so
  `RecordingConfig::for_env::<Pendulum>(seed)` simply did not compile and a
  driver had to fall back to `RecordingConfig::new(EnvFamily::Classic, seed)`.
  That literal is exactly the footgun the trait exists to remove: it can
  disagree with the env being recorded, which compiles fine and silently emits
  the wrong report-tier adapter. The remaining classic, bandit, grid, toy-text,
  `box2d`, and locomotion envs now carry impls, as does `TimeLimit<E>` (it
  forwards its inner env's family, so wrapping an env in a step cap no longer
  loses it). The bandits are generic over their arm count, so the impls are
  too — `TenArmedBandit` is a transparent alias for `KArmedBandit<10>` and is
  covered by the generic impl rather than one of its own.

**Fixed**

- **Two `DynamicObstacles` balls could merge into a single cell** (resolves
  #125) — `move_obstacles` decides every obstacle's target against a stable
  *pre-move* snapshot of the grid and only then applies the moves, so two
  obstacles adjacent to the same free cell each saw it as empty and both took
  it. The merged pair left a duplicate entry in `obstacles()` while the grid
  drew only one ball, so the environment's difficulty contract — `N`
  *independent* hazards — silently decayed toward fewer, and the tracked
  obstacle list disagreed with the rendered grid. The existing tests missed it
  because the defect is arithmetically unreachable at the default
  `num_obstacles = 1`, and no test drove a multi-obstacle episode far enough
  for two random walks to contend for one cell. Obstacle targets are now
  reconciled in index order against a claimed set: the first obstacle to claim
  a cell keeps it, and any later obstacle whose draw lands on a claimed cell
  stays at its old position — the standard vertex-conflict rule, and the same
  no-merge guarantee Farama Minigrid gets from its `place_obj` rejection loop.
  The agent's cell is claimed like every other, so exactly one obstacle can
  collide with the agent on a step (the −1.0 terminal collision is unchanged);
  `obstacles()` positions are now pairwise distinct throughout any episode
  driven per the `Environment` contract (`reset` → `step` until `done` →
  `reset`), including on the terminal collision step. (Stepping *past* a
  terminal snapshot without resetting still desyncs the tracked obstacles from
  the grid; that is the grids family's separately tracked missing
  post-terminal-`step` guard, not a property of this fix.) **Note for anyone
  comparing against earlier runs:** multi-obstacle seeded trajectories shift, so `num_obstacles >= 2`
  baselines from before #125 are not comparable. `num_obstacles = 1` is
  bit-for-bit unchanged — a conflict is impossible with one obstacle, and each
  obstacle still consumes exactly one RNG draw per step.
- **A landscape's search box could exclude its own global optimum** (resolves
  #113, ADR 0045) — `bounds()` returns a single `(lo, hi)` pair that every
  consumer applies to *each* coordinate, so for a landscape whose true domain is
  a rectangle the only correct value is the **square hull** of that rectangle.
  `Branin` instead returned the `x₁` range `(-5, 10)`, and `Trefethen` the `x₂`
  range `(-4.5, 4.5)`. Branin's box therefore **excluded the certified global
  minimum `(−π, 12.275)`** outright — `x₂ = 12.275 > 10`, so no search
  constrained to `bounds()` could ever reach one of its three equal optima, and
  a run that never found it looked like an algorithm that had converged rather
  than a box that was wrong. `Trefethen` clipped `x₁ ∈ [-6.5, 6.5]` to `±4.5`.
  Both now return the hull (`(-5, 15)` and `(-6.5, 6.5)`); `Bukin6` had this
  right already and is now the documented model. The existing tests missed it
  because they only ever asserted that `evaluate` returns `f*` **at** each
  optimum — never that `bounds()` could **reach** it. That gap is now closed by
  two obligations tested on every 2-D landscape: **O1**, the box contains every
  certified optimum on every axis (this is the test that catches #113), and
  **O2**, the box contains no point beating `f*` — the guard that makes widening
  a box safe rather than a silent way to invent a better optimum. Both widenings
  are provably safe: Branin's `f* = 10/(8π)` is the global infimum over all of
  `ℝ²`, and any point beating Trefethen's `f*` must lie within radius ≈0.817 of
  the origin. **Note for anyone comparing against earlier runs:** Branin and
  Trefethen results are now obtained over a larger box, so their baselines shift
  and are not comparable to pre-#113 numbers.
- **The Sphere showcase was running Rastrigin** — `sphere_showcase.rs` imported
  and constructed `Rastrigin`, not just mislabelled its title, so the example
  advertised as the convex-bowl baseline was in fact demonstrating a multimodal
  landscape. It now runs `Sphere` and converges to `~1e-16`, as its own docs
  promise.
- **A zero-dimensional landscape reported itself as *solved*** (resolves #110) —
  every n-D landscape constructor accepted `dim == 0` unchecked, and the
  resulting evaluator did not fail: it lied. `Sphere`, `Rastrigin`, `Alpine1`,
  `Schwefel`, `Needle` and `Griewank` evaluated over an empty slice and returned
  their own **global optimum**, so a misconfigured run read as converged —
  `Griewank` via `sum − prod + 1 = 0 − 1 + 1 = 0`, where the empty product is
  `1`. `Ackley` and `Deb1` divided by `n` and returned `NaN`, and
  `Penalized1` was worse still — `y[0]`
  indexed an empty `Vec` and `self.dim - 1` underflowed `usize`, panicking in
  debug but *wrapping* in release. `ConcatenatedTrap` accepted a zero
  `block_size`, where `chunks_exact(0)` panicked with std's anonymous "chunk
  size must be non-zero" and an all-zeros genome scored the optimum.
  `LunacekBiRastrigin` was the sharpest case: its `dim >= 2` assert lived inside
  `evaluate`, but the *public* `s()` and `mu2()` accessors bypass `evaluate`
  entirely, and below `n = 2` the depth-scaling parameter
  `s = 1 − 1/(2√(n+20) − 8.2)` goes non-positive (`s(1) ≈ −0.036`), making
  `mu2 = −√((μ₁² − d)/s)` a silent `NaN` that no assert could reach. The
  existing tests missed all of this because they only ever constructed
  *sensible* dimensions, and — per ADR 0034 — the fitness-hygiene chokepoint
  maps `NaN → −inf`, so even the NaN cases surfaced as "the optimizer failed to
  converge" rather than "the landscape is misconfigured". The guard now lives at
  construction, where it is unreachable-by-design rather than merely asserted,
  and a table-driven regression test pins all 15 constructors so a future
  landscape cannot land unguarded.
- **Post-terminal `step()` silently resurrected a finished episode across the
  whole `toy_text` family** (ADR 0044, resolves #105) — no environment tracked
  terminality, so a `step()` after a terminal snapshot kept mutating state.
  This was not a benign no-op. In `CliffWalking` the goal `(3, 11)` sits
  *adjacent to the cliff*, so a post-terminal `Left` landed on `(3, 10)`,
  teleported the agent back to the start, and emitted −100 on a **`Running`**
  snapshot — a finished episode brought back to life with a corrupted
  trajectory. In `Blackjack` a post-terminal `Hit` kept pushing cards onto the
  player's hand, and `hand_value` summed them into a `u8`: ~26 ten-valued cards
  overflowed it, panicking in debug and *wrapping* in release, where a wrapped
  sum (260 → 4) re-entered the valid range and emitted a nonsense non-terminal
  reward. `FrozenLake` walked the agent back off a hole or goal tile; `Taxi`
  kept driving after a completed dropoff. The existing tests missed all of this
  because every one of them stopped at the terminal snapshot — the bug lived
  entirely past the point the suite bothered to look. `hand_value` now
  accumulates in a `u16` and saturates, so even an unreachable oversized hand
  classifies as a bust instead of panicking.
- **`TimeLimit` manufactured a second terminal snapshot after truncation**
  (ADR 0044, resolves #105) — the wrapper delegated to the inner environment
  *before* stamping `Truncated`, so the inner env never learned it had been
  truncated and no guard it held could fire. A post-truncation `step()` mutated
  the inner env and returned a fresh, fabricated `Truncated` snapshot. The
  wrapper now owns its own guard and checks it *before* delegating; any wrapper
  that synthesizes a terminal status must do the same (ADR 0044).
- **Rapier `user_force`/`user_torque` were never cleared after a physics
  step** (ADR 0037, resolves #98) — despite rapier2d/3d 0.32's doc comment
  claiming auto-clear, forces/torques silently accumulated across steps for
  any env driving control via `add_force`/`add_torque`, corrupting control
  dynamics in a way existing determinism tests could not detect. Now cleared
  once per integration step; `bipedal_walker` (motor-driven) is unaffected.
- **`bipedal_walker`'s joint angle/speed observations were dead** (resolves
  #119) — angle read a build-time-constant local anchor and speed was
  hardcoded to `0.0`, so 8 of 24 observation dims carried no posture
  information and no policy could learn to walk; now reads live joint state.
  `with_config` also silently ignored `config.terrain`, always building flat
  ground regardless of the Rough/Hardcore preset — now dispatches correctly.
- **`HardcoreTerrain` could generate invalid geometry** (resolves #120) — a
  non-monotonic terminal point, a spawn pad that didn't span the spawn point,
  and a panic on negative/NaN roughness. Terrain fields are now private with
  validating constructors; invalid roughness now surfaces as
  `EnvironmentError::Config` instead of panicking.
- **`car_racing`'s reward and termination were miscalibrated** (resolves
  #121) — the default per-tile reward was calibrated to a phantom 200-tile
  track (the generator actually produces 60), understating a full-lap payout
  by ~3.5×; the config field is renamed `lap_reward` (default 1000) and
  per-tile reward is now derived from it. The per-step progress scan also
  marked only the single nearest tile, letting a fast car skip tiles and
  making the 95%-lap-complete termination unreachable; replaced with a
  bounded contiguous forward sweep.
- **`car_racing`'s 27 KB pixel framebuffer was deep-copied ~3–4× per step**
  (resolves #115) — `Rasterizer::take_pixels()` now moves the buffer out and
  it's stored as `Arc<[u8; N]>` instead of `Box`, dropping the hot path to
  one copy plus atomic-refcount clones.
- **`lunar_lander`'s crash termination was unreachable** (resolves #122) —
  gated on `pos.y < 0.1`, a height the hull collider never reaches (it rests
  at `y≈0.78`), so crashes never terminated the episode; replaced with a
  hull-ground contact query matching Gymnasium's `game_over` semantics.
- **Locomotion `rapier3d` backend's `apply_joint_torque` was
  `unimplemented!()`** (ADR 0041, resolves #123) — now dispatches by joint
  kind and returns `Result<(), BackendError>`; `contact_force`'s wrench sign
  was also inverted (Rapier applies contact impulses along `-normal`), which
  produced a spurious ~675 N permanent internal wrench on jointed neighbors —
  fixed and jointed-neighbor contacts are now disabled for MuJoCo parent-child
  filter parity.
- `mountain_car`/`acrobot`/`cartpole` action decoders panicked on a NaN
  policy logit via `partial_cmp().unwrap()`; routed through a NaN-safe argmax
  helper that falls back to index 0.
- **`MemoryEnv` now actually requires memory** (ADR 0043, #109). The cue was a
  compile-time constant (`Key(Yellow)`), the stored RNG was never read, and the
  reward was keyed to a coordinate independent of the cue — a reactive
  feedforward policy solved it outright. The cue type and the fork order are now
  drawn from a live persistent RNG each episode, all three objects are green so
  colour cannot leak the answer, `match_pos()` is derived from the sampled cue,
  and the layout is size-configurable on the canonical Minigrid geometry with a
  `Validate`-enforced `size >= 11` (Invariant M: with no occlusion in
  `egocentric_view`, only distance can hide the cue from the fork). A reactive
  policy is now capped at chance on the binary fork.
- **`GoToDoorEnv`'s mission now reaches the policy** (ADR 0043, #109). The
  instruction previously existed only on the env — `mission()` had zero callers
  workspace-wide — so simply sampling the target (what #109 asked for) would have
  made the task unsolvable at 25%. The mission colour is now broadcast into
  channel 3 of every observation cell, in the same ordinal encoding
  `Entity::color_u8` uses for perceived door colours, so a network can learn
  equality between the two. The four door colours are rejection-sampled distinct
  per episode and the fixed Red=North / Green=East / Blue=South / Yellow=West map
  is gone.
- Both envs re-seeded their RNG from `config.seed` inside `reset()`, violating
  ADR 0029 — harmless only because the RNG was never read, and a live landmine
  the moment sampling was added. Both re-seed lines are deleted; `reset()` draws
  from the persistent stream and `reset_with_seed` covers deterministic replay.
- **`ContextualBandit`'s ASCII render no longer transposes its arm and context
  counts** (#124). The `AsciiRenderable` impl was declared
  `impl<const K, const C> … for ContextualBandit<K, C>` while the struct is
  `ContextualBandit<C, K>`, so inside the impl `K` was bound to the context
  count and `C` to the arm count and the `"Contextual (K=…, C=…)"` label printed
  them backwards: `ContextualBandit<10, 4>` (10 contexts, 4 arms) rendered as
  `K=10, C=4`. Only the labels lied — the arm-mean lookup and `best@ctx` are
  indexed positionally and were always correct — and no test asserted on the
  label text, so the existing suite could not catch it. The impl now matches the
  struct's `<C, K>` order (as `Display` already did) and a regression test pins
  the labels to the const generics.

**Changed**

- `LocomotionSnapshot<O>` and `LunarLanderSnapshot` are now type aliases over
  `SnapshotBase` instead of hand-rolled `Snapshot` impls; construction moves
  metadata onto a `.with_metadata(...)` builder call (ADR 0042).
- `grids_solvable` integration tests for `MemoryEnv` and `GoToDoorEnv` are now
  **seed-driven oracles**: they read the sampled cue / mission back from the env
  after `reset` and derive the script, across a range of seeds. The old
  hard-coded scripts ("walk north to the Red door", "the match is on the top
  fork") encode answers that are now per-episode random — they would pass by luck.
- `Direction` moved out of `grids::core` to a crate-level module so non-grid
  classic environments (e.g. `SantaFeAntEnv`) don't depend on the Minigrid
  framework; `grids::core` re-exports it for source compatibility.
- Range-shaped config/state fields across the crate adopted `Bounds` in place
  of raw `(f32, f32)` pairs (ADR 0027).

### `rlevo-evolution`

**Added**

- Memetic algorithms — `LocalSearch<B>` trait (hill-climb, random-restart,
  simulated-annealing-style, pattern search) and `MemeticWrapper<B, S, L, F>`
  refining offspring inside `tell` (ADR 0016).
- Estimation-of-distribution algorithms — `ProbabilityModel<B>` trait and
  generic `EdaStrategy<B, M>` with four models (UMDA, PBIL, cGA, MIMIC, ADR
  0017), plus a fifth, `BayesianNetwork` (BOA), using BIC-scored greedy
  structure learning (ADR 0018).
- Co-evolutionary algorithms module — cooperative/competitive coupled-
  population evolution.
- Neuroevolution: weight-only evolution of Burn `Module` weights, bounded
  architecture NAS via enum-dispatched module variants, interpreted NEAT with
  speciation, and a tensorized/GPU-accelerated NEAT batch evaluator that
  forward-passes a whole population on-device (the interpreted path remains
  the numerical-parity oracle).
- Gene Expression Programming (GEP).
- CMA-ES and CMSA-ES strategies — host-side Jacobi eigensolver + Cholesky,
  no external linear-algebra dependency (ADR 0021).
- `proptest` adopted as a dev-dependency for input-space invariant testing
  (roundtrips, shape/length invariants, no-panic/no-NaN, `Validate`
  accept/reject boundaries), complementing rather than replacing seeded-
  `StdRng` example tests; all algorithm randomness still routes through
  `seed_stream` (ADR 0029) — proptest's own PRNG never touches Burn (ADR
  0036).

**Fixed**

- A central fitness-hygiene chokepoint sanitizes NaN/Inf fitness (NaN → -inf,
  +inf → `f32::MAX`) before every `Strategy::tell` inside
  `EvolutionaryHarness::step` (ADR 0034), closing holes previously found
  independently in EDA probability models (#129), GEP (#130), `local_search`,
  `ep`/`es_classical` sigma self-adaptation, `gp_cgp`'s uninitialized-vs.
  sanitized parent fitness, GWO's leader selector, and the non-harness NEAT /
  ArchNAS / coevolution driver seams.
- CMA-ES/CMSA-ES numerical stability: a NaN Cholesky pivot could return a
  poisoned factor past the existing guard; `sigma_i` could underflow to `0.0`
  and poison the rank-µ blend via `0/0`; a generation with fewer than `mu`
  finite fitness values could corrupt the rank-µ update; rank-µ covariance
  accumulation could drift a few ULPs off symmetric under float
  non-associativity (resolves #241, closed by a new proptest property).
- `gp_cgp` (Cartesian GP) panicked on an empty candidate pool or an empty
  fitness batch (`lambda == 0`), and an `Inf` fitness sentinel could collapse
  the `(1+λ)` loop; all three now degrade gracefully.
- `memetic.rs` had a NaN-selection bug in hall-of-fame/coverage selection
  (fitness now sanitized before `total_cmp`) and a per-row writeback upload
  that's now coalesced per contiguous covered-index run.
- Metaheuristics gained algorithm-specific divergence/overflow guards
  (resolves #156): WOA clamps its spiral exponent before `exp()`; Firefly
  derives `gamma` from the bounds extent instead of a hardcoded constant;
  Bat clamps velocity like PSO's `v_max`; Cuckoo folds a non-finite Lévy
  denominator to `0`; ACO_R falls back to uniform weights on a non-finite
  weight sum.
- `aco_r`/`firefly` panicked when `genome_dim == 1` — Burn's `squeeze::<2>()`
  strips every size-1 axis, collapsing rank below 2; fixed via axis-targeted
  `squeeze_dim` (resolves #233).
- `Normal::new(...).expect(...)` Gaussian sampling could abort evolution on a
  non-finite `std`; replaced with a NaN-safe sampling module that falls back
  to `mean` (resolves #145).
- Weight-only/NEAT neuroevolution's `ModuleReshaper` could desync between the
  strategy's reshaper and a second, convention-only instance on template
  drift; `ModuleReshaper` is now `Clone` and shared (resolves #157).
- Crate-wide sweep replacing `.unwrap_or_default()` / bare `.unwrap()` on
  tensor host-reads with `.expect(...)` — a dtype/device transfer failure
  previously substituted an empty `Vec` silently, surfacing generations later
  as a misleading out-of-bounds panic (resolves #136).

**Changed**

- Duplicated per-algorithm argmax/tournament-selection helpers (10+ copies)
  consolidated into shared `ops::selection` functions.
- `PopulationObserver` dispatch wrapped in `catch_unwind` — a panicking
  observer no longer aborts the run.
- `InterpretedPhenotype::new` rewritten from O(n²)+O(n·e) to O(n+e); `forward`'s
  per-column input clone cut from O(I²·B) to O(I·B) — no behavior change,
  removes a performance cliff on larger genomes.

### `rlevo-hybrid`

**Added**

- `StatefulPolicy<B, E>` trait (`type Hidden`, `reset`, `act`) so recurrent/
  memory policies are first-class in `RolloutFitness`, plus a `ReactivePolicy`
  blanket convenience for Markov (stateless) policies (ADR 0025).

### `rlevo-benchmarks`

**Changed**

- Record schema **v6 → v7**: `RunManifest` gains `objective_sense` (ADR 0023).

### Infrastructure

**Added**

- CI: `rlevo-environments` feature-orthogonality check — `cargo check
  --no-default-features --features box2d` and `--features locomotion` run in
  isolation to catch a type gated behind only one of the two orthogonal
  features silently breaking the other (`.github/workflows/crate-tests.yml`).
- `rustfmt.toml` added; `cargo fmt --all --check` is now enforced in CI.
- `rlevo-test-support` — dev-only, unpublished crate consolidating duplicated
  RL integration-test fixtures (ADR 0024).

**Changed**

- CI toolchain-install steps reconciled with `rust-toolchain.toml` (redundant
  explicit installs removed; the pin auto-installs on first cargo/rustc
  invocation).
- Long-running RL integration tests (DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3,
  SAC) gated behind manual/weekly CI runs instead of the default suite.
- `docs/rules.md` codifies: NaN-safe fitness comparison (sanitize-then-
  `total_cmp`), the host-RNG `seed_stream` seeding convention — sample once
  at construction, never re-seed in `reset()` (ADR 0029), a CI grep guard
  against `unwrap_or_default()` masking tensor host-read failures (ADR 0028),
  the state/params/genome struct-field-encapsulation convention, and a rule
  that deferred work must be filed as a GitHub issue before the deferring
  change lands.
- 28 new ADRs (0016–0043) recording the design decisions above; see
  [`docs/adr/README.md`](docs/adr/README.md) for the annotated index.

---

## [0.2.0] – 2026-06-07

### Breaking changes

- **`DIM` → `RANK`** — the const generic parameter on `State<D>`, `Observation<D>`, `Action<D>`, and `Environment<D, SD, AD>` is renamed to `RANK` (or `R`, `SR`, `AR` at usage sites) across all crates. Update any downstream `impl State<D>` / `impl Environment<D, …>` declarations accordingly.
- **`fn new` removed from `Environment` trait** (ADR 0011) — construction is no longer part of the shared trait contract. Replace call sites with the new `ConstructableEnv` factory trait or a concrete `new` method.

### New crates

- **`rlevo-examples`** — heavy visualisation, recording, and report examples extracted from the `rlevo` umbrella (ADR 0012). Lightweight environment/algorithm examples stay in `crates/rlevo/`.
- **`rlevo-metrics-registry`** — wasm32-compatible leaf crate that owns the canonical metric descriptor list (`CANONICAL_METRICS`, `MetricDescriptor`, domain grouping). Eliminates the hand-copied duplicate that previously existed between `rlevo-benchmarks` and `rlevo-benchmarks-report-client` (ADR 0015).
- **`rlevo-benchmarks-report-client`** — Leptos/WASM static-HTML post-run report viewer. Served from an embedded `axum` server. Shares the metric registry with `rlevo-benchmarks` without pulling in `burn` or `rand`.

### Dependency upgrades

- **burn** `0.20.0` → `0.21.0`; migrated `ndarray` backend to the new `flex` backend.
- **rand** `0.9.x` → `0.10.1`, **rand_distr** → `0.6.0`.

### `rlevo-core`

**Added**

- `ConstructableEnv` factory trait — standalone `fn new(render: bool) -> Self` replacement for the removed `Environment::new` (ADR 0011).
- `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier`, semantic `palette` module, and `AsciiRenderable`/`AsciiRenderer` hoisted from `rlevo-environments::render` into `rlevo-core::render` (ADR 0009). Import paths inside `rlevo-environments` are preserved via a re-export shim.

**Changed**

- `AsciiRenderable` demoted from a required library invariant to an optional debug helper; implementing it is no longer implied by `Environment` (ADR 0013).

### `rlevo-environments`

**Changed**

- Render types (`StyledFrame`, `AsciiRenderable`, etc.) re-exported from `rlevo-core::render`; originals removed (ADR 0009).
- `Environment::new` removed; each environment exposes its own `new` constructor and may opt into `ConstructableEnv` (ADR 0011).

### `rlevo-evolution`

**Changed**

- All EA algorithms and shared ops (`selection`, `crossover`, `mutation`, `replacement`) now draw random values through `seed_stream` on the host CPU rather than calling `B::seed + Tensor::random`, eliminating the process-wide RNG mutex contention that caused non-determinism in parallel tests.
- `SharedPopulationObserver` unified to `parking_lot::Mutex` (was split between `std::sync` and `parking_lot` lock types, causing type mismatches in recording examples) (ADR 0010).

### `rlevo-reinforcement-learning`

**Added**

- `polyak_update` hoisted as a shared utility function available to all RL algorithm crates.

### `rlevo-benchmarks`

**Added**

- Record schema **v6** (`FORMAT_VERSION` bumped `5 → 6`, ADR 0014):
  - Expanded `CANONICAL_METRICS` (explained variance, per-iteration episode-return statistics, DQN/SAC loss terms) — list now owned by `rlevo-metrics-registry`.
  - Typed run-provenance fields on `RunManifest`: algorithm name, crate versions, git ref, device, seed count, success threshold.
  - `EpisodeKind { Training, Evaluation }` field in episode headers.
  - Episode wall-clock duration as a terminal metric.
  - `checkpoints: Vec<CheckpointRef>` seam for deep-RL Burn-`Recorder` model files (EA runs unaffected).
- Metrics-only live `ratatui` TUI replaces the earlier three-tier visualisation plan (ADR 0013 supersedes ADR 0008); no environment render panel in the TUI.

**Changed**

- `CANONICAL_METRICS` constant moved to `rlevo-metrics-registry`; `rlevo-benchmarks` re-exports it for back-compat.

### `rlevo-benchmarks-report-client`

**Added**

- Interactive post-run static HTML report (Leptos + WASM):
  - Min/max downsampling for long metric series (ADR 0013 / M8.2).
  - Multi-seed mean ± std band aggregation.
  - Hover crosshair with exact raw-value tooltip.
  - Per-panel SVG export buttons.
  - Step / episode / wall-clock x-axis toggle for episode panels.
  - Eval/training split via `EpisodeKind` in the episode index and table badge.
  - Landscape heatmap background for EA optimisation landscape runs.
  - Diversity-threshold guideline line with breach-pulse highlight.
  - Strip-plot overlay toggle on the population box-plot panel.

### `rlevo` (umbrella)

**Changed**

- Lightweight examples retained; heavy viz/record/report examples migrated to `rlevo-examples` (ADR 0012).

### Infrastructure

**Added**

- GitHub Actions CI: integration-test matrix (Linux × stable toolchain) and weekly full-workspace test run.
- `BACKEND_LOCK` per-binary synchronisation for wgpu-backed integration tests; removes the previous `--test-threads=1` requirement.

---

## [0.1.0] – 2026-04-28

Initial alpha release. All crates are published together at the same version.

### `rlevo-core`

**Added**

- `State<D>` / `Observation<D>` traits for typed, const-generic environment state and agent perception.
- `Action<D>` trait hierarchy (`DiscreteAction`, `ContinuousAction`, `MultiDiscreteAction`, `MultiBinaryAction`) for compile-time action-space safety.
- `Environment<D, SD, AD>` trait with `reset` / `step` / `render` contract; `Snapshot<D>` trait with `SnapshotBase<D, O, R>` concrete type.
- `Reward` trait with `ScalarReward` and `VectorReward` implementations.
- `TensorConvertible<B, D>` bridge trait for lifting state/action types onto Burn tensors.
- `Agent` and `BenchableAgent` traits for uniform agent interaction.
- `FitnessEvaluable` and `Landscape` traits for benchmarking evolutionary algorithms.
- `BenchEnv`, `BenchError`, `BenchStep`, `Metric`, `MetricsProvider`, and `SeedStream` (moved from `rlevo-benchmarks` per ADR-0004).
- `util::seed` — deterministic `SeedStream` for reproducible multi-run experiments.
- `EnvironmentError` and `StateError` error types with `thiserror` derives.

### `rlevo-environments`

**Added**

- **Classic control** — `CartPole`, `MountainCar`, `MountainCarContinuous`, `Pendulum`, `Acrobot`.
- **Bandits** — `KArmedBandit<K>`, `ContextualBandit`, `NonStationaryBandit`, `AdversarialBandit`.
- **Toy text** — `Blackjack`, `CliffWalking`, `FrozenLake`, `Taxi`.
- **Gridworlds** (MiniGrid-style) — `Empty`, `DoorKey`, `Memory`, `FourRooms`, `Crossing`, `LavaGap`, `MultiRoom`, `Unlock`, `UnlockPickup`, `GoToDoor`, `DistShift`, `DynamicObstacles`; shared `GridCore` (grid, entity, action, direction, observation, render, reward, dynamics).
- **Box2D physics** (`box2d` feature, rapier2d) — `BipedalWalker`, `LunarLander` (discrete and continuous action spaces), `CarRacing`.
- **Locomotion** (`locomotion` feature, rapier3d) — `Reacher`, `Swimmer`, `InvertedPendulum`, `InvertedDoublePendulum`.
- **Games** — `Chess` (full move generation and board state), `ConnectFour`.
- **Optimisation landscapes** — `Sphere`, `Ackley`, `Rastrigin` for benchmarking evolutionary algorithms.
- **Wrappers** — `TimeLimit` wraps any `Environment` with an episode step cap.
- **Bench adapter** (`bench` feature) — `BenchAdapter` and preset `Suite` factories to drive any environment from `rlevo-benchmarks`.
- ASCII render backend for text-based environments.

### `rlevo-evolution`

**Added**

- `Strategy<B>` pure trait (`init` / `ask` / `tell` / `best`) — stateless, parallelism-friendly, trivially checkpointable.
- `EvolutionaryHarness<B, S, F>` — wraps any `Strategy` as a `BenchEnv`.
- `BatchFitnessFn` trait with `FromFitnessEvaluable` adapter.
- `GenomeKind` enum (`RealValued`, `Binary`, `Integer`, `Program`).
- **Classical families** — `GeneticAlgorithm` (real-valued, SBX crossover + polynomial mutation), `BinaryGeneticAlgorithm` (one-point/uniform crossover + bit-flip mutation), `EvolutionStrategy` (`(1+1)`, `(1+λ)`, `(μ,λ)`, `(μ+λ)` with self-adaptive σ), `EvolutionaryProgramming` (Gaussian perturbation + tournament), `DifferentialEvolution` (Rand/1/Bin, Best/1/Bin, CurrentToBest/1/Bin), `CartesianGeneticProgramming` (symbolic regression via CGP graph).
- **Metaheuristics** — `ParticleSwarmOptimization`, `AntColonyOptimizationReal`, `AntColonyOptimizationPermutation`, `ArtificialBeeColony`, `FireflyAlgorithm`, `BatAlgorithm`, `CuckooSearch` (Lévy flights via Mantegna), `GreyWolfOptimizer`, `SalpSwarmAlgorithm`, `WhaleOptimizationAlgorithm`.
- **Genetic operators** (`ops`) — selection (tournament, roulette, rank, SUS, elitism, NSGA-II crowding), crossover (uniform, one-point, multi-point, SBX, BLX-α, intermediate), mutation (Gaussian, uniform, polynomial, bit-flip, inversion), replacement (generational, steady-state, elitist, comma, plus).
- **Custom CubeCL kernels** (`custom-kernels` feature) — fused pairwise-attract (Firefly large-N path) and fused Lévy-flight (Cuckoo/Bat) kernels; pure-tensor fallbacks used when feature is off.
- `PopulationState` tensor wrapper; `ShapingFn` fitness shaping (linear rank, exponential rank, truncation).

### `rlevo-reinforcement-learning`

**Added**

- **Replay memory** — `PrioritizedExperienceReplay` (uniform-sampling mode in v0.1.0); `TrainingBatch` typed container.
- **Experience** — `ExperienceTuple` (s, a, r, s', done), `History` trajectory buffer.
- **Metrics** — `AgentStats` (per-step), `PerformanceRecord` (per-episode).
- **DQN** — `DqnModel`, `DqnAgent`, `DqnTrainingConfig`; ε-greedy exploration schedule; Double-DQN target option.
- **C51** — `C51Model`, `C51Agent`, `C51TrainingConfig`; Bellman projection onto N-atom support; categorical cross-entropy loss.
- **QR-DQN** — `QrDqnModel`, `QrDqnAgent`, `QrDqnTrainingConfig`; quantile Huber loss; no `[v_min, v_max]` required.
- **PPO** — `PpoAgent`, `PpoTrainingConfig`, `RolloutBuffer`, GAE advantages, `CategoricalPolicyHead` (discrete), `TanhGaussianPolicyHead` (continuous), clipped surrogate + value loss, early-stop on `approx_kl`.
- **PPG** — `PpgAgent`, `PpgConfig`, `AuxBuffer`, `PpgCategoricalPolicyHead`; interleaved policy-phase + auxiliary-phase with KL distillation.
- **DDPG** — `DdpgAgent`, `DdpgTrainingConfig`; deterministic actor + Q-critic; Polyak target sync; Gaussian exploration noise.
- **TD3** — `Td3Agent`, `Td3TrainingConfig`; twin-critic min-bootstrap; delayed actor updates; target-policy smoothing.
- **SAC** — `SacAgent`, `SacTrainingConfig`; squashed-Gaussian stochastic actor; twin critics; learnable temperature α with auto-tuning toward `-|A|`.
- Shared `EpsilonGreedy` schedule (DQN / C51 / QR-DQN) and `GaussianNoise` exploration (DDPG / TD3).

### `rlevo-benchmarks`

**Added**

- `Evaluator` — drives any `BenchEnv` for N episodes, collecting per-step and per-episode metrics.
- `Suite` — ordered sequence of `(env, evaluator)` pairs with shared reporter.
- **Metrics** — `EaMetrics` (best fitness, population diversity, convergence rate), `RlMetrics` (episode return, episode length, sample efficiency).
- **Reporters** — `JsonReporter` (newline-delimited JSON), `LoggingReporter` (tracing spans), `TuiReporter` (ratatui live dashboard, `tui` feature).
- `Checkpoint` and `Storage` traits for saving/resuming benchmark state.
- `rayon`-parallel episode evaluation for multi-seed sweeps.

### `rlevo-hybrid`

**Added**

- Stub crate establishing the dependency wiring between `rlevo-evolution` and `rlevo-reinforcement-learning`. No hybrid strategies are implemented in v0.1.0; see the crate README for the v0.2.0 roadmap.

### `rlevo` (umbrella)

**Added**

- Re-exports all public APIs from every workspace crate behind a single `rlevo` entry point.
- `keywords`: `reinforcement-learning`, `evolutionary`, `deep-learning`, `burn`, `neural-network`.
- `categories`: `science`, `algorithms`, `simulation`.
- Full example suite (35 examples across gridworlds, classic control, Box2D, locomotion, evolutionary showcases, RL algorithms, and benchmarks harness).
- Cross-crate integration tests.

---

[0.2.0]: https://github.com/anthonytorlucci/rlevo/releases/tag/v0.2.0
[0.1.0]: https://github.com/anthonytorlucci/rlevo/releases/tag/v0.1.0
