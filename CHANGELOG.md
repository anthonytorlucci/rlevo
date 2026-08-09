# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking changes

- **`ReplayBufferError` loses its `TensorConversionError(String)` and
  `BatchError(String)` variants, and becomes `#[non_exhaustive]`** (resolves
  #411). Neither variant was constructible in practice — no code in this
  workspace or its predecessor ever built one — so the only code a removal can
  break is an exhaustive `match` on the enum. Add a wildcard arm; the
  `#[non_exhaustive]` attribute now requires one anyway, and in exchange a
  future variant stops being a breaking change. No persisted data is involved:
  `ReplayBufferError` does not derive `serde`, so no wire or config format
  changes. (Other types in `replay/` — `ReplayConfig`, `PrioritizedReplaySettings`,
  `Priority`, `ImportanceExponent` — do derive it, and are untouched.)

- **All eight agent error enums lose their unconstructed variants and become
  `#[non_exhaustive]`** (resolves #1070, and subsumes #467 and #484). The six
  off-policy enums — `DqnAgentError`, `C51AgentError`, `QrDqnAgentError`,
  `DdpgAgentError`, `Td3AgentError`, `SacAgentError` — drop
  `TensorConversionFailed(String)`, `Buffer(#[from] ReplayBufferError)` and
  `Io(#[from] std::io::Error)`, keeping `InvalidAction` and `Polyak`.
  `PpoAgentError` and `PpgAgentError` drop `TensorConversionFailed(String)`,
  `InvalidConfig(String)` and `Io(#[from] std::io::Error)`, keeping
  `Environment` and so becoming one variant wide. Twenty-four variants in
  total, none of which any code in this workspace ever constructed. Migration:
  as above, an exhaustive `match` needs a wildcard arm, which
  `#[non_exhaustive]` now requires anyway and which buys future variants for
  free. The three `#[from]` impls go with their variants, so a `?` that
  converted a `ReplayBufferError` or an `std::io::Error` into an agent error no
  longer compiles — no such site exists in this workspace, and for
  `ReplayBufferError` none is possible (see below). No persisted data is
  involved: none of the eight derives `serde`.

### `rlevo-evolution`

**Fixed**

- **A `NaN` fitness permanently froze one individual in the metaheuristic
  family, silently shrinking the population by one** (resolves #131). ABC, Bat,
  Cuckoo Search and PSO keep a persistent per-slot fitness cache and accept a
  candidate only when it beats the cached value. The raw `NaN` was latched into
  that cache at the gen-0 bootstrap, and every subsequent comparison against it
  is false — so the slot became a zombie the algorithm could never replace. Bat
  and PSO have no reset mechanism at all, so the freeze was
  permanent-until-restart; ABC escaped only via the scout `limit`
  (`pop_size · genome_dim / 2` — hundreds of generations for realistic configs);
  Cuckoo escaped via abandonment, but never at `p_a = 0`, a valid configuration
  the suite already exercises. All nine metaheuristic `tell` impls now sanitize
  the fitness vector once, where it is pulled to host, so the bootstrap seed and
  the accept-store are both covered by one call.

  The issue was filed as leader/global-best poisoning; it is not that.
  `argmax_host` seeds from `−∞` and compares with `>`, so a `NaN` can neither
  win a champion scan nor seed one, and `best()` was correct throughout — which
  is precisely why the frozen slot went unnoticed: the run's *reported* optimum
  stayed right while the search quietly ran a member short. Two files were
  missed by the original triage in the other direction. `gwo.rs` sanitized on
  the read (`argtop3_max`) but stored the raw value, and `aco_r.rs` sanitized
  for its archive ranking but then re-read the *unsanitized* vector when
  materializing `archive_fitness`, so a `NaN` survived at the archive tail. Both
  had been declared clean on the strength of their read-side guard alone.

  No existing test could have caught this. The NaN-safety coverage all runs
  through `EvolutionaryHarness::step`, which sanitizes before `tell`, and the
  convergence-style assertions that do exercise these algorithms pass happily
  while a slot is frozen. The new `nan_fitness_does_not_latch_without_harness`
  tests — one per algorithm, the bypass twin of the existing
  `nan_fitness_survives_harness` — drive `init → ask → tell` directly and assert
  on the cache rather than on convergence.

  Only callers who drive `Strategy::tell` directly were exposed; harness-driven
  runs were never affected, and on that path the change is a provable no-op
  (`sanitize_fitness` is idempotent). `Strategy` is public and re-exported in the
  umbrella prelude, so this is the ADR 0034 decision-3 bypass hole, now closed at
  the per-site floor rather than left to the chokepoint above it.

  One route remains open and is tracked separately as #1064: this sanitizes the
  fitness *entering* `tell`, not the cache it is compared against, so a `NaN`
  placed directly into a state's fitness vector via the `pub` `*State::try_new`
  constructors (or `PsoState`'s `pub` fields) still freezes that slot.

- **`mean_fitness` still reported `+∞` for a population of optimal
  individuals** (resolves the metrics half of #132). ADR 0034 maps a `+∞`
  fitness to `f32::MAX` and states that, because the clamped value is finite, it
  "cannot blow a `mean`, `variance`, or reward to `+∞`". That guarantee did not
  hold: `f32::MAX` passes `is_finite()` and so joins the running total, and
  `f32::MAX + f32::MAX` saturates straight back to `f32::INFINITY`. Two
  individuals legitimately pegging the objective in one generation were enough
  to report an infinite mean. `StrategyMetrics::from_host_fitness` now
  accumulates in `f64` and narrows once, after the division — the accumulator's
  width, not the clamp, is what carries the guarantee. This also removes the
  ~1 ULP-per-addition drift the `f32` total accrued over a large population.
  Because the chokepoint is shared, every evolutionary strategy was affected,
  not only the EA-root family #132 names.

  The existing coverage could not have caught this: the one `+∞` regression test
  passed a slice with a *single* infinite member, and a single `f32::MAX` in the
  sum is exactly the case that does not overflow. The defect needed two.

- **A `NaN` at index 0 permanently stranded the champion genome in GA,
  binary GA, and EP** (resolves the champion-tracking half of #132).
  `update_best` in `algorithms/ga.rs`, `algorithms/ga_binary.rs` and
  `algorithms/ep.rs` seeded its scan with `best_f = fitness[0]` and compared
  with `>`. Every comparison against a `NaN` seed is false, so the scan kept
  index 0 and the champion-write guard `best_f > state.best_fitness` never
  fired — while the caller advanced `state.best_fitness` from the *sanitising*
  `StrategyMetrics`. The two fields desynchronised: `best_fitness` ratcheted up
  to the real winner, `best_genome` stayed `None`, and because the ratchet was
  now high, every later generation failed the guard too. `Strategy::best()`
  returned `None` for the rest of the run. All three now sanitise into a local
  buffer and order with `total_cmp`, matching the fix already carried by
  `algorithms/gp_cgp.rs` and satisfying ADR 0034's requirement that every
  champion-write site hold the per-site correctness floor.

  Only callers who drive `Strategy::tell` directly could reach this —
  `EvolutionaryHarness::step` sanitises before `tell`, so harness-driven runs
  were never exposed. `Strategy` is public and re-exported in the umbrella
  prelude, and the existing tests all went through the harness. `es_classical.rs`
  and `de.rs` were checked and are unaffected: both delegate to `argmax_host`,
  which seeds at `−∞` and is `NaN`-safe under `>`.

- **Two `+∞` fitnesses anywhere in a NEAT population erased fitness-proportional
  offspring apportionment for *every* species, not just the one holding them**
  (resolves #1062). This is the same overflow as the `mean_fitness` defect
  above, at a site that fix could not reach: `speciate` and `allocate_offspring`
  in `neuroevolution/species.rs` never route through
  `StrategyMetrics::from_host_fitness`, so widening that accumulator left these
  two untouched. Both summed sanitized fitness into an `f32`, and two members
  clamped to `f32::MAX` saturate the total to `+∞`.

  The population-wide blast radius comes from `allocate_offspring` inheriting
  the infinity. Its `if total <= 0.0` guard does not fire, because `+∞ > 0`.
  Healthy species then compute `pop_size × finite / ∞ = 0.0`; the poisoned one
  computes `pop_size × ∞ / ∞ = NaN`, and `NaN as usize` saturates to `0` in
  Rust. Every floored share lands on zero, so the largest-remainder
  reconciliation hands out all `pop_size` seats round-robin — a species holding
  a 100× fitness advantage went from 27 of 30 seats to 10, exactly the even
  split NEAT's speciation exists to avoid. Unlike #132's champion desync, no
  unusual usage is required: `NeatStrategy::tell` sanitizes and then calls
  `speciate` unconditionally, and it is NEAT's only entry point, so any
  objective that can return `+∞` twice reaches this on the normal path.

  `allocate_offspring`'s `total` also overflows *independently* of the first
  defect — three species whose adjusted sums are each individually finite still
  saturate their total — so both accumulators needed widening, and `total`
  stays `f64` rather than narrowing back.

  The blind spot was inherited too. `test_speciate_sanitizes_nan_and_inf_fitness`
  passes, and is correct: it covers the raw-`NaN` path, which ADR 0034 genuinely
  fixed. But it places a *single* `+∞` member in a species, and one `f32::MAX` in
  a sum is precisely the case that does not overflow. The new tests pin exact
  count vectors rather than asserting the total sums to `pop_size` — that
  weaker assertion holds on the buggy code, and is what let this through.

- **`shaping::z_score` returned an all-zero vector — a silent zero ES gradient —
  for any population containing a single saturated member.** Found while
  generalising the two fixes above into ADR 0069, and latent: `z_score` is `pub`
  but has no in-workspace caller yet. It squares its centred terms, so a member
  at `f32::MAX` overflowed *its own squared term* to `+∞` at `N = 1`, before any
  accumulation — no accumulator width would have helped. `var = +∞` drove
  `std = +∞` and collapsed every output element to `±0.0`. An entirely saturated
  population returned `NaN` instead. Both are finite-looking, panic-free, and
  exactly the shape of failure that gets mistaken for a converged run.

  The remedy could not be the `f64` widening the sibling fixes used:
  `Tensor::sum()` accumulates in `B::FloatElem`, which the backend fixes, and
  reaching an `f64` accumulator would force the device→host round-trip ADR 0034
  introduced `sanitize_fitness_tensor` to avoid. `z_score` now divides the
  population by its own max-abs magnitude before reducing, bounding every squared
  term. That is strictly stronger than widening would have been — it also holds
  on a narrower element type, where the old formula overflowed at fitness ≈ 256 —
  and z-scoring is invariant to a positive rescale, so ordinary inputs move by at
  most a few ULP.

  A `−∞` member still yields `+∞`/`NaN` utilities. That behaviour is unchanged,
  pre-existing, and deliberately left alone here rather than folded into an
  overflow fix; it is tracked as #1068 and pinned by a test marked "Pin, not a
  fix".

**Changed**

- **The rule that a sanitized `+∞` "cannot blow a `mean`, `variance`, or reward
  to `+∞`" has been corrected wherever it was stated** (ADR 0069). It was false —
  `f32::MAX` is finite, so it *joins* a sum — and it was the direct cause of
  #132, #1062, and the `z_score` defect above: in each case the author read the
  rule, sanitized correctly, and then accumulated in `f32` because the
  documentation said the clamp made that safe. The claim is corrected in the
  `sanitize_fitness` rustdoc (the tooltip at ~90 call sites, and the surface that
  misled the `speciate` author), in `rules.md`, in a coevolution test's doc
  comment, and via an annotation on ADR 0034's index row. ADR 0034 itself is
  unedited — its *decision* stands and only its justification was wrong.

  ADR 0069 records the corollary as a binding rule: a reduction over sanitized
  fitness accumulates in `f64` and narrows at most once, afterwards; ordering,
  comparison and argmax are excluded, since saturation is order-preserving. Two
  `pub(crate)` primitives, `sanitized_mean` and `sanitized_sum`, give the rule a
  name at the call site, and three rescale-invariance property tests enforce it
  behaviourally — a source-text guard was rejected because two of the four
  mis-widened sites never mention `sanitize_fitness` and would have gone green
  through both #132 and #1062.

### `rlevo-reinforcement-learning`

**Fixed**

- **A non-finite target quantile survived terminal transitions in QR-DQN's
  Bellman backup** (resolves #357). QR-DQN builds its target inline rather than
  through the shared helper, and it still computed the backup by *scaling* the
  bootstrap term: `rewards + (1 − terminated) · θ_target · γ`. Because IEEE-754
  gives `NaN · 0.0 == NaN` (as does `inf · 0.0`), a poisoned quantile anywhere
  in the target network's output for the bootstrap action contaminated the
  target on exactly the samples where the terminal convention says the
  bootstrap must vanish — samples whose correct value, the reward alone, is
  known with certainty. The term is now genuinely masked to `0` wherever
  `terminated == 1.0`, so a terminal target is the reward regardless of what
  the next-state estimate holds.

  This corrects the record as well as the code. The #192 entry under 0.4.0
  below states that "C51 and QR-DQN never used this helper (they mask in their
  own projection step)". That is false for QR-DQN: it did not mask, it scaled,
  and this is the fix for the gap that sentence denied. C51 genuinely is
  unaffected, for a reason worth stating precisely so nobody patches it on a
  pattern match — its `(1 − terminated)` factor multiplies the **fixed atom
  support**, finite by construction from `v_min`/`v_max` and asserted so, never
  a network output.

  This is hardening, not a repair of live divergence: the expression cannot
  *originate* a non-finite value, only preserve one, and for all-finite inputs
  the masked form is numerically identical, so no algorithm's behaviour
  changes. The known upstream NaN sources (#184, #173) are both closed, so
  there is no live trigger today. No test could have caught it, and the reason
  is the interesting part: the backup was an inline expression in the middle of
  `train_step`, with no seam a test could reach. Lifting it into a named
  function is what made the poisoned case testable at all, and the five new
  tests cover it. Four sit on the helper — including that masking stays
  *per-row*, leaving a non-finite quantile on a non-terminal row for the
  finite-loss guard rather than silently scrubbing a real divergence. The
  fifth runs through the agent and pins the backed-up target numerically in
  the reported loss, because the call site passes reward and terminal mask as
  two `(B, 1)` tensors the compiler cannot tell apart: without it, transposing
  them was a silent change.

**Added**

- **`utils::compute_target_quantiles`**, the rank-2 sibling of
  `compute_target_q_values`, backing up a whole `(B, N)` quantile vector per
  sample instead of a single bootstrap value. Introduced by the #357 fix above;
  it is public for the same reason its rank-1 sibling is, and its docs carry
  the terminal-bootstrap convention, the masking rationale, and the C51
  exclusion so both forms of the convention are findable from one place.

**Removed**

- **Two `ReplayBufferError` variants that advertised failure modes the replay
  seam cannot have** (resolves #411). `TensorConversionError(String)` and
  `BatchError(String)` were residue of the pre-ADR-0050 `TrainingBatch`, which
  assembled tensors inside the buffer. ADR 0050 §3 moved that out — a
  `ReplayStrategy` "never sees a `Tensor`, a `Backend`, or a device" — and
  `sample` is the seam's only fallible method, so after the rewrite there was
  nowhere left for either variant to be returned from. Neither was ever
  constructed, in `replay/` or in the `memory.rs` it replaced.

  No test could have caught this, and none was missing: nothing behaved
  incorrectly. The defect was the API surface itself. Read as an error domain,
  the enum told an implementor that a strategy may fail at tensor conversion or
  batch assembly, which is precisely the boundary ADR 0050 §3 draws — so the
  issue's suggested remedy of a typed `#[from] rlevo_core::base::TensorConversionError`
  payload was rejected: it would have handed out-of-crate implementors `?`
  ergonomics for the crossing, advertising it as supported.

  This mirrors the reasoning already recorded on `ReplayStrategy::sample`, where
  ADR 0051 §2 kept a bad-β variant out of the enum so `UniformReplay` "is not
  made to carry an error variant it could never produce"; and it is consistent
  with `SampledBatch::weighted`, which treats the one *real* batch-assembly
  failure (`weights.len() != ids.len()`) as a deliberate panic rather than an
  `Err`. The enum's rustdoc now carries this rationale, so the absence reads as
  a decision rather than an oversight.

  The same shape one layer up — `TensorConversionFailed(String)`, declared and
  unconstructed on all eight agent error enums — is **not** fixed here and is
  tracked as #1070. It is deliberately a separate call: agents *are* where
  staging happens, so deletion is not automatically the right remedy there.

- **Twenty-four agent error variants that advertised failure modes the agents
  cannot reach** (resolves #1070 — the deferral recorded in the entry above —
  and subsumes #467 and #484, which are per-agent slices of the same finding).
  The issue was filed against `TensorConversionFailed(String)` on all eight
  enums; verifying it turned up two more dead shapes on the same family, plus a
  third on PPO and PPG. Three arguments, one per shape.

  `TensorConversionFailed(String)` is unconstructible by signature, not by
  accident. `act`, `act_greedy`, `act_with` and `act_greedy_with` all return a
  bare action, not a `Result`, so the host-read that could fail has no channel
  to return through. What is there instead is
  `.expect("actor output is f32")` — an `.expect` on a named invariant, which
  is the form `docs/rules.md` §4 sanctions for a read that cannot fail by
  construction. So the issue's second candidate remedy, wiring the staging
  sites to return the variant, is not a fix to this enum at all: it is a change
  to four public signatures, which #317 explicitly defers. The first candidate,
  `#[from]`-ing the core error, fails for the same reason — there is nothing to
  convert *into a return value* from a function that does not return one. Each
  enum's rustdoc now records that when #317 lands the variant must come back as
  `#[from] rlevo_core::base::TensorConversionError`, not as a `String`, since
  §4 prefers structured variants and names that type as the tensor-op domain; a
  re-introduced `String` payload would reproduce exactly the defect this entry
  closes.

  `Buffer(#[from] ReplayBufferError)` was unreachable by design. Every
  `learn_step` writes
  `let Ok(batch) = self.buffer.sample(..) else { return Ok(None) };`, and after
  the removal above the only variant `sample` can produce is
  `InsufficientData` — which means "skip this learn step", not "the step
  failed". Propagating it would misreport a warm-up buffer as an error.

  `Io(#[from] std::io::Error)` anticipated checkpointing that does not exist:
  there is no `save`, `load`, `Recorder` or `std::fs` anywhere under
  `algorithms/`, and ADR 0014 defers checkpointing to Tier D. On PPO and PPG,
  `InvalidConfig(String)` duplicated validation another type already performs —
  `new` returns `rlevo_core::config::ConfigError`, so a bad config is rejected
  before an agent exists and never reaches the agent's own error type.

  As with `ReplayBufferError`, no test could have caught this and none was
  missing: nothing behaved incorrectly. The defect was the API surface. Read as
  an error domain, these enums told a caller that action selection may fail,
  which the signatures say it cannot. `PpoAgentError` and `PpgAgentError` are
  left one variant wide, which is the honest width — `Environment` is the only
  failure their train loops surface. No new ADR is written; the rationale lives
  in each enum's rustdoc, so the absence reads as a decision rather than an
  oversight, and `#[non_exhaustive]` keeps the door open at no cost. The bar
  for reopening it is a real construction site, not an anticipated failure
  mode.

---

## [0.4.0] – 2026-08-02

Minor release: contains breaking changes since 0.3.1. See the
`rlevo-core`, `rlevo-environments`, and `rlevo-reinforcement-learning`
**Breaking changes** subsections below for the full list and migration
notes; there are no changes to `rlevo-evolution`, `rlevo-hybrid`,
`rlevo-benchmarks`, or `rlevo-metrics-registry` this cycle.

### `rlevo-core`

**Breaking changes**

- **Observation production moved off `State` to a new env-side `Sensor` trait**
  (ADR 0047, supersedes ADR 0019, resolves #329). In the POMDP tuple
  ⟨S, A, T, R, Ω, O⟩ the emission model `O` is a property of the environment,
  not of a state value, so it no longer lives on `State`. `State<R>` **loses**
  its `type Observation: Observation<R>` associated type and its
  `fn observe(&self) -> Self::Observation` method; a `State` now carries only
  `RANK`, `shape()`, `numel()`, and `is_valid()`. Observation production moves to
  the new `rlevo_core::environment::Sensor<OR, AR, SR>`:

  ```rust
  pub trait Sensor<const OR: usize, const AR: usize, const SR: usize> {
      type Action: Action<AR>;
      type State: State<SR>;
      type Observation: Observation<OR>;
      fn observe(&self, action: &Self::Action, next_state: &Self::State) -> Self::Observation;
      fn observe_reset(&self, state: &Self::State) -> Self::Observation;
  }
  ```

  *Migration.* For each environment: delete `type Observation`/`fn observe` from
  its `State` impl, and implement `Sensor` on the **environment** struct (the
  same three associated types the env already names for `Environment`, with the
  ranks from its `Environment<R, SR, AR>` bound). Move the old `observe` body
  into `Sensor::observe`/`observe_reset`, taking the passed-in `next_state` /
  `state` instead of `&self`. In `step`, build the snapshot observation with
  `self.observe(&action, &next_state)`; in `reset`, with
  `self.observe_reset(&next_state)` (reset has no action). Because `&self` is now
  the environment, world-derived sensors (physics raycasts, rendered frames) read
  the simulator directly and no longer need to be cached onto the state — the
  ADR-0039 cached-sensor pattern is retired. Where an observation is a pure
  function of the state, the `Sensor` body may delegate to
  `Observable::project` (see the `pixel_grid` environment).
- **`Observable<OR>` is demoted, not removed.** It is retained as an optional
  pure-projection helper a `Sensor` may delegate to; it is no longer the
  documented home for observation or for modality change. No code change is
  required for existing `Observable` impls — only the docs changed.
- **`BeliefState` signature changed.** `BeliefState<SR, AR, S, A>` becomes
  `BeliefState<OR, SR, AR, S, A>` and gains an associated
  `type Observation: Observation<OR>`; `update`'s second parameter changes from
  `&S::Observation` to `&Self::Observation` (mirroring `HiddenState`/
  `LatentState`). No `BeliefState` implementors exist in the workspace, so this
  is a contract-only change for downstream callers.
- **`TensorConvertible` splits its row-writer onto a new backend-independent
  `HostRow<R>` supertrait** (ADR 0052, extends ADR 0028). `row_shape()` and
  `write_host_row()` move to `HostRow<R>`; `TensorConvertible<R, B>` becomes
  `TensorConvertible<R, B>: HostRow<R> + Sized` and keeps only `to_tensor` (still
  derived, still not to be overridden) and `from_tensor`. ADR 0028's decisions are
  unchanged — this moves the two methods, it does not redefine them.

  *Why.* ADR 0028 required `write_host_row` to push **plain `f32`** and never
  pre-convert to a backend element type, but the method sat on a `B`-parameterised
  trait, so that contract was prose the compiler could not check — a
  backend-specialised row-writer was permitted by the signature. Under the
  off-policy agents' `TensorConvertible<DO, B> + TensorConvertible<DO,
  B::InnerBackend>` bound, an unqualified call was ambiguous (E0284), so six
  staging sites named a backend that provably could not affect the result. Had
  anyone ever specialised a row-writer on `B`, the two qualified spellings would
  have staged **different bytes** silently: 0028's `debug_assert` checks a row's
  length, not its contents. `HostRow` has no `B`, so that divergence is now
  unrepresentable — the same invariants-in-types move as `Bounds` (ADR 0027), the
  rate newtypes (ADR 0031), and `Slot` (ADR 0046). `stack_to_tensor`'s bound
  relaxes to `T: HostRow<R>` accordingly: host-side staging never touches a
  device, so it should never have demanded a decode impl. There is **no
  performance change** — no byte moves differently and no upload count changes.

  *Migration.* Split each existing impl in two, dropping `B` from the `HostRow`
  half:

  ```rust
  // Before
  impl<B: Backend> TensorConvertible<1, B> for MyObservation {
      fn row_shape() -> [usize; 1] { [4] }
      fn write_host_row(&self, buf: &mut Vec<f32>) { /* ... */ }
      fn from_tensor(t: Tensor<B, 1>) -> Result<Self, TensorConversionError> { /* ... */ }
  }

  // After
  impl HostRow<1> for MyObservation {
      fn row_shape() -> [usize; 1] { [4] }
      fn write_host_row(&self, buf: &mut Vec<f32>) { /* ... */ }
  }

  impl<B: Backend> TensorConvertible<1, B> for MyObservation {
      fn from_tensor(t: Tensor<B, 1>) -> Result<Self, TensorConversionError> { /* ... */ }
  }
  ```

  Method bodies are unchanged; only the impl blocks move. Writing
  `impl<B: Backend> HostRow<1> for MyObservation` fails with **E0207** (`B` is
  unconstrained, since `HostRow` never mentions it) — the `HostRow` half must not
  carry a backend parameter at all. Any turbofish previously needed to
  disambiguate `write_host_row` across two backends can be deleted. `HostRow` is
  re-exported from `rlevo::prelude`. Note the new documented invariant: a domain
  type implements `HostRow` at exactly **one** rank (two ranks makes unqualified
  calls ambiguous again, E0283). **No persisted data format changes** — nothing in
  this path derives serde, and no record schema is affected.
- **`BoundedAction::low()`/`high()` return `&'static [f32]` instead of
  `[f32; R]`** (ADR 0053, resolves #253). The bounds were keyed on the tensor
  **rank** `R`, not on `ContinuousAction::COMPONENTS` — the same rank-vs-component
  conflation as #100, one layer up. A rank-1 action with `C > 1` components could
  therefore express only a *single* bound for all `C` of them, which is not
  representable for an asymmetric space such as CarRacing's
  `Box([-1,0,0], [1,1,1])`. Keying on `COMPONENTS` needs `[f32; Self::COMPONENTS]`,
  which requires unstable `generic_const_exprs`, so the length moves to the slice
  instead; the trait keeps its single const generic parameter.

  *Migration.* Change each impl's return type and wrap the array literal in a
  borrow — `fn low() -> [f32; 1] { [-2.0] }` becomes
  `fn low() -> &'static [f32] { &[-2.0] }`. Impls that must compute bounds can
  use a `OnceLock` or `Box::leak`. Generic code bounded as `A: BoundedAction<AR>`
  is **unaffected** — the trait arity is unchanged, so no downstream signature
  moves. New documented invariants: `low().len() == high().len() ==
  Self::COMPONENTS`, and `low()[i] < high()[i]` for every `i`. **No persisted data
  format changes.**
- **Also on `BoundedAction`:** the trait's doc claim that bounds may be derived
  from "a runtime env config (e.g. a `max_torque` field)" is **removed as
  false** — `low()`/`high()` are
  static methods with no `self` and cannot reach instance state. `from_slice`'s
  docs (and the matching row in `docs/rules.md`) said the slice must have exactly
  `RANK` elements; the contract has been `COMPONENTS` since ADR 0038.
- **`config::positive` and `config::in_range` reject a non-finite value; the
  paired-value checks `config::ordered` and `config::distinct` now require
  *both* arguments finite** (ADR 0060, resolves #353). The rule is now
  explicit and enforced in one place: a config **value** must be finite, a
  config **bound** may be `±∞`. `positive(f64::INFINITY)` returned `Ok`
  — `f64::INFINITY > 0.0` is true — and the same comparison-vs-usable-number
  gap ran through the sibling predicates, across call sites spanning learning
  rates, physics constants, and evolution parameters. A new
  `ConstraintKind::NotFinite { got: f64 }` is checked *before* every other
  float constraint, and `ConstraintKind` is now `#[non_exhaustive]`. No call
  site changed — the fix sits entirely inside the four predicates.
  `in_range(C, f, 0.0, f64::INFINITY, x)` is unaffected and remains the
  blessed spelling of "non-negative, unbounded above": `hi = ∞` is a bound,
  not a value.

  This is a **breaking**, not a bugfix-only, change because a downstream
  `Validate` impl inherits the stricter behaviour with no source change of its
  own. It breaks in four distinct ways:

  1. **Previously-accepted configs are now rejected.** `positive(+∞)`:
     `Ok` → `Err(NotFinite)`. `in_range(lo, ∞, ∞)`: `Ok` → `Err(NotFinite)`
     for any `lo`. `ordered(-∞, ∞)`, `ordered(x, ∞)`, `ordered(-∞, x)`:
     `Ok` → `Err(NotFinite)`.
  2. **The error *kind* changes for inputs that already failed** —
     source-compatible, but assertion-breaking. `positive(NaN)` /
     `positive(-∞)`: `NotPositive` → `NotFinite`. `in_range(0, 1, ±∞)` and
     `in_range(.., NaN)`: `OutOfRange` → `NotFinite`. `distinct(NaN, x)` and
     `distinct(∞, ∞)`: `DegenerateInterval` → `NotFinite`. `ordered(∞, ∞)`:
     `NotOrdered` → `NotFinite`.
  3. **Type-level.** The new `ConstraintKind::NotFinite { got: f64 }` variant
     and the enum's new `#[non_exhaustive]` both break a downstream `match`
     with no trailing wildcard arm. Variant *construction*, including
     `Custom`, is unaffected.
  4. **`Display` text changes** for every input in (1) and (2), which breaks
     any downstream code string-matching on error messages.

  *Migration.* If you genuinely wanted an unbounded **value** rather than a
  bound, spell it with `f32::MAX` / `f64::MAX` instead of infinity — the two
  in-workspace parameters where `+∞` was a meaningful sentinel, TD3's
  `noise_clip` and CMSA-ES's `tau_c`, both mean "effectively
  unclipped/frozen", and `f32::MAX` expresses that exactly, since
  `clip(x, −f32::MAX, f32::MAX) ≡ x` for any finite `x`. **No persisted-data
  format changed, and no call site in the workspace needed editing.**

  *Why the existing tests missed it.* `positive` had a `positive_rejects_nan`
  test but no infinity test in either direction. `NaN` is rejected by
  `got > 0.0` for free — `NaN` compares false against everything — so that
  test passed without the predicate ever having a finiteness concept.
  Infinity is the one input where "greater than zero" and "usable as a
  number" diverge, and nothing exercised it. Separately, `qrdqn_config.rs`
  had already hand-rolled its own `is_finite()` guard on κ (see #345, below)
  — a call site working around the missing shared-layer rule rather than
  reporting it.
- **`Observation<R>` loses its `Serialize + for<'de> Deserialize<'de>`
  supertraits; the contract is now exactly `Debug + Clone + Send + Sync`**
  (ADR 0064, resolves #405). The bound's own doc comment claimed it existed "for
  storage in a replay buffer" — a capability the type system never delivered to
  anyone. No function, struct, or trait in the workspace generically requires
  `O: Serialize`; `ExperienceTuple`/`History` derive only `Clone, Debug`; and the
  replay seam (ADR 0050) derives no serde at all and *erases* the action payload
  specifically so it need not impose bounds. The workspace's one real persistence
  consumer of a domain type, `RecordingTap`, asks for
  `where E::ActionType: Serialize + Clone` explicitly and asks nothing of the
  observation. So the bound constrained neither a wire format nor round-trip
  fidelity, while taxing every implementor: two hand-written serde impls exist
  only to satisfy it — `ContextualBanditObservation`'s validating `Deserialize`
  and `CarRacingObservation`'s `Visitor` over 27,648 bytes (serde derives no array
  impl above length 32) — and no consumer exercises either. It also left the real
  bound set invisible where callers read for it: `experience.rs`'s bounds doc on
  `ExperienceTuple`/`History` reasons carefully, citing the Rust API Guidelines'
  C-STRUCT-BOUNDS, about declaring no `Send`/`Sync` bounds — a narrow claim that
  was and remains correct — while the `O: Observation<D>` header silently imported
  a `Serialize + Deserialize` obligation that doc never mentions.

  This is **breaking despite being a strict relaxation for implementors.** Every
  existing `impl Observation<R>` still compiles unchanged — verified. What breaks
  is downstream *generic* code that relied on the implied bound:

  ```rust
  // Compiles today; does not compile after.
  fn save<const D: usize, O: Observation<D>>(o: &O) -> String {
      serde_json::to_string(o).unwrap()
  }
  ```

  **In-tree affected sites: 0.**

  *Migration.* Add the requirement at your own call site — `fn save<const D:
  usize, O: Observation<D> + Serialize>(..)`, or a `where O: Serialize + for<'de>
  Deserialize<'de>` clause on the type or function that persists. That is the
  shape `RecordingTap` already uses, and it is additive and reversible in a way a
  supertrait is not. **All concrete observation types keep their serde derives and
  both hand-written impls**, so `serde_json::to_string(&concrete_obs)` is
  unaffected. **No persisted data changes format** — no observation was ever
  present in any persisted payload (`capture_frame` writes action bytes, reward,
  ascii, styled, and the family payload), so there is no record-schema
  `FORMAT_VERSION` bump.

  *Why no test caught it.* There was nothing to test. No code path in the
  workspace ever serialized an observation, so the bound had no observable
  behaviour to assert on — the defect was a doc comment promising a capability
  nothing requested, which only a bound-versus-consumer audit finds.

**Added**

- **`HostRow::row_is_finite`, a provided method for detecting non-finite rows**
  (ADR 0067, resolves #1043). Every observation reaching an agent was passed to
  the network unchecked. The method is *provided*, so every existing `HostRow`
  implementor compiles unchanged; it takes a `scratch: &mut Vec<f32>` rather
  than allocating internally, because an f32-backed image observation would
  otherwise cost a ~110 KB allocation on every environment step and widening
  the signature afterwards would break every impl.

  The predicate is a branchless `u32::max` reduction over the IEEE-754 exponent
  field, not `iter().all(f32::is_finite)`, and the difference is not stylistic.
  `0x7F80_0000` is the largest value the masked field can take, so the
  reduction equals it exactly when some element is non-finite; because `max` is
  associative, LLVM lowers it to a horizontal reduction. `Iterator::all` must
  short-circuit and cannot be lowered that way — measured at ~9 GB/s against a
  ~62 GB/s memcpy, roughly seven times the cost of the write it rides — and its timing
  depends on *where* the poison sits, which quietly confounds any
  clean-versus-poisoned benchmark control.

  The four integer-backed observation types (`PixelObservation`,
  `CarRacingObservation`, `GridObservation`, `GoToDoorObservation`) override it
  to `true`, since `u8 -> f32` cannot produce a non-finite value. Each override
  carries a compile-time witness — e.g. `PixelObservation`'s
  `let _: &[u8] = &self.pixels;`, or a nested-array ascription over the field
  for the grid types — so that re-backing one of them with `f32` fails to
  compile at the site whose reasoning it invalidates, rather than silently
  turning the override into a lie. The witness must stay a concrete type
  ascription: written as an
  `Into<f32>` bound it would keep compiling forever, because
  `impl<T> From<T> for T` gives `f32: Into<f32>`.
- **Kind-level tests for `config::in_range`'s rejection of non-finite values**
  (resolves #335). `in_range` is written as `got >= lo && got <= hi`, so `NaN`
  fails both comparisons and lands in the `Err` branch — behaviour its rustdoc
  already promised. A refactor to the superficially equivalent
  `!(got < lo || got > hi)` would silently start *accepting* `NaN`.

  The issue as filed overstated the gap, and the correction is worth recording:
  `in_range_boundaries_are_inclusive` **did** already pass a `NaN`, but only
  through an `is_err()` assertion. What was genuinely missing was a
  **kind-level** assertion distinguishing "rejected as `OutOfRange`" from
  "rejected for some other reason", coverage of the infinity branches, and a
  test pinning the construction path: config fields are `pub`, so
  `DqnTrainingConfig { tau: f64::NAN, ..Default::default() }` is constructible,
  and only `DqnAgent::new`'s `validate()?` stops it from reaching an agent.
  That constructor was confirmed to be the sole entry point — there is no
  `config_mut()` and the `config` field is private — so a `NaN` `tau` remains
  unreachable in practice. This matters more since #182 made `tau` a
  control-flow switch rather than a plain coefficient: `NaN > 0.0` is `false`,
  so a `NaN` `tau` would silently select pure hard-sync mode instead of
  erroring.
- **`config::nondegenerate_bounds`, a named spelling for the strictness a
  `Bounds` field does not carry** (ADR 0068, resolves #387). Migrating a
  `config::ordered(C, field, lo, hi)` pair to a single `Bounds` field discharges
  the *ordering* half of that check — inverted is unrepresentable (ADR 0027) —
  but not the *strictness* half: `ordered` is a strict `<` and rejected
  `lo == hi`, while `Bounds` deliberately permits the degenerate zero-width
  range. So every such migration reads as a pure refactor while loosening
  validation by exactly one case, and nothing marked the choice at the call
  site.

  *Why no test caught it.* A zero-width range is *legitimate* for 30 of the
  workspace's 32 `Bounds` fields — clamping or sampling to a constant is
  well-defined — so there was no invariant to assert generally. The two fields
  where it is a misconfiguration are both a policy head's `log σ`, where zero
  width collapses σ to a constant that still trains and still reports finite
  numbers. It was caught once, on SAC, only because PPO happened to be a
  sibling to diff against; the next migration would have had no sibling.

  The helper delegates to `config::distinct` rather than testing
  `bounds.span() > 0.0`, and that is load-bearing: a span check accepts
  `Bounds::new(-20.0, f32::INFINITY)` (span is `inf`, which is `> 0.0`) and the
  SAC head has no downstream span check to catch it — the obvious
  simplification would reintroduce a loosening of exactly the kind being fixed.
  A unit test pins this, so the shortcut fails CI. Enforcement is deliberately
  asymmetric: a mechanical guard test covers `rlevo-reinforcement-learning`,
  where both strictness-needing fields live, while `rlevo-evolution` and
  `rlevo-environments` keep convention, since zero width is correct for
  essentially every `Bounds` field there.

**Changed**

- **The workspace lint table is now enforced on `rlevo-core`** (#391), adding
  `#[must_use]` to 15 public items — including trait methods
  (`ContinuousAction::clip`, `DiscreteAction::random`/`enumerate`,
  `MarkovState::is_markov`/`predict_next`, `BeliefState::update`) and inherent
  methods (`EpisodeStatus::is_done`/`is_terminated`/`is_truncated`,
  `RenderPayload::new`/`with`/`with_position`, `util::combinations`). Not
  breaking — downstream code still compiles — but a caller discarding one of
  these return values now gets a new `unused_must_use` warning.
- **`TensorConvertible`'s round-trip contract is now two clauses instead of
  one** (ADR 0061, resolves #286). The old text — *"round-trip:
  `from_tensor(x.to_tensor(device))` equals `Ok(x)` for any valid `x`"* —
  silently assumed every implementor's row covers every field, and said
  nothing about what an implementor with a partial row must do. Now:
  **(1) tensor-image fidelity** — `from_tensor(x.to_tensor(d))?.to_tensor(d)
  == x.to_tensor(d)`, decode-then-re-encode is a no-op *on the tensor* — and
  **(2) no fabrication** — any field `write_host_row` omits must decode to an
  explicit absence (`None`, or a dedicated "unknown" variant), never a
  plausible in-domain value. A type whose row covers every field gets the
  stronger `from_tensor(x.to_tensor(d)) == Ok(x)` for free and satisfies (2)
  vacuously — the expected case, and (1) is a **floor**, not a licence to be
  partial. `docs/rules.md`'s Core Trait Invariants table and §10 carry the
  matching text.

  This is **doc-only**: no signature on `HostRow` or `TensorConvertible`
  changes, and none of the ~33 other impls in the workspace needs to change —
  each already writes every field its type declares, so clause 1's stronger
  form and clause 2's vacuous case already held for them. The two grid
  observation types that motivated the amendment are the `rlevo-environments`
  entries below.
- **`Environment::step`'s post-terminal contract is now unconditional** (ADR
  0044 §8, closes #289). The rustdoc's alpha migration note — which disclosed
  that only some families enforced the contract and that post-terminal
  behaviour was **undefined** everywhere else, so callers must not rely on it —
  is deleted: with the bandit family guarded, every `Environment` implementation
  in the workspace now holds an `EpisodeGuard` and rejects a post-terminal
  `step()`. The normative "implementations **must** return
  `EnvironmentError::StepAfterEpisodeEnd`" text is unchanged; what changes is
  that it is now true of the whole workspace rather than aspirational, so
  callers may rely on it. The same section also now says the check belongs
  ahead of any state mutation **or RNG draw** — a rejected step must not
  advance the environment's RNG stream either, which ADR 0029 makes persistent,
  observable state; `EpisodeGuard`'s own docs carry the matching wording.

  This is **doc-only**: no signature changes and no behavioural change in
  `rlevo-core`. Fixture and stub environments (`rlevo-test-support`, the
  in-crate `StubEnv`/`MockEnvironment` types) remain exempt by ADR 0044 §9 —
  `TimeLimit`'s `StubEnv` is unguarded on purpose so the wrapper's tests
  exercise the wrapper's own guard.

### `rlevo-environments`

**Breaking changes**

- Every environment now implements `rlevo_core::environment::Sensor` and builds
  its `reset`/`step` snapshots through it instead of `State::observe` (ADR 0047,
  #329). For the box2d family the observation is no longer *produced* from a state
  cache — the env-side sensor reads the world directly (BipedalWalker lidar,
  CarRacing pixels). `CarRacing` drops its cached pixel buffer entirely;
  `BipedalWalker`/`LunarLander` retain only a finiteness signal for `is_valid`
  (`last_obs`/`prev_shaping` respectively), matching the locomotion states, so
  `is_valid` is unchanged. `pixel_grid` keeps `Observable<3>` and delegates its
  `Sensor` to `project()`. Behaviour (observations, rewards, termination) is
  unchanged for every family except the grids, which this same unreleased cycle
  revisits below: ADR 0047 §5 initially kept the grid family routing its shared
  egocentric projection through `GridState: Observable<3>` and the
  `build_snapshot` chokepoint, with only `GoToDoorEnv` implementing `Sensor` for
  its goal-conditioned observation. That exemption's stated premise — one
  shared, state-pure projection covers all eleven other envs — turned out to be
  false (canonical Minigrid sets `see_through_walls` per environment, not per
  family) and is reversed later in this section (ADR 0063): `Observable<3>` is
  removed from `GridState` entirely, and all twelve grid environments implement
  `Sensor`. Neither state shipped to a user, so there is one migration path
  below, not two.
- **Nine grid environments now reject sub-minimum configs at construction**
  (#106): `CrossingEnv`, `DoorKeyEnv`, `DynamicObstaclesEnv`, `EmptyEnv`,
  `FourRoomsEnv`, `LavaGapEnv`, `MultiRoomEnv`, `UnlockEnv`, `UnlockPickupEnv`.
  Each declared a minimum but enforced it only in `FromStr`, so `with_config` and
  `ConstructableEnv::new` accepted values that `from_str` refused. Configs that
  previously built — `EmptyConfig { size: 3, .. }`, `MultiRoomConfig { num_rooms:
  1, .. }`, and every other sub-floor value — now return `Err(ConfigError)`.

  *Migration.* Raise the offending field to the environment's documented minimum:
  `Empty`/`Unlock` 4, `DoorKey`/`DynamicObstacles`/`LavaGap` 5,
  `Crossing`/`UnlockPickup` 7, `FourRooms` 11 **and odd**, `MultiRoom`
  `num_rooms ≥ 2`, `room_width ≥ 3`, `height ≥ 5`. Every `Default` config already
  satisfies its floor, so `ConstructableEnv::new` is unaffected. No persisted data
  is involved — `Deserialize` remains non-validating by design (ADR 0026 §2), and
  the constructor is where the rejection now happens for a deserialized config too.
- **`ConfigError::kind` for a zero-valued grid dimension changes from
  `ConstraintKind::Zero` to `ConstraintKind::TooSmall { min, got }`** on the same
  nine environments, because a single `config::at_least` guard replaces the
  `config::nonzero` it subsumes. Code matching on `Zero` for these fields must
  match `TooSmall` instead; the new variant carries strictly more information.
  `max_steps` keeps `nonzero`/`Zero` throughout — it has no floor above 1.
  `FromStr`'s `String` error text changes correspondingly, from
  `"size must be >= 5, got 1"` to `ConfigError`'s `Display` form.
- **`GridObservation`/`GoToDoorObservation::agent_direction` changes from
  `u8` to `Option<Direction>`** (ADR 0061, resolves #286, closes #844). The
  field is public on both types, which every one of the 12 grid environments
  emits, so this reaches every caller that reads it.

  *Migration.* `obs.agent_direction` is no longer directly comparable to a
  `u8` or to `Direction::to_u8()`'s output. Compare it against
  `Some(Direction::East)` (etc.) instead of `Direction::East.to_u8()`. A
  tensor-decoded observation — `from_tensor`, the path a decode goes through
  — now reports `agent_direction: None` rather than the fabricated `North`
  it silently reported before; a real observation, from any environment's
  `reset`/`step`, always reports `Some(direction)`. **No persisted data
  breaks**: nothing in-tree serializes a grid observation — `ExperienceTuple`
  holds it by value with no persistence path, `BenchStep` isn't `Serialize`,
  and the report client's wire format carries steps and metrics, not raw
  observations. An **externally**-serialized observation's wire form does
  change, though, since both types still derive `Serialize`/`Deserialize`:
  the field was a plain integer and is now an `Option`-wrapped enum tag —
  in JSON, `3` becomes `"North"` or `null`. The exact bytes depend on the
  format (bincode, MessagePack, and JSON each encode the change differently),
  so re-serialize rather than hand-patching a stored payload.
- **Five grid environments draw a fresh layout on every `reset()`** (ADR 0062,
  which partially supersedes ADR 0029; refs #282, closes #108): `CrossingEnv`,
  `DoorKeyEnv`, `LavaGapEnv`, `FourRoomsEnv`, `UnlockPickupEnv`. Each previously
  rebuilt one board that was a pure function of `size`, so a training run was a
  single board repeated for however many episodes it lasted, and a policy could
  score well on it by memorizing one route. `reset()` now samples from the
  environment's persistent RNG and lets the stream advance — what ADR 0029
  already required of every stochastic environment — so a fixed `seed`
  reproduces a fixed *sequence* of episodes rather than one repeated episode.
  Semver-relevant in the same class as ADR 0029's own `reset()` change.

  *Migration.* Where you relied on consecutive resets returning the same board —
  a scripted rollout, a fixture, a test pinning a coordinate — call the new
  inherent `reset_with_seed(seed)` instead of `reset()`: it re-seeds the
  persistent stream and then resets, so one nominated episode replays
  bit-for-bit. Run-level reproducibility is unchanged; construct with a fixed
  `config.seed` and the whole episode sequence is reproducible exactly as
  before. The accessors that survived unchanged in signature — `DoorKeyEnv::split_col`,
  `LavaGapEnv::lava_col`, `LavaGapEnv::gap_row` and `UnlockPickupEnv::target` —
  now report *this episode's* sampled value instead of a constant, so any caller
  that recomputed one from `size` (`size / 2` and friends) or hard-coded the
  target box will now disagree with the board; read them from the environment.
  `target` in particular was never re-derived in `reset()`, which was harmless
  only while the box colour was a `const`. **No persisted config breaks.** Every grid
  `*Config` keeps its `seed: u64` field, retained for precisely this reason
  (ADR 0062 §2), and no grid config gained, lost or renamed a field, so a
  payload serialized before this change still deserializes.
- **`CrossingEnv::gap_col` returns `Vec<i32>` instead of `i32`**, and
  `strip_rows()` narrows in meaning. The old board was two horizontal strips at
  `size / 2 ± 1` sharing a single opening at `size / 2`, which one column
  described completely. Upstream `CrossingEnv` shuffles its candidate rivers and
  punches one opening per river, and a river may be vertical as easily as
  horizontal, so neither the count nor the orientation is fixed: `gap_col()[k]`
  is the opening of the horizontal river at `strip_rows()[k]`, and the vector is
  empty on an episode whose two sampled rivers are both vertical.

  *Migration.* Pair `gap_col()[k]` with `strip_rows()[k]` rather than treating
  the gap as one shared column, and read the vertical rivers through the new
  `strip_cols()`/`gap_rows()` pair — `strip_rows()` alone no longer describes the
  board. A caller that only wants to know whether a cell is passable should read
  `env.state().grid` instead of reconstructing the layout from accessors at all.
- **All 12 grid environments implement `Sensor` (ADR 0063, resolves #281);
  `impl Observable<3> for GridState` is removed.** Each environment's
  `observe`/`observe_reset` forwards to the shared public `observe_grid`
  (backed by an internal `mask_view` helper), parameterised by that
  environment's own `const VISIBILITY`, so the per-env
  difference is one constant, not eleven near-duplicated projections.
  `build_snapshot` now **takes** an already-produced `GridObservation` instead
  of producing one — its signature changes from `build_snapshot(reward, done)`
  to `build_snapshot(observation, reward, done)`. `egocentric_view` is demoted
  to `pub(crate)` and dropped from the `grids::core` re-export entirely: a raw,
  unoccluded view is semantically wrong for eight of the twelve environments
  once visibility is per-env, so it is no longer public API. No in-tree caller
  outside `grids/` used any of these, so this is a zero-migration break for
  every consumer of the public crate.
- **`Entity::type_u8` is renumbered to canonical Minigrid's `OBJECT_TO_IDX`**
  (ADR 0063, #281): `Empty` 0→1, `Wall` 1→2, `Floor` 2→3, `Door` 5→4, `Key`
  6→5, `Ball` 7→6, `Box` 8→7, `Goal` 3→8, `Lava` 4→9, with `0` now reserved for
  an unseen (masked) cell rather than doubling as `Empty`. This is channel 0 of
  every grid observation's encoding, across all 12 environments.

  *Migration.* **Persisted configs and recorded runs still load** — `Entity`
  derives `Serialize`/`Deserialize` by variant name, not by `type_u8`, and
  `render.rs`/the WASM report client match on the `Entity`/`GridTile` variant,
  never on the raw byte — so nothing in the existing record/report pipeline
  needs to change. What does not survive is anything keyed to the **byte
  table** itself: a saved observation tensor, a trained checkpoint whose input
  layer learned the old numbering, or an external baseline comparing raw
  channel-0 values must be regenerated against the new table. Note channel 1
  (color) is unaffected and therefore now **mixed-parity** with channel 0:
  `Entity::color_u8` still reserves `0` for "no color" rather than shifting by
  one the way `type_u8` did (`Color::to_u8`'s own indices are unchanged), which
  will surprise anyone porting a Minigrid baseline expecting both channels to
  move together.
- **The inherent `KArmedBandit::is_done()` is removed** (ADR 0044, #295). It was
  a second source of truth for episode termination, which `docs/rules.md` §10
  forbids: done-ness is read from the snapshot, never from the environment by
  other means. The method could not have been the real answer in any case — it
  returned a private `done: bool` that `step()` wrote and nothing ever read, so
  it reported termination from a field the episode's actual lifecycle had no
  dependence on. The three sibling bandits never exposed an equivalent, so the
  removal also ends an inconsistency inside the family.

  *Migration.* Read done-ness from the snapshot: `snapshot.is_done()`
  (`rlevo_core::environment::Snapshot::is_done`), which is what every in-tree
  caller already did. No in-workspace caller of the inherent method existed, so
  the break is nominal for anyone not depending on it externally.

**Added**

- **`BoundedAction<1>` for the five multi-component continuous actions**
  (`BipedalWalkerAction`, `CarRacingAction`, `LunarLanderContinuousAction`,
  `ReacherAction`, `SwimmerAction`; ADR 0053, #253). These were the environments
  the rank-keyed bounds representation had kept out of DDPG/TD3/SAC: each is
  rank-1 with more than one component, so it could not state its bounds at all
  under the old signature. `CarRacingAction` is the workspace's only action whose
  components disagree — steering ∈ [-1, 1] but gas and brake ∈ [0, 1].
- **`reset_with_seed(seed)` on `CrossingEnv`, `DoorKeyEnv`, `LavaGapEnv`,
  `FourRoomsEnv` and `UnlockPickupEnv`** — the inherent replay hatch ADR 0029 §1
  mandates, matching the three grid environments that already had it
  (`GoToDoorEnv`, `MemoryEnv`, `DynamicObstaclesEnv`). It re-seeds the persistent
  stream and then resets, which is the only way to reproduce one *specific*
  episode now that plain `reset()` advances the stream.
- **Accessors that read the sampled board instead of recomputing it**:
  `DoorKeyEnv::door_row`, `FourRoomsEnv::openings` (with the public
  `OPENING_COUNT`), `UnlockPickupEnv::door_pos` and `::door_color`, and
  `CrossingEnv::strip_cols`/`gap_rows`. A layout that varies per episode is no
  longer derivable from `size`, so anything that needs to locate the door, the
  doorways or the one passable cell — a scripted rollout, a planner, a test —
  must ask the environment rather than recompute.
- **`grids::core::Visibility`** (pub enum) and **`grids::core::observe_grid`**
  (pub fn) — the per-environment occlusion mode and the shared
  observation-building helper behind the `Sensor` migration above (ADR 0063,
  #281). Every grid environment's `Sensor` impl now reduces to one
  `const VISIBILITY: Visibility` plus a call into `observe_grid`, which is the
  new public surface an external environment implementor following the same
  pattern would need to name.
- **`grids::core::placement`**, the shared placement sampler the family never
  had: `is_free`, `sample_pos`, `place_obj`, `place_agent`, `random_direction`,
  `no_reject`, a `Rect` region type and a `PlacementError`. `grids/core/`
  previously offered
  `Grid::{new, in_bounds, get, set, draw_walls}` and nothing else — no free-cell
  predicate, no position sampler — so the only sampling code in the family was
  `GoToDoorEnv::sample_door_colors`, a hand-rolled rejection loop. Randomizing
  several environments at once off that base means one chance per environment to
  get the free-cell predicate wrong. `DoorKeyEnv`, `FourRoomsEnv` and
  `UnlockPickupEnv` place through it; `CrossingEnv` and `LavaGapEnv` deliberately
  do not, and say why in-file — `Crossing` draws lines rather than cells and
  punches openings *into* obstacle rows that `is_free` would reject outright,
  while `LavaGap`'s column and row are two independent range draws that
  `sample_pos` would collapse into one draw over a rectangle and silently
  re-weight. Two design choices are documented at the sampler: exhaustion returns
  `Result` and converts to `ConfigError` at the environment boundary rather than
  panicking, because a region can exhaust on an unlucky draw from an entirely
  valid config (`DoorKey` at `size = 5` has a 3×3 interior and the draws consume
  two cells of it), so the ADR 0026 chokepoint cannot rule it out; and the sampler
  materializes the candidate cells and draws a uniform index rather than porting
  upstream's unbounded rejection loop, whose failure mode is a `reset()` that
  never returns.
- **`tests/rng_seeding_guards.rs`**, a source-text guard over the **whole**
  crate's `src/`: every `seed_from_u64` must sit inside an allowlisted
  constructor or replay hatch, and the allowlist is checked in both directions so
  a row matching nothing on disk fails too. The guard exists because the two
  halves of #282 were only dangerous together — a re-seed in `reset()` is a
  genuine no-op while the RNG is unread, and silently pins every episode the
  moment sampling is added, and *both* states pass a test that asserts only "the
  environment draws from its RNG". Resolving the enclosing `fn` is the mechanism,
  not an optimization: `reset_with_seed` carries `seed_from_u64` three lines above
  `self.reset()`, so a line-level grep flags the very method the convention
  mandates. Scope is crate-wide rather than `grids/`-only because #104 was
  crate-wide, and the limits are recorded in-file — it reads source text, so it
  is defeated by aliasing or by a constructor invoked *from* `reset`, and it
  catches the accident rather than the adversary.

**Changed**

- **Eight of twelve grid environments now occlude; four stay see-through,
  matching upstream exactly** (ADR 0063, #281): `crossing`, `door_key`,
  `four_rooms`, `lava_gap`, `memory`, `multi_room`, `unlock`, and
  `unlock_pickup` run the canonical shadow cast; `empty`, `dist_shift`,
  `dynamic_obstacles`, and `go_to_door` stay fully visible, per canonical
  Minigrid's own per-env `see_through_walls` values (whose default is
  occlusion — an environment opts *out*, not in). `FourRoomsEnv`,
  `MultiRoomEnv`, `DoorKeyEnv`, and `UnlockPickupEnv` get harder, since the
  agent can no longer see into a room it has not entered — but say this
  honestly, not from the floor plan alone: `FourRooms`'s four cross openings
  let the shadow cast's flood fill spread sideways into a neighbouring room,
  so from most poses the goal in an unentered room stays visible exactly as
  before, and across 12 seeds the goal was maskable from any pose in only 2 of
  them. Stronger still, and measured while writing the per-env tests: **three
  of the eight occluding environments mask no in-grid cell in any pose at
  all.** `LavaGapEnv` and `CrossingEnv` at its default lava kind hide nothing,
  because lava is transparent to the shadow cast (canonical `see_behind()` is
  overridden only by `Wall`, and by `Door` when shut) and a convex room of
  transparent cells is entirely lit by a flood fill — only
  `CrossingKind::Wall`, the `SimpleCrossing` family, occludes anything.
  `UnlockEnv` hides nothing for a different reason: rlevo draws one perimeter
  room with the door in the outer wall, where upstream `RoomGrid` has two
  rooms with the door between them (#1020). In all three the shadow cast is
  running and correct; there is simply nothing positioned for it to hide. So
  "eight environments now occlude" describes what they *run*, not what any of
  them *hides*. `grid_memory_rl`'s bench numbers are not comparable across
  this change — they now measure a different (occluded) task, not a
  regression.
- **`EmptyEnv` and `DistShiftEnv` are deterministic on purpose, and their docs
  now say so** (ADR 0062 §1, §2b); the unread `_rng` field is deleted from both.
  Each was reconciled against upstream and found faithful: `MiniGrid-Empty-*`'s
  `_gen_grid` makes no draws at all — only the separately registered
  `-Empty-Random-*` ids call `place_agent` — and `DistShift`'s lava row is a
  registration constant because determinism *is* the experiment, a train/test
  distributional-shift probe over two fixed, known boards that randomizing either
  half would destroy. Both keep their `seed` field for config-surface uniformity
  across the family, and each doc now states outright that the stored value
  cannot affect any observation, reward or transition, in place of the
  "reserved for future stochastic variants" promise that had no owner.
- **`MultiRoomEnv` and `UnlockEnv` ship knowingly non-conformant with ADR 0062
  §1, and now record it at the call site** (#1021, #1020). `MultiRoom` keeps its
  fixed equal-width strip: upstream resamples the room count and every room's
  size and position through a recursive placer with backtracking, which is
  procedural generation rather than a handful of draws, and bundling it would
  have held the five tractable environments hostage. `Unlock` keeps its fixed
  board pending a two-room topology change that moves `MIN_SIZE`, the layout and
  the solvability oracle together — and its `seed` doc now also carries the
  placement defect found on the way past: upstream `Unlock` puts the locked door
  on the wall *between* two rooms, whereas `UnlockEnv` draws one
  perimeter-walled room and then writes the door into that perimeter at
  `(1, 0)`. A door in the perimeter is not the `Unlock` task. Both deviations are
  stated in the config's `seed` doc rather than only in a tracker, which is why
  #282 stays open on a 7/9 checklist instead of closing against work that was
  not done.
- **The six planner-driven solvability oracles became seed loops over sampled
  boards**, each sweeping `PLANNED_SEEDS` episodes at two board sizes, and gained
  an `assert_boards_varied` guard that fails a run whose seeds all produced one
  board. The per-env `build_places_*` assertions they replace pinned exact cells
  on a fixed board and never asked the question procedural generation actually
  fails — whether a board the environment *generates* can be solved at all.
- **`grid_door_key_scripted` pins its seed**, and its module docs explain why: a
  fixed action script is a property of one board, not of the environment, so an
  example built around a script has to nominate its episode through
  `reset_with_seed`. The documented run command was also wrong
  (`-p rlevo-environments`, where the example does not live) and is now
  `-p rlevo`.

**Fixed**

- **Post-terminal `step()` drew a fresh random reward and re-emitted
  `Terminated` across the whole bandit family** (ADR 0044, resolves #295 and
  closes the #289 sweep) — `KArmedBandit`, `AdversarialBandit`,
  `ContextualBandit` and `NonStationaryBandit`. Each already carried a
  `done: bool` that `step()` **wrote and nothing read**, so the guard was
  half-built and inert: a call made after `max_steps` kept incrementing the
  step counter, produced a **fresh reward** — a new draw from the arm's
  distribution for the three stochastic bandits, a further slide along the
  adversary's `steps`-indexed schedule for `AdversarialBandit` — and emitted a
  **new `Terminated` snapshot** — every time, without bound. A
  trainer that stepped one call past the budget did not get an error and did
  not get a repeat of the terminal transition; it got fabricated experience
  that looks indistinguishable from real experience once it is in a replay
  buffer. This is exactly the silent data contamination ADR 0044 rejects
  absorbing semantics to avoid, except here it was not even absorbing — the
  reward was freshly random on each call.

  All four now hold an `episode::EpisodeGuard`, `check()` it as the **first**
  statement of `step()`, and `record()` the emitted snapshot's own
  `EpisodeStatus` on a single exit, so the guard and the snapshot cannot drift
  apart. The `done: bool` field is gone from all four, along with the inherent
  `KArmedBandit::is_done()` accessor it backed (see *Breaking changes* above):
  `EpisodeStatus` is the single source of truth (`docs/rules.md` §10), and a
  bool could not have carried the `Terminated`/`Truncated` distinction the
  error reports anyway.

  The guard sits **ahead of** the action-validity check, because the episode
  being over is a fact about the call sequence and is independent of whether
  the action was well-formed. One observable consequence: an out-of-range arm
  index replayed past a terminal snapshot now reports `StepAfterEpisodeEnd`
  rather than `InvalidAction` — the caller is told about the sequencing bug
  they have, not handed the wrong diagnosis.

  The existing tests missed all of this for two compounding reasons. Nothing
  asserted on post-terminal behaviour — every test stopped on the terminal call
  and checked its status, never asking what a further step returns — and
  because `done` was **write-only**, no test *could* have observed that the
  lifecycle tracking was half-implemented; the field's value was correct and
  simply never consulted. A dead field is invisible to behavioural testing by
  construction. Each environment gained the shared
  `assert_rejects_post_terminal_step` conformance check, a
  rejected-before-`is_valid()` test, a reset-reopens-the-episode test, and a
  no-mutation regression: the three stochastic bandits pin that a rejected step
  does not advance the **RNG stream** (ADR 0029 makes that stream persistent,
  observable state, so consuming a draw on a rejected call would desynchronise
  every subsequent episode), and `AdversarialBandit` — whose reward is a pure
  function of `steps` — pins the deterministic analogue, that the schedule does
  not slide.
- **Post-terminal `step()` kept moving the agent — and paid a *negative* goal
  reward — in `pixel_grid`** (ADR 0044, resolves #294). `PixelGridEnv::step`
  opened with `self.steps += 1; self.state.apply_move(action);` and had no
  terminal check at all, so a step taken past a done snapshot advanced the
  latent, ticked the counter and emitted a fresh reward.

  Neither ending is self-limiting. An agent standing on the goal keeps
  satisfying `at_goal()` — it is a live read, not a latch — so a finished
  episode re-emitted `Terminated` with a fresh success reward on every
  subsequent call. Past `max_steps` the truncation predicate keeps holding
  while `steps` climbs beyond the budget the caller asked for. The two
  compound into the real defect: Minigrid's success formula
  `1 - 0.9 * (step / max_steps)` is only non-negative for `step <= max_steps`,
  so walking onto the goal *after* truncation paid a **penalty** for reaching
  it. Measured on the pre-fix code with `max_steps = 3`, fixed placement: the
  episode truncated at step 3, a further 16 steps to the goal terminated at
  step 19 with reward **−4.7**.

  `PixelGridEnv` now holds an `episode::EpisodeGuard`, `check()`s it as the
  **first** statement of `step()` — ahead of every mutation — and `record()`s
  the emitted snapshot's own `EpisodeStatus` on a single exit. Because the
  counter can no longer climb past `max_steps`, the negative branch of
  `success_reward` is unreachable **by construction**; the separately-tracked
  `success_reward` negativity is resolved without a `step.min(max_steps)`
  clamp, which would have hidden the call-sequence bug rather than rejecting
  it. `reset_with_seed` reseeds and then delegates to `Environment::reset`, so
  there is exactly one `guard.reset()` site and a future reset variant cannot
  forget it.

  Existing tests missed the defect because none of them stepped past a done
  snapshot: `reaching_goal_terminates_with_positive_reward` and
  `step_limit_truncates_with_zero_reward` both stop on the terminal call and
  assert its status, never asking what a further step does. Six new tests
  cover it — the shared `assert_rejects_post_terminal_step` conformance check
  on **both** done paths, a no-mutation regression, the `success_reward`
  reachability bound, and reset/`reset_with_seed` re-opening the episode.
- **Post-terminal `step()` kept integrating the Rapier sim across the whole
  `locomotion` family** (ADR 0044, resolves #292) — `InvertedPendulum`,
  `InvertedDoublePendulum`, `Reacher` and `Swimmer` gated `step()` only on
  `action.is_valid()`, so a step taken past a done snapshot ran another
  physics tick and the observation drifted past termination.

  The two pendulums are the sharp case: their healthiness predicates are live
  reads of the current pose, not latches. A toppled pole that swings back
  through `|θ| < 0.2` (`InvertedPendulum`) or a tip that swings back above the
  healthy z floor (`InvertedDoublePendulum`) re-earns the alive bonus (+1,
  +10 respectively) on a **`Running`** snapshot — a finished episode was
  silently resurrected, not merely drifted. `Reacher` and `Swimmer` have no
  `Terminated` path at all — their only done status is `Truncated` at `steps
  >= max_steps` — so for them the defect was unbounded stepping past the
  truncation boundary rather than reward re-firing.

  All four now hold an `episode::EpisodeGuard`, `check()` it as the **first**
  statement of `step()` — ahead of the `action.is_valid()` bounds check,
  because the episode being over is a call-sequence fact independent of
  whether the action was well-formed — and `record()` the emitted snapshot's
  own status on a single exit. One observable side effect of the ordering: a
  replayed malformed action taken past a terminal now reports
  `StepAfterEpisodeEnd` rather than `InvalidAction`.

  Each environment gained an `assert_rejects_post_terminal_step` conformance
  test and a reset-reopens-the-episode test. Existing tests missed the defect
  because none of them stepped past a done snapshot; `Reacher`'s
  `reward_distance_is_nonpositive` and `reward_control_is_nonpositive` each
  drive exactly 50 unconditional steps against a default `max_steps == 50` —
  they land precisely **on** the truncation boundary, and one more iteration
  in either would have caught it.
- **Post-terminal `step()` kept integrating the physics sim across the whole
  `box2d` family** (ADR 0044, resolves #293) — `BipedalWalker`, `CarRacing`,
  `LunarLanderDiscrete` and `LunarLanderContinuous` checked only
  `action.is_valid()`, so a `step()` taken past a done snapshot ran another
  Rapier tick and emitted a fresh reward. The four terminal predicates are live
  tests over the world, not latches, and every one of them keeps holding after
  the episode ends, so the environments did not merely drift — they re-fired.

  `LunarLander` is the sharp case, and the one #122 left exposed. That issue
  made the terminal reward *overwritten* rather than accumulated: a crash sets
  `reward = -100.0` exactly, discarding the shaping delta. A crashed hull stays
  in ground contact, so `hull_in_contact()` was still `true` on the next call
  and the unguarded environment re-emitted `Terminated` with a **fresh −100
  every time**. Measured on the default config at `seed = 0`, free fall,
  `DoNothing`: terminal at step 135 with −100, then −100 on each of five further
  steps — **−600 banked for a single crash**, identically for both the discrete
  and continuous variants, and unbounded because the contact never clears. A
  rollout loop that steps once more before checking `is_done()` pays that
  penalty again with no signal that anything went wrong. `BipedalWalker` had the
  same shape one level deeper: a fallen hull re-pays the −100 fall penalty into
  `total_reward`, which is itself the accumulator its `total_reward < -100.0`
  termination rule reads.

  All four now hold an `episode::EpisodeGuard`, `check()` it as the **first**
  statement of `step()` — ahead of the `action.is_valid()` bounds check, because
  the episode being over is a call-sequence fact independent of whether the
  action was well-formed, and ahead of `world.step()`, the step counter and
  `prev_shaping` — and `record()` the emitted snapshot's own status on a single
  exit. Ordering the guard first also matters for `LunarLander`'s
  `WindMode::Stochastic`: `step_common` calls `apply_wind` before anything else,
  which draws from `wind_rng`, and ADR 0029 requires a rejected step to leave
  every seed stream where it was. `BipedalWalker::reset` clears the guard only
  *after* its fallible `rebuild_world()?` succeeds (ADR 0044 §6, the
  `TimeLimit::reset` precedent); `LunarLander` clears it in the infallible
  `LunarLanderCore::rebuild`, the single point both variants' `reset` funnels
  through, so a future third variant cannot forget.

  Each environment gained a `assert_rejects_post_terminal_step` conformance
  test, a reset-reopens-the-episode test, and a state-untouched test that
  compares the observation across the rejected call — for `CarRacing` that is
  the full 96×96×3 frame, which is the direct evidence the world did not
  integrate. `LunarLander` additionally pins the #122 regression on both
  variants. Existing tests missed all of this because none of them stepped past
  a terminal snapshot — they unwrapped every call and never asked what the
  status was. `test_joint_obs_not_dead` is the one that shows it: it drove 30
  unconditional `.unwrap()`ed steps with an asymmetric action that topples the
  walker inside those 30, and only the absence of the guard kept it green. It
  now breaks on `is_done()`, with both assertions intact on the terminal
  snapshot.
- **Post-terminal `step()` was an unbounded reward pump across the whole `grids`
  family** (ADR 0044, resolves #291) — all twelve gridworlds derived termination
  from a predicate over the *current* board rather than latching it, and the
  shared dynamics consume nothing on success: `step_forward` moves the agent
  **onto** `Entity::Goal` and leaves the cell intact. So a finished episode was
  re-enterable. Stepping off the goal and back on re-raised
  `StepOutcome::ReachedGoal` and paid `success_reward` again, on a loop as short
  as two steps. Measured, not inferred: `EmptyEnv` banked `0.955 + 0.901 = 1.856`
  for a single goal; `GoToDoorEnv` returned **5.649 on a task whose maximum is
  1.0** (one win plus five replayed `Done`s). Because `success_reward` is
  step-count-discounted and deliberately unclamped, replays past `max_steps` go
  **negative** — a below-zero reward on a snapshot the caller reads as a win.
  The intermediate calls emitted **`Running`** snapshots *after* a `Terminated`
  one, so any rollout loop watching `is_done()` simply kept collecting.

  Several envs lost a task-defining property rather than just reward hygiene. A
  `LavaGapEnv` / `CrossingEnv` death was reversible — the agent stands *on* the
  lava, and one more `Forward` walks it off and re-emits `Running`, converting a
  `0.0` death into a positive-return episode. In `MemoryEnv` a wrong commit could
  be retracted and the other object taken for full reward, making guessing free
  and dissolving the recall property the 11-cell layout exists to enforce; the
  same retry defeated `GoToDoorEnv`'s 25% cap on a mission-blind policy.
  `UnlockEnv` was farmable because `toggle` maps `Open -> Closed` with no key
  check, and `UnlockPickupEnv` because `Drop` returned the box and flipped
  `has_target()` false. `DynamicObstaclesEnv` is the one that broke a documented
  *invariant*: a collision-terminal step leaves an obstacle tracked on the
  agent's cell with no `Ball` drawn, so re-entering `move_obstacles` violates its
  SOUNDNESS premise and obstacles merge — reproduced at `seed = 37`,
  `size = 5`, `num_obstacles = 4`, where `[(2,2),(2,1),(1,1),(1,3)]` becomes
  `[(1,2),(1,1),(1,1),(2,3)]` on the *first* post-terminal step. #125 shipped
  with that caveat documented rather than fixed; the guard now makes the
  invariant unconditional, and the hedged prose on `obstacles()` and
  `move_obstacles` has been tightened accordingly.

  Every one of the twelve now holds an `episode::EpisodeGuard`, `check()`s it as
  the first statement of `step()` — before the step counter, before
  `apply_action`, and before `move_obstacles` draws (ADR 0029: a rejected step
  must not advance the seed stream) — and `record()`s the emitted snapshot's own
  status on a single exit. `DynamicObstaclesEnv`'s three return paths were
  collapsed to one to make that reachable. Where `reset()` runs a fallible
  `Self::build(..)?` (`FourRoomsEnv`, `DoorKeyEnv`, `UnlockPickupEnv`) the guard
  is cleared **only on success**, per ADR 0044 §6 and the `TimeLimit::reset`
  precedent: clearing first would re-open a finished episode over a board that
  never returned to its initial state.

  Note the guard is orthogonal to #1028, still open: `build_snapshot` maps a
  step-limit cutoff to `Terminated` rather than `Truncated`, so the grids family
  still disagrees with `SantaFeAnt`, which emits `Truncated` for its time limit.
  The new conformance tests deliberately end their episodes on a task terminal
  (goal, lava, mission, collision) rather than the step limit, so fixing #1028
  will not require rewriting them.
- **Post-terminal `step()` silently resurrected a finished episode across the
  whole `classic` family** (ADR 0044, resolves #290) — the contract #105 made
  normative on `Environment::step` was enforced only by `toy_text` and
  `TimeLimit`; the six `classic` environments each recomputed terminality from
  the *current* state on every call, with no memory that the episode had already
  ended. Because that test is a predicate and not a latch, the failure was not a
  benign no-op: a post-terminal `step()` on `CartPole` re-emitted a fresh
  `Terminated` snapshot and paid the `+1.0` again while advancing `steps` past
  the true episode length, and on `MountainCarContinuous` it re-paid the `+100`
  goal bonus on every extra call. `Acrobot` was worse still — the RK4 integrator
  keeps running past the goal, the tip swings back *below* the height threshold,
  and the next snapshot reads **`Running`** with reward `−1`, so a finished
  episode came back to life with a corrupted return. `SantaFeAnt` could keep
  eating pellets past a budget it had already exhausted. Every one of these
  environments now holds an `episode::EpisodeGuard`, `check()`s it before any
  state mutation or RNG draw (so a rejected step cannot desynchronise the seed
  stream — ADR 0029), and `record()`s the emitted status on a single exit path.
  The existing tests missed it because they all stopped at the first `is_done()`
  snapshot — the standard rollout shape — and so never exercised the one call
  sequence that fails; the regression tests now step deliberately past it via
  the shared `assert_rejects_post_terminal_step` conformance helper.
  `Pendulum` has no terminal condition and so cannot be covered by that helper;
  it takes the guard for family uniformity, verified only to stay open, so a
  future termination rule cannot reintroduce the hole. The remaining families
  (`grids`, `locomotion`, `box2d`, `pixel_grid`, bandits) were still unguarded
  at this point in the cycle, tracked by #289 — all five land later in this
  same `[Unreleased]` section (below), closing out the sweep and making the
  `rlevo-core` migration-note removal (above) true workspace-wide.
- **`ContinuousAction::from_slice` now accepts exactly `COMPONENTS` values on all
  five multi-component continuous actions**, matching what the trait and
  `docs/rules.md` §3 have always documented. `ReacherAction` and `SwimmerAction`
  had **no** length check at all — a short slice panicked with a bare
  index-out-of-bounds and a long one was silently truncated — while
  `CarRacingAction`, `BipedalWalkerAction` and `LunarLanderContinuousAction`
  asserted only `len() >= COMPONENTS` and so accepted (and truncated) a long
  slice. All five now `assert_eq!` and carry a matching `# Panics` line. No
  in-workspace caller passed a non-exact slice, so nothing else changes.
- **`from_tensor` no longer fabricates the agent's facing on decode**
  (ADR 0061, resolves #286). `write_host_row` on both `GridObservation` and
  `GoToDoorObservation` never wrote a facing byte at all — `row_shape()` is
  `[7,7,3]`/`[7,7,4]`, sized for the view alone — yet `from_tensor` on both
  types hard-coded `agent_direction: Direction::North.to_u8()` on every
  decode: a value indistinguishable from a real measurement that was never
  one. `GoToDoorObservation` carried the identical, previously unfiled
  defect. Both now report `agent_direction: None` on decode (see the
  breaking-change entry above).

  *Why the existing tests missed it.* Each type's `view_round_trips_through_tensor`
  test asserted the lossiness directly —
  `assert_eq!(round_tripped.agent_direction, Direction::North.to_u8())` —
  so the test could not fail on the exact defect it sat on top of; it
  encoded the bug as the expected value. Nothing in the workspace decodes a
  grid observation outside tests or doc examples (`ExperienceTuple` stores
  observations by value, and the replay modules never call `from_tensor`),
  which is also why this was never corrupting a live training run.
- **Grid construction no longer panics or silently builds a corrupt board for an
  undersized config** (#106). The nine environments above split their invariant
  across two doors: `FromStr` checked the minimum, `Validate::validate` — the one
  the ADR 0026 chokepoint actually calls — checked only `nonzero`. Anything
  reaching `with_config` by struct literal, `..Default::default()`, or
  `Deserialize` therefore skipped the guard entirely and ran `build()` on a size
  the layout code cannot express. The failure was inconsistent, which is why it
  survived: `EmptyEnv`/`LavaGapEnv` at `size = 1` panicked with a `usize`
  underflow, `DoorKeyEnv`/`UnlockEnv`/`UnlockPickupEnv` panicked on an
  out-of-bounds `Grid::set`, and the rest built quietly broken boards —
  `UnlockPickupEnv` at `size = 3` produced a board with **no door at all** (the
  key overwrote it), `FourRoomsEnv` at `size = 7` punched a hole through all four
  perimeter walls, and `MultiRoomEnv` at `num_rooms = 1` returned a single-room
  environment. The floors now live in `Validate`, so all four construction paths
  share one guard, and `FourRoomsEnv` gained `const _` assertions pinning the
  derivation of its bound at compile time.

  The existing conformance test could not have caught this: every grid case in
  `tests/config_validation_chokepoint.rs` used a **zero** value, which trips the
  generic `nonzero` guard and so cannot distinguish a real minimum from an
  incidental one. Each environment now also has a non-zero below-floor case,
  which is the input that actually pins the constraint.
- **Nine grid environments held an RNG they never read, and re-seeded it on every
  `reset()`** (#282, closes #108; ADR 0062). `rg 'self\._rng\.'` over `grids/`
  returned zero hits: all nine stored an `_rng: StdRng`, rebuilt it from
  `config.seed` inside `reset()`, and never drew a value from it. Their
  `build(config)` functions took no RNG parameter at all, so they were
  structurally incapable of sampling — the `seed` field was inert across almost
  the whole family while the docs said otherwise. `crossing`, `door_key` and
  `four_rooms` promised that "using the same seed always produces the same
  episode layout" on a layout no seed could reach, and `empty`, `lava_gap`,
  `multi_room`, `unlock` and `unlock_pickup` held a `seed` "reserved for future
  stochastic variants" — a promise with no owner. Both wordings are gone; every
  `seed` doc now states what its environment does, naming the upstream id it
  departs from where it departs.

  *Why this survived two reviews.* ADR 0029 carved the family out on the
  reasoning that re-seeding an unread RNG is a no-op. That is true, and it was
  the wrong question — it classified nine environments as deterministic on the
  evidence that their code does not sample, without asking whether their upstream
  `_gen_grid` does. Reconciled env by env against Farama Minigrid, seven of the
  nine deviate. The reading behind the carve-out was a category error worth
  naming: the Minigrid paper's determinism claim is about the *transition
  function* — `step` has no slip probability, which `grids/core/dynamics.rs`
  reproduces correctly and this change does not touch — not about `reset`, whose
  shipped `_gen_grid` implementations sample.

  *Why the existing tests missed it.* Each of the five randomized environments had
  a `reset_is_deterministic` test that constructed two environments from one
  config and compared **one** `reset()` each. A fixed board satisfies that, and so
  does a correct persistent stream — the assertion cannot distinguish them, and it
  never looked at a *second* `reset()` on the same environment, which is where the
  defect lived. Those tests still pass unmodified and are now meaningful rather
  than tautological, joined by consecutive-reset and board-variability assertions
  that the old code would have failed.
- **The grid solvability oracles could pass on an episode the agent died in**
  (#282). `assert_solvable!` and `run_script` both ran the entire action list and
  asserted only the *final* snapshot's reward. At the time this was written, no
  environment in the grid family rejected a post-terminal `step` (ADR 0044
  recorded ~44 non-conformant environments workspace-wide) — the grid family's
  own `EpisodeGuard` sweep lands later in this section (above) — so an episode
  could end badly mid-script, keep stepping on a finished episode, and hand
  back a healthy last snapshot. This was observed, not
  hypothesized: a `DistShift` script that walked into lava at action 5 — `done`,
  reward `0.0` — and then strolled on to the goal for `0.874` passed the old
  helper. Both now stop at the first snapshot reporting `is_done()` and assert on
  that one, and treat any action left in the script after termination as a defect
  in the script or the planner rather than tolerating it.
- **`egocentric_view` applied no masking at all, so `see_through_walls` was
  effectively `true` crate-wide** (ADR 0063, resolves #281). Every cell of the
  rotated `7×7` window was read straight from the grid, walls included, the
  opposite of canonical Minigrid's own default. A new crate-private
  `grid::process_vis` — a direct port of `Grid.process_vis` — flood-fills from
  the agent's own cell outward and masks any cell an opaque cell (a wall, or a
  closed door) stands in front of; `mask_view` is the one place that dispatches
  on each environment's `Visibility`.

  *Why the existing tests missed it, and why this shipped safely anyway.*
  Commit 040e057 changed eight environments' observations on every single
  step and broke **zero** existing tests. `memory.rs`'s Invariant-M tests —
  the ones written to prove the cue is unreadable from the decision region —
  called `egocentric_view` directly rather than the observation an environment
  actually emits, so they were asserting a property of a raw projection no
  environment ever produces; nothing anywhere asserted that the *emitted*
  observation was occlusion-free, because nothing expected occlusion to exist.
- **Masked cells were briefly encoded identically to seen-empty cells**, which
  made the fix above invisible to any policy at exactly the pose it mattered
  most. Because `UNSEEN_TYPE == Entity::Empty.type_u8() == 0` under the byte
  table in place before this cycle's renumbering (see the Breaking changes
  entry above), a masked `Empty` cell and a confirmed-empty cell wrote the same
  bytes. Measured on `MemoryEnv`'s default board: of 9,053 masked cells across
  its occlusion sweep, 2,928 were `Entity::Empty`, and at the fork decision
  cell facing West — the pose at which the agent must actually answer — the
  occluded and unoccluded observations were **byte-identical**. Occlusion was
  running correctly and encoding it wrong made it disappear on the wire; the
  `type_u8` renumbering above is what makes it observable.
- **ADR 0043's predicted `MemoryEnv::MIN_SIZE` 11→7 relaxation was tested and
  refuted, not merely left undone.** ADR 0043 called the relaxation "a
  one-line change" once occlusion landed. An executed sweep over every
  decision-region cell and facing, at sizes 7, 9, 11, and 13, found the
  occluded and see-through violation sets **identical** at every size (5
  violating poses at 7 and 9, none at 11 or 13) — the shadow cast reaches the
  cue by routing around the corridor's walls through the open start room, so
  occlusion buys this environment nothing. `MIN_SIZE` is therefore unchanged at
  `11`, and canonical `MiniGrid-MemoryS7-v0`/`S9-v0` remain unreproducible in
  rlevo. Recorded explicitly because a reader of ADR 0043 alone would expect
  S7/S9 to have arrived in this cycle; they have not.

### `rlevo-reinforcement-learning`

**Breaking changes**

- **`PpoUpdateStats` gains a `max_log_std` field and becomes
  `#[non_exhaustive]`** (part of the #347 fix, below). The struct is a
  report-only type — the agents write it, callers read it — so the
  exhaustiveness break is taken now, while the type is already breaking for the
  new field, rather than paying for it again at the next diagnostic. This
  follows ADR 0060's `ConstraintKind` precedent.

  *Migration.* Read-only consumers that destructure exhaustively add `..` to the
  pattern. Code that **constructs** the struct with a literal — test fixtures and
  mocks, mostly — uses the new `PpoUpdateStats::default()` (all counters zero,
  both `log_std` fields `None`) plus field assignment. Nothing persisted changes:
  the type derives neither `Serialize` nor `Deserialize`, so no records need
  migrating. All four in-workspace construction sites are inside
  `rlevo-reinforcement-learning` itself, where `#[non_exhaustive]` does not
  apply, so there is no in-repo migration.

- **`tau` and `target_update_frequency` are replaced by a single
  `target_update: TargetUpdate` field on all six off-policy configs** (ADR 0058
  + ADR 0059, resolves #334, closes #455). The two fields did not describe two
  mechanisms — they described one, badly, and they did not agree on what they
  meant. `target_update_frequency` gated a **hard copy** in DQN/C51/QR-DQN but
  the **soft** Polyak update in SAC, so a τ carried from one family to the other
  silently produced a different training regime. Worse, and not what the issue
  reported: DDPG and TD3 had no such field at all but gated their Polyak update
  on `policy_frequency`, making the actor-delay knob an *undeclared alias* for
  target cadence — you could not change how often the actor updated without also
  changing how often the target moved. Three regimes, one field name, no error.

  The new type says it once: the cadence decides *when* an update fires, τ
  decides *how far* the target moves, and `τ = 1.0` is a hard copy by degeneracy
  rather than a separate mode. That formulation is not borrowed from
  Stable-Baselines3 — it is Haarnoja et al. 2018a's own "SAC (hard target
  update)" ablation (τ = 1, interval = 1000) and TD3's Algorithm 1, which gates
  the soft update on a period. What was rejected is SB3's *two-flat-fields
  shape*, which is the mechanism by which the three families drifted apart.

  ```rust
  // before
  .tau(0.005).target_update_frequency(500)
  // after
  .target_update(TargetUpdate::polyak(0.005, 1))   // soft, every gradient update
  .target_update(TargetUpdate::hard(10_000))       // full copy, τ = 1.0
  ```

  *Migration.* Replace the paired `.tau(..)` / `.target_update_frequency(..)`
  builder setters with one `.target_update(..)`, and the paired struct fields
  with `target_update`. Every existing call site becomes a **compile error**
  rather than silently changing meaning — no config in the workspace derives
  `Serialize`/`Deserialize`, so there is no persisted data to migrate and no
  record `FORMAT_VERSION` bump. **Defaults are bit-identical to the previous
  behaviour** (`polyak(0.005, 1)` for DQN/C51/QR-DQN/SAC, `polyak(0.005, 2)` for
  DDPG/TD3, matching the `policy_frequency` it decouples from), so a run left at
  defaults is unchanged. Do **not** transcribe an old `target_update_frequency`
  literally: under `tau > 0.0` it was inert, so carrying `10_000` across would
  collapse the Polyak cadence 10,000-fold. The default-*value* question is
  tracked separately by #337.

  This also makes a frozen target unrepresentable rather than merely rejected.
  `PolyakTau`'s invariant is the half-open `(0, 1]` and the cadence is a
  `NonZeroUsize`, so `τ = 0.0` — which passed `config::in_range`'s closed
  interval and froze the target network (#455) — no longer type-checks, in all
  six configs and including via struct-literal `..Default::default()`
  construction. Six `config::in_range("tau", ..)` checks, SAC's
  `config::at_least("target_update_frequency", .., 1)`, and three cross-field
  frozen-target guards are deleted as redundant (ADR 0027 §3).
- **`sync_target()` is removed from `DqnAgent`, `C51Agent` and `QrDqnAgent`**
  (ADR 0059). The target update now happens inside `learn_step`, so the three
  train loops no longer call it and a hand-written loop has no target-sync stage
  to forget. This comes with a unit change that is easy to miss: the cadence now
  counts **gradient (optimizer) updates**, not environment steps. Every
  canonical source counts it that way — Nature DQN's `C` is "measured in the
  number of parameter updates", SAC's interval sits inside "for each gradient
  step do", TD3's `d` is "updates to the critic" — and the env-step reading was
  the reason the shipped default was 4× more frequent than the Nature value it
  claimed to match. `learning_starts` and `train_frequency` still count
  environment steps, so a config now deliberately carries two units; the field
  rustdoc says which is which. The counter advances even when the ADR-0056
  non-finite-loss guard skips an optimizer step, so a diverging run cannot drift
  the cadence.

  *Why the old tests missed all of this.* Nothing in the workspace could read a
  target network — there was no accessor — so the three integration tests
  asserted only that rewards were finite and that two seeded runs agreed. A
  wrongly-scheduled target update is finite and perfectly deterministic, so both
  the correct and the defective regimes passed. The agents now expose a
  target-network observation seam, and the new tests assert the arithmetic
  directly: that a fired update moves each parameter to exactly
  `(1 − τ)·target + τ·live`, that no update fires between cadence boundaries,
  and — the configuration that was previously unexpressible — that an actor
  cadence of 1 and a target cadence of 2 are now independent.
- **The target soft-update path is now fallible** (ADR 0057, resolves #341,
  partially #317). A `ParamId`-topology mismatch between a network and its target
  is a recoverable configuration error — the target was built wrong — so it is
  now surfaced as a typed `Result` instead of a panic, matching ADR 0056's
  skip-don't-crash posture. Three signatures change:
  - `polyak_update` now returns `Result<M, PolyakError>` (was `M`).
  - the `soft_update` trait method on `DqnModel`, `C51Model`, `QrDqnModel`, and
    the DDPG actor (`DeterministicPolicy`) and critic (`ContinuousQ`) traits now
    returns `Result<Self::InnerModule, PolyakError>` (was `Self::InnerModule`);
    SAC and TD3 reuse the DDPG traits, so no new decls.
  - `learn_step` on all six off-policy agents (DQN, C51, QR-DQN, DDPG, TD3, SAC)
    now returns `Result<Option<LearnOutcome>, XAgentError>` (was
    `Option<LearnOutcome>`). Each agent error enum gains one
    `#[error(transparent)] Polyak(#[from] PolyakError)` variant. `Ok(None)` still
    means "step skipped" (warm-up or non-finite loss, ADR 0056); `Ok(Some(o))`
    means applied; `Err` means the update failed.

  Every in-tree target is built by cloning its active network, so the `Err`
  paths are unreachable in practice and the healthy-path behaviour is unchanged.
  `act()` and the on-policy PPO/PPG agents stay panic-based (residual under
  #317).

  *Migration.* Callers of `learn_step` add `?` — the training loops already
  return the agent error type, so they need only the `?`. Direct callers that
  cannot propagate double-unwrap: `agent.learn_step(rng).expect("no polyak
  error")` yields the `Option<LearnOutcome>` you handled before. `soft_update`
  and `polyak_update` implementors/callers likewise propagate or unwrap the
  `Result`. No persisted data or on-disk format is affected.

- **`DqnMetrics::value_loss` removed — it was an exact mirror of `policy_loss`**
  (resolves #415). DQN optimizes a single TD loss; unlike the actor-critic
  algorithms there is no separate policy/value pair to report. The field was
  populated with `last_loss`, the same value already assigned to `policy_loss`
  on the line above it, and its own doc-comment described it as a "mirror ...
  kept for parity with actor-critic algorithms". Dashboards consuming
  `DqnMetrics` therefore plotted one curve twice under two names.

  `QrDqnMetrics` never carried the field, so this removal brings the two
  Q-learning metric structs into agreement rather than splitting them.

  *Migration.* Read `policy_loss` instead — it holds the TD loss and always did.
  Delete any `value_loss:` initializer from struct literals; those call sites
  fail to compile. No persisted data or on-disk format is affected.

- **`PpoTrainingConfig::max_grad_norm` removed — it was dead state advertising a
  feature the crate does not have** (resolves #183). The field defaulted to
  `0.5`, was validated as positive, and had a public builder setter, but nothing
  in the workspace ever read it: `PpoAgent::update` and `PpgAgent` build their
  optimizers from `clip_grad` alone. A user reading `max_grad_norm: 0.5` in the
  default config reasonably concluded gradient clipping was on. It never was —
  `clip_grad` is the only functional knob and it defaults to `None`, so **stock
  PPO and PPG perform no gradient clipping whatsoever**.

  PPG inherited the defect verbatim through `PpgConfig::ppo`, and the deletion
  propagates to it automatically.

  Setting `clip_grad` is *not* an equivalent substitute for what the field
  claimed. Burn's `GradientClipping::Norm` rescales **each parameter tensor
  independently** (`burn-optim`'s `clip_by_norm` takes the L2 norm of one
  tensor; `SimpleOptimizerMapper` applies it per-parameter), whereas Huang et
  al. detail #10 clips the **global** norm across the whole flattened parameter
  vector. The per-tensor form neither bounds the global norm nor preserves the
  gradient's direction, so wiring the old field into the optimizer would have
  been a different algorithm wearing the documented name. True global-norm
  clipping needs a reduction over `GradientsParams` before `step_with` and is
  tracked separately in #328.

  *Migration.* Delete any `.max_grad_norm(..)` builder call or struct-literal
  field — the call sites fail to compile, which is the point, since they were
  silently no-ops. To opt into per-tensor clipping, set `clip_grad`, but do not
  record it as detail #10. No persisted data is affected.

  Four doc sites that asserted the missing behavior were corrected: the crate
  README's defaults table (which also miscited the detail as #11), both
  `max_grad_norm` and `clip_grad` doc-comments in `ppo_config.rs`, and the
  `ppo/README.md` implementation-details table, where #10 moves from
  "Implemented" to the documented-gaps list. The tests missed this because no
  test asserted that a configured clip actually changes a gradient — the field
  was only ever exercised through its own validator.

- **GAE read episode done-ness one step late, mis-timing the bootstrap cut for
  *every* PPO/PPG run** (resolves #170, part 1). `RolloutBuffer::push_step`
  stores `obs[t]` alongside the status of the transition *out of* `obs[t]`, so
  `terminated[t]` means "transition `t` ended the episode" — which is exactly
  what decides whether `values[t + 1]` belongs to the same episode.
  `compute_gae` instead consulted `terminated[t + 1] || truncated[t + 1]`, while
  its own final-step branch used `last_done` on the *correct* `[t]` convention.
  Two conventions in one loop; only `[t]` is right. Every episode boundary
  zeroed the bootstrap one step early and the true terminal step kept a
  bootstrap it should not have had.

  This is **not** confined to time-limited environments — it mis-weights
  genuinely *terminated* episodes too, so it affects every PPO and PPG user.

  `compute_gae` and `RolloutBuffer::finish` **lose their `last_done`
  parameter**. Once each step's status is read at `[t]`, the final step's
  done-ness is already recorded in the buffer; the parameter existed only to
  paper over `[t + 1]` running off the end, and was the precise site where the
  two conventions collided. `RolloutBuffer::last_step_ended()` replaces it.

  *Migration.* Drop the `last_done` argument from any `compute_gae` or
  `finish` call. **All seeded PPO/PPG results change — re-measure baselines
  rather than re-fitting thresholds to them.**

  The existing `gae_handles_terminated_mid_rollout` test asserted the wrong
  values and its comment recorded the author reasoning toward them ("Wait — the
  convention is…"), which is why the defect survived review; it has been
  rewritten from a fresh hand-computed expectation rather than adjusted.

- **Truncated steps now bootstrap `V(s_continuation)` instead of being treated
  as terminations** (ADR 0048, resolves #170, part 2). Per Pardo et al.,
  "Time Limits in Reinforcement Learning" (ICML 2018) Eq. 6, a time-limit
  cutoff ends the *trajectory*, not the *task*: the GAE delta must bootstrap
  from the value of the state the episode was cut at, while the λ-recursion is
  still cut at the boundary. These are two distinct masks, and the single
  `next_nonterminal` term could not express both — which is why the previous
  code could not be fixed by reworking the existing flags alone.

  This is a **deliberate divergence from CleanRL's default PPO**, which ORs
  `terminations` and `truncations` before the recursion; `rlevo` now follows
  Stable-Baselines3 and the source literature instead. Results on
  `TimeLimit`-wrapped environments are no longer directly comparable to
  CleanRL's. The prior behaviour was a documented, accepted tradeoff rather
  than an oversight — ADR 0048 records the reversal and its justification.

  `RolloutBuffer` replaces `truncated: Vec<bool>` with
  `truncation_value: Vec<Option<f32>>`, so "is truncated" and "has a bootstrap
  value" cannot disagree by construction — a parallel `Vec<f32>` would make an
  unset `0.0` indistinguishable from a legitimate zero bootstrap, reproducing
  the very bug being fixed. `push_step` correspondingly takes a new
  `StepEnd { Running, Terminated, Truncated { bootstrap_value } }` rather than a
  `(EpisodeStatus, Option<f32>)` pair, which would still admit `Truncated` with
  no value. `compute_gae`'s `truncated: &[bool]` becomes
  `truncation_value: &[Option<f32>]`.

  *Migration.* `PpoAgent::record_step` and `PpgAgent::record_step` gain a new
  `next_obs: &O` parameter, inserted before the trailing `status`. Pass the
  observation from the snapshot the
  environment just returned — **not** the observation from a subsequent
  `reset()`. The agent computes the continuation value itself, and only when
  the status is `Truncated`, so a hand-written loop cannot forget to: it never
  computes a value at all. Cost is one extra value forward per truncation and
  none per ordinary step.

- **`ExperienceTuple.is_done` renamed to `terminated`** (resolves #170, part 4).
  The field is the Bellman bootstrap mask, so it may only ever hold
  `Snapshot::is_terminated` — but it was named after `is_done`, and its one
  caller obligingly passed `is_done()`. Parts 1–3 corrected the semantics and
  the rustdoc; leaving the name behind would have left the module `CLAUDE.md`
  cites as *the* RL replay buffer telling the next reader two different things
  at once, with the name winning at the call site every time. A bootstrap mask
  that says "done" collects truncations, and every Q-value learned through it is
  biased toward the pessimistic assumption that time running out is the same as
  the task ending.

  `PrioritizedExperienceReplay::add` and `History::add` rename their
  corresponding `is_done` parameter to `terminated`, and `TrainingBatch.dones`
  becomes `TrainingBatch.terminated` — the sampled tensor is the same mask one
  hop downstream, and a bundle whose field still said "done" would hand the
  learning algorithms back the misreading the rest of this entry removes.

  *Migration.* Rename both fields in any `ExperienceTuple` or `TrainingBatch`
  struct literal or field read, and — this is the part that changes results,
  not just compilation — pass
  `snapshot.is_terminated()` rather than `snapshot.is_done()` at every `add`
  call site. The parameters are positional `bool`s, so a call site left
  unexamined still compiles and still trains, just wrongly.

  **No production agent used PER**, so the blast radius was latent rather than
  live: each of the six off-policy agents carries its own private `Transition` +
  `VecDeque`, all six corrected in part 3. The only caller was an integration
  test asserting batch tensor shapes, which never read the flag back. This is a
  trap disarmed before anyone wired PER into an agent, not a bug that was
  corrupting shipped training runs.

- **`TanhGaussianPolicyHeadConfig` gains required `log_std_min` / `log_std_max`
  fields** (resolves #173, ADR 0049). The bounds live on the policy-head config
  — the one actually consumed to build the head — rather than on
  `PpoTrainingConfig`, following the convention #185 names for SAC. The struct
  has no `Default` and every construction site uses a full struct literal, so
  there is no partial migration.

  *Migration.* Add the bounds to every `TanhGaussianPolicyHeadConfig` literal.
  (Later in this same release, #386 collapsed the two fields into a single
  `log_std: Bounds` — write `log_std: Bounds::new(-20.0, 2.0)`.) `validate()` now rejects an inverted
  interval, a `log_std_min` below `-35`, a span of `40` or more, and a
  `log_std_init` outside the bounds — all four are construction-time errors,
  not silent coercions. The floor and the span guard **different** failures and
  neither implies the other: the span bounds the ratio `σ_old/σ_new`, while the
  floor bounds `σ` itself, so `(-120, -100)` — ordered, spanning only `20` — is
  rejected because `exp(-110)` is exactly `0.0` in f32. `-35` is derived from
  `|z − μ|/σ ≤ sqrt(f32::MAX)`; it sits six orders of magnitude below the
  default `-20` and constrains no usable configuration. Note the two numerical
  checks jointly imply `log_std_max < 5`. **Persisted
  records still load**: the bounds are plain `f32` constants on the head, not
  `Param`s, so no saved weights are invalidated. Seeded results are unchanged
  at the default bounds, which never bind on a healthy run (verified: the
  Pendulum end-to-end run passes unchanged at avg −1167.78).

- **`SacTrainingConfig::log_std_min` / `log_std_max` removed — they were dead
  state that silently did nothing** (resolves #185). Both fields were public,
  defaulted to CleanRL's `-5.0` / `2.0`, were ordering-validated against each
  other, and had public builder setters — but **no runtime path ever read
  them**. `SacAgent`, `sac/train.rs`, and `SacModel` never touch them; the only
  reads were `validate()` and the config's own tests. A user who called
  `.log_std_min(-10.0)` got a config that accepted the value, reported it back
  on the struct, and trained with the bounds entirely unchanged.

  The clamp that actually runs has always come from
  `SquashedGaussianPolicyHeadConfig` (`sac_policy.rs`), which carries its own
  copy of the same two names — and that copy is untouched here. The duplication
  was the defect; the head is the surviving owner, matching the convention ADR
  0049 established for PPO's `TanhGaussianPolicyHeadConfig`. There was no seam
  to wire the training-config fields through in the first place:
  `SacAgent::new` takes an already-built `actor`, so the head's bounds are
  fixed before the training config is ever consulted.

  *Migration.* Set the bounds on the `SquashedGaussianPolicyHeadConfig` you use
  to build the actor passed to `SacAgent::new`, and drop any
  `.log_std_min(..)` / `.log_std_max(..)` calls on
  `SacTrainingConfigBuilder`. Because the removed setters never had an effect,
  this changes no training behaviour — code that compiled and trained before
  trains identically after. `SacTrainingConfig::validate()` no longer emits a
  `log_std_max` ordering error. `SquashedGaussianPolicyHeadConfig` carries the
  equivalent check — after #386 below, an inverted range is unrepresentable
  (`Bounds`) and a zero-width one is rejected (`config::distinct`), which
  together match the removed strict-`<` check exactly. When #185 landed that
  check was only reached if a caller
  invoked `.validate()` on the head config explicitly — neither `init()` nor
  `SacAgent::new` did. **That gap is closed later in this same release** by
  #386, which replaced `init()` with a validating `try_init()`; see the entry
  below. **Persisted configs are unaffected**: `SacTrainingConfig` derives only
  `Clone, Debug` and has no serde impl, so nothing on disk encodes these
  fields.

- **`PpoTrainingConfig::action_log_std_init` / `action_scale` removed — the same
  write-only defect, one algorithm over** (resolves #385). Both fields were
  public, defaulted (`0.0` / `1.0`), had public builder setters, and
  `action_scale` was even positivity-validated — but **no runtime path ever read
  them**. `PpoAgent`, `PpgAgent`, `ppo/train.rs`, and every policy head left
  them untouched; the only reads were `validate()` and the config's own
  round-trip test. Calling `.action_scale(2.0)` returned a config that accepted
  and reported the value while the actions reaching the env were scaled by
  whatever the *head* said. `PpgConfig` embeds `PpoTrainingConfig` verbatim, so
  it inherited both dead fields and is fixed by the same removal.

  The values that actually run have always come from
  `TanhGaussianPolicyHeadConfig` (`ppo/policies/gaussian.rs`), which carries its
  own `log_std_init` and `action_scale` — and that copy is untouched here. The
  duplication was the defect; the head is the surviving owner, which is what ADR
  0049 already established for PPO's `log σ` bounds. As with SAC, removal rather
  than delegation is the only viable fix: `PpoAgent::new` and `PpgAgent::new`
  both take an already-built `policy`, so the head's scale is fixed before the
  training config is ever consulted — there is no seam to wire through.

  *Migration.* Set `log_std_init` and `action_scale` on the
  `TanhGaussianPolicyHeadConfig` you use to build the policy passed to
  `PpoAgent::new` / `PpgAgent::new`, and drop any `.action_log_std_init(..)` /
  `.action_scale(..)` calls on `PpoTrainingConfigBuilder`. Because the removed
  setters never had an effect, this changes no training behaviour — code that
  compiled and trained before trains identically after.
  `PpoTrainingConfig::validate()` no longer emits an `action_scale` error; the
  head config carries the equivalent check. **The trap this closes is worth
  naming**: both in-repo call sites (`crates/rlevo/tests/ppo_integration.rs`,
  `crates/rlevo/benches/pendulum_rl.rs`) set the identical value on *both* the
  dead builder and the live head literal, so they were correct by accident. A
  maintainer retuning the Pendulum torque limit through the prominent, fluently
  named builder alone would have silently no-op'd while the head kept the old
  scale. **Persisted configs are unaffected**: `PpoTrainingConfig` derives only
  `Clone, Debug` and has no serde impl, so nothing on disk encodes these fields.

- **Policy-head configs now validate on the construction path: `init()` is
  replaced by `try_init()`, and the Gaussian heads' `log_std_min` /
  `log_std_max` pair becomes a single `log_std: Bounds`** (resolves #386,
  ADR 0026, ADR 0027, ADR 0049).

  **The defect.** All four policy-head configs implemented `Validate`, and the
  checks were correct — but **no production path ever called them**. `init()`
  did not validate, and the agent constructors validate only the *training*
  config, so every call to `validate()` on a head config in the entire
  workspace sat inside a `#[cfg(test)]` module. The bounds that feed the live
  `log σ` clamp were therefore unenforced: `validate()` was, in effect,
  documentation that happened to compile. #185 and #385 above removed the dead
  *duplicates* of these fields; this entry closes the gap on the *surviving*
  copy, the one that actually feeds the clamp.

  **The failure was backend-divergent**, which is why it merits a breaking
  change rather than a doc note. Building a head with inverted bounds
  (`log_std_min: 5.0, log_std_max: -5.0`) reaches `Tensor::clamp`, and the two
  backends disagree about what that means. On the actor path (`Autodiff<Flex>`)
  the default `float_clamp` is `clamp_min(clamp_max(x, max), min)`, which with
  an inverted range pins **every** `log σ` to the constant `log_std_min` — a
  deterministic, gradient-dead collapse with no NaN, no panic, and no signal.
  On the target/critic path (raw `Flex`) `float_clamp` delegates to
  `core::f32::clamp`, which asserts `min <= max` and **panics**. Same config,
  same op: silent corruption on one backend, a crash on the other.

  *Migration, part 1 — `init` → `try_init`.* Every head config's `init()` is
  **removed** and replaced by
  `try_init<B>(&device) -> Result<Head<B>, ConfigError>`, whose first statement
  is `self.validate()?`. `init()` is not kept alongside it: an unvalidated
  constructor would simply reinstate the bypass. Applies uniformly to all four
  heads — `SquashedGaussianPolicyHeadConfig`, `TanhGaussianPolicyHeadConfig`,
  and both `CategoricalPolicyHeadConfig`s. The categorical heads have no
  `log_std`, but they share the *structural* bypass: `LinearConfig::new(0,
  hidden)` builds a zero-width layer without complaint, so a zero `obs_dim`
  produced a silently degenerate head. A "Gaussian heads validate, categorical
  ones don't" carve-out is a convention nobody retains. Replace
  `cfg.init::<B>(&device)` with `cfg.try_init::<B>(&device)?` where a `Result`
  is available, or `.expect("valid head config")` in benches, examples, and
  other non-`Result` contexts. The `try_` prefix is deliberate: Burn's own
  `*Config::init` methods are all infallible, and this crate does not
  contradict that idiom under its own name.

  *Migration, part 2 — `log_std: Bounds`.* On the two Gaussian head configs the
  `log_std_min: f32` / `log_std_max: f32` **pair is replaced by a single
  `log_std: rlevo_core::bounds::Bounds`** field. Write
  `log_std: Bounds::new(-20.0, 2.0)` (PPO) or `Bounds::new(-5.0, 2.0)` (SAC)
  in place of the two scalars. `Bounds` exists precisely because `f32::clamp`
  panics when `min > max`, so this makes the inverted case **unrepresentable**
  rather than merely rejected — the invariant travels with the value instead of
  being re-checked at each boundary (ADR 0027).

  **`Bounds` does not replace `validate()`.** It subsumes exactly one check —
  ordering — which is why `config::ordered` is gone from both Gaussian
  `validate()` impls. It subsumes only *half* of it: `config::ordered` was
  strict (`lo < hi`) and so also rejected a zero-width range, whereas
  `Bounds::try_new` deliberately permits `lo == hi`. **Both** Gaussian head
  configs therefore carry an explicit
  `config::nondegenerate_bounds(C, "log_std", self.log_std)` check (the named
  helper introduced in the `rlevo-core` section above) to preserve the old
  semantics; a degenerate range reports `field: "log_std"` with
  `ConstraintKind::DegenerateInterval`. The
  consequence differs by algorithm — on PPO a zero-width range freezes the
  shared `log_std` parameter and its gradient from step 0 with no path back,
  while on SAC it pins the per-observation `σ` to a constant and flattens the
  entropy term the temperature is tuned against — but in both cases it is a
  silent collapse, not a usable setting.

  Every other invariant remains and is now, for the first
  time, actually reached: ADR 0049's absolute floor (`log_std_min >= -35`) and
  span (`log_std_max - log_std_min < 40`, now expressed via `Bounds::span()`),
  the `log_std_init`-within-bounds range check, `action_scale > 0`, and the
  non-zero dimension checks. The floor and span are *not* expressible as an
  ordering: ADR 0049's own counterexample `(-120, -100)` is a perfectly
  well-ordered range that still reaches `NaN`, because `exp(-110)` is exactly
  `0.0` in f32.

  Two `ConfigError::field` values change on `TanhGaussianPolicyHeadConfig`: the
  floor and span violations now report `"log_std"` rather than `"log_std_min"`
  / `"log_std_max"`, since there is one field where there were two.

  **Behaviour is unchanged for every valid configuration.** No construction
  site in the repo used inverted bounds, so nothing that trained before trains
  differently now — the change converts a reachable-but-unreached check into an
  enforced one. **Persisted records still load**: the bounds remain plain `f32`
  constants on the built head (the clamp site takes two scalars, and keeping
  them as `f32` leaves `#[derive(Module)]`'s plain-data classification and the
  module record untouched), so no saved weights are invalidated.

- **The `memory` module — `PrioritizedExperienceReplay`, its builder, and
  `TrainingBatch` — is removed outright, with no deprecation shim** (ADR 0050,
  resolves #188). The docs advertised the module as the sanctioned
  replay-integration path, but nothing in the workspace constructed it beyond
  one shape-assertion test, and it carried four independent defects: no
  `update_priorities`, so priorities were insert-time constants never fed back
  from TD error — alpha-weighted sampling over static values, not PER; an
  internal `rand::rng()` no seed could reach; a one-hot `Float` action tensor
  that cannot express DQN's `Int` gather index; and without-replacement
  sampling where every agent draws with replacement. A `#[deprecated]` shim
  would assert "this works, prefer the new thing" — it did not work. The gap
  survived because the only usage example was an ```` ```ignore ```` doctest
  that nothing ever compiled; its replacement on `PrioritizedReplaySettings`
  is a real, running doctest.

  *Migration.* The `replay` module is the integration path: `UniformReplay`
  is what every agent already does by default, and prioritization is enabled
  per agent via the DQN/C51/QR-DQN config builders'
  `prioritized_replay(PrioritizedReplaySettings)`. `crate::memory::ReplayBufferError`
  is gone with the module — import it from `crate::replay` instead. No
  persisted data is affected; the removed types never serialized anything.

- **`buffer_capacity` is renamed `replay_buffer_capacity` on the DDPG, TD3,
  and SAC configs and their builder setters** (ADR 0050). The discrete three
  already spelled the same knob `replay_buffer_capacity`; six agents feeding
  one replay seam should not name it two ways. *Migration.* Rename the field
  or setter call — capacity semantics are unchanged.

- **`categorical_cross_entropy` and `quantile_huber_loss` are renamed
  `categorical_cross_entropy_per_sample` and `quantile_huber_loss_per_sample`
  and return the unreduced `[batch]` loss** (ADR 0050). Callers reduce with
  `.mean()` — or with an importance-weighted mean, which is the point: a
  per-sample loss is what an IS weight can scale. The rename is load-bearing,
  not cosmetic. The signature is `Tensor<B, 1>` before and after, so a stale
  caller would compile clean and silently backpropagate a different gradient;
  the new name turns that into a compile error. *Migration.* Append `.mean()`
  to restore the previous value bit-for-bit.

**Added**

- **`dropped_transitions()` on all six off-policy agents** (ADR 0065, part of
  the #352 fix above). Returns how many transitions `remember` discarded for
  carrying a non-finite reward. Public rather than test-only because
  `remember` is public API driven directly from outside the crate (the
  cross-crate integration tests and the benches all call it), so a caller
  hand-driving the agent would otherwise have data silently dropped with no
  programmatic way to detect it. A non-zero count means those environment
  steps never entered the buffer.

- **A replay-strategy seam — `replay::ReplayStrategy<T>` — with uniform and
  prioritized implementations, and opt-in prioritized replay for the
  value-based agents** (ADR 0050, ADR 0051, resolves #188). `UniformReplay`
  absorbs the six agents' hand-rolled `VecDeque` buffers bit-identically —
  the guarantee is a pinned contract test asserting the sampler leaves the
  RNG in the same state as the verbatim pre-seam expression, so seeded
  baselines did not move. `PrioritizedReplay` is a paper-faithful rebuild of
  Schaul et al. 2016's proportional variant: sum-tree storage, stratified
  one-draw-per-equal-mass-segment sampling (not k i.i.d. draws — the paper
  presents the balancing as deliberate variance reduction), running-max
  insert priority, IS weights max-normalized over the sampled minibatch, and
  `Priority`/`ImportanceExponent` newtypes that make the old
  NaN-into-`powf` path unrepresentable.

  Enable it per agent with the DQN/C51/QR-DQN builders'
  `.prioritized_replay(PrioritizedReplaySettings::default())` — defaults
  `priority_exponent 0.6`, β annealing `0.4 → 1.0` (Schaul Table 3,
  proportional). Two fidelity notes are encoded in code and rustdoc rather
  than left to convention: C51 prioritizes by the **KL divergence** — what
  the algorithm minimizes, per Rainbow — not by its cross-entropy loss (they
  differ by the target entropy, which is theta-constant but varies per
  sample, so it changes replay ranking; a test pins a case where CE and KL
  rank two transitions in opposite order). QR-DQN's quantile-Huber priority
  is an **uncited extrapolation** of Rainbow's principle — Dabney et al.
  explicitly evaluated QR-DQN without prioritization — and its rustdoc says
  so instead of inventing a citation. DDPG/TD3/SAC deliberately keep uniform
  replay: Panahi et al. (RLJ 2024) find no prioritized variant consistently
  beats uniform in control, and Saglam et al. (JAIR 2022) give the
  actor-gradient mechanism, so prioritization there would be a fidelity
  defect, not a feature.

- **`algorithms::c51::projection::atom_spacing`** — the single source of truth
  for the atom spacing `Δz = (v_max − v_min) / (N − 1)` (Bellemare et al. 2017,
  §4.1). `C51TrainingConfig::delta_z()` now delegates to it, so the support
  tensor built in `C51Agent` and the index scale used by the projection can no
  longer drift apart. Exposed as a free function taking scalars rather than a
  `C51TrainingConfig` method, so a future Rainbow agent can call
  `project_distribution` without depending on C51's config type.

**Changed**

- **`C51TrainingConfig::delta_z()` returns `f32::NAN` for `num_atoms < 2`,
  where it previously returned `±inf`.** The old body divided by
  `num_atoms.saturating_sub(1)`, i.e. by zero. Both values are degenerate and
  the builder's `validate()` rejects `num_atoms < 2` before either can be
  observed, but `NaN` propagates visibly through downstream arithmetic whereas
  `inf` silently yields a `b` coordinate of `0`. Reachable only by constructing
  the config through a struct literal, which bypasses `validate()` — the
  general problem tracked as #326. The signature and `#[must_use]` are
  unchanged.

- **`target_update_frequency` default tuning — superseded within this same
  release.** An interim change here raised the DQN/C51/QR-DQN default from
  `100` to `10_000`, on the reasoning that it should match Stable-Baselines3's
  `target_update_interval` (measured in environment steps, ≈4× more frequent
  than Nature DQN's `C = 10,000` parameter-update figure once `train_frequency:
  4` is accounted for). That field no longer exists: the ADR 0058/0059
  `tau`/`target_update_frequency` → `TargetUpdate` unification (above)
  replaces it and the shipped default for DQN/C51/QR-DQN is
  `TargetUpdate::polyak(0.005, 1)` — bit-identical to the *pre-this-cycle*
  Polyak behavior, not the `10_000`-step hard-sync value this entry describes.
  See that entry for the actual migration. The default-*value* question
  (Atari- vs classic-control-scaled cadence) remains open, tracked by #337.

- **`polyak_update` now keeps every soft update on-device instead of
  round-tripping each parameter through host memory** (resolves #322). The
  collector materialised every active parameter with `param.val().to_data()` — a
  blocking device→host readback — and the mapper rebuilt each one with
  `Tensor::from_data(.., &device)`, an upload straight back to the same device.
  Both networks always share one backend and device, so the entire host
  round-trip was gratuitous. It now stores the rank-erased on-device
  `TensorPrimitive<B>` handle (`into_primitive`) and rewraps it with
  `from_primitive`, so the value never leaves the device. Every off-policy agent
  that maintains a target network (DQN, C51, QR-DQN, DDPG, TD3, SAC) soft-updates
  on its configured cadence, so this fired on essentially every training step —
  twice per step for the twin-critic agents (SAC, TD3). The update is
  numerically identical (the blend arithmetic is unchanged). The mechanism is the
  removal of a per-parameter host round trip — a blocking device→host readback and
  matching upload — on every soft update; on any future accelerator backend it also
  removes a per-parameter GPU sync stall. A standalone benchmark on the `Flex` CPU
  backend measured ~1.4–1.9× on the soft-update step (see issue #322). The public
  signature and `PolyakError` are unchanged.

**Fixed**

- **A non-finite observation entered the replay buffer and was fed to the
  network, and on the CPU backend nothing downstream could ever detect it**
  (ADR 0067, resolves #1043). ADR 0065 closed reward finiteness at `remember`;
  the observations passing through the same call, in the same struct literal,
  were still unvalidated across all six off-policy agents. `remember` now drops
  and counts a transition whose `obs` or `next_obs` is non-finite, exposing
  `dropped_observations()`, and the action-selection path reports and counts
  through `degenerate_action_selections()` without substituting an action.

  The reason existing tests could not have caught this is the whole point of
  the fix. The issue predicted a NaN observation would produce a NaN action.
  Measured, that only happens on wgpu/Metal. On `flex` — the backend CI runs —
  `relu` *rescues* NaN to `0.0`, so a fully non-finite observation yields a
  finite, in-bounds, `is_valid() == true` action from a bias-only Q row; the
  one-NaN and all-NaN rows come out bit-identical, because the first ReLU
  erased the observation entirely. There is no NaN left anywhere downstream,
  so ADR 0056's `FiniteLossGuard` cannot fire, a Q-value check cannot fire, and
  an action check cannot fire. Only a check on the observation itself, before
  it becomes a tensor, can observe this at all. For discrete agents there is a
  second silent path: `argmax` over a *partly* NaN row returns the index of the
  first NaN rather than the finite maximum on flex, while wgpu returns the
  correct index — so the same input is silently wrong on CPU and correct on GPU.

  The guard runs at `remember` and not at batch staging. Issue #1043 assumed
  the check could ride the existing `write_host_row` traversal at near-zero
  cost; benchmarking refuted that, because `remember` stores the typed
  observation and never flattens, so there is no traversal there to ride.

  `act` deliberately does **not** substitute a fallback action. A plausible
  in-domain substitute is the same failure this entry describes — a
  legal-looking action that makes a broken run appear healthy — so the guard's
  value there is attribution, not correction.
- **The C51 projection produced a clean, plausible, and wrong target
  distribution on a GPU backend where the host backend failed loudly, because
  `Tensor::clamp`'s NaN behaviour differs between them** (ADR 0066, resolves
  #1044). `burn-flex` clamps with `f32::clamp` and propagates a NaN;
  `burn-cubecl`/wgpu lowers to the WGSL `clamp` builtin, which on Metal
  *rescues* a NaN to the lower bound. Neither behaviour is documented —
  `Ordered::clamp` says nothing about NaN — and `project_distribution` was
  relying on the host one. Measured on an Apple M2 Pro: a NaN reward yields a
  row of `[NaN, 0, …]` summing to `NaN` on Flex, which ADR 0056's
  `FiniteLossGuard` catches, but `[1.0, 0, …]` summing to **exactly 1.0** on
  Metal — a well-formed probability vector asserting certainty of the worst
  representable return, containing no NaN for any guard to fire on. Both
  clamps now go through a shared `clamp_preserving_nan`, so the failure is
  loud on every backend.

  The issue's own hypothesis was **refuted**, and the refutation is the
  interesting part. It predicted that a NaN index would reach `scatter` and
  panic on GPU. It does not — the clamps hold on both backends. But once the
  NaN is *preserved* through them, it does: on Metal `NaN.floor().int()` is
  `i32::MIN`, not the host's saturating `0`. So the fix creates the very
  out-of-range index the issue feared, and the derived `Int` indices are now
  clamped to `[0, num_atoms-1]` as well — the two halves are only correct
  together. That guard matters more than a panic would suggest: `burn-cubecl`'s
  scatter kernel is `launch_unchecked` and `cubecl-wgpu` sets
  `bounds_checks: false`, so an out-of-range index does not panic on wgpu, it
  writes into whatever tensor occupies that address. Deleting just the index
  clamp was measured to do exactly that.

  Nothing caught this because nothing could: every CI workflow is
  `ubuntu-latest` with no GPU, and the workspace's only cross-backend test is
  `#[ignore]`d for that reason. The obvious postcondition would not have
  helped either — the corrupted row sums to exactly 1.0, so a row-sum check
  passes it. The new tests assert *finiteness*, not sums, and the added
  cross-backend parity test is likewise `#[ignore]`d: regression protection
  for this class is a manual GPU run, not CI.

  Exposure is narrow and was already narrowing: ADR 0065's `FiniteRewardGuard`
  closed the reward ingress at `remember`, and `tz` is structurally derived
  only from the reward, terminal mask, and fixed support — never from network
  output — so a NaN *observation* cannot reach it. What remains is a direct
  `Transition` push bypassing `remember`, a future offline-RL loader, and
  direct callers of the `pub fn` such as the bench. ADR 0066 also corrects the
  record in ADR 0065 §Context and ADR 0056 §Out-of-scope, both of which assert
  the host clamp's NaN semantics as universal fact; both ADRs are immutable, so
  the correction lives in 0066.

- **A non-finite reward from the environment was stored in the replay buffer
  unchecked, where it silently cost every future minibatch that resampled it —
  it is now dropped at ingestion, counted, and warned on an escalating
  schedule** (ADR 0065, resolves #352). All six off-policy agents' `remember`
  pushed the caller's `f32` straight into the buffer with no finiteness
  contract anywhere between the environment and the Bellman target. The
  defect's *shape* is not what the issue reported: ADR 0056's `FiniteLossGuard`
  already stops a NaN reward from reaching `backward()`, so weights and the
  target network are **not** corrupted. What remained is quieter and lasts
  longer — the poisoned transition sits in the FIFO buffer until capacity
  eviction, and every minibatch that resamples it produces a non-finite loss
  whose update 0056 skips. Because that guard's `warn!` is a one-shot latch,
  only the *first* such skip is ever logged, so the run bleeds training steps
  while reporting nothing, and the reward that caused it is never surfaced at
  all.

  Nothing caught this because no test has ever asserted anything about replay
  buffer *contents*. The cross-crate suite asserts `*_produces_finite_rewards`
  — the *environment's* output, which is the input side of this very boundary —
  and the reproducibility tests assert same-seed self-consistency, which a
  deterministic NaN satisfies perfectly.

  The issue named four agents; there are **six**. C51 (`c51_agent.rs`) and
  QR-DQN (`qrdqn_agent.rs`) were added later by copying an unguarded
  `remember` and had the identical hole, which is why the guard is one shared
  `FiniteRewardGuard` with a test in each of the six files rather than six
  inline checks. A non-finite reward's transition is not pushed; the drop
  fires on **every** occurrence and is never latched, while the warning
  escalates at 1, 10, 100, … drops carrying the running total — a dropped
  transition is unbounded data loss, so unlike 0056's self-limiting skip the
  operator needs the magnitude, not just the fact.

  Deliberately unchanged, so they are not later "fixed": `remember`'s
  signature (fallibility is #317's), `ScalarReward::new` (a core check cannot
  close the hole — `Reward` is a trait and an environment may ship a type that
  never touches `ScalarReward`), and the episode-return accumulator, which
  still adds the NaN because an episode return is the primary scientific
  measurement and omitting a step from it would report a return the agent
  never earned.

- **PPO's `log_std` clamp warning latched once per *head*, so a second bound
  crossing on another action dimension was silent — it now latches once per
  *bound*, and a ceiling-pinned dimension is finally visible in the metrics**
  (resolves #347). ADR 0049 §4 shipped telemetry alongside the clamp precisely
  because the clamp trades a loud failure (`NaN`) for a quiet one (a
  state-independent `log_std` pinned at a bound, its gradient permanently zeroed,
  with no path back). Three independent defects each partly reinstated that quiet
  failure. Verified by execution with a counting `tracing::Subscriber`, not by
  reading the source.

  The `swap` on a single `Arc<AtomicBool>` meant the latch was spent by whichever
  dimension crossed whichever bound first. A 2-dim head driven through four
  crossings — dim0 below the floor, dim1 above the ceiling, dim1 above again,
  then both bounds at once — emitted **exactly one** warning; the other three were
  silent. Second, and not what the issue reported: the `if below { … } else { … }`
  arm meant that even on a *fresh* latch, a call violating both bounds at once
  named only the floor. A σ of `exp(9) ≈ 8103` was dropped from the very first
  warning, so the defect never needed a prior crossing to lose information.
  Third, `min_log_std` is a *minimum* and is therefore structurally blind to
  ceiling drift: while dim1 sat pinned at `log_std_max`, the metric reported a
  healthy-looking `0.0`. Both telemetry channels were blind to the same
  dimension simultaneously — which is exactly the state ADR 0049 argues must not
  be reachable.

  The latch is now a two-field `ClampWarnLatch { below, above }`, one `swap` per
  bound, so the floor and the ceiling warn independently and each still warns at
  most once. Per-`(dim, bound)` latching — what the issue proposed — was
  rejected: the clamp's zeroed gradient is a one-way door per dimension, so a
  floor-pinned dim can never later reach the ceiling and half those latches are
  unreachable by construction; worse, a 17-DoF head collapsing across all dims
  would emit 17 copies of a six-line warning, and log volume that large is itself
  a signal-loss mechanism. The offending dimension indices are emitted as a
  structured `dims` field on the single per-bound event instead, which is
  strictly more information at none of the cost. The two messages are now
  distinct prose — σ collapsing and σ diverging are different pathologies with
  different remedies — and the floor message no longer asserts that the whole run
  is dead, which overclaimed for any head with more than one action dimension.

  `PpoUpdateStats` gains `max_log_std` (see Breaking changes) and `PpoPolicy` a
  defaulted `max_log_std()`; the value costs no extra device→host traffic,
  because the existing read already computed the maximum and discarded it.

  **Why no existing test caught any of this**, and it is two failures, not one.
  Every `log_std` telemetry test drove a single dimension, or drove several to
  the *same* bound — none drove two dimensions to *different* bounds, which is
  the only configuration in which any of the three defects is observable. More
  importantly, the tests asserted on a `#[cfg(test)]` accessor for the latch
  *flag*, never on the emitted event, and two opaque booleans cannot distinguish
  two atomics from one atomic exposed under two names. A merged latch therefore
  survived the obvious regression test. The tests now install a hand-rolled
  `tracing::Subscriber` and assert on the events themselves — their count, their
  `bound`, and the `dims` payload — which needs no new dependency, since
  `tracing` is already a direct one. Each was confirmed to fail against a
  deliberately reverted implementation rather than assumed to cover it.

- **PPO and PPG training loops now emit `min_log_std` / `max_log_std` in their
  periodic progress events** (found while fixing #347). ADR 0049 §4 named two
  telemetry channels, a one-shot warning and a per-update metric, the metric's
  stated purpose being to make drift visible *before* the bound pins. The metric
  was written to `PpoUpdateStats` on every update and then read by nothing:
  `emit_progress` enumerated eighteen fields in each of the two loops and omitted
  this one. So the second channel existed on the struct but was unreachable to
  anyone who did not call `agent.update()` by hand and inspect the return value —
  the drift-warning half of ADR 0049 §4 was, as shipped, not delivered.

- **QR-DQN's Huber threshold κ is now required to be finite and strictly
  positive, closing two NaN paths that `validate()` waved through** (resolves
  #345). `QrDqnTrainingConfig::validate` checked κ with
  `config::in_range("kappa", 0.0, f64::INFINITY, …)`, and `config::in_range` is
  inclusive at *both* ends — so the one value that must never reach the loss was
  the one the check explicitly admitted. κ is the divisor of Dabney et al.
  (2018) Eq. (10), `ρ^κ_τ(u) = |τ − 𝟙{u<0}| · L_κ(u) / κ`, which
  `quantile_loss.rs` implements literally as `(weight * huber_u).div_scalar(κ)`.
  Correcting the issue as filed: κ = 0 does **not** produce `inf`. `huber()`
  returns exactly `0.0` at κ = 0 (the `|u| ≤ 0` mask is false for nonzero `u`,
  leaving the linear branch `(|u| − 0)·0`), so the division is `0/0` and the
  loss is `NaN` — verified by execution, not by reading. The inclusive *upper*
  bound was the second hole and went unreported: the Huber mask selects the
  quadratic branch everywhere, but the masked-out linear branch `(|u| − 0.5κ)·κ`
  is still evaluated eagerly, so once it overflows f32 the blend computes
  `0 · (−inf)` = `NaN` in **every** element, including at `u = 0`. That happens
  at κ = `+∞` and also at any κ large enough that `0.5·κ²` exceeds `f32::MAX`,
  so an `is_finite` check alone is not sufficient. `NaN` itself was never
  reachable — `NaN >= lo` is false, so `in_range` already rejected it.
  Also correcting the severity this issue was filed under: an invalid κ does
  **not** corrupt weights. `FiniteLossGuard` (ADR 0056, #318) checks the loss
  scalar in `qrdqn_agent.rs` before `backward()`, skips the optimizer step, and
  fires a one-shot `tracing::warn!` — so the actual pre-fix consequence was a
  QR-DQN run that *silently stalled*, every update a no-op while the step
  counter advanced, discoverable only from a single warn line. That is a milder
  and quite different failure from the silent poisoning the issue describes, and
  it is what makes the config-boundary rejection the right fix: fail fast with a
  named field error instead of leaning on a downstream generic guard and a
  stalled run. Validation now uses `config::positive` plus an explicit
  finiteness-and-overflow guard, and the fix sits at the config boundary rather
  than inside the loss: both
  `QrDqnTrainingConfigBuilder::build` and `QrDqnAgent::new` call `validate()`,
  so a hand-constructed config with `pub` fields cannot route around it. No
  behavior changes for any valid κ, and no call site in the repo moved — every
  one already passed `1.0`. The existing tests missed this because the config
  suite covered `num_quantiles == 0` and the τ/cadence cross-field case but had
  no κ boundary test at all; the new `rejects_*_kappa` tests fence the whole
  invalid domain, and were confirmed to fail against the pre-fix validator
  before the fix landed. The root cause of the upper-bound half lives in
  `huber()`'s eagerly-evaluated masked-out branch, not in κ — that pattern is
  filed separately, since it is reachable through the public
  `quantile_huber_loss_per_sample` regardless of what the config accepts. Note
  for anyone expecting the paper's κ = 0 variant: QR-DQN-0 is *not* Eq. (10)
  evaluated at zero. The paper's "as κ → 0 the quantile Huber loss reverts to
  the quantile regression loss" is a limit statement, and its κ = 0 experiments
  substitute the separate unsmoothed Eq. (8), `ρ_τ(u) = u(τ − 𝟙{u<0})`, which
  this crate does not implement — so `kappa = 0.0` was never a way to select it.
  The field docs and the crate README now say so.

- **PPG's auxiliary phase no longer anneals one iteration ahead of the policy
  phase it accompanies** (resolves #324). `maybe_aux_phase` read
  `current_learning_rate()`, but `policy_phase_update` had already incremented
  the iteration counter on its way out — so every auxiliary phase stepped at the
  *next* tick's rate, and whenever `total_iterations % n_iteration == 0` the
  final one landed on `iteration == total_iterations`, where the linear anneal
  is exactly `0.0`. That closing phase ran its full complement of forward and
  backward passes and moved no parameter at all: wasted compute, plus a
  `last_aux_phase().policy_kl` of bit-exactly `0x00000000` that reads as a
  broken auxiliary phase to anyone instrumenting it (it is what raised the false
  alarm in #319). The rate the policy phase applies is now snapshotted before
  the increment and reused by the auxiliary phase, matching CleanRL's
  `ppg_procgen.py` — whose `# AUXILIARY PHASE` block is nested inside the phase
  body, shares the policy optimizer, and never rewrites its `lr` — and Algorithm
  1 of Cobbe et al. (2021), which places the auxiliary phase inside the phase
  rather than after a schedule tick. `current_learning_rate()` is unchanged and
  still reports the rate the *next* policy phase will use; its doc now says so.
  The existing tests missed this because the PPG suite asserts that the
  auxiliary phase *fires* and that its metrics are *finite* — and zero is
  finite — while the annealing tests live in `ppo_config` and only check the
  arithmetic, never the phase ordering that consumes it. `ppg_aux_phase_actually_runs`
  demonstrably passed against the bug it was positioned to guard; it and the new
  regression test now assert `policy_kl > 0.0`, guarded by an explicit
  `minibatches > 1` precondition (the first minibatch of the first auxiliary
  epoch forward-passes the same weights `π_old` was snapshotted from, so its KL
  term is structurally zero and a single-minibatch phase would report `0.0` at a
  perfectly healthy rate). Training impact is one phase per run and small in
  absolute terms; the diagnostic trap was the real cost.
- **`polyak_update` no longer mis-updates target networks with tied or subset
  parameter topologies** (ADR 0057, resolves #341, partially #317). The mapper
  pairs `active` and `target` parameters by [`ParamId`], and its bookkeeping was
  wrong for two topologies and uninformative on a mismatch — a real *tied-weights*
  module (two fields holding clones of one `Param`, sharing one id) hit every
  defect at once. First, it consumed each active entry with `.remove()`, so tied
  weights panicked when the second field's lookup found the id already gone;
  lookup now uses `.get().cloned()`, letting one active entry blend into every
  field that references it. Second, a `target` parameter absent from `active`
  panicked; it is now the typed `PolyakError::MissingActive(ParamId)`, naming the
  offending id and explaining the independent-init cause. Third, a `target` that
  was a **strict subset** of `active` (some active parameters had no counterpart)
  silently applied a partial update; unconsumed active parameters are tracked in a
  `seen` set and the smallest leftover is now the typed
  `PolyakError::MissingTarget(ParamId)` (deterministic via `ParamId: Ord`) instead
  of vanishing without a signal. The existing polyak tests missed all three
  because they only exercised the blend arithmetic (`tau = 0`/`1`/fractional) over
  a single same-`ParamId` two-field fixture — the id-topology paths were never
  entered, so no value assertion could observe the defect, which lived entirely in
  how *mismatched* modules were handled. Unlike the interim #341 landing, the
  detection is now reported as a recoverable `Result` rather than a panic: see the
  **Breaking changes** entry above for `polyak_update` / `soft_update` /
  `learn_step`. `act()` and the on-policy agents keep the panic shape, so #317 is
  only partially addressed here.
- **A non-finite loss no longer silently poisons the weights** (ADR 0056,
  resolves #318). Burn does not panic on `NaN`/`±Inf` — it propagates it. A loss
  that went non-finite (a PPO `ratio = exp(new − old)` overflow when
  `new − old > ~88`, a degenerate `log`/`div` in an entropy or log-prob term, an
  exploding gradient) was fed straight into `backward()` and the optimizer step,
  corrupting every weight while training continued and reported finite-looking
  bookkeeping. Every agent now runs its already-host-resident loss scalar through
  a `FiniteLossGuard` *before* `backward()`; on a non-finite value it **skips the
  backward pass and the optimizer step** (the skip re-fires every occurrence, so
  a persistently-diverging run is protected every step, not just once) and emits a
  **one-shot** `tracing::warn!` naming the site and the likely cause. The check
  rides the read every agent already did for metrics, so it adds no device→host
  sync and runs unconditionally in release. Skipped values are excluded from the
  reported loss means, so a single `NaN` can no longer masquerade as a finite
  average. Covers all eight agents (PPO, PPG, DQN, C51, QR-DQN, SAC, DDPG, TD3),
  generalizing the SAC-α guard from #184. The existing tests missed this because
  the cross-crate suite asserts *reward* finiteness (a fully `NaN`-poisoned
  network still emits finite rewards) and the reproducibility suite only checks
  same-seed self-consistency (a deterministic `NaN` reproduces perfectly). This
  is a loss-level guard: it fully prevents *loss-origin* `NaN` but only surfaces
  (does not recover from) the rarer *finite-loss → `NaN`-gradient* case, which is
  tracked separately under #328; reward-ingress finiteness (#352) is fixed
  above by ADR 0065.
- **The `BoundedAction` construction-time check now enforces the ordering half of
  the contract too**, not just the length half. `low()[i] < high()[i]` is stated
  in the trait docs and in `docs/rules.md` §3, but only the lengths were verified
  at agent construction; an inverted pair surfaced mid-episode as an empty-range
  panic inside `Rng::random_range` or a `min > max` panic inside `f32::clamp`,
  naming the agent rather than the offending impl, and an *equal* pair produced a
  degenerate action space that reported nothing at all.
- **DDPG, TD3 and SAC truncated every action to its tensor rank** (ADR 0053,
  resolves #253). The `act`/`act_with` paths looped `0..A::RANK` and
  `.take(A::RANK)` over the actor's output — the rank is the number of *axes*,
  not the number of scalar components — so for any rank-1 action with `C > 1`
  components the warm-up sample, the policy mean, and the greedy action were each
  cut to a single value before `from_slice` asserted the true length and panicked.
  This is #100's panic resurfacing inside three algorithms. All eight sites now
  key on `A::COMPONENTS`, and the three constructors assert
  `A::low().len() == A::COMPONENTS` so a mis-declared impl fails at construction
  rather than mid-episode.

  No shipped configuration could reach it: every `BoundedAction` impl that existed
  had `RANK == COMPONENTS`, which is also why the tests missed it — the fixtures
  were rank-1/1-component and rank-3/3-component, so rank and component count were
  numerically equal in every case exercised. The regression test is now a
  deliberately rank-1, 3-component fixture, the one shape that distinguishes them.
- **DDPG and TD3 clipped target actions against a single scalar bound** (ADR
  0053, #253). Both collapsed the bound vector to `low[0]`/`high[0]`, citing
  CleanRL convention. That is correct only while every component shares a bound;
  for an asymmetric space such as CarRacing's `Box([-1,0,0], [1,1,1])` it admits
  negative gas and brake into the target action — precisely the "values of
  impossible actions" that TD3's clip exists to exclude (Fujimoto et al. 2018,
  arXiv:1802.09477, Eq. 14). Clipping is now per-component via `max_pair`/
  `min_pair` against `[1, C]` bound tensors cached at construction. TD3's
  `noise_clip` stays scalar by design: it bounds noise *magnitude*, which is a
  property of the smoothing rather than of the action space.
- **`AgentStats::new(0)` silently degenerated into a one-record window instead
  of rejecting the argument** (resolves #191). `record` pops the front whenever
  `recent_history.len() >= window_size`, so with `window_size == 0` the pop
  fires on every call and the window pins at exactly one entry forever.
  `avg_score` then divides that single score by one and returns the *latest*
  episode's score under the name "moving average" — a plausible-looking number
  with none of the smoothing a caller asked for, and no error to signal it.
  `new` now asserts `window_size > 0` and documents the panic.

  No call site is affected: all eight agents construct their stats with a
  hardcoded `100`. The guard matters because `AgentStats` is genuinely wired
  into every agent, so the first call site that takes a window from user
  config would otherwise inherit a silent misreport rather than a loud failure.
  The existing tests missed it because none constructed a zero window, and the
  degenerate case is invisible from `avg_score`'s return type — `Some(f32)`
  looks identical whether it averaged one record or a hundred.

  The companion claim filed against this module — that a single `NaN` score
  permanently poisons `best_score` — was **refuted** and no change was made:
  `f32::max` is NaN-ignoring, so `best_score` self-heals on the very next
  record, and the proposed "sanitize `NaN` to −∞" fix would have been a no-op.
  `avg_score` does propagate a `NaN` through its sum, but only until the value
  slides out of the window; filtering it there stays out of scope because both
  upstream NaN origins are already guarded (#184's SAC alpha optimizer, #173's
  Gaussian `log_std` clamp — both closed), which makes the filter a backstop
  against sources that no longer produce, not a deferred repair.
- **`History::new(0)` silently degenerated into a one-record buffer instead of
  rejecting the argument** (resolves #190). The same eviction arithmetic as
  `AgentStats` above: `add` pops the front whenever `trace.len() >= capacity`,
  so with `capacity == 0` the pop fires on every call and the buffer pins at
  exactly one transition forever — while `capacity()` keeps reporting `0` and
  `is_full()` returns `true` from the first insert. That is a standing violation
  of the `len() <= capacity` invariant `docs/rules.md` §3 states for this crate's
  buffers, and it discards every experience the caller collects without
  signalling anything. `new` now asserts `capacity > 0` and documents the panic.

  `History` has no call sites at all today — `memory.rs`, its only former
  consumer, was deleted by ADR 0050 — which is why no test caught it and why the
  fix is cheap now. It is reachable from user code the moment the POMDP work
  adopts `HistoryRepresentation` or `SufficientStatistic`, both of which take
  `&History` in their signatures; fixing it before a call site exists is the
  point, not an argument against it.

  The companion claim — that `Send`/`Sync` "isn't guaranteed" for the stored
  `A`/`R` and needs explicit bounds plus an ADR — was **refuted**. `Send` and
  `Sync` are auto-traits: `History` and `ExperienceTuple` hold owned fields in a
  plain `VecDeque` with no interior mutability, so both are already `Send`/`Sync`
  exactly when `O`, `A`, and `R` are. Declaring the bounds on the struct would
  have *restricted* the types without adding a guarantee, against Rust API
  Guidelines C-STRUCT-BOUNDS. What was genuinely missing is that the property was
  neither documented nor pinned, so a later field addition (an `Rc` for cheap
  clones, a `Cell` for a cached statistic) could strip it silently; both types now
  document the propagation and a static assertion in the test module holds it.

- **A non-finite `next_q_max` survived terminal transitions in
  `compute_target_q_values`, defeating the terminal-bootstrap convention**
  (resolves #192). The target was computed by *scaling* the bootstrap term,
  `rewards + gamma * next_q_max * (1.0 - terminated)`, and because
  `NaN * 0.0 == NaN` (as does `Inf * 0.0`), a poisoned `next_q_max` propagated
  into the target on exactly the samples where the convention says the
  bootstrap must vanish. The term is now genuinely masked to `0` wherever
  `terminated == 1.0`, so a terminal target is the reward regardless of what
  the next-state estimate holds.

  This is hardening, not a repair of live divergence: the expression cannot
  *originate* a NaN, and for all-finite inputs the new form is numerically
  identical, so no algorithm's behaviour changes. It amplified a defect whose
  only known trigger — SAC's alpha optimizer poisoning itself on a single
  non-finite gradient — was fixed in #184. The existing tests missed it because
  they only ever fed finite Q-values, where the two formulations agree
  exactly. The four affected call sites are DQN, DDPG, TD3, and SAC; C51 and
  QR-DQN never used this helper (they mask in their own projection step).

- **PPO/PPG progress logging fired at `lcm(num_steps, log_every)`, not
  `log_every` — and not at all for plausible configs** (resolves #321). Both
  loops gated on `global_step.is_multiple_of(log_every)`, but the check sits
  *after* the inner rollout `for` loop, so `global_step` was only ever observed
  at multiples of `num_steps` (default `128`). Logging therefore required
  `log_every` to divide the rollout stride. With `log_every = 100` the first
  line landed at step 3200 instead of 100 — 32× too sparse — and with
  `log_every = 500` at step 16000, so any run shorter than that emitted
  **nothing at all**, silently, while the rustdoc promised "a progress line
  every this many global steps". The six off-policy loops were never affected:
  they check `(step + 1) % log_every == 0` *inside* the step loop.

  The trigger is now a last-logged watermark
  (`global_step − last ≥ log_every`), which is robust to the stride instead of
  depending on divisibility. It stays at the rollout boundary by necessity —
  the log payload reports `PpoUpdateStats` from `update()`, which does not
  exist mid-rollout — so the realised cadence is bounded by
  `[log_every, log_every + num_steps)` rather than being exactly `log_every`.
  Rounding *up* to the next boundary is the deliberate choice: advancing the
  watermark by `log_every` instead of to `global_step` would let it fall behind
  and then fire on consecutive boundaries to catch up, which is burstier.

  A second defect surfaced while fixing the first: because the final rollout is
  usually partial, the terminal boundary can sit less than `log_every` past the
  watermark and never fire, dropping the last `update()`'s statistics
  (`total_timesteps = 10000, log_every = 500` logged last at 9728 and reported
  nothing for 10000). Both loops now emit a final progress line when the run
  ends, unless the boundary already logged it.

  Existing tests missed all of this because nothing anywhere observed a log
  line: there was no log-capture test in the workspace, and every integration
  call site passed `log_every = 0`, so the trigger was pure control flow that
  no assertion touched. Both halves are now closed. The decision itself moved
  into a `LogWatermark` helper in `algorithms::shared` with direct unit tests,
  and `crates/rlevo/tests/` gained `tracing`-subscriber capture tests for PPO
  and PPG that run the real training loops with logging on and assert a line is
  emitted, that the last one reports `total_timesteps`, and that spacing stays
  within `[log_every, log_every + num_steps)`. Those tests were verified to
  fail against the old gate — the first draft used a 700-step budget, which the
  old divisibility check satisfied by coincidence at the terminal boundary, so
  the run length is deliberately 650 (no boundary of which is a multiple of
  `log_every = 100`). `log_every == 0` still disables logging.
- **One non-finite gradient permanently bricked SAC's temperature controller for
  the rest of the run** (resolves #184). `LogAlpha::adam_step` folded
  `g = −(log π̄ + H̄)` straight into its hand-rolled Adam moments with no
  finiteness check. Those moments are exponential moving averages, so
  `β₁ · NaN = NaN`: a single pathological batch poisoned `m` and `v`
  **permanently**, and every subsequent α was NaN no matter how healthy later
  gradients were. The actor and critic optimizers rebuild from fresh gradients
  each step and self-heal; `m`/`v` carry state across steps and never recover.

  A collapsed squashed-Gaussian policy legitimately emits `log π → −Inf` on
  out-of-distribution actions and a diverging critic can feed NaN back through
  the reparameterised actor, so this needed no exotic configuration to fire —
  and it fired on the *first* bad gradient. From there the NaN α propagated into
  both critic losses via the Bellman target and into the actor loss, taking down
  the rest of the agent with it. The run kept reporting finite-looking
  bookkeeping throughout.

  `adam_step` now skips the update in full when `g` is not finite — `m`, `v`,
  `t` and `log α` are all left untouched — and emits a one-shot `tracing::warn!`
  naming the likely cause. A separate backstop clamps `log α` to `[−88, 88]` so
  `α = exp(log α)` cannot overflow to `+Inf` down the *other* path into those
  same losses. The two are independent: clamping the parameter does nothing for
  already-poisoned moments, which is why the guard is the actual fix.

  **A finite gradient is not enough**, and review of the first guard turned up a
  second route to the same permanent corruption. `(1 − β₂) · g · g` is
  left-associative, so it overflows to `+Inf` from about `|g| ≳ 1e21` while `g`
  itself is still an ordinary finite float and the finiteness check passes.
  `v = +Inf` is absorbing under the moving average, so `v̂.sqrt()` is `Inf` and
  every later step size is exactly `0`: the controller freezes **silently** —
  no NaN, no odd-looking `log α`, nothing to notice — which is strictly harder
  to diagnose than the NaN it replaces. This is reachable rather than
  adversarial: the policy's `log σ` is clamped but its Gaussian *mean* is an
  unclamped `Linear` output, so a mean that has run away against a near-floor σ
  makes `((a − μ)/σ)²` huge but finite and `log π` follows. Both moments are
  now computed into locals and committed only once known finite, under their own
  one-shot warning — a separate latch, so whichever failure fires first cannot
  silence the other.

  A non-finite `alpha_lr` is rejected by the same guard. The `[−88, 88]` clamp
  appears to cover it, but only for `g ≠ 0`: at `g = 0` the step is `Inf · 0 =
  NaN`, and `NaN.clamp(..)` propagates rather than rescuing. (Such a value
  could reach the optimizer at all because `config::positive` treated `+Inf`
  as positive — fixed by #353, see the `rlevo-core` finiteness entry above.)

  A third variant sits one level further down, in the **bias-corrected** values:
  `v̂ = v / (1 − β₂ᵗ)` can overflow to `+Inf` while `g`, `lr` and both raw
  moments are finite, because the divisor is only ~`0.001` at `t = 1`. Same
  silent-freeze signature. This one is *bounded* rather than permanent — the
  divisor grows and the controller recovers on its own — but the reachable band
  `|g| ∈ (1.84e19, 5.83e20)` drops between 27 and 686 consecutive updates, and
  the band's lower edge moves with `t`. Guarded by rolling the whole step back;
  keeping the raw moment update and skipping only the parameter subtraction was
  measured to be bit-for-bit identical to no guard at all, since the freeze is
  caused by the committed finite-but-large `v` rather than by the skipped
  subtraction.

  Each of the three failures carries its own warning latch, so whichever fires
  first cannot silence the others — they point at different subsystems and only
  the second indicates a NaN source.

  Both hardenings are deliberate `rlevo` deviations — softlearning (the SAC
  authors' own code), rlkit, CleanRL and Stable-Baselines3 all leave `log α`
  unbounded and none guards the α optimizer against a non-finite gradient. The
  bounds are wide enough to be provably non-binding in a healthy run (SAC's
  legitimate α range is ~`[0, 10]`, i.e. `log α ≤ 2.3`), so no converging run's
  numbers change. The module docs record the deviation against Haarnoja et al.
  (arXiv:1812.05905) Eq. 18 rather than presenting it as standard practice.

  The three existing tests only ever drove `adam_step` with finite log-probs and
  asserted the *direction* α moved, so nothing exercised the failure path at
  all. The new coverage asserts that the controller still **moves** after a
  poisoned step, not merely that its state is finite — a frozen controller is
  perfectly finite, which is exactly how the overflow variant would have slipped
  past a finiteness-only assertion.

- **C51 crashed on roughly 4% of valid atom supports — f32 rounding pushed the
  projection's atom index one past the end of the support** (resolves #180).
  `project_distribution` clamps the Bellman shift `Tz` to `[v_min, v_max]`, so
  the continuous atom coordinate `b = (Tz − v_min) / Δz` is *mathematically*
  confined to `[0, N−1]`. Bellemare et al. 2017 assert exactly that, as an
  inline comment in Algorithm 1 — and it is exact in ℝ. It is not exact in
  IEEE-754. When `Tz` saturates at `v_max`, the division can round `b` a few
  ULPs **above** `N−1`, `ceil` then yields `N`, and the `scatter` indexes off
  the end of a size-`N` axis and panics.

  With `v_min = −10, v_max = 0.1, N = 8`, `b = 7.000000477` and any reward
  `≥ 0.1` panics with `index 8 out of bounds for dimension of size 8`. A sweep
  over `v_min ∈ [−20, 0)`, `v_max ∈ (v_min, 20]` and `N ∈ [2, 64]` found
  **165,092 of 3,786,300 supports** affected. Every one of them passes
  `validate()`; none is exotic. `b` is now clamped to `[0, N−1]` before
  `floor`/`ceil` — a no-op in real arithmetic, and the same guard CleanRL's
  `c51.py` carries.

  **Why the tests missed it.** The default support `(−10, 10, 51)` lands on
  `b = 50.0` exactly, and so do every unit test's `(−1, 1, 3)` and `(−2, 2, n)`
  and every benchmark's `(−10, 10, {21, 51, 101})`. The suite contained a
  `projection_clamps_above_support` test aimed squarely at this boundary, but on
  an exactly-landing support it cannot observe the defect no matter how it is
  written. The regression tests added here use non-default supports chosen
  *because* they round badly.

  Note this is distinct from the exact-atom-landing case (`l == u`, where both
  distance weights are zero and mass would be dropped), which the existing
  `l_eq_u_mask` already handled correctly and which is unchanged.

- **`project_distribution` silently returned a corrupted target distribution
  for a degenerate support, instead of failing** (found while fixing #180).
  With `v_min == v_max` the spacing `Δz` is `0`, so `b = (Tz − v_min)/0` is
  `NaN`. `f32::clamp` **propagates** `NaN` rather than rescuing it, and Rust's
  saturating float→int cast maps `NaN` to `0` — so every index collapsed to
  atom 0 and the function returned a plausible-looking distribution with all
  mass on the bottom atom. No panic, and no `NaN` in the output to signal it.
  Silent corruption is a worse failure than the out-of-bounds panic above.

  The pre-existing `assert!(num_atoms >= 2)` does not cover this: `num_atoms`
  can be perfectly valid while `v_max − v_min == 0`. `project_distribution` now
  asserts that the spacing is finite and strictly positive, and documents both
  panic conditions under `# Panics`.

  This is guarded in the projection operator rather than only in config
  validation because `project_distribution` is public, re-exported, and takes
  **raw `f32` scalars** — it is reachable with no `C51TrainingConfig` involved,
  so it has to defend its own contract. The related gap where a config struct
  literal bypasses `validate()` entirely is tracked as #326.

- **PPO's Gaussian `log_std` was unbounded, so a long continuous-control run
  could collapse it until `σ` underflowed to zero and NaN poisoned every
  weight** (resolves #173, ADR 0049). The gradient of the Gaussian log-prob
  w.r.t. `log_std` is `((z − μ)/σ)² − 1`, which is `≈ −1` for a high-advantage
  action near the mean — exactly the case the surrogate rewards. Every such
  update pushed `log_std` down, linearly and without limit. Below `≈ −87`,
  `σ = exp(log_std)` underflows f32 to exactly `0.0`, `centered / σ` becomes
  `±inf`, and `backward()` corrupts the parameters permanently. At the Pendulum
  benchmark's `lr = 3e-4` that is on the order of 290k updates — inside a normal
  training budget, and with no error signal until the run visibly diverges.

  **The entropy bonus does not save it.** Gaussian entropy here is linear in
  `log σ`, so its restoring force is a constant `entropy_coef · lr` — roughly
  300× weaker than the drift at the default `entropy_coef = 0.01`, and *zero*
  in the workspace's only continuous-control benchmark. That zero is correct,
  not an oversight: it is the published SB3 rl-zoo tuned Pendulum-v1 config, and
  PPO's own MuJoCo benchmark (Schulman et al. 2017, Table 3) also ran without an
  entropy bonus. The reference-faithful configuration is precisely the one with
  no restoring force.

  **Why the tests missed it.** Every existing test evaluated the head at or near
  `log_std_init = 0.0`, where the arithmetic is unremarkable; nothing exercised a
  collapsed `log_std`, and nothing ran long enough to reach one. The failure is a
  slow drift over tens of thousands of updates, which no unit test and no
  non-`#[ignore]`d integration test covers.

  Note this is a **deliberate deviation from reference PPO**, not a correction of
  an oversight: CleanRL, Stable-Baselines3 (`DiagGaussianDistribution`) and
  Spinning Up all leave PPO's `log_std` unclamped, and SB3 clamps only in its
  SAC path — so the previous PPO-unclamped/SAC-clamped asymmetry *matched* both
  references. (Issue #173 as filed asserted the asymmetry existed "for no stated
  reason"; it did have one.) The bound is justified by numerical totality — it
  makes `log_prob` a total function on f32 — not by a claim that bounding trains
  better. Andrychowicz et al. 2021 found a minimum std "matters little, if it is
  not set too large," and separately observed that exponentiating an unbounded
  `log_std` "occasionally produced NaN values". The default `[-20, 2]` is far
  below any healthy policy, so the only runs it changes are runs already
  producing garbage. Issue #173's claim that PPG inherits the defect is also
  false — PPG is discrete-only in v1 and has no Gaussian head.

- **DQN, C51, QR-DQN, DDPG, TD3 and SAC zeroed the Bellman bootstrap on
  time-limit truncation, biasing Q-values downward on every time-limited
  environment** (resolves #170, part 3). All six training loops masked the
  target with `snapshot.is_done()`, which is true for `Truncated` as well as
  `Terminated`. Zeroing on truncation tells the agent the trajectory genuinely
  ended with no future value; the error is systematic, always downward, and
  compounds silently over long runs. The canonical target is
  `r + γ · ¬terminated · max_a Q(s′, a)` (Pardo et al. 2018 Eq. 6; Gymnasium
  Eq. 2) — the mask is `¬terminated`, never `¬done`. TD3's own paper specified
  this in 2018 (Fujimoto et al., Appendix D), so the implementation diverged
  from its own primary source.

  The loops now bind `terminated` for the replay mask and keep `done` for
  episode bookkeeping and `env.reset()`. The private `Transition.done` field is
  renamed `terminated` throughout the batch path, so nothing downstream still
  calls a terminated-mask "dones".

  **No signature or storage change was required.** Each loop already cloned
  `next_snapshot.observation()` *before* the `env.reset()` inside its `if done`
  branch, so the replay buffer was already storing the true continuation state
  — `max_a Q(next_obs, a)` was already `V(s_continuation)`. Only the mask was
  wrong. (Issue #170 as filed asserted the opposite, claiming the fix required
  plumbing a continuation observation through `remember` and `Transition`;
  triage refuted that.)

  Nothing caught this because none of the six training loops had a
  `#[cfg(test)]` module at all, and every replay-level test used an environment
  that terminates rather than truncates — so no test could distinguish the two
  masks. The new coverage pins the episode-end *cadence* alongside the mask, so
  a truncation assertion cannot pass vacuously by the episode simply never
  ending.

- **The default DQN, C51 and QR-DQN config ran two target-update mechanisms at
  once, and the hard one erased the soft one** (resolves #182) — `sync_target`
  gated its full `target ← policy` copy on `target_update_frequency` alone and
  never read `tau`, despite every config doc promising the opposite ("When
  `tau > 0.0` … this field is ignored"). Because the `Default` shipped at the
  time set **both** `tau = 0.005` and `target_update_frequency = 100`, and each
  `train` loop called `sync_target()` unconditionally, a default run performed
  a Polyak soft update every learn step *and* slammed the target onto the
  policy every 100 steps — destroying the target lag that Polyak exists to
  provide, which is the whole mechanism these algorithms rely on for bootstrap
  stability. This was not a config-exotic edge case; it was the default path
  for all three agents.

  No published algorithm runs both schemes on one target network: Mnih et al.
  2015 (DQN), Bellemare et al. 2017 (C51) and Dabney et al. 2018 (QR-DQN) all
  specify a *pure* periodic hard copy, while Lillicrap et al. 2015 (DDPG)
  specifies pure Polyak explicitly as a **replacement** for it — the point of
  soft updates being that targets are "constrained to change slowly". Stable-
  Baselines3 and CleanRL expose both knobs but as a *single* gated mechanism
  (frequency says when, `tau` says how far), never as two independent schedules.

  The tests could not catch it, for a sharper reason than #167/#168: nothing in
  the workspace ever reads `target_net` — there is no accessor and never was
  one. The three integration tests set **both** knobs (e.g. `.tau(0.005)` with
  `.target_update_frequency(500)`), thereby *encoding the defective config*,
  then assert only that rewards are finite and that two seeded runs agree. A
  scheduled hard copy is both finite and perfectly deterministic, so those
  assertions hold identically under either regime.

  **Superseded within this same release.** The interim fix here made
  `sync_target` return early whenever `tau > 0.0`, and added regression tests
  pinning that no-op plus the `tau == 0.0` hard-sync branch. Neither
  `sync_target` nor the `tau`/`target_update_frequency` pair it read exist
  anymore: ADR 0058/0059 (the `target_update: TargetUpdate` entry above)
  replaces both fields with a single cadence+τ type under which this
  dual-mechanism bug — and the `tau == 0.0` / `target_update_frequency == 0`
  frozen-target case this entry also fixed — are structurally unrepresentable,
  and folds in the #334 cross-family (SAC vs. DQN/C51/QR-DQN) divergence this
  entry originally left open. Read that entry for the shipped mechanism and
  its own regression tests; the two `sync_target`-specific tests described
  above were removed along with `sync_target` itself.
- **A panic inside a learn step could permanently brick an agent** (ADR 0046,
  resolves #167) — all eight gradient-based agents (`dqn`, `c51`, `qrdqn`,
  `ddpg`, `td3`, `sac`, `ppo`, `ppg`) stored their trainable networks as
  `Option<M>` and `.take()`d the field for the *entire* learn step: forward
  pass, loss, `backward()`, and gradient reduction all ran while the field
  was `None`. Any panic in that window — a malformed batch, a shape
  mismatch, a device transfer failure — left the field `None` forever.
  Every subsequent `act()` and learn call hit its own `.expect(...)` on an
  empty `Option` and re-panicked; only killing the process and rebuilding
  the agent from scratch recovered it. 17 call sites across the eight
  agents shared this shape. On TD3 and SAC the blast radius was wider than
  a single field: both critics were taken out of their fields up front and
  stepped sequentially from the same wide window, so a panic stepping
  `critic_1` also destroyed `critic_2` in the same unwind — a single fault
  bricking two networks. The existing test suite could not see any of this:
  the in-crate unit tests cover only pure functions, config validation, and
  metrics bookkeeping and never drive a numeric `learn_step`; the
  cross-crate integration tests that actually exercise training
  (`crates/rlevo/tests/*_integration.rs`) are `#[ignore]`d by default; and
  the reproducibility tests only assert that the *same* seed reproduces the
  *same* output on a clean run — none of the three drives a panic path at
  all. All eight agents now hold their networks in a crate-internal
  `Slot<M>` newtype: `forward`/loss/`backward` run against a borrow, and the
  module leaves the field only for the single Burn `Optimizer::step` call
  itself, so `critic_1` and `critic_2` now step through disjoint windows. A
  panic strictly inside that one `step` call remains unrecoverable **by
  design** — Burn's `Optimizer::step` consumes the module by value, so
  neither a drop-guard nor `catch_unwind` can hand it back once `step` has
  been entered and the module has moved into its frame — but the
  poisoned-slot panic message now says so plainly and tells you to rebuild
  the agent, instead of pointing at `learn_step`, a method that does not
  even exist on PPO or PPG (theirs are `update()` and
  `policy_phase_update()`). `ppg` was not named in issue #167 but carried
  the identical defect at 5 call sites and is fixed under the same change.
- **A panic during a target soft update silently hard-synced the target onto
  its live network** (resolves #168) — the six off-policy agents (`dqn`,
  `c51`, `qrdqn`, `ddpg`, `sac`, `td3`) built a throwaway `.valid()` snapshot
  of the *active* network and `std::mem::replace`d it into the target field
  purely to keep the field populated while `soft_update` consumed the target
  by value. On the happy path the placeholder was overwritten on the very next
  line and cost nothing observable. On a panic inside `soft_update` it was
  never overwritten, so the agent unwound with the target field holding a full
  copy of the policy — a hard sync the caller never asked for, and the exact
  failure mode τ exists to avoid. On TD3 the window spanned three fields, so
  one fault could corrupt the target actor and both target critics. The tests
  could not catch it for the same reason they missed #167: nothing in the
  suite drives a panic through a learn step. `M::InnerModule` is `Clone`
  (`Module<B>: Clone` is a supertrait in Burn 0.21), so all 10 call sites now
  pass `self.<target>.clone()` and leave the field untouched until
  `soft_update` returns. Numerics are unchanged — the discarded snapshot never
  reached the Polyak average.

  Note that the *performance* premise in the original text of #168 was wrong
  and has been retracted: `.valid()` is not a deep copy (it moves refcounted
  backend primitives), so its cost scales with `Param` count, not network
  size. The device→host→device round-trip in `polyak_update` is where the
  real per-step cost lives — fixed later in this same release, resolving
  #322 (see the `polyak_update`/on-device entry under **Changed**, above).

- **Every agent's minibatch staging round-tripped each observation through the
  device before ever using it** (resolves #362, completing the #187 sweep).
  `stack_to_tensor` was added in #195 as the single batched host→device upload
  path, nominally consumed by `memory.rs::sample_batch` — but no agent ever
  called `sample_batch` (the dead-`PrioritizedExperienceReplay` defect tracked
  as #188), so the helper had no live caller from the day it landed. ADR 0050
  retired that consumer deliberately, leaving staging to each agent because
  "each agent stages differently". Eight agents — `dqn`, `c51`, `qrdqn`,
  `ddpg`, `td3`, `sac`, `ppo`, and `ppg` — had always kept a hand-rolled
  staging loop that called
  `TensorConvertible::to_tensor` on one sampled transition, immediately called
  `.into_data()` on the result, and copied the floats back into a host `Vec`.
  The observation was already on the host: the loop uploaded it to the
  accelerator and downloaded it unchanged, then dropped the tensor without a
  single operation ever running on it. The staging loops now write straight
  into the preallocated flat buffer with `write_host_row` — the same primitive
  `to_tensor` and `stack_to_tensor` are both built on — and the one batched
  `Tensor::from_data` upload per minibatch is unchanged.

  The cost is worst on `wgpu`, where `into_data()` is a synchronization point:
  a `learn_step` at `batch_size = 64` stalled the pipeline 128 times (state and
  next-state) before any real work began, on top of 128 discarded buffer
  allocations. The `.expect("float data")` panic each read carried is gone from
  the hot loop with it.

  No test caught this because there was nothing to catch: `write_host_row` is
  the primitive `to_tensor` itself uses, so the staged bytes are bit-identical
  either way. The defect was pure throughput, invisible to any correctness
  assertion, and the existing tests pass unmodified — which is the acceptance
  criterion here rather than evidence of a gap.
- **`PpoTrainingConfig::minibatch_size()` could return `0`** (#166). The accessor
  guarded only the divisor (`num_minibatches.max(1)`) but not the quotient, so a
  config with `num_minibatches > batch_size` (e.g. `num_steps = 10,
  num_minibatches = 20`) reported a minibatch size of `0` while `PpoAgent::update`
  clamped its own `mb_size` to `1` — the public API and the training loop
  disagreed, and a caller pre-sizing a buffer from the accessor got a zero-length
  allocation. The quotient is now floored to `1`, matching the loop. The existing
  config tests only exercised the well-formed case (`batch_size` a multiple of
  `num_minibatches`), where the two agree, so the divergence never surfaced. PPG
  inherits the fix through its wrapped `PpoTrainingConfig`.

**Added**

- **`AuxPhaseStats::learning_rate`** — the rate a PPG auxiliary phase's two
  optimizer steps actually ran at, carried alongside the existing loss fields.
  Lets a caller distinguish a phase that *ran but moved nothing* (`0.0`, the
  #324 defect) from one that moved the policy imperceptibly; the loss fields
  cannot, because at `lr == 0.0` every parameter is bit-exactly unchanged and
  `policy_kl` collapses to `0.0` — which is also its value on a healthy
  single-minibatch phase. Additive: `AuxPhaseStats` is constructed only inside
  `maybe_aux_phase`.

  This replaces the load-bearing assertion in `ppg_integration.rs`'s #324
  regression test, which previously decided the bug through
  `policy_kl > 0.0`. That form is correct but is an `f32` mean of
  log-differences between near-identical logits, measured at ~4.1e-7 — about
  3.4× `f32::EPSILON` — so a backend with a different reduction order could
  round a *healthy* phase to zero. Exposure that mattered once #519 put the
  crate on shared CI hardware. The behavioral half of the check moved to a new
  in-crate test, `ppg_aux_phase_at_nonzero_lr_moves_policy_parameters`, which
  measures a host-side weight delta across the terminal auxiliary phase —
  `lr · step`, linear in the rate and clear of `f32` resolution by orders of
  magnitude. Both assertions were verified to fail against a deliberately
  reintroduced #324 (`max |Δw| = 0` exactly).

- **`PpoUpdateStats::min_log_std` and a one-shot warning when the `log_std`
  bound binds** (resolves #173, ADR 0049). Bounding `log_std` trades a *loud*
  failure for a *quiet* one: before, a collapsing policy produced NaN and the run
  visibly died; now it can sit silently pinned at `σ ≈ 2·10⁻⁹`, emitting
  near-deterministic actions with no crash. Worse, because `log_std` is a
  state-independent `Param` rather than a per-state network output, `clamp`
  zeroes its gradient **permanently** once it crosses — there is no recovery
  path, unlike SAC, where the `Linear` layer keeps learning from in-range
  observations. Shipping the clamp without telemetry would have been a net
  downgrade in debuggability, so the two land together.

  `PpoPolicy` gains a defaulted `min_log_std() -> Option<f32>`; the categorical
  head keeps the `None` default, the Gaussian head reports its clamped minimum
  across action dims. It is read **once per update**, not per forward pass —
  detecting a bind needs a host-side read, and doing that in the forward pass
  would force a device→host sync every step on wgpu. Deferring costs nothing:
  the bound is reported at the end of the update in which it binds, and since
  the parameter can never leave the bound, no crossing is missed.

  Not yet surfaced in `PpoMetrics`, so the value does not reach the TUI or the
  recorded metric stream — it is returned from `update()` only. Tracked
  separately.

- **Contract tests for `polyak_update`** (resolves #336). `utils.rs` had no
  `mod tests` at all, leaving the single arithmetic primitive beneath every
  off-policy agent's target update — `dqn`, `c51`, `qrdqn`, `ddpg`, `sac`,
  `td3` — entirely unexercised. The `soft_update` impls in the integration
  fixtures merely delegate to it and assert nothing about it, and the tests
  that use them check only finite rewards and seeded reproducibility, both of
  which hold for any deterministic update rule, correct or not. An
  implementation returning `target` unchanged — or returning `active` outright,
  which is the #182 defect expressed one layer down — passed the entire suite.

  Five tests now pin the contract on constant-weight fixtures with
  hand-computed expected values: `tau = 0.0` is identity, `tau = 1.0` is an
  exact hard copy (a promise `utils.rs`'s own rustdoc already made and nothing
  checked), fractional `tau` is the exact convex combination, shapes and
  parameter counts are preserved, and repeated application converges
  monotonically toward `active` without overshoot. Each test asserts that
  `active` and `target` genuinely differ *before* the update, so none can go
  vacuous — without that precondition a do-nothing implementation satisfies the
  `tau = 0`, `tau = 1` and blend cases simultaneously. Both mutations above
  were confirmed to fail three tests each.

  One non-obvious constraint surfaced while writing the fixtures, recorded here
  because it is easy to trip over: `PolyakMapper` looks parameters up by
  `ParamId`, so a target net built independently of the active net does not
  blend cleanly. *At the time these tests were written a `ParamId` mismatch
  panicked; ADR 0057 (above) later made the whole soft-update path fallible,
  so the same mismatch is now reported as `Err(PolyakError)` instead of
  aborting.* Real agents get the matching IDs for free by cloning the policy
  net; hand-built fixtures must reuse the active net's `ParamId`s explicitly.

  A second qualification, added when #182 landed: the phrase "the exact
  failure mode τ exists to avoid" above overstates the practical exposure on
  `dqn`, `c51` and `qrdqn` specifically. Under the `tau`/`target_update_frequency`
  field pair that existed at the time — since replaced by `TargetUpdate` (see
  above) — those three hard-synced the target on schedule anyway whenever a
  hard-sync cadence was configured, so under the shipped `Default` the
  panic-path residue was byte-identical to what the scheduled hard sync was
  about to do regardless. The fix was nonetheless fully load-bearing on
  `ddpg`, `sac` and `td3` — which have no hard-sync path at all, so nothing
  would ever have overwritten the residue — and on pure-Polyak configs of all
  six agents.

### Docs

**Changed**

- **`*Config` types keep their `pub` fields — this is now a recorded decision,
  not an accident** (ADR 0055, closes #326). All 71 workspace `*Config` structs
  expose `pub` fields, so `Config { lr: 3e-4, ..Default::default() }` compiles
  without ever calling `validate()`. That was read as a validation hole, but
  ADR 0026 deliberately placed the obligation on the **consumer**: the
  constructor that takes the config by value calls `validate()?`. ADR 0055
  writes down the allocation rule that was previously spread across
  `docs/rules.md` §2/§4 and ADRs 0026/0027/0031 — `*Config` keeps `pub` fields
  and is validated at consumption; `*State`/`*Params`/`*Genome` encapsulate;
  an invariant that must survive struct-literal construction is encoded in a
  validated newtype (`Bounds`, `Probability`, `NonNegativeRate`) rather than by
  hiding the field; `#[non_exhaustive]` is reserved for enums. **No API
  changes** — the struct-update idiom is explicitly supported and stays
  supported. `docs/rules.md` §2 previously left the `*Config` exemption
  inferable only from its absence in a list of three suffixes, which is why the
  rule could not be reconstructed and the issue was filed.

**Added**

- Conformance test pinning the ADR 0026 consumption chokepoint across 31
  `rlevo-environments` configs, asserting each `with_config` rejects an invalid
  config with the expected structured `ConfigError` field and `ConstraintKind`.
  Adding an environment now means adding a case here. `Taxi`/`Blackjack`/
  `CliffWalking` are excluded with a stated reason — their `Validate` is
  unconditionally `Ok(())`, having no numeric invariant to check.
  Nothing previously failed if a new environment's constructor forgot
  `validate()?` — that silent-regression risk is what #326 was groping at, and
  it is the part of the issue that was real. Also characterizes the dormant
  `Deserialize` gap: plain `derive(Deserialize)` accepts an out-of-domain
  config (executed: `bincode` decodes `GoToDoorConfig { size: 1 }` below its
  `MIN_SIZE = 5`), which is per ADR 0026 the loader's obligation — and no
  config loader exists in the workspace today.

### Infrastructure

**Fixed**

- **Every workspace crate now runs its tests in the pull-request gate, and a
  meta-check keeps it that way** (resolves #519). `crate-tests.yml`'s matrix was
  hand-maintained and named five crates; six were missing. The consequence was
  worst for `rlevo`, whose 22 `crates/rlevo/tests/*.rs` binaries were reachable
  only through `weekly-tests.yml` — which runs `-- --ignored` and therefore
  *filters out* every non-ignored test in them. 38 non-ignored tests across 16
  binaries executed in no workflow at all, including the regression tests
  written for #321 and #324 specifically so they would gate pull requests.
  Un-ignoring a test never made it run.

  The gap was not `rlevo`-only: `rlevo-benchmarks-report-client` (80 tests),
  `rlevo-metrics-registry` (17), `rlevo-test-support` (6),
  `rlevo-hybrid` (6) and `rlevo-examples` (5) were absent from the matrix for no
  recorded reason. All six crates are now in it; `rlevo` carries
  `--features viz-report`, without which `recording_episode_count.rs` and
  `cartpole_report_smoke.rs` fail to *compile* (neither is `#![cfg]`-gated, and
  the feature is not in `rlevo`'s defaults). Measured cost of the added
  coverage: ~55 s of test execution for `rlevo`, under a second each for the
  other five.

  The new `test-matrix-coverage` job fails the build when a crate under
  `crates/` is absent from the matrix, on the model of the #391 lint-opt-in
  job — the omission itself is the bug, and nothing in cargo reports it.

---

## [0.3.1] – 2026-07-17

Patch release: no breaking changes since 0.3.0.

### `rlevo-metrics-registry`

**Added**

- `Trend` reading (`HigherIsBetter` / `LowerIsBetter` / `Diagnostic`) and a
  one-line interpretation hint per canonical metric (`trend_for`, `hint_for`),
  surfaced above each plot in the benchmarks TUI's Separate layout so a
  reader can tell whether a rising sparkline is good news without leaving the
  dashboard. `MetricDescriptor` gains two fields with const-default
  constructors; existing consumers stay source-compatible.

### `rlevo-benchmarks`

**Added**

- Combined-layout TUI metric labels are now prefixed with a trend glyph
  (↑ / ↓ / •), matching the Separate layout's enriched titles.

### `rlevo-reinforcement-learning`

**Fixed**

- `SacAgent`'s default target entropy now uses `COMPONENTS` instead of
  `RANK` — the Haarnoja et al. (2018b) heuristic it cites is
  `-dim(action_space)`, not `-rank`. Behavior-identical today, since every
  existing `BoundedAction` impl has `RANK == COMPONENTS`; this closes a
  latent bug for any future multi-component bounded action type (part of
  the ADR 0038 RANK/COMPONENTS blast radius).

### Docs

**Fixed**

- KaTeX now renders backtick-wrapped inline `` $...$ `` math and fenced
  ` ```math ` blocks correctly on docs.rs; both forms previously rendered as
  raw, unprocessed LaTeX text (rustdoc emits them as `<code>` /
  `<pre class="language-math"><code>`, which the header's original selectors
  missed).
- Sub-crate `readme` fields now resolve to each crate's own `README.md`
  instead of all inheriting the workspace-root `README.md` via
  `readme.workspace = true` (which does not re-relativize per member).
- Over 20 misattributed or inaccurate citations corrected across crate
  READMEs, ADRs 0043 and 0045, and the user-book (reference-verification
  audit, issue #313).

### Examples & user-book

**Added**

- `backend_sweep_neuroevolution` example and a "Choosing a Backend"
  user-book chapter, illustrating CPU-vs-GPU backend selection via Burn's
  backend genericity on a batched neuroevolution fitness function.

### Infrastructure

- Project logo migrated from SVG to PNG for consistent rendering on GitHub
  and crates.io.
- `justfile`'s report-client build recipe self-heals a missing
  `wasm32-unknown-unknown` target after toolchain updates.

---

## [0.3.0] – 2026-07-13

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
