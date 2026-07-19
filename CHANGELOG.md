# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

**Added**

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

### `rlevo-environments`

**Breaking changes**

- Every environment now implements `rlevo_core::environment::Sensor` and builds
  its `reset`/`step` snapshots through it instead of `State::observe` (ADR 0047,
  #329). For the box2d family the observation is no longer *produced* from a state
  cache — the env-side sensor reads the world directly (BipedalWalker lidar,
  CarRacing pixels). `CarRacing` drops its cached pixel buffer entirely;
  `BipedalWalker`/`LunarLander` retain only a finiteness signal for `is_valid`
  (`last_obs`/`prev_shaping` respectively), matching the locomotion states, so
  `is_valid` is unchanged. The grid family routes its shared egocentric projection
  through `GridState: Observable<3>` and the `build_snapshot` chokepoint;
  `GoToDoorEnv` implements `Sensor` for its goal-conditioned observation.
  `pixel_grid` keeps `Observable<3>` and delegates its `Sensor` to `project()`.
  Behaviour (observations, rewards, termination) is unchanged.

### `rlevo-reinforcement-learning`

**Breaking changes**

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

  *Migration.* `PpoAgent::record_step` and `PpgAgent::record_step` gain a
  trailing `next_obs: &O`. Pass the observation from the snapshot the
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

  *Migration.* Add `log_std_min: -20.0, log_std_max: 2.0` to every
  `TanhGaussianPolicyHeadConfig` literal. `validate()` now rejects an inverted
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

- **`target_update_frequency` default raised from `100` to `10_000` for DQN,
  C51 and QR-DQN.** The old value had no basis in the literature or in any
  reference implementation. The new one matches Stable-Baselines3's
  `target_update_interval`, which — like our `step` counter — is measured in
  **environment steps**, so the two are directly comparable.

  This is deliberately *not* a literal match to Nature DQN, and the field docs
  now say so. Mnih et al. 2015 specify `C = 10,000` measured in **parameter
  updates** (Extended Data Table 1); at our `train_frequency: 4` that would be
  ≈40,000 env steps. Our 10,000 env steps is ≈2,500 parameter updates — 4×
  more frequent than Nature, and exactly SB3's convention. The field rustdocs
  now state the unit explicitly, since "target update frequency" is measured
  differently by different sources and the ambiguity is a live trap.

  **No behavioral change under the shipped defaults**, which keep `tau = 0.005`
  — after the #182 fix below, `target_update_frequency` is inert whenever
  `tau > 0.0`. This affects only runs that explicitly opt into hard-sync mode
  with `tau = 0.0`. Such runs will now sync the target 100× less often; if you
  had been relying on the old `100` implicitly, set it explicitly. Note the new
  default is Atari-scaled: on a short classic-control run it may sync only a
  handful of times (see #337 for per-scale tuning guidance).

**Fixed**

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
  reaches the optimizer at all because `config::positive` treats `+Inf` as
  positive — tracked separately in #353.)

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
  time set **both** `tau = 0.005` and `target_update_frequency = 100` (that
  frequency is now `10_000`, see **Changed** above), and each `train`
  loop calls `sync_target()` unconditionally, a default run performed a Polyak
  soft update every learn step *and* slammed the target onto the policy every
  100 steps — destroying the target lag that Polyak exists to provide, which is
  the whole mechanism these algorithms rely on for bootstrap stability. This
  was not a config-exotic edge case; it was the default path for all three
  agents. `sync_target` now returns early whenever `tau > 0.0`, so the hard
  sync fires only under `tau == 0.0`, exactly as documented.

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
  assertions hold identically under either regime. Each agent now carries two
  in-source unit tests asserting on the target's **parameter tensors** — one
  pinning that `sync_target` is a no-op under `tau > 0.0`, one covering the
  `tau == 0.0` hard-sync branch, which the fix would otherwise leave entirely
  untested since no in-tree config sets it. Assertions deliberately avoid
  Q-values and greedy actions: argmax is a lossy hash of the weights, so a
  0.005 Polyak step and a full hard copy frequently yield the same action, and
  such a test would have passed against the unfixed code.

  Also fixed: `tau == 0.0` together with `target_update_frequency == 0` meant
  the target network never updated at all — silently trainable against a frozen
  random bootstrap. All three configs now reject it in `validate()`. The
  converse (both set) remains legal, since the library `Default` relies on it.

  Note that `target_update_frequency` still means *soft* cadence in SAC and
  *hard* cadence in these three; that cross-family divergence is tracked
  separately as #334.
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
  real per-step cost lives; that is tracked separately as #322.

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

**Added**

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
  `ParamId` and panics on a miss, so a target net built independently of the
  active net does not blend — it aborts. Real agents get the matching IDs for
  free by cloning the policy net; hand-built fixtures must reuse the active
  net's `ParamId`s explicitly.
  A second qualification, added when #182 landed: the phrase "the exact
  failure mode τ exists to avoid" above overstates the practical exposure on
  `dqn`, `c51` and `qrdqn` specifically. Until #182, those three hard-synced
  the target on schedule anyway under any `target_update_frequency > 0`, so
  under the shipped `Default` the panic-path residue was byte-identical to
  what `sync_target` was about to do regardless, and was re-applied within at
  most `target_update_frequency` env steps. The fix was nonetheless fully
  load-bearing on `ddpg`, `sac` and `td3` — which have no hard-sync path at
  all, so nothing would ever have overwritten the residue — and on
  pure-Polyak configs (`target_update_frequency == 0`) of all six agents.

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
