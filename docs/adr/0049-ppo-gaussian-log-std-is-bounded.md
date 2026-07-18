---
project: rlevo
status: active
type: decision
date: 2026-07-18
tags: [adr, decision, reinforcement-learning, ppo, gaussian, log-std, numerical-stability, issue-173]
---

# ADR 0049: PPO's Gaussian `log_std` is bounded, not free

## Status

**Accepted (2026-07-18).** Resolves issue #173 ("[rl] PPO/PPG Gaussian head:
clamp `log_std` like SAC does — unconstrained param → std→0 → NaN"). Supersedes
no ADR.

## Context

PPO's continuous policy head
(`crates/rlevo-reinforcement-learning/src/algorithms/ppo/policies/gaussian.rs`,
`TanhGaussianPolicyHead`) learns `log_std` as a **state-independent `Param`**
of length `action_dim` — one shared vector, not a per-observation network
output. `σ = exp(log σ)`, and `log_prob` evaluates `((z − μ)/σ)²` where `z` is
the pre-squash sample stored in the rollout buffer.

### The defect

The gradient of the Gaussian log-probability with respect to `log_std` is
`((z − μ)/σ)² − 1`. For a high-advantage action near the mean — the case the
PPO surrogate objective rewards — this is `≈ −1`. Every such update therefore
applies sustained, **unbounded downward** pressure on `log_std`, linear in
update count at rate `≈ lr`. Nothing in the surrogate objective opposes it on
its own.

Two failure modes follow, and they are not the same defect — the distinction
determines which one this ADR must actually guard against:

1. **Absolute underflow (the reachable one).** At `log_std < ≈ −87`, `σ` falls
   out of f32's normal range into the subnormals, and by `≈ −104` it is
   exactly `0.0`; well before that, `|z − μ|/σ` alone overflows. `centered / σ`
   is then `±inf` regardless of the numerator, and `NaN` propagates through
   `backward()` and
   permanently corrupts every weight the optimizer touches next. At the
   Pendulum benchmark's `lr = 3e-4`, drifting from `log_std_init = 0.0` to
   `−87` at the `≈ −1`-per-update rate takes on the order of `290,000`
   updates — well inside a normal training budget.
2. **Ratio overflow.** `scaled²` can overflow f32 while `σ` is still
   representable, whenever the **span** between the `σ` that produced a
   buffered sample and the `σ` currently evaluating it is large: writing
   `scaled = k · σ_old / σ_new` for a `k`-sigma sample, this overflows once
   `2·(log_std_old − log_std_new) + 2·ln k > ln(3.4e38) ≈ 88.7` — a span of
   about `44`. Because the rollout buffer is refreshed every iteration,
   `σ_old/σ_new` normally stays close to `1` in ordinary training, so this
   mode is **not** reached by ordinary gradient drift. It **is** reachable
   through a misconfigured, overly wide `[log_std_min, log_std_max]`
   interval — which is the entire reason the span validation in Decision §3
   exists.

**The entropy bonus does not prevent mode 1.** Gaussian entropy here is
`Σ (log σ + ½ log(2πe))` — linear in `log σ` — so its restoring force is a
constant `entropy_coef · lr`. At the crate's default `entropy_coef = 0.01`
that force is roughly `300×` weaker than the `≈ −1`-per-update drift above.
Worse: the only continuous-control benchmark in the workspace
(`crates/rlevo/benches/pendulum_rl.rs:198`, `.entropy_coef(0.0)`) sets it to
**zero** — and that is the *correct* setting, not an oversight. It is the
published SB3 rl-zoo tuned Pendulum-v1 configuration, and PPO's own MuJoCo
benchmark (Schulman et al. 2017, Table 3) likewise ran without an entropy
bonus. So the reference-faithful configuration for this library's one
continuous-control benchmark is exactly the configuration with **no
restoring force at all**.

### This is a deviation from reference PPO, and must be stated as one

Bounding `log_std` is **not** canonical PPO. The literature and every
mainstream reference implementation leave it free:

- TRPO (Schulman et al. 2015, arXiv:1502.05477, Appendix D) defines
  `stdev = exp(r)` with `r` a state-independent free parameter vector — no
  bound.
- PPO (Schulman et al. 2017, arXiv:1707.06347, §6.1) inherits this verbatim:
  "outputting the mean of a Gaussian distribution, with variable standard
  deviations, following [Sch+15b; Dua+16]." No clamp appears anywhere in the
  paper; its Roboschool showcase (Table 4) instead used a
  `LinearAnneal(−0.7, −1.6)` **schedule**, not a clamp.
- CleanRL's `ppo_continuous_action.py`:
  `actor_logstd = nn.Parameter(torch.zeros(...))`, unclamped.
- Stable-Baselines3's `ActorCriticPolicy` / `DiagGaussianDistribution`: a
  state-independent `nn.Parameter`, `log_std_init=0.0`, unclamped. SB3 clamps
  only in `SquashedDiagGaussianDistribution` — its **SAC** path.
- OpenAI Spinning Up: an `nn.Parameter` initialised to `−0.5`, unclamped.
- Huang et al. 2022, *37 Implementation Details of PPO* — detail #2 of the
  continuous-action section documents the state-independent `log_std`
  initialised to `0`, explicitly contrasts it with SAC's state-dependent
  choice, and never mentions clamping.

So `rlevo`'s prior PPO-unclamped / SAC-clamped asymmetry **matched** both
reference implementations it was drawn from. Issue #173's original framing —
that the asymmetry existed "for no stated reason" — is incorrect, and this
ADR corrects that framing rather than repeating it: the asymmetry had a
reason, and this decision knowingly removes it.

### Why deviate anyway

Andrychowicz et al. 2021, *What Matters In On-Policy Reinforcement Learning?*
(arXiv:2006.05990, §3.2 / App. B.8) is the one large-scale empirical study of
this exact knob, and two of its findings carry this decision:

- Exponentiating an unbounded `log_std` "occasionally produced NaN values" —
  published corroboration of precisely the defect described above.
- "The minimum action standard deviation seems to matter little, **if it is
  not set too large**" — which licenses a non-binding floor as
  performance-neutral, and is the empirical cover for the choice below.

The justification is therefore **numerical totality, not training
stability**: the bound exists to make `log_prob` a total function on f32, not
because bounding `log_std` is believed to train better. A healthy PPO run
lives around `log_std ∈ [−3, 1]`; a floor at `−20` is `σ ≈ 2·10⁻⁹`. The set of
runs whose numbers the clamp changes is exactly the set of runs already
producing garbage. Andrychowicz's own wording — "matters little" — is taken
at face value: this ADR does not claim the bound improves learning.

## Decision

### 1. Clamp `log_std` to `[log_std_min, log_std_max]` everywhere it is read

Default `[−20, 2]`. `−20` (`σ ≈ 2·10⁻⁹`) sits far below any healthy converged
policy so the floor never binds on a working run; `2` (`σ ≈ 7.4`) matches the
SAC head's existing ceiling.

### 2. Bounds live on the policy-head config, not the training config

`log_std_min` / `log_std_max` are new fields on `TanhGaussianPolicyHeadConfig`
— the config that is actually consumed to build the head — mirroring the
convention already established for SAC (see the References note on #185):
`SquashedGaussianPolicyHeadConfig` (`sac_policy.rs`) is where the clamp is
genuinely applied;
`SacTrainingConfig` (`sac_config.rs`) carries a second, unconsumed pair of
fields with the same names. This ADR does not touch the SAC duplication (out
of scope, tracked separately) but deliberately does not repeat it here: PPO
gets exactly one home for these two numbers.

### 3. `validate()` rejects an inverted interval, a floor below `−35`, and an oversized span

Three checks, and the last two guard **different** failure modes — neither
implies the other:

- **Ordering.** `log_std_min < log_std_max`.
- **Absolute floor → mode 1.** `log_std_min ≥ −35`. This bounds `σ` itself, so
  `σ = exp(log_std_min)` can never underflow to `0.0`. Derived from the
  requirement that `((z − μ)/σ)²` stay finite, i.e.
  `|z − μ|/σ ≤ √f32::MAX = 1.8447·10¹⁹`, hence
  `log_std_min ≥ ln|z − μ| − 44.36`. The floor is therefore a function of the
  largest pre-squash residual that must remain representable: at
  `|z − μ| ≤ 10²` the bound is `−39.75`, and `−35` is that rounded up with
  margin (it admits `|z − μ|` up to `≈ 1.16·10⁴`). `−35` is `σ ≈ 6.3·10⁻¹⁶`,
  six orders of magnitude below the default floor of `−20`, so it constrains no
  usable configuration.
- **Span → mode 2.** `log_std_max − log_std_min < 40`. This bounds the *ratio*
  `σ_old/σ_new`, derived from the mode-2 overflow condition above
  (`2·Δ + 2·ln k > 88.7` at `Δ ≈ 44`), with margin held back so that even a
  large-`k` outlier sample stays within f32 range.

The floor is not redundant with the span, and assuming otherwise was the
original defect in this decision: `log_std_min = −120, log_std_max = −100` is
correctly ordered, spans only `20`, and admits `log_std_init = −110` — yet
`exp(−110)` is **exactly** `0.0` in f32, so `(z − μ)/σ` is `±inf` and `NaN`
reaches `backward()`. The span check bounds the ratio between two `σ`s; it says
nothing about either one's absolute magnitude. Mode 1 needs its own guard.

Together the two numerical checks also cap the *upper* bound: `log_std_min ≥
−35` with a span under `40` forces `log_std_max < 5`. That is intended and
free, since a converged continuous-control policy sits near `log σ ∈ [−3, 1]`.

### 4. Clamp-plus-telemetry ship together, not the clamp alone

PPO's per-iteration stats gain the min-across-dims `log_std`, and a one-shot
`tracing::warn!` fires the first time the bound binds on a run. See
Consequences below for why telemetry is not optional here.

### 5. No `Default` impl

`TanhGaussianPolicyHeadConfig` was already a plain struct with no `Default`
and every construction site uses a full struct literal; that does not change.
A `Default` is deliberately not added now either: `obs_dim` / `action_dim` of
`0` fails `config::nonzero`, so any `Default` would itself be rejected by the
config's own `validate()` — a `Default` that cannot validate is worse than no
`Default`.

### 6. Scope is PPO only

PPG is discrete-only in v1
(`crates/rlevo-reinforcement-learning/src/algorithms/ppg/policies/mod.rs:3-9`
— "A tanh-Gaussian variant is a follow-up") and has no Gaussian head today, so
nothing there changes. A future continuous PPG head is expected to reuse
`TanhGaussianPolicyHead` (or its config shape) and inherits this decision by
construction.

## Consequences

### Positive

- `log_prob` is total on f32 for any `(z, μ, log_std)` triple reachable from a
  config that **passes `validate()`**, provided `|z − μ| ≤ 1.16·10⁴` — the
  residual budget the `−35` floor buys. Totality is a joint property of the
  clamp *and* the validation: the clamp alone does not deliver it, because it
  bounds `log_std` to an interval the caller chooses, and a caller may choose
  an interval in which `σ` is `0`. Mode 1 is closed by the absolute floor
  (Decision §3), not by the clamp.
- Mode 2 (ratio overflow via a misconfigured wide bound interval) is rejected
  at construction time by the span check, rather than surfacing as an
  intermittent `NaN` deep in a training run.
- Matches the one empirical study of this knob (Andrychowicz et al. 2021)
  rather than either reference implementation's silence on it.

### Negative / accepted costs — do not soften these

- **Trap-door gradient, and the SAC analogy actively misleads.** SAC clamps
  a per-step **network output**: its `log_std` `Linear` layer keeps
  receiving gradient from every in-range observation in the batch, so a
  policy that saturates on some states still has a path back. Here
  `log_std` is a **state-independent `Param`**: once it crosses a bound,
  `clamp` zeroes its gradient **permanently**, including the entropy
  bonus's restoring force. It is stuck at the bound for the remainder of
  training with no route back. This is accepted — a run that reached `−20`
  was already producing degenerate output — but it must be stated plainly,
  because a reader reasoning from the SAC precedent will wrongly expect
  recovery.
- **A loud failure becomes a quiet one.** Previously: `NaN`, unmistakable,
  the run visibly dies. Now: a policy can be silently pinned at
  `σ ≈ 2·10⁻⁹` forever, producing plausible-looking (near-deterministic)
  actions with no crash. This is why the clamp does not ship alone —
  Decision §4's telemetry (min `log_std` in PPO stats, a one-shot warning
  the first time a bound binds) is part of this decision, not a follow-up;
  the clamp by itself would be a net downgrade in debuggability.
- **Breaking change.** `TanhGaussianPolicyHeadConfig` gains two required
  fields. It is a plain struct with no `Default`, and every construction
  site in the workspace uses a full struct literal, so every one of them
  must be updated in the same change (no partial migration is possible).
  Accepted: pre-1.0, all call sites are in-repo.
- **Scope is PPO only**, per Decision §6 — recorded here as a limit, not an
  omission: PPG's discrete head needs no change, and a future continuous PPG
  head is the one that inherits this decision.

## Alternatives considered

- **Softplus / `min_std` soft floor** (Andrychowicz's own recommended
  alternative). Rejected: it is a non-canonical reparameterisation that
  perturbs the optimisation geometry smoothly but **everywhere** in the
  operating region, including healthy runs. A non-binding hard clamp
  perturbs nothing in-distribution — it is the smaller deviation from
  reference PPO, not the larger one.
- **`Option`-able bounds** (`Option<f32>` for either end, `None` meaning
  "unbounded"). Rejected: `None` means "opt into `NaN`," and there is no
  research value in a `log_std` region that is unrepresentable in f32 at
  all. A user who genuinely wants maximal permissiveness sets
  `[−35, 2]` — the widest floor `validate()` accepts, and still total.
- **Warn instead of clamp.** Rejected: by the time a non-finite `log_std` or
  `NaN` gradient is observable, the weights are already poisoned and the run
  is unrecoverable. A warning after the fact documents the corpse; it does
  not prevent it.
- **Rely on the entropy bonus as the restoring force.** Rejected: quantified
  above as roughly `300×` too weak at the crate's default
  `entropy_coef = 0.01`, and exactly zero in the one reference-faithful
  continuous-control configuration this workspace ships
  (`pendulum_rl.rs:198`).

## References

- Issue #173 — "[rl] PPO/PPG Gaussian head: clamp `log_std` like SAC does —
  unconstrained param → std→0 → NaN."
- Schulman, Levine, Abbeel, Jordan, Moritz. *Trust Region Policy
  Optimization.* ICML 2015. arXiv:1502.05477 —
  <https://arxiv.org/abs/1502.05477>. (Appendix D: `stdev = exp(r)`,
  state-independent, unclamped.)
- Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization
  Algorithms.* 2017. arXiv:1707.06347 —
  <https://arxiv.org/abs/1707.06347>. (§6.1 continuous-control setup, no
  clamp stated; Table 3 MuJoCo hyperparameters, no entropy bonus; Table 4
  Roboschool `LinearAnneal(−0.7, −1.6)` log-std schedule.)
- Andrychowicz, Raichuk, Stańczyk, Orsini, Girgin, Marinier, Hussenot,
  Geist, Pietquin, Michalski, Gelly, Bachem. *What Matters In On-Policy
  Reinforcement Learning? A Large-Scale Empirical Study.* ICLR 2021.
  arXiv:2006.05990 — <https://arxiv.org/abs/2006.05990>. (§3.2 / App. B.8:
  exponentiating an unbounded `log_std` "occasionally produced NaN values";
  minimum std "matters little, if it is not set too large.") **Primary
  empirical source for this decision.**
- Huang, Dossa, Ye, Braga, Chakraborty, Mehta, Araújo. *The 37
  Implementation Details of Proximal Policy Optimization.* ICLR Blog Track,
  2022. (Continuous-action detail #2: state-independent `log_std`, init `0`,
  contrasted with SAC's state-dependent head; no clamp.)
- CleanRL — `ppo_continuous_action.py`
  (`actor_logstd = nn.Parameter(torch.zeros(...))`, unclamped).
- Raffin et al. *Stable-Baselines3.* JMLR 22(268):1–8, 2021.
  (`DiagGaussianDistribution` — unclamped `nn.Parameter`,
  `log_std_init=0.0`; `SquashedDiagGaussianDistribution` — the SAC path,
  the only place SB3 clamps.)
- OpenAI Spinning Up — `spinup/algos/pytorch/ppo/core.py`
  (`log_std` `nn.Parameter` initialised to `−0.5`, unclamped).
- Issue #185 — SAC's dead `log_std_min`/`log_std_max` fields on
  `SacTrainingConfig`; this ADR follows the same convention it names —
  bounds live on the **policy-head** config that is actually consumed
  (`SquashedGaussianPolicyHeadConfig`/`TanhGaussianPolicyHeadConfig`), not on
  the training config.
- ADR [0026](0026-shared-config-validation-convention.md) /
  [0027](0027-bounds-newtype-for-closed-ranges.md) — the `Validate` /
  ordered-range convention this ADR's `validate()` change follows.
- Code: `crates/rlevo-reinforcement-learning/src/algorithms/ppo/policies/
  gaussian.rs` (`TanhGaussianPolicyHead`, `TanhGaussianPolicyHeadConfig`,
  `log_prob_entropy`); `.../sac/sac_policy.rs`
  (`SquashedGaussianPolicyHeadConfig`, the live SAC clamp at `:129`) and
  `.../sac/sac_config.rs` (`SacTrainingConfig`, the
  unconsumed duplicate fields, #185); `.../ppg/policies/mod.rs:3-9`
  (discrete-only v1, no Gaussian head); `crates/rlevo/benches/
  pendulum_rl.rs:198` (`entropy_coef(0.0)`, the reference-faithful
  continuous-control benchmark configuration).
