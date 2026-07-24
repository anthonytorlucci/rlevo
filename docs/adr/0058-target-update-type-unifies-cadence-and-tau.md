---
project: rlevo
status: active
type: decision
date: 2026-07-24
tags: [adr, decision, target-network, polyak, cadence, dqn, c51, qrdqn, ddpg, td3, sac, issue-334, issue-455, issue-182]
---

# ADR 0058: τ and cadence are one target-update mechanism — `TargetUpdate` is its type

## Status

**Accepted (2026-07-24).** Resolves issue #334 ("`target_update_frequency`
means 'soft cadence' in SAC and 'hard cadence' in DQN/C51/QR-DQN"), **correcting
its own premise** to a three-way divergence (§Context). Closes issue #455
(`tau = 0.0` passing `config::in_range`) **by construction**, for all six
configs. Defers the default-*value* question (Atari-scaled cadence against
classic-control `steps_per_episode`) to issue **#337**. Builds on ADR 0057
(`polyak_update` → `Result`) — `TargetUpdate` names the *when*/*how far*
contract that `soft_update`'s `Result` already assumed a caller had settled.
Cadence *unit* (env steps vs. gradient updates) is deliberately a separate
decision — see ADR 0059.

**Chosen shape.** One type,

```rust
pub struct TargetUpdate {
    tau: PolyakTau,       // f32, invariant 0.0 < tau <= 1.0 — zero EXCLUDED
    every: NonZeroUsize,  // cadence, unit fixed by ADR 0059
}

impl TargetUpdate {
    pub const fn hard(every: usize) -> Self;                   // tau = 1.0; panics on 0
    pub const fn polyak(tau: f32, every: usize) -> Self;        // panics on invalid tau/cadence
    pub fn try_polyak(tau: f32, every: usize) -> Result<Self, TargetUpdateError>;
    pub const fn tau(self) -> f64;
    pub const fn every(self) -> usize;
    pub const fn is_hard(self) -> bool;                        // tau == 1.0
    pub const fn fires_at(self, updates: usize) -> Option<f64>; // Some(tau) iff this update fires
}
```

in `crates/rlevo-reinforcement-learning/src/target.rs`. Cadence (`every`) gates
*when* an update fires; τ gates *how far* it moves the target at that firing.
`τ = 1.0` is a hard copy **by degeneracy of the shared formula**, not by a
separate variant. Adopted as one field, `target_update: TargetUpdate`, on all
six off-policy agents' configs (DQN, C51, QR-DQN, DDPG, TD3, SAC), replacing
the two flat `pub tau` / `pub target_update_frequency` fields (or, for
DDPG/TD3, the aliased use of `policy_frequency` — see §Context) on each.

The **half-open** τ interval is load-bearing, not a stylistic choice: `τ = 0.0`
is precisely the frozen-target state, so admitting it would falsify the
§Consequences claim below that a never-updating target becomes unrepresentable —
and would make the deletion of the three cross-field validation blocks a
regression rather than a simplification. `PolyakTau` exists *because*
`rlevo_core::Probability` admits `0.0` (§Alternatives).

`fires_at` returns `Option<f64>` rather than `Option<f32>` because all five
`soft_update` model-trait methods take `f64` (`dqn_model.rs:65`,
`ddpg_model.rs:59,100`); returning `f32` would put a widening cast at every one
of the nine call sites. `try_polyak` takes `every: usize` rather than
`NonZeroUsize` so that a runtime cadence of `0` is *handled* here rather than
pushed onto the caller, and returns a single `TargetUpdateError` covering both
failure modes (`Tau { got }`, `ZeroEvery`) — a τ-only error has no name for the
cadence failure. `is_hard()` keeps hard-vs-Polyak **queryable** without
reintroducing a variant.

## Context

### The problem is a three-way divergence, not the two-way one the issue states

Issue #334's title and body describe DQN/C51/QR-DQN vs. SAC. Verified against
current source, the actual split is three-way, and the third arm is worse than
either named case because it is silent.

**DQN / C51 / QR-DQN** run `tau: f64` and `target_update_frequency: usize` as
two independently-gated mechanisms on one target network. Polyak fires
**ungated, every learn step**, whenever `tau > 0.0`
(`dqn_agent.rs:558-566`, `c51_agent.rs`, `qrdqn_agent.rs`, same shape). The hard
path, `sync_target()`, early-returns when `tau > 0.0` and otherwise hard-copies
on `self.step.is_multiple_of(target_update_frequency)`
(`dqn_agent.rs:385-400`). At the shipped default `tau = 0.005`,
`target_update_frequency` is **inert** — `sync_target()` is called
unconditionally from every train loop (`dqn/train.rs:137`, `c51/train.rs:130`,
`qrdqn/train.rs:124`) and is a documented no-op the whole time. This is the
tail of the #182 fix (`docs/.private/research/target-network-update-semantics.md`):
the two-schedule defect was closed by making the hard path self-gate, not by
removing the second field.

**SAC** (`sac_agent.rs:757-771`) uses `target_update_frequency` to gate the
**soft** update itself, on `self.critic_updates` — the SB3/CleanRL unified
convention (§Literature below). Default `1`, so the gate is a no-op at
defaults and the semantics are untested in-tree by anything that varies it.

**DDPG / TD3 have no `target_update_frequency` field, but they are not the
"sidesteps the question" case the issue claims.** Both gate the Polyak update
on `policy_frequency` (default `2`), inside the *same* delayed-actor block that
also fires the actor step — `ddpg_agent.rs:566-605`, `td3_agent.rs:670-714`,
both `if self.critic_updates.is_multiple_of(self.config.policy_frequency)`.
So `policy_frequency` is an **undeclared alias for target cadence**. Issue #334
states: "DDPG and TD3 sidestep the question by omitting the field." They do
not — they answer it silently with someone else's knob, which is worse: a
reviewer checking the issue's claim against `ddpg_config.rs` sees no frequency
field, agrees with the issue, and misses the aliasing entirely. (Recorded
verbatim as the load-bearing correction in
`docs/.private/research/target-network-update-semantics.md`, "CORRECTION
(2026-07-24, while working #334)".) TD3's *behaviour* here matches Fujimoto
§5.2 exactly — critic and target update together every `d` critic steps — what
is not canonical is that the cadence is unnameable and unsettable independently
of the actor delay.

So the true divergence is three regimes, not two: soft-every-gradient-step
(DQN family), soft-every-`target_update_frequency`-critic-update (SAC), and
soft-every-`policy_frequency`-critic-update, tied to the actor delay
(DDPG/TD3).

### Two corrections to the issue's own premises

**1. No config in the workspace derives `Serialize`/`Deserialize`.** All seven
`*TrainingConfig` types (`DqnTrainingConfig`, `C51TrainingConfig`,
`QrDqnTrainingConfig`, `DdpgTrainingConfig`, `Td3TrainingConfig`,
`SacTrainingConfig`, `PpoTrainingConfig`, plus `PpgConfig`) are
`#[derive(Clone, Debug)]` only — verified by grep across every
`algorithms/*/*_config.rs`. The issue's critical open question — "Is
`DqnTrainingConfig` serialized into any `EpisodeRecord` / `RunManifest`
provenance field? If so, the migration includes a record `FORMAT_VERSION`
bump" — answers **no**. `Hyperparameters` in `rlevo-benchmarks`
(`record/schema.rs:541`) is a free-form `BTreeMap<String, String>` populated by
callers at report time, not a serialized config; there is no config loader
anywhere in the workspace (ADR 0055 §Consequences, negative cost 3, already
records this as a dormant, unowned obligation). So there is **no persisted
config to migrate and no `FORMAT_VERSION` bump** — the ADR 0014/0023 precedent
for a schema bump does not apply here. The real stale-config vector is
struct-literal `..Default::default()` construction and the fluent builders,
both of which the field removal turns into a **compile error** — strictly
stronger than the deserialization error the issue hoped a schema bump would
force.

**2. The unified operator is canonical, not an SB3/CleanRL convention that
rlevo is merely borrowing.** The issue asks the ADR to record "why the
SB3/CleanRL *semantics* were adopted but their *field shape* was not." That
framing under-attributes the mechanism. Haarnoja et al. 2018a (SAC, ICML,
arXiv:1801.01290), Appendix D Table 1, runs a **"SAC (hard target update)"**
ablation — τ = 1, target update interval = 1000 — beside standard SAC — τ =
0.005, interval = 1 — inside the SAC paper itself: the degenerate hard copy is
obtained purely by setting τ = 1 on the soft rule, a controlled ablation, not a
different code path. Fujimoto et al. 2018 (TD3, ICML, arXiv:1802.09477),
Section 3, states the same generality directly: "The weights of a target
network are either updated periodically to exactly match the weights of the
current network, or by some proportion τ at each time step
θ′ ← τθ + (1 − τ)θ′." Algorithm 1 then runs the gated soft form, and §5.2:
"The modification is to only update the policy and target networks after a
fixed number of updates *d* to the critic." So the unified operator predates
SB3/CleanRL by four years and appears in both the SAC and TD3 papers
themselves. What is rejected here is SB3/CleanRL's **two-flat-`pub`-field
shape**, not their semantics — that shape is the concrete mechanism by which
the three in-tree families diverged, since nothing ties the two fields'
meanings together at the type level.

The theory literature does not contest this unification, but it does not
endorse it as a mathematical identity either, and the ADR does not overclaim
one. Lee 2026a (*Target Updates May Stabilize Linear Q-Learning: Periodic and
Soft Dynamics*, arXiv:2606.02645, **preprint**) and Lee 2026b (*Geometrically
Averaged Hard Target Updates for Linear Q-Learning*, arXiv:2606.10835,
**preprint**) treat periodic-hard and soft-Polyak as separate switching-system
families with separate joint-spectral-radius convergence proofs; Kobayashi &
Ilboudo 2020/2021 (*t-Soft Update of Target Network for Deep RL*, Neural
Networks 136:63-71, arXiv:2008.10861) and Zhang, Yao & Whiteson 2021 (*Breaking
the Deadly Triad with a Target Network*, ICML, PMLR v139, arXiv:2101.08862)
both frame soft as an alternative to hard ("instead of"), never as its limit.
None of the four engages with API design, and none claims the two mechanisms
are the same operator — only SAC's own ablation and TD3's own generality
statement do that, and both are engineering statements about one paper's
method, not a proven equivalence. Full provenance and confidence notes:
`docs/.private/research/2026-07-24-issue-334-target-update-cadence-units.md`.

## Decision

1. **`TargetUpdate { tau: PolyakTau, every: NonZeroUsize }`** lives in
   `crates/rlevo-reinforcement-learning/src/target.rs`. `tau` is a private
   validated `f32` newtype (`PolyakTau`, invariant `0.0 < tau <= 1.0`,
   mirroring the ADR 0027/0031 shape) so `τ = 1.0` is representable — it is a
   legal Polyak coefficient, not a distinct case — while `τ = 0.0`, `τ` outside
   the interval, NaN, and infinity are not. **The lower bound is exclusive
   because `τ = 0.0` is the frozen-target state**, the exact condition the
   deleted cross-field checks existed to reject; admitting it would make
   §Consequences' unrepresentability claim false. `every: NonZeroUsize` makes
   "never fires" unrepresentable, reusing the standard library's own
   validated-nonzero type rather than adding a bespoke one for an invariant
   `std` already names.
2. **Constructors:** `hard(every)` (`tau = 1.0`), `polyak(tau, every)`
   (panicking, for literals — the ADR 0026/0031 convention for a config
   constant), `try_polyak(tau, every) -> Result<Self, TargetUpdateError>` (for a
   value that did not arrive as a source literal). Both take `every: usize` and
   validate it, so a runtime `0` is rejected here rather than at the caller.
   `TargetUpdateError` is one `Copy` enum with `Tau { got }` and `ZeroEvery`;
   τ is validated before cadence, and a test pins that ordering.
3. **`fires_at(self, updates: usize) -> Option<f64>`** is the seam every call
   site consumes: `Some(tau)` when `updates` is a multiple of `every`,
   `None` otherwise. This is a **predicate**, not an applier — see
   §Alternatives for why the type deliberately does not also call
   `polyak_update`. It reproduces `usize::is_multiple_of` exactly, so
   `fires_at(0)` is `Some(τ)` for every cadence; callers pass a **post-increment**
   counter, so index `0` is unreachable in practice. `is_hard()` reports
   `τ == 1.0` so a caller (e.g. a metrics label) can branch on hard-vs-Polyak
   without the type carrying a variant to branch on.
4. **Adoption.** All six off-policy configs (DQN, C51, QR-DQN, DDPG, TD3, SAC)
   gain `pub target_update: TargetUpdate`, replacing `pub tau` +
   `pub target_update_frequency` (DQN/C51/QR-DQN, SAC) or the `policy_frequency`
   alias for target cadence (DDPG/TD3). **DDPG and TD3 keep `policy_frequency`
   unchanged** as the actor-delay knob (TD3's `d`, Fujimoto §5.2) — it now
   governs the actor/α-analogue cadence only. `target_update` is a new,
   independently settable field, so the aliasing identified in §Context is
   removed: an operator can change how often the actor updates without
   changing how often the target moves, and vice versa.

## Consequences

### Config validation shrinks, by construction (ADR 0027 §3 / ADR 0055 §4)

A validated newtype **removes** its paired `config::` check rather than
replacing it. Deleted: `config::in_range(C, "tau", 0.0, 1.0, ...)` from all six
configs (`dqn_config.rs:166`, and the equivalent line in c51/qrdqn/ddpg/td3/sac);
`config::at_least(C, "target_update_frequency", ..., 1)` from SAC
(`sac_config.rs:100-105`); and the three `tau <= 0.0 &&
target_update_frequency == 0` cross-field blocks from DQN/C51/QR-DQN
(`dqn_config.rs:182`, `c51_config.rs:189`, `qrdqn_config.rs:195`). The
combination those blocks rejected — a target that can never update — becomes
**unrepresentable**: `every: NonZeroUsize` cannot be `0`, `PolyakTau` cannot be
`0.0`, and any valid τ combined with any valid `every` always fires eventually
and always moves the target when it does. Both halves of the invariant are
required; a closed-interval τ would leave `polyak(0.0, n)` as a frozen target
and make this deletion a regression. This is strictly stronger than "rejected at
construction," per the ADR 0027 pattern.
The three regression tests that pinned the old rejection
(`rejects_frozen_target_network` in `dqn_config.rs:373`,
`rejects_config_where_target_never_updates` in `c51_config.rs:428`,
`rejects_frozen_target_when_tau_and_frequency_are_both_zero` in
`qrdqn_config.rs:422`) are **replaced**, not deleted, by `PolyakTau`/
`TargetUpdate` constructor unit tests asserting the same states are
unreachable via the type rather than via `validate()`.

### Closes #455 by construction, for all six configs

Issue #455 ("`tau = 0.0` passes `config::in_range`, which is inclusive at both
ends") is closed the same way, and in the direction #455 asked for:
`PolyakTau`'s half-open `(0, 1]` invariant means `TargetUpdate::polyak(0.0, n)`
**panics** and `try_polyak(0.0, n)` returns `Err(TargetUpdateError::Tau { got:
0.0 })`. The frozen-target state #455 identified is unreachable in all six
configs, including via struct-literal `..Default::default()` construction, which
no `validate()`-based fix could have achieved. `config::in_range` itself is **deliberately
left unchanged**: every other caller (`gamma`, `epsilon_start`, `epsilon_end`,
`epsilon_decay`) has legitimate closed-interval endpoints, and narrowing the
shared helper for one field's sake would be the wrong lever (ADR 0055 §4: the
newtype is the lever, not the shared range check).

### Breaking, alpha, no external consumers

All six off-policy `*TrainingConfig` types change shape. In-tree construction
sites move from `.tau(x).target_update_frequency(n)` builder calls (or the
DDPG/TD3 `.policy_frequency(n)` overload for target cadence) to
`.target_update(TargetUpdate::polyak(x, n))`. Struct-literal and
`..Default::default()` construction of a stale field name is a **compile
error**, not a silent behavioural drift — the strictly-stronger outcome named
in §Context's correction 1.

## Alternatives Considered

### The two-variant enum the issue proposes

```rust
pub enum TargetUpdate {
    Hard   { every: usize },
    Polyak { tau: Probability, every: usize },
}
```

Rejected. `Hard { every: n }` and `Polyak { tau: 1.0, every: n }` denote the
*same target-update behaviour* — representable-but-equivalent, which
reintroduces one level up the exact defect the type exists to remove (two
distinct-looking configurations meaning the same thing, this time by variant
choice rather than by a second field going inert). It also hands the first
serde adopter two encodings of one config, and it breaks a derived
`PartialEq`-as-config-identity check: `Hard { every: 5 }` and
`Polyak { tau: 1.0, every: 5 }` should compare equal as configurations and
would not, under `#[derive(PartialEq)]` on the enum.

### A disjoint enum with τ restricted to the open interval `(0, 1)`

Buys disjointness by asserting something mathematically false: `τ = 1` **is**
a legal Polyak coefficient (Haarnoja's own hard-update ablation runs it), so
excluding it from `Polyak` is not a domain fact, it is a modelling error
disguised as type safety. It also forces a `match` at every apply site where
both arms perform identical arithmetic (`(1−τ)·target + τ·active` at `τ=1` is
exactly a copy — see `polyak_update`'s own doc comment: "Pass `tau = 1.0` for a
hard copy"), and it makes a continuous τ sweep over `[0.001, 1.0]` —
inexpressible across an enum boundary — unavailable to a hyperparameter search,
which is a real operation in a research library.

### Reusing `rlevo_core::Probability`

`Probability`'s invariant is `0.0 <= p <= 1.0` — admits `p = 0.0`, which for
τ is the frozen-target state (`fires_at` would still gate correctly, but the
degenerate case a reader most needs flagged, `τ = 0`, gets no distinguishing
name from `τ = 0.7`). A dedicated `PolyakTau` costs one small type and buys a
name (`PolyakTau`, `TargetUpdateError::Tau`) that a diagnostic or a doc comment can
point at specifically, matching the ADR 0031 precedent of a dedicated rate
newtype per distinct semantic domain rather than a shared generic
`0.0..=1.0` type for every fraction in the codebase.

### Putting the type in `rlevo-core`

ADR 0031 placed `Bounds` in `rlevo-core` because two crates consumed it
(`rlevo-evolution` and `rlevo-environments`). `TargetUpdate`/`PolyakTau` has
exactly one consumer — the six off-policy RL agents — and `polyak_update` /
`PolyakError` already live in `rlevo-reinforcement-learning::utils` (ADR 0057).
Co-locating `target.rs` in the same crate keeps the type next to the operator
it configures.

### A generic helper that *applies* a `TargetUpdate`, not just gates it

Five distinct traits declare an identical-shaped `soft_update` method —
`DqnModel`, `C51Model`, `QrDqnModel`, and DDPG's `DeterministicPolicy`
(actor) / `ContinuousQ` (critic), the latter two reused by SAC and TD3 (ADR
0057 §Decision). Unifying "gate + apply" behind one generic function needs a
new supertrait spanning all five (an ADR-0052-scale refactor: 0052 split
`HostRow` from its backend for exactly this kind of cross-cutting-trait churn)
or a macro. `fires_at` is the seam that works with **zero** trait plumbing —
every call site already has its own `soft_update` to call and already clones
the field before calling it (ADR 0057's documented `.clone()` + `?` +
reassignment idiom, retained across nine call sites: DQN, C51, QR-DQN
one target each; DDPG two; TD3 three; SAC two). Collapsing that idiom behind a
macro was considered and rejected: ADR 0057 records it as **load-bearing** —
the duplication is what lets each site's comment explain, in place, why the
early `?` return leaves the target's prior weights intact. A macro would hide
the invariant its comments exist to state.

## References

- Issue #334 — "`target_update_frequency` means 'soft cadence' in SAC and
  'hard cadence' in DQN/C51/QR-DQN"; resolved with the three-way divergence
  corrected in §Context.
- Issue #455 — `tau = 0.0` passes `config::in_range`'s closed interval; closed
  by construction (§Consequences).
- Issue #337 — target-update *default values* are Atari-scaled against
  classic-control `steps_per_episode: 1000`; deferred, not answered here.
- Issue #182 — the original two-independent-schedules defect on the DQN
  family; the tail of its fix (self-gating `sync_target`) is what makes
  `target_update_frequency` inert at the DQN-family default today.
- ADR [0026](0026-shared-config-validation-convention.md) — `Validate`/
  `ConfigError`.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the valid-by-
  construction newtype shape this ADR follows for `PolyakTau`.
- ADR [0031](0031-probability-rate-newtypes.md) — `Probability`/
  `NonNegativeRate`; precedent for a self-validating field removing its
  `config::in_range` line, and for a dedicated rate newtype per semantic
  domain rather than a shared generic one.
- ADR [0052](0052-hostrow-supertrait-splits-layout-from-backend.md) — cited as
  the scale of refactor a cross-trait `soft_update` unification would require.
- ADR [0055](0055-config-invariant-enforcement-allocation.md) — `*Config`
  types keep `pub` fields; struct-literal-surviving invariants go in a
  newtype (§4), which is exactly what `TargetUpdate`/`PolyakTau` does here.
- ADR [0057](0057-target-soft-update-path-is-fallible.md) — `PolyakError`,
  `soft_update: Result`, and the `.clone()` + `?` idiom this ADR's
  "generic applier" alternative declines to collapse.
- `docs/.private/research/target-network-update-semantics.md` — the #182
  note; its "CORRECTION (2026-07-24, while working #334)" block is the source
  of the DDPG/TD3 `policy_frequency` aliasing finding quoted in §Context.
- `docs/.private/research/2026-07-24-issue-334-target-update-cadence-units.md`
  — the #334 note; source of the SAC/TD3 primary-literature citations and the
  theory-literature non-contestation survey in §Context.
- Code: `crates/rlevo-reinforcement-learning/src/algorithms/dqn/dqn_agent.rs:385-400,558-566`;
  `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_agent.rs:757-771`;
  `crates/rlevo-reinforcement-learning/src/algorithms/ddpg/ddpg_agent.rs:566-605`;
  `crates/rlevo-reinforcement-learning/src/algorithms/td3/td3_agent.rs:670-714`;
  `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_config.rs:51-56,88-108`;
  `crates/rlevo-reinforcement-learning/src/utils.rs` (`polyak_update`,
  `PolyakError`).
