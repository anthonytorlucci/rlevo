---
project: rlevo
status: active
type: decision
date: 2026-07-24
tags: [adr, decision, target-network, polyak, cadence, units, dqn, c51, qrdqn, ddpg, td3, sac, issue-334, issue-337]
---

# ADR 0059: The target-update cadence counts gradient updates, not environment steps

## Status

**Accepted (2026-07-24).** Companion to ADR 0058 (`TargetUpdate` unifies τ and
cadence into one type). Recorded as a **separate** ADR because it is
separately reversible: `TargetUpdate` could have been adopted with its `every`
field counting environment steps instead, and this decision — and only this
one — would then need superseding. Resolves the unit question issue #334
leaves open (`docs/.private/research/2026-07-24-issue-334-target-update-cadence-units.md`,
§Q2). Defers the default-*value* question to issue **#337**.

**Chosen shape.** `TargetUpdate::every` (ADR 0058) counts **gradient /
optimizer updates**, uniformly, on all six off-policy agents (DQN, C51,
QR-DQN, DDPG, TD3, SAC). The unit is **not** configurable. It is named in the
accessor's rustdoc, since `learning_starts` and `train_frequency` correctly
stay in environment steps — so a `*TrainingConfig` after this ADR carries two
distinct units, and only the rustdoc distinguishes them.

## Context

### The in-tree divergence

The two agent families currently count different things when they gate a
target update:

| Family | counter | unit | site |
|---|---|---|---|
| DQN / C51 / QR-DQN | `self.step` | environment steps, incremented in `on_env_step` | `dqn_agent.rs:349-351`, gated in `sync_target()` called from each train loop |
| SAC / DDPG / TD3 | `self.critic_updates` | gradient (optimizer) updates | incremented inside `learn_step`, e.g. `sac_agent.rs:683`, `ddpg_agent.rs:564`, `td3_agent.rs:668` |

### What the canonical sources count

All three foundational papers define their cadence against gradient/parameter
updates, not environment steps:

- **Mnih et al. 2015** (Nature DQN) — Extended Data Table 1 defines `C`
  ("Target network update frequency") explicitly as "measured in the number of
  *parameter updates*."
- **Haarnoja et al. 2018a** (SAC, arXiv:1801.01290) — the target-update line
  `ψ̄ ← τψ + (1−τ)ψ̄` sits inside "for each gradient step do," nested under a
  separate "for each environment step do"; Table 1 lists "target update
  interval" beside a distinct "gradient steps" hyperparameter.
- **Fujimoto et al. 2018** (TD3, arXiv:1802.09477) — §5.2: "only update the
  policy and target networks after a fixed number of updates *d* to the
  critic," i.e. `d` is defined against critic (gradient) updates.

**The only primary-source dissenter is a library, not a paper.**
Stable-Baselines3's `dqn.py` docstring reads "update the target network every
``target_update_interval`` environment steps," and its counter
(`target_update_interval // self.n_envs`) is an artifact of vecenv plumbing,
not a claim about the DQN algorithm. CleanRL uses env steps for both `dqn.py`
and `sac_continuous_action.py`, which is correct only under its
1-gradient-step-per-env-step default and diverges from SAC's own definition
(Haarnoja's "for each gradient step") the moment the update-to-data ratio
exceeds 1.

Rainbow (Hessel et al. 2018) adds a **third** unit: Table 1 reports "Target
Network Period 32K frames" — raw Atari frames, ≈8K frame-skipped agent steps
at the standard 4× skip — which is itself neither of the above two, and is the
evidence that "environment steps" is already an ambiguous unit even within the
Atari-benchmark literature, not a clean fallback.

### The engineering argument the papers do not need to make

Under env-step units, changing `train_frequency` from `4` to `1` silently
quadruples the number of gradient steps that elapse between target refreshes
— *without the target-update field itself changing*. One field's effective
meaning depending on the value of a second, independent field is structurally
the same defect issue #334 exists to fix, one field over.

### This dissolves the "unit trap," it does not just document it

`docs/.private/research/target-network-update-semantics.md` (the #182 note)
names a "unit trap": the workspace's `target_update_frequency = 10_000` (env
steps) is **4× more frequent** than Nature's `C = 10,000` (parameter updates),
because Nature's own update frequency is 4 (one gradient step per 4 actions,
matching `train_frequency: 4`), so `C = 10,000` parameter updates is 40,000 env
steps — not 10,000. Counting gradient updates removes the mismatch at the
unit level; the field no longer needs a conversion footnote to be compared
against the paper it is nominally citing.

## Decision

1. **Count gradient updates, on all six agents, unconditionally.** DQN, C51,
   and QR-DQN each gain a gradient-update counter (parallel to
   `self.critic_updates` on SAC/DDPG/TD3) and gate `TargetUpdate::fires_at` on
   it instead of on `self.step`. SAC/DDPG/TD3 keep using `self.critic_updates`
   — already the correct unit, unchanged by this ADR.
2. **The unit is not a config choice.** No `Cadence::EnvSteps(n) |
   GradientSteps(n)` alternative is offered — see §Alternatives.
3. **`learning_starts` and `train_frequency` stay in environment steps.**
   Those two gate *when the agent is allowed to act on the environment /
   collect a transition at all*, a question that has no gradient-update
   framing (there is no gradient step to count before the first one exists).
   Only the target-update cadence changes unit. The accessor/field rustdoc for
   `target_update` names its unit explicitly so a reader does not assume it
   matches `learning_starts`'/`train_frequency`'s.
4. **Counter-advance semantics, stated explicitly because it is subtle.**
   SAC/DDPG/TD3 already advance `critic_updates` **unconditionally**, including
   on an ADR-0056 non-finite-loss skip (`sac_agent.rs:683`,
   `ddpg_agent.rs:564`, `td3_agent.rs:668` — each increments before the
   finite-loss-gated backward/step block, or independently of it). DQN, C51,
   and QR-DQN currently **early-return** on a non-finite loss
   (`dqn_agent.rs:546-548`, and the equivalent lines in `c51_agent.rs`,
   `qrdqn_agent.rs`) before any counter in that function could advance. The new
   gradient-update counter on the DQN family **must advance unconditionally**,
   matching the other three agents — specifically, it must be incremented even
   on the branch that returns `Ok(None)` for a non-finite loss. Gating the
   counter on a successful step instead would let the cadence silently drift
   on a diverging run (more attempted updates than counted ones), which is the
   same defect class — a field whose effective meaning depends on a condition
   the caller cannot see — in a new location.

## Consequences

### `sync_target()` is deleted

`sync_target()` is removed from `DqnAgent`, `C51Agent`, and `QrDqnAgent`, along
with its three unconditional train-loop call sites (`dqn/train.rs:137`,
`c51/train.rs:130`, `qrdqn/train.rs:124`). The target update moves inside
`learn_step`, next to the counter it is now gated on — matching where
SAC/DDPG/TD3 already do it. This removes the "the train loop forgot to call
`sync_target()`" failure mode entirely: after this change there is no
train-loop-owned call whose omission can silently freeze the target network.
This is a public-API removal; `sync_target()` has no caller outside the RL
crate's own train loops and tests.

### Defaults are behaviour-preserving, deliberately, and only at defaults

`TargetUpdate::polyak(0.005, 1)` for DQN, C51, QR-DQN, and SAC;
`TargetUpdate::polyak(0.005, 2)` for DDPG and TD3 (matching the existing
`policy_frequency = 2` default that used to double as the target-cadence
alias, per ADR 0058 §Context). These are **bit-identical to today's behaviour
at every agent's current default configuration** — DQN-family Polyak already
runs every gradient step (`tau = 0.005`, hard path inert); SAC already gates
on 1 critic update; DDPG/TD3 already gate on `policy_frequency = 2` critic
updates.

**The existing `target_update_frequency: 10_000` default (DQN/C51/QR-DQN) is
deliberately not carried across**, and the reason is recorded here so a future
reader does not "fix" this by porting the old number: `10_000` was inert under
the shipped `tau = 0.005` (the hard path never fired in a default run).
Adopting it literally as the new `every` under the unified contract would fire
Polyak once per 10,000 gradient updates *instead of* once per update — a
10,000× cadence collapse, with τ still at 0.005, which would visibly break
every existing default-config training run. SB3 pairs `10_000` with `τ = 1.0`
precisely because its interval is that long *for a hard copy*; the pair
(τ = 0.005, every = 10_000) is incoherent under the unified operator — it is
not "the same value the field already held," it is a different mechanism's
value borrowed by name collision. Bundling a type change (ADR 0058) with a
cadence-*value* change here would make the combined diff behaviourally
unbisectable: a training-run regression could not be attributed to "the type
changed" or "the number changed" independently. The default-*value* question
— including whether `every = 1` is even the right cadence to ship broadly,
given `docs/.private/research/target-network-update-semantics.md`'s residual
scale concern that Atari-scaled defaults (`10_000`) sit oddly against
classic-control's `steps_per_episode: 1000` — is deferred whole to **#337**.

## Alternatives Considered

### Make the unit configurable: `Cadence::EnvSteps(n) | GradientSteps(n)`

Rejected. Every agent would have to maintain **both** an env-step counter and
a gradient-update counter regardless of which one a given config selects, so
the "unused" counter's upkeep cost is paid unconditionally. Every consumer of
the cadence value would need to `match` both arms. And
`Cadence::EnvSteps(n)` is **meaningless at SAC's, DDPG's, and TD3's own
update site** — `learn_step` there has no env-step counter in scope at all
(the agent does not know how many environment steps preceded this particular
gradient step; that bookkeeping lives in the train loop, not the agent). A
config value that is well-formed but has no coherent interpretation at half
its own call sites is a representable-but-meaningless state — the ADR 0058
defect (an inert or aliased field) wearing a different hat.

### Leave DQN/C51/QR-DQN on env steps, normalise nothing

This is what the literal reading of issue #334 (which only asks about the
soft-vs-hard field semantics, not the unit) would leave in place. Rejected on
the evidence in §Context: every canonical paper's cadence is a gradient/
parameter-update quantity, `train_frequency`-relative drift is a live defect
under env-step counting, and the only precedent for env-step counting is a
library's vecenv-plumbing artifact, not an algorithmic claim. Keeping the
DQN family on env steps would also leave `TargetUpdate::every` meaning two
different things depending on which agent holds it — undermining the very
point of giving all six agents one shared type in ADR 0058.

### Gate the new DQN-family counter on *applied* (non-skipped) updates only

Considered briefly as "more honest" — count only updates that actually moved
the weights. Rejected: it makes the cadence a function of run health (a
diverging run that emits non-finite losses would advance the counter more
slowly, changing the target-update rhythm exactly when stability matters
most), and it breaks parity with SAC/DDPG/TD3, which already advance
unconditionally by explicit ADR 0056 §3 decision. Matching that precedent
(§Decision, point 4) is the correct generalisation, not a new rule.

## References

- Issue #334 — the field-semantics question this ADR's companion (0058)
  resolves; this ADR answers the unit question #334 itself does not ask.
- Issue #337 — target-update cadence *default values*, deferred here.
- Issue #182 — origin of the "unit trap" observation this ADR dissolves.
- ADR [0056](0056-non-finite-loss-skip-and-warn-guard.md) — the unconditional-
  counter-advance-on-skip precedent (§3) this ADR extends to the DQN family's
  new gradient-update counter.
- ADR [0058](0058-target-update-type-unifies-cadence-and-tau.md) — the
  `TargetUpdate` type whose `every` field this ADR fixes the unit of.
- `docs/.private/research/target-network-update-semantics.md` — the #182
  note; source of the "unit trap" table (Nature `C = 10,000` parameter
  updates ≡ 40,000 env steps vs. the workspace's `10_000` env-step default).
- `docs/.private/research/2026-07-24-issue-334-target-update-cadence-units.md`
  — the #334 note, §Q2; source of the per-source unit table (Mnih 2015,
  Hessel 2018/Rainbow, Haarnoja 2018a/b, Fujimoto 2018, SB3, CleanRL) and the
  verdict that gradient/parameter units are the literature-dominant choice.
- Code: `crates/rlevo-reinforcement-learning/src/algorithms/dqn/dqn_agent.rs:349-351,385-400,546-548`;
  `crates/rlevo-reinforcement-learning/src/algorithms/sac/sac_agent.rs:683,757-771`;
  `crates/rlevo-reinforcement-learning/src/algorithms/ddpg/ddpg_agent.rs:564,566-605`;
  `crates/rlevo-reinforcement-learning/src/algorithms/td3/td3_agent.rs:668,670-714`;
  `crates/rlevo-reinforcement-learning/src/algorithms/dqn/train.rs:137`;
  `crates/rlevo-reinforcement-learning/src/algorithms/c51/train.rs:130`;
  `crates/rlevo-reinforcement-learning/src/algorithms/qrdqn/train.rs:124`.
