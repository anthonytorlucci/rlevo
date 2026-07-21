---
project: rlevo
status: active
type: decision
date: 2026-07-21
tags: [adr, decision, polyak, soft-update, target-network, dqn, c51, qrdqn, ddpg, td3, sac, issue-341, issue-317]
---

# ADR 0057: The target soft-update path is fallible — `PolyakError`, `soft_update`, and off-policy `learn_step` return `Result`

## Status

**Accepted (2026-07-21).** Resolves issue #341 ("[rl] `polyak_update` panics on
`ParamId` mismatch and silently mis-updates on subset mismatch; tied weights hit
both") and **partially** addresses issue #317 (surface the target-update failure
mode as a recoverable error rather than a panic) for the six off-policy agents.

**References** ADR 0046 — it does **not** supersede it. ADR 0046 rejected a
`Result` return on `learn_step` *as a panic-safety mechanism for #167* and
**explicitly deferred this change to its own issue** (`0046`, "Change
`learn_step`'s signature to return `Result<_, E>`", lines ~251-257: "Rejected as
the fix for *this* issue specifically … Deferred to its own issue if wanted").
This ADR is that issue. The two are orthogonal: 0046 shrank the ownership window
a *panic* unwinds through; 0057 turns a *specific, recoverable* configuration
error into a typed `Result` instead of a crash. Nothing in 0046 is reversed.

## Context

`polyak_update` (`utils.rs`) pairs `active` and `target` parameters by
[`ParamId`], not by position or name. A target network built by cloning the
active one shares its ids and blends correctly. A target built independently (a
fresh `init` mints new ids) does not match, and three distinct topology defects
were reachable — all three of which a real *tied-weights* module (two fields
holding clones of one `Param`, one shared id) triggers at once:

1. **`ParamId` mismatch (defect 1).** A target parameter with no counterpart in
   active. The pre-#341 code `panic!`ed inside the module mapper.
2. **Double-consume on tied weights (defect 2).** The mapper used `.remove()`,
   so the *second* field sharing an id found it already consumed and panicked
   even for a correctly-cloned target. Fixed under #341 by switching to
   `.get().cloned()` plus a `seen: HashSet<ParamId>` exhaustion check.
3. **Silent strict-subset partial update (defect 3).** When active carried an id
   that no target field consumed, the overlap was blended and the leftover
   **silently dropped** — a partial target update with no signal.

The #341 branch landed the *correctness* half (defects 2 and 3 detected) but in
a **panic** form: `panic!` on defect 1, `assert!` on defect 3. Arithmetic-only
tests missed all of this because they only ever exercised correctly-cloned
fixtures (`fixture()` shares ids by construction), so the id-topology paths were
never entered — the failure lived entirely in how *mismatched* modules were
handled, which no blend-value assertion can observe.

A `ParamId`-topology mismatch is a **configuration error** — the caller built
the target network wrong — not an unrecoverable invariant violation. It is
detectable, nameable, and the natural response is to reject the update and let
the caller fix construction, not to abort the process.

## Decision

Make the whole target soft-update path fallible with a typed error.

- **`PolyakError`** (new, `utils.rs`) — `Copy + PartialEq + Eq`, `thiserror`:
  - `MissingActive(ParamId)` — a target param absent from active (defect 1).
  - `MissingTarget(ParamId)` — an active param never consumed by any target
    field, i.e. target is a strict subset of active (defect 3). Reported as the
    **smallest** leftover id (`ParamId: Ord`, `.min()`) so the error is
    deterministic and testable.
- **`polyak_update` → `Result<M, PolyakError>`.** The mapper records the *first*
  `MissingActive` miss and returns the parameter untouched (the `ModuleMapper`
  method is infallible, so it cannot short-circuit mid-walk); `polyak_update`
  surfaces that recorded error, then the exhaustion check, after the walk.
- **`soft_update` → `Result<Self::InnerModule, PolyakError>`** on all five model
  trait declarations (`DqnModel`, `C51Model`, `QrDqnModel`, DDPG's
  `DeterministicPolicy` actor and `ContinuousQ` critic; SAC/TD3 reuse the DDPG
  traits). Every impl returns the `polyak_update(...)` `Result` directly.
- **`learn_step` → `Result<Option<LearnOutcome>, XAgentError>`** on the six
  off-policy agents (DQN, C51, QR-DQN, DDPG, TD3, SAC). `Ok(None)` = step skipped
  (warm-up or non-finite loss, per ADR 0056); `Ok(Some(o))` = applied; `Err` =
  failure. Each per-agent error enum gains one `#[error(transparent)]
  Polyak(#[from] PolyakError)` variant. Soft-update call sites use `?`.

The `.clone()` at each `soft_update` call site is retained: `soft_update`
consumes `target` by value, so on `Err` the `?` returns *before* the field
reassignment and the target keeps its prior weights — no silent hard-sync onto
the live network. The invariant that held for a panic (early unwind before
reassignment) now holds equally for an early `Result` return.

## Why `Result` over panic

- A param-topology mismatch is a **recoverable config error**: the caller can
  rebuild the target by cloning active and retry. A panic makes that
  indistinguishable from a genuine bug and unwinds a whole training run.
- It matches ADR 0056's **skip-don't-crash** posture for the learn step: 0056
  already made `learn_step` a "may or may not apply an update this step" call
  (`Ok(None)` on a non-finite loss); returning `Result<Option<_>, _>` extends
  the same surface with a third, honestly-typed outcome rather than a crash.
- The error is **part of the public contract**: `PolyakError` names the offending
  `ParamId`, so a caller (or test) can assert on the exact defect.

## Scope

Fallible: the off-policy target soft-update path only. **Out of scope, still
panic-based (residual under #317):** `act()` on every agent, and the on-policy
PPO/PPG agents (no target network, no `polyak_update`). The device→host→device
round-trip inside `polyak_update` (#322) is untouched.

## Consequences

- **Breaking API change** across the six off-policy agents and the model traits.
  In-tree callers add `?` (train loops already return the agent error); direct
  callers double-unwrap (`.expect("no polyak error")` then handle the `Option`).
  Every in-tree target is built by cloning, so `Err` is unreachable in practice
  — the change is a contract/robustness improvement, not a behaviour change on
  the healthy path. No persisted data is affected.
- The three #341 topology tests move from `#[should_panic]` to exact
  `assert_eq!` on `PolyakError` variant + id (now possible since
  `PolyakError: PartialEq`), which is a strictly stronger assertion.

### Known limitation: the multi-target sequence is not atomic

DDPG/TD3/SAC apply two to three independent `soft_update(...)?` calls per
`learn_step` (actor + one or two critics). *Each* assignment is atomic — the
target is cloned in, blended, and reassigned only on `Ok`, so an `Err` returns
via `?` before that field is touched — but the *sequence* of calls is not. If
the k-th call returns `Err`, the first k-1 target fields have already been
updated and the remaining ones have not. This is unreachable for in-tree agents:
every target on a given agent is cloned identically from its policy net, so they
all share the same mismatch-or-not verdict, and the sequence either fully
succeeds or fails on the first call. A caller that *catches* the error and
continues training anyway, however, could observe partially-advanced targets.
Pre-`Result` this was moot — a panic unwound the entire `learn_step` call, so no
partial state was observable. We accept this as a documented limitation rather
than adding all-or-nothing staging (stage every blended target, then commit the
batch), which would buy nothing on the healthy path and only matters for a
caller that already stepped outside the contract by swallowing a config error.

## References

- Issue #341 — "[rl] `polyak_update` panics on `ParamId` mismatch and silently
  mis-updates on subset mismatch; tied weights hit both".
- Issue #317 — surface the target-update failure mode as a recoverable error.
- ADR 0046 — `Slot` take/restore window; deferred this `Result` change by name.
- ADR 0056 — non-finite loss ⇒ skip the step; the skip-don't-crash posture this
  ADR extends to the topology-mismatch case.
