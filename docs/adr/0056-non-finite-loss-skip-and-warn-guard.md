---
project: rlevo
status: active
type: decision
date: 2026-07-21
tags: [adr, decision, numerical-stability, nan, loss, ppo, ppg, dqn, c51, qrdqn, sac, ddpg, td3, issue-318]
---

# ADR 0056: Non-finite loss ⇒ skip the step + warn once, keyed on the already-host-resident loss scalar

## Status

**Accepted (2026-07-21).** Resolves issue #318 ("[rl] NaN silently propagates
through training instead of surfacing"). **Generalizes** the SAC-α optimizer
guard (#184, `sac_alpha.rs:248`) from one hand-rolled optimizer to every agent's
learn step. **Builds on** ADR 0046 (`Slot` take/restore window) by placing the
guard strictly *before* that window is entered. Purely additive — supersedes
nothing, changes no public API, changes no healthy-step numerics.

**Chosen shape.** A `FiniteLossGuard` in `algorithms/shared.rs` performs a
host-side `f32::is_finite()` check on the loss scalar every agent already reads
for metrics *before* `backward()`. On a non-finite value it makes the caller
**skip `backward()` and the optimizer step** for that step, and fires a
**one-shot `tracing::warn!`**. The skip runs **every** occurrence; only the
`warn!` is latched. It runs **unconditionally in release** — the loss scalar is
already host-resident at every site, so the check adds **no** device→host sync.
Each distinct loss site owns its own latch (one failure mode must not silence
another's diagnostic). Skipped values are excluded from epoch-mean accumulators
so a single NaN cannot re-poison an otherwise-finite reported mean.

## Context

Burn does not panic on NaN — it propagates it silently (unlike the #167 defect,
which failed loudly). A NaN entering a loss (PPO `ratio = exp(new−old)` overflow
when `new−old > ~88`; degenerate `log`/`div` in an entropy or log-prob term;
exploding gradients) is folded into the weights by the optimizer step. Training
continues, reports finite-looking bookkeeping, and learns nothing. Existing tests
miss it structurally: cross-crate `*_produces_finite_rewards` checks *reward*
finiteness (a fully NaN-poisoned network still emits finite rewards), and
reproducibility tests only assert same-seed self-consistency (a deterministic
NaN reproduces perfectly and passes).

The load-bearing fact that decides the design: **every agent already reads its
loss scalar host-side (`into_scalar().elem::<f32>()`) before calling
`loss.backward()`** — for metrics. So the guard rides an existing sync. The
issue's own "a per-step host sync may be too costly for release builds" caveat is
a false premise at these sites; #184's comment thread established the same for the
loss scalar specifically.

### Literature

Skip-the-step-and-continue (not abort, not sanitize) is the **canonical** response
shape. PyTorch AMP `GradScaler` skips `optimizer.step()` when unscaled gradients
contain inf/NaN "so the params themselves remain uncorrupted" and continues [1];
its origin is Micikevicius et al., *Mixed Precision Training*, ICLR 2018 — "skip
the weight update when an overflow is detected and simply move on to the next
iteration" [2]. Provenance and full citations:
`docs/.private/research/2026-07-21-issue-318-nonfinite-loss-guard.md`.

## Decision

1. **Home & shape.** `FiniteLossGuard { warned: bool, label: &'static str }` in
   `algorithms/shared.rs`, with `check(&mut self, loss: f32) -> bool` (returns
   `true` ⇒ proceed to backward+step; `false` ⇒ skip both) and a
   `#[cfg(test)] warning_fired(&self) -> bool` accessor mirroring
   `sac_alpha.rs:385`. Each agent holds one guard field per distinct loss site.

2. **Guard sits before `backward()`, never inside `Slot::step_with`.** The point
   is to skip the backward graph traversal that turns a NaN loss into NaN grads;
   by the time grads exist, poisoning has already happened. This also keeps
   `Slot::step_with` closure-free and single-purpose, and strictly *strengthens*
   ADR 0046's invariant: on a skip the `Slot` is never emptied, so the
   take/restore poison window is never entered.

3. **Skip semantics.** The **skip re-fires on every non-finite occurrence** — a
   run that emits NaN every step must be protected every step. Latching the skip
   would be a correctness defect. Only the `warn!` is one-shot (per-run, per loss
   site). Cadence/LR/iteration counters advance independently of skips (the
   actor/α cadence is a rhythm over update *attempts*; LR anneals per `update`
   call, not per minibatch). A skipped loss value is **excluded** from its
   epoch-mean accumulator (denominator counts healthy contributions; `denom==0`
   is guarded) so the reported mean is not re-poisoned — the `warn!`, not a NaN
   metric, is the surfacing mechanism, matching #184 which added no stats field.

4. **Runs unconditionally in release.** No `debug_assert`/config gate: the host
   read already exists, so a gate would strip the guard from exactly the long
   release runs that diverge.

5. **Scope: all 8 agents, one latch per loss site (17 total).** PPO (policy,
   value); PPG (policy, value, aux-value, aux-total — the aux-total input is
   host-*derived* from its two already-read summands, adding no sync); DQN; C51;
   QR-DQN; SAC (critic_1, critic_2, actor — α keeps its own #184 guard); DDPG
   (critic, actor); TD3 (critic_1, critic_2, actor).

## Consequences & honest limits

- **This is a loss-level *proxy*, not a gradient-level check.** It fully
  *prevents* poisoning when the NaN originates in the loss computation — the
  dominant real-world mode (PPO `exp` overflow, `log(0)`, TD-target blowup):
  `backward()` never runs, weights stay clean, the next minibatch can recover.
  It is **blind to finite-loss → NaN-gradient** cases (`torch.where`/masked
  `0·∞` in `Normal`/`Categorical` backward; PyTorch issues #16317, #52248,
  #156212 [4]); there it catches the NaN one step *late*, once poisoned weights
  make the next loss NaN, and can then only surface-and-stop, not recover. That
  gradient-origin case, and grad-norm handling generally, are **#328's**
  territory (grad-norm clipping is itself *not* a NaN guard: a NaN element makes
  the norm NaN). Surface-and-stop is still strictly better than today's silent
  propagation.
- **Out of scope, deliberately:** reward finiteness at `remember` (**#352**);
  grad-finiteness / grad-norm clipping (**#328**); a PPO `log_ratio` clamp before
  `exp` (non-canonical — no reference PPO impl clamps it [7][8][9][10]; and
  `min(unclamped, clamped)` masks `Inf` but not `NaN`, so the loss guard is the
  correct catch-all). The existing SAC-α guard is **not** refactored into
  `FiniteLossGuard`: it guards closed-form Adam moment overflow with bespoke
  messages, a different shape in a different module.

## References

[1] PyTorch AMP `GradScaler` — https://docs.pytorch.org/docs/stable/amp.html
[2] Micikevicius et al., *Mixed Precision Training*, ICLR 2018, arXiv:1710.03740
[4] PyTorch `torch.where` backward-NaN cluster (issue #156212)
[7]–[10] OpenAI Baselines ppo2, CleanRL, Stable-Baselines3, Tianshou (none clamp `log_ratio`)

Full citation list in the vault research note referenced under *Context → Literature*.
