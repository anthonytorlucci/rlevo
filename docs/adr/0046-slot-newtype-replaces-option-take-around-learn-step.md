---
project: rlevo
status: active
type: decision
date: 2026-07-17
tags: [adr, decision, reinforcement-learning, burn, panic-safety, optimizer, issue-167]
---

# ADR 0046: `Slot<M>` replaces `Option<M>` + wide `take()` around a learn step

## Status

**Accepted (2026-07-17).** Resolves issue #167 ("[rl] Panic safety:
`Option::take()` on target/policy nets bricks the agent on any mid-learn-step
panic"). Supersedes nothing.

## Context

All eight gradient-based agents in `rlevo-reinforcement-learning`
(`dqn`, `c51`, `qrdqn`, `ddpg`, `td3`, `sac`, `ppo`, `ppg`) stored every
trainable network as a plain `Option<M>` field and, at the top of a learn
step, wrote:

```rust
let net = self.net.take().expect("net present");   // field is now None
let loss = net.forward(batch);                      // ← any panic here
let grads = loss.backward();                        // ← or here
let grads = GradientsParams::from_grads(grads, &net);
self.net = Some(self.optimizer.step(lr, net, grads));
```

The field was `None` for the **entire** window from `take()` to the
write-back — forward pass, loss computation, `backward()`, and
`GradientsParams::from_grads` all ran while the network was not in the
struct. Any panic anywhere in that window — a shape mismatch, a device
transfer failure, a NaN assertion, an OOM — left the field permanently
`None`. There was no recovery path: every subsequent `act()` and `learn_step()`
hit its own `.as_ref().expect(...)` and panicked again, with a message naming
the wrong method. 17 call sites across the eight agents shared this shape.

Seven of the eight agents' networks were reviewed independently (one
maintainer per file); four of those seven reviews independently proposed an
RAII drop-guard or `std::panic::catch_unwind` as the fix. TD3's review put it
most strongly: a drop-guard was called "the only way to make `learn_step`
truly panic-safe." Both proposals are provably wrong for this specific
failure mode, and the wrongness is not obvious without reading the exact
signature of the API being wrapped — hence this ADR.

### Why a drop-guard and `catch_unwind` cannot close the window

Burn's `Optimizer::step` **consumes the module by value**
(verified — `burn-optim-0.21.0/src/optim/base.rs:77`):

```rust
fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M;
```

Once `step` is called, the module lives inside `step`'s stack frame, not the
caller's. If `step` panics, the module is dropped during unwinding **before
control ever returns to the caller** — there is no owned value left anywhere
for a guard to restore or for `catch_unwind` to recover.

- **A drop-guard can only restore a value it still holds.** By the time
  `step` is invoked, the caller has already moved the module out; the guard
  has nothing left in its hand to give back.
- **`catch_unwind` only tells you the call failed.** It does not resurrect a
  value that was moved into the panicking frame and dropped there. `step`
  returns `M` on its one success path and nothing on any other path.
- **The only mechanism that could work is cloning the weights before every
  step**, so there is a spare copy to restore from regardless of what
  happens inside `step`. This is rejected: it is a full copy of every
  parameter tensor on **every step of the hot training loop**, paid to guard
  against a failure (device OOM, a NaN assertion, a shape bug) that already
  means the run is dead. The cost is not proportional to the risk.

So a panic **strictly inside** `Optimizer::step` is an **irreducible,
by-design residual window**. It poisons the slot, and the read side reports
that honestly rather than papering over it. This is a deliberate trade, not
an oversight left for a future contributor to "finish."

### The real fix is forward-before-take, centralized

Two Burn facts make it possible to shrink the window to exactly the call to
`step` itself, with no ownership transfer required for anything before it:

- `forward(&self, ...)` on every model in this crate takes `&self`. The
  forward pass and the loss it produces need only a borrow.
- `GradientsParams::from_grads(grads, module: &M) -> Self`
  (`burn-optim-0.21.0/src/optim/grads.rs:33`) takes a **reference** and
  carries **no lifetime parameter** on the return type — so under NLL, the
  borrow of the module ends at `from_grads`'s last use, well before `&mut
  self` is needed for the actual `step` call.

So `forward`, the loss, `backward()`, and `from_grads` can all run against a
`&M` borrow. The *only* point at which the module must be owned, rather than
borrowed, is the call into `Optimizer::step` itself — which is exactly the
irreducible window identified above, and nothing wider.

## Decision

### 1. `pub(crate) struct Slot<M>(Option<M>)` in `algorithms/shared.rs`

`Slot<M>` is a newtype over `Option<M>` whose entire purpose is to make the
wide "module is temporarily nowhere" window **unrepresentable**, per the
usual `rules.md` posture (mirrors the private-field-plus-accessor pattern of
ADR 0026/0027/0039). It exposes exactly three operations:

- `new(module: M) -> Self` — never empty at birth; there is no `Default` and
  no dangling constructor.
- `get(&self) -> &M` — the **single read path**, and the only read-side
  `.expect(...)` in the crate (previously there were 17, one hand-copied
  into every agent). Used for `forward`, for
  `GradientsParams::from_grads`, and for post-step reads such as refreshing
  a target network via `self.net.get().valid()` — every one of those
  consumers takes `&self`, so an owned module was never actually needed on
  the read side.
- `step_with<B, O>(&mut self, opt: &mut O, lr: LearningRate, grads:
  GradientsParams)` — moves the module out, hands it to `Optimizer::step`
  (which consumes it), and writes the result back. This is the **only**
  place a module ever leaves the field, and it does so for the duration of
  one call and nothing else.

`step_with` is deliberately **closure-free**. An `update_with(f: impl
FnOnce(M) -> M)` would look like the "flexible" version of the same idea,
and it would reopen exactly the hole this type exists to close: a caller
could put `forward`/loss/`backward` back *inside* the closure, i.e. back
inside the region where the module is out of the slot, and every panic
there would poison the agent again. The absence of a closure-taking
variant is load-bearing, not an oversight.

### 2. Every agent reorders to forward-before-take

All eight agents (`dqn`, `c51`, `qrdqn`, `ddpg`, `td3`, `sac`, `ppo`, `ppg`)
now hold their trainable networks as `Slot<M>` fields and follow the same
shape at every learn-step call site:

```rust
let loss = self.net.get().forward(batch);
let grads = loss.backward();
let grads = GradientsParams::from_grads(grads, self.net.get());
self.net.step_with(&mut self.optimizer, lr, grads);
```

Every realistic panic source — a malformed batch, a device transfer
failure, a shape mismatch in the forward pass, a NaN assertion in the loss
— now happens while the module is borrowed, not moved. The field is never
observably absent at any point a caller (including a panic handler
unwinding through this code) can inspect it, except during the literal
`step` call.

### 3. The convention for a ninth algorithm

Any future agent added to this crate holds its trainable network(s) in a
`Slot<M>`, does all forward/loss/backward/`from_grads` work against
`Slot::get()`, and calls `Slot::step_with` as the sole point of ownership
transfer. This is now the crate's one shape for "a network trained via a
Burn optimizer," not a pattern to be reinvented per algorithm.

### 4. The `Slot::get()` poison message names the cause and the remedy, not a method

Agents disagree on what to call their learn step (`learn_step` for the DQN
family and DDPG/TD3/SAC, `policy_phase_update`/`maybe_aux_phase` for PPG,
`update` for PPO), so a poisoned-slot panic message cannot correctly name
"the failing method" — half the crate would get the wrong name. The shared
message instead names the mechanism ("poisoned by a panic inside an
earlier optimizer step ... the module was moved into `step` and lost during
unwinding") and the remedy ("rebuild the agent from a fresh network"),
which is true regardless of which agent or which method hit it. This
incidentally fixes a real, pre-existing defect: PPO and PPG's old messages
literally named `learn_step`, a method that does not exist on either
agent.

## Consequences

### Positive

- The wide bricking window is closed for all 17 former call sites across
  all eight agents. A panic in the realistic failure region (forward,
  loss, backward, host-tensor reads inside `from_grads`) is now fully
  recoverable — the agent's networks are untouched and training can
  continue after a caller catches the unwind.
- **A latent multi-field bug is fixed as a side effect.** TD3 and SAC both
  took *both* critics (`critic_1`, `critic_2`) out of their fields up
  front and stepped them sequentially before writing either back. A panic
  stepping `critic_1` therefore destroyed `critic_2` in the very same
  unwind — a single failure bricking two networks, not one. Under `Slot`,
  `critic_1.step_with(...)` and `critic_2.step_with(...)` are independent,
  disjoint windows: a panic inside one leaves the other (and the actor)
  fully intact and usable. This was not caught by any of the seven
  file-level reviews; it fell out of applying the same fix uniformly.
- Read-side `.expect(...)` calls collapse from 17 hand-copied sites (one
  per agent field) to one, inside `Slot::get()`. A future change to the
  poison message, or to the panic condition itself, is now a one-line edit
  instead of an eight-file sweep.
- The residual, irreducible window (a panic strictly inside
  `Optimizer::step`) is now honestly reported: `Slot::get()` panics with a
  message naming the cause and the remedy, rather than a generic
  `Option::unwrap` failure pointing nowhere useful.
- **Behaviour-preserving.** `take()`/`Slot::step_with` is a pure host-side
  move — no tensor operation, no operation order, and no RNG draw changed.
  Every migrating engineer independently A/B'd seeded training runs
  against pristine `HEAD` and got bit-identical output: DDPG (101 values),
  SAC (50 learn steps, including α auto-tune and Polyak averaging), PPO
  and PPG (114 values each), TD3 (51 values), and DQN/C51/QR-DQN (299
  values each).

### Negative / accepted costs

- **The residual window is permanent by design.** A panic strictly inside
  `Optimizer::step` still poisons the `Slot` forever; there is no code
  path that recovers from it short of rebuilding the agent. This is
  accepted, not deferred — see Alternatives considered for why the two
  usual fixes (drop-guard, `catch_unwind`) cannot touch it, and why
  cloning the weights every step to hedge against it is not worth the
  cost.
- **`pub(crate)` scope means no downstream crate benefits directly.**
  `Slot` is internal to `rlevo-reinforcement-learning`; a hypothetical
  future crate implementing its own Burn-trained agent outside this crate
  would need to reinvent it (or this crate would need to make it `pub`,
  which is out of scope here — no external consumer exists today).

## Alternatives considered

- **An RAII drop-guard that restores the module on unwind.** Rejected: the
  module is moved into `Optimizer::step`'s own frame before it can panic;
  a guard constructed by the caller no longer holds anything to restore
  once that move has happened. This was independently proposed in four of
  the seven per-agent code reviews and is provably incapable of closing
  the actual window, which is documented here specifically so it is not
  re-proposed for a ninth algorithm.
- **`std::panic::catch_unwind` around the optimizer step.** Rejected for
  the same reason: catching the unwind confirms the step failed but does
  not hand back a module that was moved-and-dropped inside the callee's
  frame. There is nothing to catch that contains the value.
- **Clone the module's weights before every step, to have a spare to
  restore.** The only mechanism that is actually correct. Rejected on
  cost: it is a full parameter-tensor copy on every iteration of the hot
  training loop, to defend against a failure mode (OOM, NaN assertion,
  device loss) that already means the run cannot continue meaningfully.
- **Drop `Option`/`Slot` entirely in favor of `std::mem::take` on a
  `Default`-implementing module.** Proposed in SAC's review. Rejected on
  two independent grounds, one fatal on its own: `#[derive(Module)]` does
  **not** generate a `Default` impl (verified), so this does not compile
  as stated. And even supposing a hand-written `Default` existed, using
  `mem::take` would leave a **default-weighted, untrained module sitting
  in the field** on the success path of every step until the real value is
  written back — turning a loud, immediately diagnosable panic into
  **silent training corruption** that a caller would only notice much
  later, if at all, when returns stopped improving. Strictly worse than
  the pre-existing bricking behaviour, which at least fails loudly.
- **Change `learn_step`'s signature to return `Result<_, E>`.** Proposed
  in several reviews as a general robustness improvement. Rejected as the
  fix for *this* issue specifically: a `Result` return cannot catch a
  panic — the two are orthogonal error-handling mechanisms, and adding one
  does nothing to shrink the ownership window a panic unwinds through. It
  is also a breaking public API change across eight agents, which this fix
  does not require. Deferred to its own issue if wanted.
- **Leave the wide `take()` window in place and merely improve the panic
  message.** Rejected: it treats the symptom (an unhelpful message) while
  leaving the actual defect (agent permanently unusable after any
  mid-step panic) in place.

## References

- Issue #167 — "[rl] Panic safety: `Option::take()` on target/policy nets
  bricks the agent on any mid-learn-step panic."
- ADR [0026](0026-shared-config-validation-convention.md) /
  [0027](0027-bounds-newtype-for-closed-ranges.md) /
  [0039](0039-box2d-states-own-markov-dofs.md) — the
  make-invalid-states-unrepresentable posture (private field behind
  accessors and a narrow, checked construction path) that `Slot` follows.
- Code: `crates/rlevo-reinforcement-learning/src/algorithms/shared.rs`
  (`Slot<M>`, its module-level docs, and the poison message); every agent
  under `crates/rlevo-reinforcement-learning/src/algorithms/{dqn,c51,qrdqn,
  ddpg,td3,sac,ppo,ppg}/`; `burn-optim-0.21.0/src/optim/base.rs:77`
  (`Optimizer::step` consumes `module: M` by value); `burn-optim-0.21.0/
  src/optim/grads.rs:33` (`GradientsParams::from_grads(grads, module: &M)`
  — reference argument, no lifetime parameter on the return type).
