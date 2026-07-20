---
project: rlevo
status: active
type: decision
date: 2026-07-20
tags: [adr, decision, reinforcement-learning, policy-head, validation, config, bounds, log-std, error-handling]
---

# ADR 0054: Policy-head construction is fallible

## Status

**Accepted (2026-07-20).** Resolves issue #386 ("SAC/PPO policy-head configs:
`Validate` implemented but never called on the construction path (`log_std`
bounds unenforced)"). Extends ADR 0026 (`Validate`/`ConfigError`) and ADR 0027
(`Bounds`); constrained by ADR 0011 (construction lives off the behavioural
trait, not on it); preserves ADR 0049's `log_std` bound semantics rather than
replacing them.

**Chosen shape:** every policy-head config in `rlevo-reinforcement-learning`
exposes exactly one public constructor, `try_init<B: Backend>(&self, device:
&B::Device) -> Result<Head, ConfigError>`, whose first statement is
`self.validate()?`. The infallible `init()` is **removed**, not retained
alongside — no unvalidated construction path exists once this lands. The rule
binds all policy-head configs uniformly, including the discrete/categorical
heads that carry no `log_std` bounds at all, so "does this head validate?" is
never a per-head question. Additionally, the two Gaussian head configs replace
their raw `log_std_min: f32` / `log_std_max: f32` pair with a single
`log_std: Bounds` field (ADR 0027), making the inverted-range case
**unrepresentable** rather than merely rejected.

## Context

### The defect

`Validate` is implemented on all four policy-head configs —
`SquashedGaussianPolicyHeadConfig` (SAC), `TanhGaussianPolicyHeadConfig`
(PPO), and the two discrete `CategoricalPolicyHeadConfig`s (PPO, PPG) — but is
never called on any production construction path. `init()` builds the head
straight off the raw fields; it does not validate. Every non-test
`.validate()` call in the crate targets a *training* config
(`sac_agent.rs`, `ppo_agent.rs`, `ppg_agent.rs`, and the three builder
`build()` paths) — precisely the configs that are **not** consumed to build a
head. Every head-config `.validate()` call, without exception, is inside
`#[cfg(test)]`.

`gaussian.rs` already carries a doc comment admitting the gap: a rejected head
"can still be built directly." The bypass was known and never closed. This
ADR closes it.

### The failure is backend-divergent — the sharpest argument for the change

An inverted `log_std_min > log_std_max` interval does not fail uniformly; it
fails **differently depending on which Burn backend evaluates it**, with no
common symptom to grep for:

- On the default `Autodiff<Flex>` clamp path, Burn's default `float_clamp`
  implementation is `clamp_min(clamp_max(x, max), min)`. With inverted
  bounds this silently **pins every `log σ` to `log_std_min`** — a
  gradient-dead policy collapse with no `NaN` and no panic. A run in this
  state looks like a stuck policy, not a bug.
- On the raw `Flex` path, the backend's overridden `float_clamp` delegates
  to `core::f32::clamp`, which **asserts `min <= max`** and panics
  immediately.

Same config, same op, silent corruption on one backend and a crash on the
other. Neither outcome is acceptable, and the divergence itself is the
argument: a validation gap that only ever panicked would already have been
caught by a test suite exercising any backend; this one hides depending on
which backend the caller happens to run.

### Relationship to existing ADRs

- **Extends ADR 0026.** `Validate`/`ConfigError` already exist and are
  already implemented on all four head configs — the gap this ADR closes is
  purely that `validate()` is never *called* on the path that matters. This
  ADR is a construction-chokepoint fix, not a new validation primitive.
- **Extends ADR 0027.** `Bounds` already makes `lo > hi` unrepresentable
  wherever it is held; the two Gaussian head configs did not yet hold one for
  `log_std`. Folding `log_std_min`/`log_std_max` into a single `log_std:
  Bounds` field removes the *ordering* check from `validate()` the same way
  ADR 0027 §3 describes for every other adopter — it does not remove
  `validate()` itself (see Consequences).
- **Constrained by ADR 0011.** Construction lives off the *behavioural*
  trait, not on it — `ConstructableEnv` is a standalone factory trait, not a
  supertrait of `Environment`. This ADR's rejected "agent constructors build
  the head internally" alternative would have re-coupled construction onto a
  behavioural policy trait the same way ADR 0011 explicitly declined to do
  for `Environment`; see Alternatives.
- **Preserves ADR 0049.** ADR 0049 established the *floor* (`log_std_min >=
  -35`) and *span* (`log_std_max - log_std_min < 40`) checks for PPO's
  Gaussian head, on top of ordering. This ADR does not touch those two
  checks — see Consequences for why `Bounds` cannot subsume them.
- **Related, not superseded: #185 and #385.** Both removed dead duplicate
  copies of these same bounds from the *training* configs
  (`SacTrainingConfig` and the PPO/PPG action configs, respectively) — the
  copy that was never read. This ADR concerns the surviving head-config
  copy, the one that *is* read. #185/#385 made the validation gap visible by
  removing the decoy that made it look covered; they did not create the gap.

## Decision

### 1. `try_init` is the one public constructor; `init` is removed

Each policy-head config gains:

```rust
pub fn try_init<B: Backend>(&self, device: &B::Device) -> Result<HeadType<B>, ConfigError> {
    self.validate()?;
    // ... existing init() body, unchanged ...
}
```

`self.validate()?` is the **first statement**, before any tensor or `Param`
is materialised. The previous infallible `init()` is deleted outright — it is
not kept as an unchecked convenience alongside `try_init`. Keeping both would
leave exactly the bypass this ADR exists to close: any caller reaching for
the shorter name skips validation again.

### 2. The rule binds all four head configs uniformly

`CategoricalPolicyHeadConfig` (PPO, PPG) has no `log_std` and nothing today
that fails to construct from an in-range struct literal — but it gets
`try_init`/`validate()` too, on the same signature as the two Gaussian
configs. The alternative — validate only the configs with a currently-known
failure mode — makes "does this head validate?" a fact a reader has to look
up per head, and quietly reopens the same bypass the moment a future
constraint is added to a discrete head (e.g. a nonzero action-count check)
without anyone remembering to wire it through construction. Uniformity is the
point: one answer, not four.

### 3. Gaussian heads: `log_std_min` / `log_std_max` → `log_std: Bounds`

`SquashedGaussianPolicyHeadConfig` and `TanhGaussianPolicyHeadConfig` drop
their raw `log_std_min: f32` / `log_std_max: f32` pair for a single
`log_std: Bounds` field (ADR 0027). This removes the *ordering* half of each
config's `validate()` — `Bounds` cannot hold `lo > hi` — while `validate()`
keeps the ADR 0049 floor and span checks, which `Bounds` cannot express (see
Consequences). The removal covers only half of the old check: `config::ordered`
was strict (`lo < hi`) and therefore also rejected the degenerate `lo == hi`,
whereas `Bounds::try_new` deliberately permits it. Both Gaussian configs
therefore carry an explicit `config::distinct(C, "log_std", ..)` line to
preserve the prior semantics — a zero-width `log σ` range is a silent σ
collapse (a frozen shared `Param` on PPO, a state-independent constant σ on
SAC), not a usable setting. This is a field-shape change to both configs' public surface
and their serialized form; accepted pre-1.0.

### 4. The method is named `try_init`, deliberately not `init() -> Result<..>`

Burn's own `*Config::init` idiom is infallible everywhere it appears —
roughly 30 instances across `burn-nn`, none of them fallible. This ADR
deliberately departs from that idiom under a **distinct** name,
`try_init`, rather than by changing `init`'s own return type to
`Result<_, _>`. The reasoning: the ecosystem owns the meaning of `init` —
every `burn-nn` config's `init(device) -> Module` is a promise that
construction cannot fail, and a caller reading `SomeConfig::init` anywhere
in a Burn codebase is entitled to rely on that promise. Nothing in this
workspace owns the right to redefine what `init` means for types that
otherwise look like every other Burn config. `try_init` is a new name for a
new (weaker) promise, and does not silently change the meaning of a name
Burn readers already trust.

## Consequences

### Positive

- The construction bypass is closed: there is no longer an infallible path
  from an invalid policy-head config to a constructed head, on any backend.
  The backend-divergent failure (silent collapse on one, panic on the
  other) cannot occur through `try_init`, because it never gets past
  `validate()`.
- The rule is uniform across all four heads, so a reviewer or a future author
  adding a fifth head config has exactly one convention to follow, not a
  choice.
- `Bounds` removes the *inverted*-range check as a hand-written line in two
  configs' `validate()`, per ADR 0027 §3 — one less place an inverted pair can
  be spelled correctly by construction, one less line to keep in sync with the
  invariant it enforces. The degenerate `lo == hi` case is not covered by
  `Bounds` and stays as an explicit `config::distinct` line in both (§Decision
  3).

### Negative / costs — do not soften these

- **rlevo's policy heads are now the one place in this workspace where
  Burn's `Config → init(device) → Module` chain does not compose.** Every
  other Burn-shaped config in this codebase and in the wider ecosystem
  constructs infallibly; these four require a `?` or a `.expect(..)` a
  caller must remember to add, forever, against a failure mode
  (`log_std_min > log_std_max`) with **zero occurrences anywhere in this
  repo today**. That is a permanent every-user tax paid against a defect
  that has never actually fired in production code. This ADR accepts that
  tax; it is not free, and stating it as free would misrepresent the
  tradeoff.
- **31 call sites migrated.** 10 outside the `rlevo-reinforcement-learning`
  crate (cross-crate integration tests, four benches, two examples) and 21
  inside it, all within `#[cfg(test)]`. Zero non-test call sites existed in
  `crates/rlevo-reinforcement-learning/src`, and no downstream consumer
  outside the workspace exists — this is entirely an in-repo migration, no
  partial/staged rollout is possible or needed.
- **Non-`Result` contexts use `.expect("valid head config")`.** Benches and
  examples cannot propagate `Result` through their harness signatures, so
  they call `try_init(..).expect("valid head config")` — the panic is now
  explicit and named at the call site rather than absent, but it is still a
  panic a reader must recognise as "this is fine, it's a fixed literal
  config in an example," not a genuine runtime hazard.
- **`Bounds` does NOT subsume `validate()`.** Adopting `log_std: Bounds`
  removes only the *ordering* check. ADR 0049's floor (`log_std_min >=
  -35`) and span (`< 40`) checks are not expressible as a `Bounds`
  invariant — `Bounds` only knows `lo <= hi`, not "a floor below which `σ`
  underflows to `0.0` in f32" or "a span past which a ratio of two `σ`s
  overflows." Both checks stay in `validate()`, with `Bounds::span()` now
  serving the span check directly instead of `hi - lo` computed by hand.
  ADR 0049's own counterexample makes the point concrete: `log_std_min =
  -120, log_std_max = -100` is a perfectly well-ordered `Bounds` (`-120 <=
  -100`) that still reaches `NaN`, because `exp(-110)` is exactly `0.0` in
  f32. A reader who assumes "it's a `Bounds` now, so it's safe" is wrong;
  `validate()` still does real work after this ADR, not less than it looks.

### Neutral

- The four head configs' `validate()` bodies are otherwise unchanged; only
  the inverted-range check moves into the `Bounds` field's own invariant (the
  degenerate case remains an explicit line), and the call site (`try_init`) is
  new.
- No schema/serde migration beyond the two Gaussian configs' field shape
  change (§Decision 3), which is already covered above.

## Alternatives considered

- **Panicking `init()` that calls `validate().expect(..)`.** The cheapest
  option — zero call-site churn — and genuinely better than the status quo,
  since it turns the backend-divergent failure into a uniform, early panic
  on every backend instead of a silent collapse on one of them. Rejected
  because ADR 0026 already rejected this exact shape by name: "reintroduces
  panic-on-deserialized-data — the exact §4 violation this ADR exists to
  remove." Adopting it here would require *superseding* ADR 0026 rather
  than extending it, and this ADR does not have grounds to do that. Recorded
  honestly: the margin is narrower than 0026's text on its own suggests —
  none of the four head configs derives `Deserialize` today, so the
  "user-supplied runtime data" trigger that motivates ADR 0026's rule is not
  currently *met* on 0026's own factual criterion. The decisive point that
  survives that narrowing is forward-looking, not present-tense:
  `#[derive(Deserialize)]` is a one-line addition that would silently turn a
  panicking `init()` into a genuine §4 violation with **no compiler
  signal** — nothing would fail to build, it would simply become wrong. And
  `Bounds`, adopted here per Decision §3, is itself `Deserialize`
  (`#[serde(try_from = "(f32, f32)")]`, ADR 0027 §4) — the moment a head
  config picks up `Deserialize` for any other field, the whole config
  becomes user-supplied runtime data under ADR 0026, and a panicking
  `init()` would already be non-compliant on day one of that change. Fallible
  construction is the version that does not depend on that not happening.
- **Docs-only convention** ("head configs should be validated before use,"
  no enforcement). Rejected on the same ground ADR 0026 already used to
  reject it: "exactly the drift that produced 87 un-validated configs" — and
  this issue *is* that drift arriving. A convention nobody's signature
  enforces is exactly how `validate()` ended up implemented on all four
  configs and called on none of them.
- **Agent constructors take the head config and build it internally**
  (`SacAgent::new`/`PpoAgent::new`/`PpgAgent::new` accept a head config
  instead of a pre-built head). Rejected on a structural ground, recorded
  carefully because it is not a strawman: these constructors are generic
  over **open** behavioural policy traits (`SquashedGaussianPolicy`,
  `PpoPolicy`), not over one concrete head type, so the agent cannot
  construct an arbitrary policy from a concrete head config without first
  knowing which concrete head the trait's implementor actually is. Making
  it work would require welding an associated-config or
  `BuildableFrom<Cfg>` bound onto the *behavioural* policy trait — precisely
  what ADR 0011 removed from `Environment` (construction off the
  behavioural trait, onto a standalone factory), and it would bite the same
  way: wrapper policies and test mocks would need degenerate construction
  stubs purely to satisfy the bound, the exact failure mode ADR 0011's
  `RecordingTap`/`TuiEnvTap` stubs already demonstrated once. It also would
  not actually close the seam this ADR closes, since the head configs stay
  `pub` with `pub` fields either way — a caller could still struct-literal
  one and hand it to something else that skips the agent constructor
  entirely.
- **A `PolicyHeadConfig` trait carrying `try_init`.** Deferred, not
  rejected. A shared trait would give "every future head must validate" real
  compiler teeth — a fifth head config that forgets `try_init` would fail to
  satisfy the trait bound wherever one is required, rather than silently
  compiling with a bare inherent method missing. But it adds a generic seam
  that nothing in the workspace currently consumes: there is no code today
  that is generic over "some policy-head config," only four concrete types
  each reached from one concrete agent. The decision here is inherent
  `try_init` methods now, with the trait revisited the moment a fifth head
  config actually appears and a real generic consumer needs the bound. This
  is recorded as an explicitly deferred one-way-door decision, not a closed
  question: adding the trait later is additive; not needing it because a
  fifth head never appears costs nothing.

## References

- Issue #386 — "SAC/PPO policy-head configs: `Validate` implemented but
  never called on the construction path (`log_std` bounds unenforced)."
- Issue #185 — removed `SacTrainingConfig`'s dead duplicate `log_std_min`/
  `log_std_max` fields; named the convention this ADR's Decision §2 follows
  (bounds live on the config that is actually consumed).
- Issue #385 — removed the same dead-duplicate-bounds pattern from the
  PPO/PPG action configs; together with #185, made this ADR's gap visible
  without creating it.
- ADR [0011](0011-lift-construction-off-environment-trait.md) — construction
  lives off the behavioural trait, on a standalone factory; the shape the
  rejected "agent constructors build the head" alternative would have
  undone.
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate` /
  `ConfigError` convention this ADR enforces at the construction chokepoint
  it was missing from; source of the panicking-`init()` rejection this ADR
  reapplies.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the `Bounds`
  newtype `log_std` migrates onto; source of the "`Bounds` complements
  `Validate`, it does not replace it" rule this ADR's Consequences restates
  for the floor/span checks specifically.
- ADR [0049](0049-ppo-gaussian-log-std-is-bounded.md) — the floor (`>=
  -35`) and span (`< 40`) checks this ADR preserves unchanged, and the
  `log_std_min = -120, log_std_max = -100` counterexample this ADR's
  Consequences reuses to show `Bounds` alone is insufficient.
- Code: the four policy-head config modules in
  `crates/rlevo-reinforcement-learning/src/algorithms/` —
  `sac/sac_policy.rs` (`SquashedGaussianPolicyHeadConfig`),
  `ppo/policies/gaussian.rs` (`TanhGaussianPolicyHeadConfig`, and the doc
  comment acknowledging a rejected head "can still be built directly"),
  and the PPO/PPG `policies/categorical.rs` modules
  (`CategoricalPolicyHeadConfig`); the three agent constructors
  (`sac_agent.rs`, `ppo_agent.rs`, `ppg_agent.rs`) whose `.validate()` calls
  target the training config, not the head config.
