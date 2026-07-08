---
project: rlevo
status: active
type: decision
date: 2026-07-09
tags: [adr, decision, testing, proptest, property-testing, rng, seeding, rlevo-evolution]
---

# ADR 0036: Adopt `proptest` for property/invariant tests

## Status

**Accepted (2026-07-09).** Filed under issue #239 (adopt `proptest` as a
dev-dependency). Lands the dependency-wiring and convention foundation only; the
property tests themselves are authored separately. **Builds on ADR
[0029](0029-host-rng-seeding-convention.md)** — the host-RNG `seed_stream`
boundary is the load-bearing rule that keeps proptest's generative model from
touching Burn's process-wide RNG. Complements, does not replace, the existing
seeded-`StdRng` / `seed_stream` example-based test convention.

## Context

`rlevo-evolution`'s test suite is entirely example-based: a fixed seed feeds
`seed_stream`, the algorithm runs, and specific assertions check a known outcome
(`*_converges_on_sphere_d10`, known-answer operator results, regression cases).
This is the right shape for *specific scenarios*, but it under-covers the
*input space*: an operator or `Strategy<B>` should uphold invariants (finiteness,
shape, length, ordering, roundtrip identity, `Validate` accept/reject) across a
**range** of configs, sizes, and seeds — not just the handful a maintainer hand-
picked. Property-based testing (generate many valid inputs, assert an invariant,
shrink any counterexample to a minimal reproducer) is the standard tool for that
gap, and `proptest` is the mature Rust choice.

The hazard is the interaction with two existing conventions:

1. **The host-RNG seeding convention (ADR 0029).** All EA randomness is host-
   sampled through `seed_stream(base, generation, SeedPurpose)`. `B::seed` +
   `Tensor::random` is banned because Burn's process-wide RNG mutex makes tensor-
   sampled draws depend on thread schedule, not the seed — which raced the
   parallel test runner. A property framework that reached for `Tensor::random`
   to fabricate inputs would reintroduce exactly that flakiness.

2. **Process forking.** `proptest`'s default features pull `fork` and `timeout`
   (via `rusty-fork` / `tempfile` / `wait-timeout`), running each case in a
   forked subprocess. In this workspace a forked subprocess would re-initialise
   the Burn backend alongside the rayon threadpool / GPU state — expensive and
   hazardous.

## Decision

### 1. Workspace-declared, `rlevo-evolution`-only dev-dependency

`proptest` is declared once in the root `[workspace.dependencies]` (under the
`# Testing` group, next to `approx`) and consumed only by `rlevo-evolution`'s
`[dev-dependencies]`. It is not added to any other crate. Promote elsewhere only
when a concrete second need appears.

### 2. Version and features — defaults stripped

```toml
proptest = { version = "1", default-features = false, features = ["std", "bit-set"] }
```

Default features are dropped to avoid `fork` / `timeout` → `rusty-fork` /
`tempfile` / `wait-timeout` (per-case process forking would re-init the Burn
backend in a subprocess — see Context). `std` is kept (the workspace is not
`no_std`); `bit-set` is kept because it improves shrinking of index/collection
strategies.

### 3. The RNG boundary is a HARD RULE (cite ADR 0029)

**proptest generates host config only** — `λ`, `D`, structural sizes, and a
`seed: u64`. The test body routes **all** algorithm randomness through
`seed_stream(seed, generation, SeedPurpose::_)`, exactly as production does.
`B::seed` + `Tensor::random` is forbidden in property bodies as it is everywhere
else (ADR 0029). **proptest's own PRNG never touches Burn** — it only fabricates
the host config that then seeds the deterministic stream.

Because `fork` is disabled, a property's cases run **sequentially** within one
process — no new intra-property parallelism is introduced, so there is no
worsening of the process-wide-RNG-mutex race ADR 0029 guards against.

### 4. When proptest vs seeded-example (the boundary rule)

- **proptest** — for INPUT-SPACE properties: invariants that must hold across a
  range of configs / sizes / seeds. Roundtrips, shape/length invariants,
  monotonicity, "no panic / no NaN across the valid domain", and `Validate`
  accept/reject boundaries.
- **Seeded `StdRng` example tests** — for SPECIFIC-SCENARIO assertions:
  known-answer, `*_converges_on_sphere_d10`, regression cases. Keep these.
  **Do not** rewrite passing example tests into properties.

Strategies **must be bounded to the valid input domain** — respect the
`Bounds` / `Probability` / `NonNegativeRate` newtypes (ADR 0027/0031) and the
`Validate` ranges (ADR 0026). Generate *valid* configs to test behaviour; test
the invalid → `Err` path as a **separate, deliberate** property. Property
assertions must be **thread-count-invariant** (GEP evaluation is row-parallel via
rayon; assert finiteness / shape / bounds / ordering, not bit-exact reduction
values) or pin the threadpool inside the property.

### 5. CI cost policy

`proptest` has **no config-file mechanism** — configure per-`proptest!` via a
`ProptestConfig` literal (the source of truth) or `PROPTEST_*` env vars. Policy:

- **Backend-heavy properties** (instantiate a Burn backend + run a
  CMA-ES/CMSA-ES step or an EDA/GEP fit→sample cycle per case) use
  `ProptestConfig { cases: 16, max_shrink_iters: 256, ..Default::default() }`.
  Cap `max_shrink_iters` here so a failing-case shrink cannot dwarf forward cost.
- **Cheap structural properties** (shape / length / roundtrip, no full run) use
  64–128 cases.
- Optional `PROPTEST_CASES` env as a smoke-vs-deep lever (nice-to-have).

### 6. Commit `proptest-regressions/`

Regression files are **committed**, not gitignored. Each records the exact
shrunken counterexample so every future CI run replays it — a durable regression
guard that aligns with the repo's resolve-don't-just-document posture. Reversible
to gitignore later if the file churn proves unacceptable.

## Consequences

### Positive

- Input-space invariants become first-class and continuously exercised, not
  limited to hand-picked seeds; shrinking yields minimal reproducers for free.
- Committed regression files replay every past counterexample on every run.
- The RNG boundary (§3) means adopting proptest introduces **no new flakiness**:
  determinism still flows from `seed_stream`, and no forking means no new
  parallelism against Burn's RNG mutex.

### Neutral

- Scope is limited to `rlevo-evolution` dev-deps; other crates are untouched.

### Negative / accepted costs

- CI wall-clock grows with `cases`, bounded per-property by the §5 policy (16 for
  backend-heavy, 64–128 for cheap).
- Committing `proptest-regressions/` couples the repo to shrunken counterexamples
  (reversible — see §6).
- The transitive dependency surface grows, but `rusty-fork` / `tempfile` /
  `wait-timeout` are avoided by dropping default features (§2). Any `rand` /
  `bitflags` duplicate-version pull-in is absorbed by the workspace's existing
  `multiple_crate_versions = "allow"` clippy setting (root `Cargo.toml`).
- Adversarial generated inputs may surface genuine latent bugs. Per `rules.md`,
  each such finding is a **filed issue**, not a silenced or weakened property.

## Alternatives considered

- **Keep example-only tests.** Rejected: leaves the input space under-covered;
  every regression is a hand-authored seed a maintainer had to think of first.
- **`proptest` with default features (forking on).** Rejected: per-case forking
  re-inits the Burn backend / threadpool / GPU state in a subprocess — expensive
  and hazardous. `default-features = false` + `std` + `bit-set` is the safe cut.
- **`quickcheck`.** Rejected: weaker shrinking and no persisted regression
  corpus; `proptest`'s `proptest-regressions/` is the durable guard we want.
- **Let proptest generate tensors / drive Burn RNG directly.** Rejected outright:
  violates ADR 0029 (process-wide RNG mutex, thread-schedule dependence). proptest
  generates host config only; `seed_stream` owns algorithm randomness.
- **Gitignore `proptest-regressions/`.** Rejected as the default: discards the
  replay guard. Reserved as the reversible fallback if churn becomes a problem.
- **Adopt across all crates now.** Rejected: no concrete second consumer yet.
  `rlevo-evolution` has the invariant-rich operator/`Strategy` surface; promote on
  a demonstrated need.

## References

- Issue #239 — adopt `proptest` as a dev-dependency.
- ADR [0029](0029-host-rng-seeding-convention.md) — host-RNG `seed_stream`
  convention; the hard boundary property bodies must respect.
- ADR [0026](0026-shared-config-validation-convention.md) — `Validate` ranges
  strategies must stay within.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) /
  [0031](0031-probability-rate-newtypes.md) — `Bounds` / `Probability` /
  `NonNegativeRate` domains strategies must respect.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md) — the finiteness
  invariants ("no NaN across the valid domain") properties assert.
- `docs/rules.md` — the "filed issue, not silenced property" posture for
  adversarial findings.
- Root `Cargo.toml` `[workspace.dependencies]`;
  `crates/rlevo-evolution/Cargo.toml` `[dev-dependencies]`;
  `crates/rlevo-evolution/src/rng.rs` (`seed_stream`, `SeedPurpose`).
</content>
</invoke>
