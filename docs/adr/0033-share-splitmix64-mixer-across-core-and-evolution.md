---
project: rlevo
status: active
type: decision
date: 2026-07-06
tags: [adr, decision, rng, reproducibility, rlevo-core, rlevo-evolution, dry]
---

# ADR 0033: Share one `splitmix64` mixer across `rlevo-core` and `rlevo-evolution`

## Status

**Accepted (2026-07-06).** Resolves issue #161 §4.1. **Partially supersedes ADR
[0004](0004-move-bench-traits-into-rlevo-core.md)** — specifically decision point
#6 and the "Neutral" consequence (line 95) that chose to *keep* the local
`splitmix64` in `rlevo-evolution/src/rng.rs`. All other ADR 0004 decisions (the
trait moves, typed `BenchError`, the strict-DAG dep-graph shape) remain in force;
ADR 0004 stays `active`.

**Chosen shape:** promote the existing private mixer to a single documented
`pub const fn rlevo_core::util::seed::splitmix64`, pin it with a golden-value
test, and delete the byte-identical copy in `rlevo-evolution`, which now imports
the core one. The two *seed-derivation schemes* stay independent — they share the
mixer, not a derivation contract.

## Context

`rlevo-evolution/src/rng.rs` and `rlevo-core/src/util/seed.rs` each defined a
private `const fn splitmix64(mut x: u64) -> u64` with **byte-identical** bodies
(same three canonical constants, same shifts). ADR 0004 §6 accepted the
duplication, reasoning that the two APIs are distinct (`(base, generation,
SeedPurpose) -> StdRng` vs `(env_idx, trial_idx) -> u64`) and that they "share an
algorithm — a consequence of using a well-known mixer — not a dependency."

Issue #161 (🔴) revisits that call. The concern is **silent drift**: two
hand-copied frozen constants can diverge under a well-meaning edit (a typo, or an
"improvement" applied to one copy), and because each crate's reproducibility is
self-contained, such a divergence breaks one crate's stored-seed reproducibility
with no compile error and no cross-crate test to catch it.

The honest weighing of the two views:

- **ADR 0004's view (coupling risk):** sharing one function couples both crates'
  reproducibility contracts to a single definition — a change intended for one
  now affects both.
- **Issue #161's view (drift risk):** two copies of a frozen constant can silently
  diverge, and nothing guards against it.

The deciding observation is that `splitmix64` here is a **frozen reference
algorithm**: its output must *never* change in either place, because any change
breaks the reproducibility of every stored seed, golden test, and recorded run.
For a value that must never change, "coupling" is not a cost — it is the *desired*
invariant (both crates must agree on the mixer forever). A single source of truth
with one golden pin test is therefore strictly safer than two copies that can
drift: the pin test makes an accidental edit fail loudly in one place, and there
is no longer a second copy to fall out of step.

`rlevo-evolution` already depends on `rlevo-core`, and the mixer's module chain
(`util` → `seed`) is already fully `pub`, so exposing the function requires no new
dependency edge and no module restructuring.

## Decision

1. **Promote** `rlevo_core::util::seed::splitmix64` from private to
   `pub const fn` with `#[must_use]` and a **stability-contract** docstring: the
   output must never change; it is the single mixer shared by `SeedStream` and
   `rlevo_evolution::rng::seed_stream`; a failure of its pin test means "do not
   merge / revert," not "update the expected value."
2. **Pin** the mixer with `splitmix64_golden_values_are_frozen` in
   `rlevo-core` — golden outputs for inputs `0`, `1`, `u64::MAX` captured from the
   current implementation.
3. **Delete** the copy in `rlevo-evolution/src/rng.rs` and
   `use rlevo_core::util::seed::splitmix64;`. The `seed_stream` body is otherwise
   untouched, so its output is bit-identical to before this change.
4. The two seed-derivation **schemes remain independent**. Only the inner mixer is
   shared; `SeedStream`'s `(env, trial)` fan-out and `seed_stream`'s
   `(base, generation, purpose)` derivation are unchanged.

Because the mixer body is byte-identical, this change alters **no runtime output**:
all existing determinism/golden tests (`rlevo/tests/determinism.rs`, the `rng`/
`seed` unit tests) pass unchanged.

## Consequences

### Positive
- **Single source of truth.** Drift is now impossible — there is one mixer, one
  golden tripwire. An accidental edit fails `splitmix64_golden_values_are_frozen`
  immediately.
- **No behavioural change.** The dedup is a pure refactor; reproducibility of every
  stored seed and recorded run is preserved by construction.

### Neutral
- One additional `pub` free function under `rlevo-core`'s `util::seed`. The umbrella
  `rlevo` crate does not re-export `util::seed`, and `rlevo-core` is an internal
  crate ("use `rlevo` for the full API"), so third-party surface is effectively
  unchanged. The function is conceptually apt next to `SeedStream`, which already
  documents itself as consuming it.

### Negative / accepted costs
- A mechanical edit to the mixer now affects **both** crates by construction. This
  is intentional (the invariant is that they must always agree) and is guarded by
  the golden pin test.

## Alternatives considered

- **Keep two copies + a cross-crate "assert equal" guard test.** Honors ADR 0004
  (no dedup) and kills the drift *risk* by asserting the two bodies agree on golden
  inputs. Rejected: still two edit sites and two functions to maintain; strictly
  weaker than one source of truth for a value that must never change.
- **A shared macro that expands the body in both crates.** Rejected: no advantage
  over a plain `pub fn`, and it obscures the single-definition intent.
- **Do nothing (keep ADR 0004 as-is).** Rejected: the drift concern is valid and
  cheaply eliminated; the "coupling" ADR 0004 protected against is, for a frozen
  algorithm, the property we actually want.
- **Change `SeedPurpose::Other`'s constant** (which coincides with φ64). Out of
  scope and rejected here: it would break the reproducibility of every strategy
  currently using `Other`. Issue #161 §8.1 asks only to *document* the caveat,
  which the same PR does on the `Other` variant.

## References
- Issue #161 — this ADR resolves §4.1 (and the same PR lands §7.1 exhaustive
  distinctness test and §8.1 `Other` caveat docs).
- ADR [0004](0004-move-bench-traits-into-rlevo-core.md) — decision #6 / "Neutral"
  consequence (keep the local mixer), partially superseded here.
- ADR [0029](0029-host-rng-seeding-convention.md) — the host-RNG seeding convention
  the shared mixer serves.
- Code: `crates/rlevo-core/src/util/seed.rs` (the `pub` mixer + pin test),
  `crates/rlevo-evolution/src/rng.rs` (import + exhaustive distinctness tests +
  `Other` caveat).
