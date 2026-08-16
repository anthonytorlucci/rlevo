---
project: rlevo
status: active
type: decision
date: 2026-08-11
tags: [adr, decision, docs, testing, guard-test, rules-md, panic-contracts]
---

# ADR 0074: The `rules.md` panic-contract table is mechanically checked in the dead-row direction only, workspace-wide

## Status

**Accepted (2026-08-11).** Resolves issue #1109. `docs/rules.md`
"Documented Panic Contracts" table is the repo's single index of every place
a `rlevo` API is allowed to panic, and until now it was prose: nothing
compiled it, nothing failed when the code underneath a row moved. Two
failures already happened because of that:

- **#1085**: a module was deleted and two rows — `Builder with_capacity(n)`
  and `Builder with_alpha(x)` — outlived it by months, naming a type that no
  longer existed anywhere in the workspace. ADR [0050](0050-replay-strategy-seam.md)
  even wrote the remediation down at the time (`with_alpha` retires with
  the builder, `with_capacity` survives renamed to the replay builders) and it
  went unexecuted, because nothing failed when it did not happen.
- **#1108**: a whole module's four panic contracts (three
  `HillClimbingParams::with_*` setters plus `local_search::hill_climbing`'s
  `refine`/`refine_with_known_fitness`) were never listed at all.

**Supersedes nothing.** It is a sibling of ADR [0068](0068-bounds-strictness-enforcement-is-crate-asymmetric.md),
applying ADR [0062](0062-grid-layout-fidelity-and-no-dead-rng.md)
precedent — a source-text guard's scope is an architectural decision — to a
population that is workspace-shaped rather than crate-shaped, which is why
this ADR's scope call is the mirror image of 0068's.

**Chosen shape.**

1. A workspace-wide source-text guard,
   `crates/rlevo/tests/panic_contract_table_guard.rs` (5 tests), reading
   `docs/rules.md` and every `crates/*/src/**/*.rs` file at test time.
2. Scope is the table's **Site** column only — never the surrounding
   prose, never the Condition column.
3. Only the **dead-row direction** (#1085: does every row still name
   something?) is mechanized. The **missing-row direction** (#1108: does
   every panic site have a row?) stays review-enforced.
4. A one-row, bidirectional exemption list (`UNRESOLVABLE_SITES`), checked
   both for staleness and for having quietly become resolvable.

Additive: no production code changes, no public API changes. The guard can
be deleted at zero cost if its assumptions stop holding — see the section 
*Assumptions*.

## Context

### The population is workspace-shaped, not crate-shaped

ADR 0068 scoped its `Bounds`-strictness guard to one crate,
`rlevo-reinforcement-learning`, and its reasoning was explicit about why: a
workspace-wide scan of that property would have needed roughly 30 allowlist
rows, 94% of them restating "zero width is fine here, see ADR 0027" — and
"a guard whose rows are 94% 'this one is fine' trains its readers to add a
row without thinking, which is the state in which it stops catching
anything" (0068 - Context, Alternatives).

The panic-contract table does not have that shape. Its 30 rows already span
`rlevo-core` (`DiscreteAction`, `ContinuousAction`, `MultiDiscreteAction`,
`Grid`), `rlevo-reinforcement-learning` (`UniformReplay`, `History`,
`ImportanceExponent`, `Priority`), and `rlevo-evolution`
(`SimulatedAnnealingParams`, `HillClimbingParams`, six `ops::*` families, two
`local_search::*` rows) by construction — the table's subject *is* "every
place in the library allowed to panic", and the library is the workspace.
The one row this guard cannot resolve to an item (`Batch rank assertion`, a
const-generic relation asserted at each batching call site rather than owned
by one function) is **1 row out of 30 — about 3%**, the near-opposite ratio
of 0068's rejected workspace-wide alternative. 0068's argument against a
workspace scan and this ADR's argument for one are the same argument,
reaching opposite conclusions because the two populations are shaped
oppositely — which is the point ADR 0062 makes generally and 0068 Decision
2 restates for its own case.

**Reopen trigger:** a second `UNRESOLVABLE_SITES` row, or the exemption list
crossing 10% of the table's rows, means the table has drifted away from
naming individual items toward naming un-nameable relations, and this
guard's premise — that the table is mostly a list of resolvable identifiers
— needs re-examining rather than growing the list.

### The table's prose is out of scope, and that has a known, named cost

`docs/rules.md` prose around the table is dense with illustrative
backticked identifiers — `config::ordered`, `Bounds::try_new`,
`.unwrap_or_default()`, `EnvironmentError` — that are *mentions*, not
contract claims. A resolver that could not tell a claim from a mention would
need a prose allowlist of "this backtick is not a claim" rows, which is
exactly the density failure §Context above already rejected for the
workspace-vs-crate question, one level down. So the guard reads only the
Site column of rows between `### Documented Panic Contracts` and the first
non-`|` line after it.

This has a real, already-observed cost, and it is recorded here rather than
implied: `docs/rules.md:116` cites `C51Config::delta_z` as the reference
implementation for "an accessor on a config must be total," and the type is
actually named `C51TrainingConfig`
(`crates/rlevo-reinforcement-learning/src/algorithms/c51/c51_config.rs`).
That citation sits in prose, not in the table, so this guard does not
and will not catch it, or any future citation error like it, anywhere
outside the table.

### `clippy::missing_panics_doc` does not substitute for the missing direction

#1108's direction — every panic site has a table row — is not mechanized
here, and a reader must not infer from a passing test suite that it is
guaranteed by some other mechanism. It is not. Panics arrive through at
least six unrelated syntactic forms (`panic!`, `assert!`, `assert_eq!`,
`.unwrap()`, `.expect()`, slice indexing, integer overflow in a debug
build), so a source-text check for "a table row exists for this" would need
a large, hand-maintained allowlist of every panic site judged *not* to be a
documented contract — the same density problem as the prose case above, at
a much larger scale. `clippy::missing_panics_doc` is not a working
substitute: it fires on inherent methods only and is silent on trait-impl
methods, which is where several of this table's rows live (`History::index`
is `Index::index`; `ContinuousAction::from_slice` is a trait method on
several implementors). The #1108 direction stays a review-enforced
convention, same as `rules.md`'s config-value-must-be-finite convention was
before ADR 0060 and the same asymmetry ADR 0068 accepted for its own missing
direction.

## Decision

### 1. Scope: workspace-wide, not `rlevo-reinforcement-learning`-only or `rlevo-core`-only

Settled in §Context. Every crate under `crates/*/src` is scanned; the table
is checked against all of it, not against the crate that happens to define
the ADR being written. `NON_PRODUCTION_CRATES` (`rlevo-benchmarks`,
`rlevo-benchmarks-report-client`, `rlevo-examples`, `rlevo-test-support`) is
not an exemption from scanning — a spec that resolves *only* inside one of
those crates is a **hard failure**, because `docs/rules.md` §1 fixes the
dependency direction (production crates never depend on the benchmark,
example, or test-support layer) and a panic contract that lives only there
is a promise about code no library user can reach. Zero rows hit this today.

### 2. Scope: the table's Site column, not §4's surrounding prose

Settled in §Context, including the named cost: `docs/rules.md:116`'s
`C51Config` (should be `C51TrainingConfig`) is a real, present citation
error, and this guard will never see it, because it lives in prose outside
the table. Its successor — the next prose citation that drifts — will not
be caught by this guard either. That is a stated limit, not an oversight.

### 3. Direction: dead rows only, never missing rows

Settled in §Context. #1108's direction remains review-enforced.
**The table is not thereby proven complete**, and this ADR says so
explicitly rather than leaving it to be inferred from five green tests: a
row can be accurate and the table can still be missing a panic contract
that belongs in it. The guard's name —
`every_panic_contract_row_names_a_live_item` — is deliberately not
`every_panic_contract_is_documented`; it could not honestly be named the
second.

### 4. Resolution strategy: two knowingly weak tiers, and why the naive alternative is worse

A Site cell's backtick spans normalize to *specs* (`qualifier::item`, brace
expansion, trailing-argument stripping, empty-qualifier inheritance within a
cell), tiered by the qualifier's case — uppercase first letter is a type,
otherwise a module, reliable because every crate sets
`[lints] workspace = true` and so `non_camel_case_types` /
`non_snake_case` are live.

A **module-tier** spec resolves against a crate-anchored file or directory
subtree and accepts a name found *anywhere in that subtree* — proof that a
name exists in the module, not that it exists in the file the row's author
had in mind. A **type-tier** spec locates the type's unique declaration,
then looks for the item's `fn` in the declaring file (Step B) or, failing
that, in any file whose production source carries a matching `impl … T` or
wrapped `for T` header (Step C, see §Decision 7). For the three
trait-declaration rows the table names by trait rather than by
implementation — `DiscreteAction::from_index`, `ContinuousAction::from_slice`,
`MultiDiscreteAction::from_indices` — resolution against the trait's own
declaration proves the *method exists on the trait*, not that any concrete
implementor actually panics under the stated condition. Both weaknesses are
accepted rather than closed, and are recorded here so a future reader does
not mistake either kind of row for a stronger claim than it is.

Weighed against those two named weaknesses is what the tiering buys over
the naive alternative — bare-name matching, ignoring the qualifier
entirely. Measured against the workspace as it stands today, a bare-name
resolver for `fn new` alone would have to disambiguate among roughly 200
production `fn new` declarations across `crates/*/src` (found by
`rg '^\s*(pub(\(crate\))?\s+)?fn new\b' --glob '**/src/**/*.rs'`, restricted
to the same `crates/*/src` tree this guard scans). Anchoring on the
declaring file — via the type's unique declaration for type-tier rows, via
the crate-anchored subtree for module-tier rows — collapses that
disambiguation to the one file (or module) the row's qualifier actually
names, which is the entire reason a qualifier is required in the table's
spelling convention rather than being decorative.

### 5. Glob rows require at least two production matches, with today's margin at zero

`ops::selection::tournament_*` and `ops::selection::truncation_*` each
resolve as globs and must prefix-match **two or more** production `fn`
names in the named module, counting only production lines — three of
`tournament_*`'s five raw matches are `#[cfg(test)]` fixtures, and without
that filter the check is vacuous (both production functions could be
deleted and the glob would still pass). With the filter, each glob matches
**exactly two** today: zero headroom above the floor. The floor is set at
two, not one, on the reasoning that a glob matching a single item should
have named that item directly — which is exactly what the adjacent row,
`ops::selection::{tournament_select, truncation_select}`, already does.
Deleting either `tournament_indices_host` or `tournament_select` fails this
guard today; the fix is to edit the table, not to weaken the floor.

### 6. The exemption is keyed on verbatim Site-cell text, checked in both directions it can be

`UNRESOLVABLE_SITES` holds one entry, `"Batch rank assertion"`, with a
written reason (a const-generic relation asserted at many call sites, owned
by none of them). The key is the cell's byte-for-byte text, which gives the
exemption two independent failure modes rather than one silent one:
`exemption_rows_match_a_live_table_row` fails if the row is deleted or
reworded out from under the key, and `no_exempted_row_names_an_identifier`
fails if the row instead grows a backticked identifier that could have been
resolved — at which point the exemption should be deleted, not extended.
Neither test is a way to silence a rename; both exist to stop the exemption
list from becoming the kind of stale, half-checked allowlist ADR 0068
warned against.

### 7. Step C is deliberately unreached by every row today, and the design's own tripwire prediction was wrong

The design review for this guard predicted `History::index` would be the
row exercising type-tier Step C (the widened `impl … T` search), and
specified an armed-ness tripwire requiring some row to reach it. That
prediction does not hold: `struct History` is declared at
`crates/rlevo-reinforcement-learning/src/experience.rs:98` and `fn index`
at `:119`, in the same file, so `History::index` resolves at Step B. In
fact **no row in today's table reaches Step C** — every type-tier item's
`fn` happens to sit in its type's own declaring file. Writing the tripwire
as originally specified turns the suite red on a correct implementation,
and this ADR records the correction rather than silently adopting it:
`the_guard_is_armed` asserts a live row through `TypeDeclaringFile`,
`Module`, and `Glob`, and *not* through `TypeImplWiden`.

What the prediction was protecting is real, and is pinned a different way.
`experience.rs` has no single-line `impl … History` header — both of its
impls wrap across lines (`:103-104`, the `Index<usize>` impl, and
`:124-125`), with the type name appearing only on the `for History<…>`
continuation line — so `mentions_impl_of`'s `for T`-continuation
alternative is the only thing that would make this file a Step C candidate
at all, for the ordinary case of a method living outside its type's
declaring file. Asserting that fact via a tier tripwire would assert *which
file a method sits in*, and that flips on any legitimate refactor that
moves `index` out of `experience.rs`. Instead, `the_wrapped_impl_header_is_recognized`
pins the mechanism directly: it asserts `mentions_impl_of("History")` is
true today, and that a single-line-only `impl … History` scan would *not*
have found it — the second assertion is what makes the first informative.
Step C itself is kept in the resolver, unreached, for the ordinary
different-file case; a future reader who finds it uncovered by any live row
should read this section rather than "fix" it by deleting it or forcing a
row through it.

### 8. Known limits, recorded rather than rediscovered

Following the ADR 0062 §4 / ADR 0068 formula:

- **Text-based.** A `\bfn\s+name\b` match inside a `/* … */` block comment
  reads as code — the safe direction, a false *pass*, never a false
  accusation — and hand-formatting that breaks `fn name` across two lines
  defeats the scanner. CI's `cargo fmt --all --check` is what makes the
  single-line assumption hold; this guard is downstream of that gate, not a
  replacement for it.
- **Defeatable by a same-file name collision.** The scanner cannot tell an
  inherent method from an unrelated free function of the same name in the
  same file.
- **Blind to `#[cfg(test)]` regions**, deliberately — the alternative is a
  test-only function satisfying a production contract row, which is how the
  glob check in §Decision 5 would go vacuous. `cfg_test_regions` fails loud
  on an unterminated or misindented region rather than silently swallowing
  the rest of a file.
- **Cannot distinguish an inherent method from a same-file free function**
  for the purposes of module-tier resolution; matching is kind-agnostic by
  design (six of the table's specs, the `local_search::*` rows, are
  trait-impl methods spelled as module paths).
- **It catches the accident, not the adversary.** That is the same threat
  model 0062 and 0068 accept for their own guards: the failure this exists
  to prevent is a contributor renaming or deleting an item and not noticing
  the table still names it, not a determined attempt to defeat the check.
  Deleting this guard costs nothing beyond the maintenance it was doing, so
  it is a cheap two-way door.

**A deliberate cost taken to avoid the guard's most likely spurious
failure:** module-tier and type-tier candidates treat a matched directory
as its entire recursive subtree, not the single file a row's author had in
mind. `local_search/hill_climbing.rs` is 732 lines; splitting it into
`local_search/hill_climbing/{mod.rs, params.rs, refine.rs}` is the ordinary
next refactor, and the recursive subtree is what keeps that refactor from
turning red — a file-exact resolver would fail it, accuse the diff of stale
docs, and be the first thing deleted the next time someone needed to touch
that file. The price, paid on purpose: this guard proves "this name exists
somewhere in this module," not "in this file." A row pointing at the wrong
file inside the right module passes.

### 9. `cfg_test_regions` is a deliberate third copy, not a shared helper

The `#[cfg(test)]`-region scanner is lifted verbatim from
`rlevo-reinforcement-learning/tests/bounds_strictness_guard.rs:811-865`,
itself lifted from `rlevo-environments/tests/rng_seeding_guards.rs`. This is
the third copy in the workspace, and it is not factored into
`rlevo-test-support`. ADR 0062 §4 and ADR 0068 §Decision 2 both rest on each
source-text guard being independently deletable when its premise stops
holding; a shared helper would make deleting any one of the three guards a
cross-crate change, which is exactly the friction that keeps a guard alive
past its usefulness rather than the friction that protects it.

## Assumptions and the reopen trigger

The table's shape is a fact about the workspace today, not a law. The
reopen conditions:

1. A second real `UNRESOLVABLE_SITES` row, or the exemption list crossing
   10% of the table's rows (§Context) — the table has stopped being mostly
   nameable identifiers.
2. A row that needs to name something inside `rlevo-benchmarks`,
   `rlevo-examples`, or `rlevo-test-support` and cannot be restated to name
   a production item instead — a real instance of the `rules.md` §1
   dependency-direction problem this guard is built to catch as a hard
   failure, not a config error in the guard.
3. A glob row's match count dropping to one — today's zero-margin floor
   (§Decision 5) means this is the most likely single-line change to break
   the guard, and the fix is always to name the surviving item, not to
   lower the floor.
4. Someone builds the #1108 direction (every panic site has a row) as a
   mechanical check. If that ever lands, this ADR's §Decision 3 disclaimer
   — "not thereby proven complete" — should be revised to describe what the
   new check adds, not deleted.

## Consequences

### Positive

- **#1085 cannot recur silently.** A row naming a deleted item now fails
  `cargo test -p rlevo` instead of surviving until a human happens to read
  the table carefully.
- **The remediation ADR 0050 §8 wrote down and nobody executed is now
  enforced, not merely documented a second time.** The `with_alpha` /
  `with_capacity` rows this guard's own workspace scan found already
  cleaned up are the concrete instance; the mechanism generalizes to every
  future row.
- **#1108's four missing hill-climbing rows are now present and stay
  checked** — as ordinary rows subject to the same dead-row guarantee as
  every other row, even though the guard did not (and could not, alone)
  cause them to be added.
- **The guard's own armed-ness is asserted**, not assumed: row and spec
  floors, a live resolution through every reachable tier, and a
  production-`fn`-count floor under `rlevo-evolution/src/ops` so a scanner
  that has stopped parsing anything cannot pass by finding nothing to
  check.
- **Cheap to delete.** No production code depends on this guard's
  existence; if its assumptions stop holding (§Assumptions), removing it
  costs nothing beyond restoring the status quo ante.

### Negative / accepted costs — do not soften these

- **Only one of the table's two failure directions is mechanized.** A row
  can be perfectly accurate and the table can still be silently missing a
  contract, exactly as #1108 was for four months before it was noticed by
  review. This ADR does not close that gap; §Decision 3 says so in the
  guard's own module doc and here.
- **The table's surrounding §4 prose is unchecked**, with a named, present
  instance of the cost: `docs/rules.md:116`'s `C51Config` citation (the
  type is `C51TrainingConfig`) is not caught by this guard and will not be,
  because it is prose, not a table row.
- **Two tiers are knowingly weak.** A trait-declaration row proves the
  method exists on the trait, not that any implementor panics as claimed. A
  module-tier row proves the name exists somewhere in the module's
  recursive subtree, not in a particular file.
- **The glob floor ships at zero margin.** Deleting either
  `tournament_indices_host` or `tournament_select` fails this guard today;
  that is intended (§Decision 5), but it means the very next refactor of
  either function is where this cost is first paid.
- **Text-based, not semantic**, with every limit that implies (§Decision
  8): defeatable by a comment, by cross-file reformatting, by a same-file
  name collision. It catches the accident, not the adversary — the correct
  threat model for this failure class, and the same one 0062 and 0068
  accept.
- **A third copy of `cfg_test_regions` now exists**, by deliberate choice
  (§Decision 9), rather than a shared helper. Any future improvement to the
  region scanner (a new terminator shape, say) must be applied to three
  files by hand, or the three guards drift apart in what they can parse.

### Neutral

- No new dependency, no proptest — the invariant this guard checks
  ("this Site cell names something that resolves in the workspace") is a
  source-text property of `docs/rules.md` and `crates/*/src`, not an
  input-space property (ADR [0036](0036-adopt-proptest-for-property-tests.md)).
- No production signature, type, or behavior changes anywhere.

## Alternatives considered

- **Crate-scoped guards, one per crate, mirroring ADR 0068.** Rejected on
  the population argument in §Context 1: the table's rows are already
  scattered across four production crates by construction, so a per-crate
  guard would need to either duplicate the table-parsing machinery per
  crate or split the table by crate — a maintenance burden the single
  workspace-shaped population does not justify. 0068's precedent transfers
  in reasoning, not in answer, because the two populations are shaped
  oppositely (§Context).

- **Mechanizing the #1108 direction alongside this one.** Rejected for now
  on the density argument in §Context: distinguishing "a panic site that is
  a documented contract" from "a panic site that is an internal invariant"
  across `panic!`/`assert!`/`unwrap`/`expect`/indexing/overflow is a
  judgement this guard's text-scanning approach cannot make reliably, and a
  guard that makes it badly earns itself a deletion rather than trust. Left
  as a review-enforced convention, and named on the reopen list
  (§Assumptions 4) rather than closed off.

- **Extending scope to the table's surrounding §4/§3 prose.** Rejected on
  the same density argument: illustrative backticked mentions
  (`config::ordered`, `.unwrap_or_default()`, `EnvironmentError`) vastly
  outnumber contract claims outside the table, and a prose allowlist large
  enough to separate them would itself become the noise ADR 0068
  §Alternatives warned a 94%-allowlist guard becomes. The known cost
  (`C51Config`/`C51TrainingConfig` at `rules.md:116`) is accepted and
  recorded rather than chased with a wider scan.

- **A single-line `impl … T` scan for type-tier Step C, dropping the
  wrapped-header (`for T`) alternative.** Rejected: `History`'s impl
  headers wrap by construction (`experience.rs:103-104`, `:124-125`), so
  this alternative would find zero Step C candidates for `History` and
  degrade silently — a degradation that, because Step C is a *widening*
  step used only when Step B fails, would surface much later as a spurious
  failure in some unrelated future diff that moved a method out of its
  type's declaring file. See §Decision 7.

- **A tier tripwire requiring a live row through Step C.** This was the
  original design and it does not hold against the workspace as built — no
  row reaches Step C today (§Decision 7). Rejected in favor of pinning the
  wrapped-header mechanism directly with a dedicated test, because a tier
  tripwire on Step C would assert a fact about which file a method
  currently lives in, and any legitimate refactor moving a method into its
  type's declaring file would make that tripwire fail for a reason having
  nothing to do with the guard's actual purpose.

- **Factoring `cfg_test_regions` into `rlevo-test-support`.** Rejected on
  ADR 0062 §4 / ADR 0068 §Decision 2's shared reasoning: each source-text
  guard must stay independently deletable, and a shared helper turns
  deleting one guard into a cross-crate change. See §Decision 9.

- **A glob floor of one match instead of two.** Rejected: a glob matching
  exactly one production function should simply name that function, as the
  adjacent `{tournament_select, truncation_select}` row already does for
  the pair the two globs also cover individually. See §Decision 5.

## References

- Issue **#1109** — the docs-drift report this ADR resolves: the panic
  contract table cites deleted code (#1085) and is missing rows (#1108),
  with no mechanism to notice either.
- Issue **#1085** — a module deleted, two `Builder with_*` rows outliving
  it by months. The remediation ADR 0050 §8 wrote down (`with_alpha`
  retires, `with_capacity` renames) and did not enforce.
- Issue **#1108** — `local_search::hill_climbing`'s four panic contracts
  never listed. The direction this guard does **not** mechanize
  (§Decision 3).
- ADR [0050](0050-replay-strategy-seam.md) §8 — the prior remediation for
  the `with_alpha`/`with_capacity` rows, written down and unexecuted until
  this guard existed to enforce it.
- ADR [0062](0062-grid-layout-fidelity-and-no-dead-rng.md) §4 — the
  precedent that a source-text guard's **scope** is itself an
  architectural decision, and the source of the bidirectional-allowlist /
  stated-limits shape this guard and ADR 0068's both follow.
- ADR [0068](0068-bounds-strictness-enforcement-is-crate-asymmetric.md) —
  the closest sibling: same guard shape, same "scope is architectural"
  precedent, opposite scope conclusion because the two populations are
  shaped oppositely (crate-concentrated there, workspace-shaped here). Its
  `#[cfg(test)]`-region scanner is this guard's second-generation ancestor
  (§Decision 9).
- `docs/rules.md` §4, "Documented Panic Contracts" — the table this guard
  checks, and (uncaught, §Decision 2) the surrounding prose, including the
  `C51Config`/`C51TrainingConfig` citation at `rules.md:116`.
- Code —
  `crates/rlevo/tests/panic_contract_table_guard.rs`: the guard. Its
  module doc (`//!`) carries the resolution algorithm in full; this ADR
  records the decisions and their reasons, not the mechanism.
- Code —
  `crates/rlevo-reinforcement-learning/src/experience.rs:98,103-105,119,124-125`
  — `History`'s wrapped impl headers, the live example behind §Decision 7.
- Code —
  `crates/rlevo-reinforcement-learning/src/algorithms/c51/c51_config.rs:25`
  — `C51TrainingConfig`, the type `rules.md:116` misnames.
- Code — `crates/rlevo-reinforcement-learning/tests/bounds_strictness_guard.rs:811-865`,
  `crates/rlevo-environments/tests/rng_seeding_guards.rs` — the two prior
  `cfg_test_regions` copies this guard's is a third of (§Decision 9).
