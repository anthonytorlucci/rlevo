---
project: rlevo
status: active
type: decision
date: 2026-10-13
tags: [adr, decision, architecture, naming, harness, crate-boundaries, publishing, breaking-change]
---

# ADR 0088: The harness crate is named `rlevo-harness`

## Status

**Accepted (2026-10-13).**

Renames a crate that 35 earlier ADRs name: `rlevo-benchmarks`, library
`rlevo_benchmarks`. Those ADRs stand as written. Where one says
`rlevo-benchmarks`, read `rlevo-harness`; where one says
`rlevo_benchmarks::`, read `rlevo_harness::`; where one cites a file under
`crates/rlevo-benchmarks/`, the file is under `crates/rlevo-harness/`. The
name `rlevo-benchmarks-report-client` is a different crate and is not
renamed (Decision 5).

Supersedes one clause of ADR
[0004](0004-move-bench-traits-into-rlevo-core.md): its Decision 5 kept
the modules `agent` and `seed` in this crate as aliases for items it moved
to `rlevo-core`, and Decision 8 below removes them. The third alias of that
decision, `env`, went with ADR [0077](0077-remove-benchenv.md). The rest of
ADR 0004 is unchanged here. This ADR takes the decision ADR
[0084](0084-the-umbrella-carries-the-evaluation-harness.md) Decision 2 left
open ("renaming the crate to match is a separate decision"), and it is the
second half of what ADR
[0085](0085-the-generation-loop-is-named-evolutionary-loop.md) began: that
ADR made "harness" mean one thing, and this one gives the thing the name.
The separation ADR
[0001](0001-keep-environments-and-benchmarks-separate.md) decided, between
the environments and the harness, is unchanged.

## Context

The crate holds `Suite`, `Evaluator`, the two trial shapes, the metrics and
the reporters: it runs an agent or a generation loop over many seeds and
reports the result. Its name described an earlier content. The trait
surface it was named for (`BenchEnv`, `BenchStep`, `BenchError`) moved to
`rlevo-core` under ADR [0004](0004-move-bench-traits-into-rlevo-core.md)
and was removed by ADR [0077](0077-remove-benchenv.md).

The old name cost three things.

**It read as cargo's `[[bench]]`.** `docs/rules.md` section 5 carried a
boxed warning, "Terminology trap", that "benchmark" meant two different
things in the workspace and that nobody should add a `[[bench]]` target to
the crate "to match the name". ADR 0084 Decision 2 refused `benchmarks` as
the name of the umbrella's re-export for that reason, and found that
`bench`, as a plain intra-doc link, fails the documentation build because
it is also the built-in `#[bench]` attribute.

**The crate's name and its path disagreed.** Since ADR 0084 a consumer
reaches the crate as `rlevo::harness`, and the line that says so read
`pub use rlevo_benchmarks as harness;`. Every document had to say both
names and which was which.

**It would have been frozen at the next release.** No release has shipped
`rlevo::harness`. The published 0.5.0 has `rlevo-benchmarks` as a crate the
umbrella does not carry. The release that first ships the path is the last
point at which the crate under it can be renamed without that path having
existed under two crate names.

**How much names it.** Counted at the base commit with the pattern
`rlevo[-_]benchmarks` over tracked files: 573 occurrences in 119 files.
Of those, 248 are in 36 files under `docs/adr/` (35 ADRs and the index),
which do not change. Another 61 outside `docs/adr/` are the report
client's name, which does not change either. Of the remaining 264, 19
keep the old name because they state a fact from before the rename: the 16
in the released sections of `CHANGELOG.md`; one in its unreleased
section, `rlevo_benchmarks::record::RecordingTap::with_landscape_payload`,
the path of an item removed before the rename; and two in sentences
reworded to say what the crate was called, the root `README.md`'s note
that a 0.5.0 user adds the harness under that name and the note in
`docs/rules.md` on what ADR 0004 moved out of it. The other 245 change.

**The name is free.** The crates.io API answers 404 for `rlevo-harness`
(2026-10-13). crates.io neither deletes a crate nor renames one, so the old
name stays where it is whatever is decided here.

## Decision

1. **The package is `rlevo-harness` and the library is `rlevo_harness`.**
   The directory is `crates/rlevo-harness/`. Apart from the two alias
   modules of Decision 8, nothing else about the crate changes: its other
   modules, its items, its features (`json`, `tui`, `record`, `report`),
   its `export-report` binary and its place in the dependency graph are
   what they were.

   What the crate says about itself follows the name. The `description` in
   its manifest reads "Evaluation harness for rlevo" where it read
   "Benchmarking harness for rlevo", in the same form as the other library
   crates' descriptions. Its README and its crate documentation open the
   same way and say what the crate was called through 0.5.0, and the README
   gains a section, "Moving from `rlevo-benchmarks`", for a project that
   depends on it directly: the dependency line, the feature entries and the
   paths to change, Cargo's `package` key for keeping the old local name,
   and the log target of Decision 4. The description of
   `rlevo-metrics-registry`, which called the harness "the benchmarks
   recorder", calls it "the harness's recorder".

2. **A consumer of `rlevo` changes nothing.** `rlevo::harness` is the same
   path, and the line behind it reads `pub use rlevo_harness as harness;`.
   The umbrella's `viz-tui` and `viz-report` features keep their names.

3. **There is no alias.** The workspace keeps no crate under the old name
   and no re-export under the old path. A crate that depends on the
   harness directly changes its dependency line, every feature that names
   the crate, and the first segment of its paths. ADR
   [0083](0083-episode-guard-is-a-core-contract-helper.md) and ADR 0085
   made the same choice for a moved type and a renamed one.

4. **Strings the crate emits follow its name.**

   | Where | Was | Is |
   |---|---|---|
   | the `tracing` target of every harness event | `rlevo_benchmarks` | `rlevo_harness` |
   | the `tracing` target of a recording error | `rlevo_benchmarks::record` | `rlevo_harness::record` |
   | the static report's `<meta name="generator">` | `rlevo-benchmarks/report` | `rlevo-harness/report` |
   | the static report's `<noscript>` note, in a report with a client | `` see the rlevo-benchmarks `report` docs `` | `` see the rlevo-harness `report` docs `` |

   A log filter written against the old target matches nothing after the
   upgrade, and says nothing about it. The changelog names this under
   breaking changes. No field of the record format carries the crate's
   name: `FORMAT_VERSION` stays at 7, and a recording written before the
   rename is read as before.

5. **What keeps its name.**

   - The accepted ADRs, as the Status section says.
   - Released sections of `CHANGELOG.md`, and the entries of the
     unreleased section that name an item a 0.5.0 user would search their
     code for.
   - `rlevo-benchmarks-report-client`, the unpublished WebAssembly report
     client. Nothing outside the workspace can depend on it, and what
     becomes of it is a separate decision. Its documentation now names the
     crate it mirrors as `rlevo-harness`.
   - Public items whose names contain the old word: `BenchmarkReport` in
     the harness, the module `rlevo_environments::bench`, and
     `BenchableAgent` in `rlevo-core`. This ADR is about the crate's name.

6. **On crates.io the old name gets one last release, and nothing is
   yanked.** `rlevo-harness` is published for the first time with the
   release that first ships `rlevo::harness`. With it, `rlevo-benchmarks`
   gets one more release whose description and README say that the crate
   continues as `rlevo-harness`. Version 0.5.0 of `rlevo-benchmarks` is
   left as it is. A yank frees no name and hides the version from new
   resolves, which would break a fresh build for anyone who depends on
   the crate at 0.5.0 directly, who turns on `bench` or `record` in
   `rlevo-environments` 0.5.0, or who turns on `viz-report` in `rlevo`
   0.5.0, which enables that `record`.

   Left to that release, because it is cut from a tree that no longer
   has the old crate: whether the last release is a crate with no items
   and a README, or one that re-exports `rlevo_harness` and forwards its
   features.

7. **The warning box leaves `docs/rules.md`.** Section 5 keeps the rule on
   what a `[[bench]]` target is for, which never depended on the crate's
   name. Section 1's inventory lists `rlevo-harness` and says what it was
   called through 0.5.0.

8. **The modules `agent` and `seed` are removed.** `agent` re-exported
   `BenchableAgent`, `FitnessEvaluable` and `Landscape` from
   `rlevo_core::fitness`, and `seed` re-exported `SeedStream` from
   `rlevo_core::util::seed`. ADR 0004 Decision 5 kept them when it moved
   those items to the core, so that code importing through
   `rlevo_benchmarks::agent` and `rlevo_benchmarks::seed` went on
   compiling. The rename breaks those paths for every dependent that takes
   the new name. The one dependent the aliases would still serve is one
   that keeps the old local name with Cargo's `package` key, and that
   dependent is already editing its manifest; for it the cost is one import
   per item. Against that, the aliases are a second public path to four
   items, the duplicate ADR
   [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md) asks a
   public item to answer for, kept for good. The workspace's seven imports
   through them, in the umbrella's tests and examples and in
   `rlevo-examples`, now name `rlevo_core::fitness`, or
   `rlevo::core::fitness` in the umbrella's examples, as a consumer writes
   it. The crate-root re-export of `GenerationProbe` is not one of these
   aliases and is not decided here.

## Consequences

**Positive.**

- One word for one thing. The crate is `rlevo-harness`, the path is
  `rlevo::harness`, and since ADR 0085 nothing else in the workspace is
  called a harness.
- A document says one name. `CLAUDE.md` and `docs/rules.md` each had to
  say that `rlevo-benchmarks` is re-exported as `rlevo::harness`.
- The rule against a `[[bench]]` target in the harness no longer needs
  stating as a trap.
- Each of the four items Decision 8 concerns has one public path, in the
  crate that defines it.

**Costs.**

- A breaking change for a crate that depends on the harness directly: its
  dependency line, every feature that names the crate, and every path. In
  the workspace that is three dependent manifests (`rlevo`,
  `rlevo-environments`, `rlevo-examples`), with a dependency line each and
  five feature entries between them, and 16 Rust files outside the crate
  whose code names a path in it. The other files that change name the
  crate in comments and documentation.
- An import through `agent` or `seed` changes its path, not only its
  first segment, and Cargo's `package` key does not spare it: a dependent
  that keeps the old local name still writes `rlevo_core::fitness` or
  `rlevo_core::util::seed` for those four items.
- The log target and two strings of the static report change without a
  compile error (Decision 4).
- crates.io holds two names for one crate from the next release on. The
  new crate starts with no download history and no earlier versions on
  docs.rs, and a search for the old name finds a crate that says to go
  elsewhere.
- Thirty-five ADRs name a crate that no longer exists under that name. The
  Status section is the key to them, and the index repeats it.
- `git log` on a file under `crates/rlevo-harness/` stops at this commit
  unless it is given `--follow`.

**Unchanged.**

- Every path under `rlevo::` but `rlevo::harness::agent` and
  `rlevo::harness::seed`, which Decision 8 removes.
- The record format and every metric name. The static report differs in
  two strings: its generator tag and, in a report with a client, its
  `<noscript>` note.
- The published crates, eight, and the order they package in: the harness
  sits where it sat, after `rlevo-metrics-registry` and before
  `rlevo-environments`.
- `Cargo.lock`, apart from the package's name in its own entry and in the
  three entries that depend on it.

**Surface delta**, from `tools/public-surface.sh` with all features on: the
base commit's listing for `rlevo-benchmarks`, with the crate's name
substituted, against the listing for `rlevo-harness`. Six lines leave and
none enter, from 632 to 626: the modules `rlevo_harness::agent` and
`rlevo_harness::seed`, and the four re-exports in them. The other 626 are
the base commit's, line for line.

**How it was checked.** The substitution is a script that leaves
`docs/adr/`, `CHANGELOG.md` and `Cargo.lock` alone and skips a match that
begins the report client's name; the unreleased section of `CHANGELOG.md`
was edited by hand. Run again in checking mode it reports the mentions of
the old name that remain outside those three and fails unless each is one
of the sentences that say what the crate used to be called. The
workspace compiles with all targets and all features. `cargo package
--workspace --no-verify` packages the eight crates in the order above,
which shows that a workspace member crates.io has never seen packages
beside its dependents. The four outside-in probes build small consumer
crates against the packaged workspace and pass, the thermostat with the
two means it printed before. The fast, heavy and GPU test tiers pass. The
five `katex-header.html` files, whose comment lists the crates that carry
one, are still byte-identical.

## Alternatives considered

**Keep the name and the warning.** Rejected. The cost is paid by every
reader of every document for as long as the crate exists, and after the
next release the path `rlevo::harness` would have a history under the old
crate name that a later rename could not remove.

**Rename later, at 1.0.** Rejected for the same reason, and because the
count only grows in the one place it cannot be reduced. ADR 0084,
accepted the day before this one, names the old crate 21 times and
cannot be edited.

**Keep `rlevo-benchmarks` in the workspace as a crate that re-exports
`rlevo-harness`.** Rejected. It is a ninth published crate with a manifest,
a README and a place in the publish order, kept to spare dependents an edit
to their manifest and to the first segment of their paths. The single
pointer release of Decision 6 tells them, once.

**Yank the old crate.** Rejected; Decision 6 gives the reason.

**Another name.** `rlevo-eval`, `rlevo-evaluation` and `rlevo-evaluator`
are free as well. `evaluation` is already a module of `rlevo-core`, and
the umbrella's path has been `harness` since ADR 0084, so any other crate
name would restore the disagreement this ADR removes.

**Keep the old log target, so existing filters keep working.** Rejected. A
target that names a crate which does not exist is a second trap, and the
filter a new user writes, on the crate's name, would be the one that
matches nothing.

**Keep the `agent` and `seed` modules.** Rejected. They were kept for
code that imports through the old crate's paths. After the rename such an
import compiles only for a dependent that keeps the old local name with
Cargo's `package` key, and only until it changes the name. Kept, the
modules would be a second path to four items, for good, that every reader
of the harness's documentation has to learn is the same as the first.

**Rename the report client as well.** Not done. It is unpublished, so the
rename would help no dependent, and it would add about 70 occurrences to a
change whose point is the published name.

**Rename `BenchmarkReport`, `rlevo_environments::bench` and
`BenchableAgent` in the same change.** Not decided here. Each is a public
item with its own callers and its own reason for the word, and one of them
is a trait in the contract crate. A crate rename that also renamed types
would no longer be a substitution a reviewer can check by its predicate.

## References

- ADR [0001](0001-keep-environments-and-benchmarks-separate.md), whose
  separation stands.
- ADR [0004](0004-move-bench-traits-into-rlevo-core.md) and ADR
  [0077](0077-remove-benchenv.md), which took away what the crate was
  named for.
- ADR [0084](0084-the-umbrella-carries-the-evaluation-harness.md), whose
  Decision 2 named the re-export and left the crate's name open.
- ADR [0085](0085-the-generation-loop-is-named-evolutionary-loop.md),
  which freed the word.
- `docs/rules.md` sections 1 and 5.
- Code: `crates/rlevo-harness/`; `crates/rlevo/src/lib.rs`, the re-export.
