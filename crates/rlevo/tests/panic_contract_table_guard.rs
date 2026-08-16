//! Workspace-scoped guard for issue — the "Documented Panic Contracts"
//! table in `docs/rules.md` is **prose**, and prose does not fail CI.
//!
//! The table is the repo's single index of every place a `rlevo` API is allowed
//! to panic. Nothing compiles it, so nothing notices when the code underneath a
//! row moves.
//!
//! This guard closes the **first** direction only: *every row names something
//! that resolves in the workspace*. It deliberately does **not** attempt the
//! second (every panic site appears as a row) — that requires deciding which
//! `panic!`/`assert!`/`unwrap` is a *contract* rather than an internal
//! invariant, which is a judgement no text scan makes well, and a guard that
//! makes it badly gets deleted. Missing rows — a `HillClimbingParams::with_*`
//! setter panicking exactly like its tabled `SimulatedAnnealingParams::with_*`
//! sibling but never earning a row — stay a review concern this guard does not
//! check.
//!
//! # Scope: the table, and nothing but the table
//!
//! Checked: the rows between `### Documented Panic Contracts` and the first
//! non-`|` line after them, **Site column only**.
//!
//! Not checked: `include_str!("../../docs/rules.md")` surrounding prose. Its backticked identifiers
//! (`Box<dyn Error>`, `map_err`, `.unwrap_or_default()`, `EnvironmentError`, …)
//! are *illustrative mentions*, not contract claims. Feeding them to a resolver
//! would force an allowlist of "this backtick is prose" rows — and a guard whose
//! bulk is allowlist trains its readers to add a row instead of reading the
//! failure, which is precisely the state in which it stops catching anything.
//!
//! The **Condition** column is likewise never parsed. It contains things that
//! cannot resolve and must not be attempted: `ACTION_COUNT` (an associated const
//! on a user's own type), `u64::MAX`, \\(\lambda < \mu\\). Only field index 1 of each row —
//! the Site cell — reaches the resolver.
//!
//! # Why `fs::read_to_string` and never `include_str!`
//!
//! `rlevo` is **published**. `include_str!("../../docs/rules.md")` reaches
//! outside the package root, so it works locally and forever, and then breaks
//! exactly once: inside `cargo package`'s verify build, which compiles from a
//! tarball that cannot contain files above the crate root.
//! `rlevo-core/Cargo.toml` already records this failure class for the
//! `KaTeX` header (`--html-in-header` with a `../../` path "silently works
//! locally but 404s on docs.rs"). Same trap, same crate family, second
//! occurrence. [`RULES_MD`] is therefore read from disk **at test time**, where
//! a missing file is a loud, local, developer-machine failure and never a
//! packaging one. Do not "simplify" it to `include_str!`.
//!
//! # How a Site cell becomes a checked claim
//!
//! Every backtick-delimited span in the Site cell is one **spec**, normalized in
//! order:
//!
//! 1. A trailing argument list is dropped — `ContinuousAction::from_slice(v)`
//!    becomes `ContinuousAction::from_slice`.
//! 2. Braces expand — `ops::crossover::{blx_alpha, uniform_crossover}` becomes
//!    two specs. A `{` surviving expansion is a **hard failure**, never a skip.
//! 3. The path splits on `::`: last segment is the *item*, the rest the
//!    *qualifier*.
//! 4. A spec with an **empty qualifier inherits the preceding spec's qualifier
//!    within the same cell**. This is what turns the trailing
//!    `` (via `refine_impl`) `` on the two `local_search::*` rows into a
//!    *checked* claim rather than an exemption: `refine_impl` is resolved
//!    against `local_search::hill_climbing`, the module its cell already named.
//! 5. The qualifier's last segment is tiered **by case**: an uppercase ASCII
//!    first character means a type, anything else a module. That is reliable
//!    here only because every crate sets `[lints] workspace = true`, so
//!    `non_camel_case_types` / `non_snake_case` are live and the convention is
//!    machine-enforced.
//!
//! The three multi-item spellings the table uses — ` / `, `{a, b}`, and the
//! parenthetical — all fall out of those five rules. **Nothing is exempted for
//! the way it is spelled**, which is the property that keeps the exemption list
//! at one row.
//!
//! # Resolution, and the false positive it is designed against
//!
//! A **module-tier** qualifier `a::b` is anchored at each crate root — `src/a/b.rs`
//! *or* `src/a/b/`, never suffix-matched — and a directory candidate contributes
//! **every `.rs` beneath it, recursively**.
//!
//! That recursion is the point. `hill_climbing.rs` is 732 lines;
//! `hill_climbing/{mod.rs, params.rs, refine.rs}` is the ordinary next move, and
//! it is a **pure refactor** — no contract changes, no rename, no panic added or
//! removed. A file-exact resolver would turn that diff red, accuse its author of
//! stale docs, and earn itself a deletion. The recursive subtree is the
//! mitigation.
//!
//! **The cost, taken deliberately and stated here so it is not rediscovered as a
//! bug:** this guard proves *"this name exists somewhere in this module"*, not
//! *"in this file"*. A row pointing at the wrong file inside the right module
//! passes.
//!
//! A **type-tier** qualifier resolves its type declaration first (unique across
//! the workspace, or the guard fails and names every site), then looks for the
//! item's `fn` in the declaring file — widening to any file whose production
//! source carries an `impl … T` or a bare `for T` line. The second alternative
//! is not redundant: `History`'s impl header **wraps**, and
//! `rlevo-reinforcement-learning/src/experience.rs:103-105` reads
//! `impl<…> Index<usize>` / `for History<D, AD, O, A, R>` / `{`. Any owner parse
//! that assumes one impl per line drops it.
//!
//! ## Step C is unreached by today's table, and that is recorded, not hidden
//!
//! Every type-tier row in the current table resolves at **step B**: each item's
//! `fn` happens to sit in its type's own declaring file — `History::index`
//! included, since `experience.rs` declares both `struct History` (line 98) and
//! `fn index` (line 119). Step C exists for the ordinary case where a method
//! lives in a different file from its type, and it is kept because that case
//! costs nothing to support and is expensive to diagnose when unsupported.
//!
//! [`the_guard_is_armed`] therefore requires a live row through step B, the
//! module tier, and the glob tier — but **not** through step C, because that
//! would be asserting a fact about which file a method happens to live in,
//! which any legitimate refactor may flip. Step C's one subtle part, the wrapped
//! header, is pinned directly instead by
//! [`the_wrapped_impl_header_is_recognized`].
//!
//! Matching is **kind-agnostic** — a production `fn <item>` counts whether it is
//! a free function, an inherent method, or a trait-impl method. That is load
//! bearing rather than lax: the six `local_search::…::{refine,
//! refine_with_known_fitness, refine_impl}` specs are spelled as module paths but
//! are *associated items* (`impl<B: Backend> LocalSearch<B> for HillClimbing` at
//! `hill_climbing.rs:355`, its methods at `:361` and `:377`, the inherent
//! `refine_impl` at `:230`). A free-function-only resolver fails 6 of the 37
//! specs on day one. The right response is **not** to rewrite those cells as
//! `HillClimbing::refine`: the module path tells a reader where the code lives,
//! which is information the type path does not carry.
//!
//! # Glob rows, and why they need two matches
//!
//! `ops::selection::tournament_*` prefix-matches production `fn` names and
//! requires **at least two**. A glob that matches one item should have named
//! that item — the more so because
//! `ops::selection::{tournament_select, truncation_select}` is already its own
//! row directly below.
//!
//! **`tournament_*` matches five names in `ops/selection.rs`, three of which are
//! `#[cfg(test)]` functions.** Without production-line filtering the check is
//! *vacuous*: both production functions could be deleted and the glob would
//! still match. With it, the count is exactly **2**, for both globs — this ships
//! green with **zero margin above the floor**. That is intentional and it is
//! recorded here so the next person is not surprised: deleting either
//! `tournament_indices_host` or `tournament_select` fails this guard and the fix
//! is to edit the *table*, naming whichever function survives.
//!
//! # `#[cfg(test)]` regions are skipped
//!
//! A panic contract is a claim about shipped code. A `#[cfg(test)]` fixture
//! named `tournament_size_one_is_uniform_random` is not an implementation of
//! `tournament_*`, and letting it satisfy a row is how a guard goes quietly
//! vacuous (see the glob paragraph — this is not hypothetical here).
//!
//! [`cfg_test_regions`] is **lifted verbatim** from
//! `rlevo-reinforcement-learning/tests/bounds_strictness_guard.rs:811-865`,
//! including its fail-loud behaviour: an unterminated region panics rather than
//! swallowing the rest of the file, and each region must end on a line at the
//! attribute's own indentation, so a brace miscount reports itself instead of
//! hiding production code.
//!
//! **The duplication is deliberate — do not factor it into `rlevo-test-support`.**
//! ADR 0062 §4 and ADR 0068 §Decision 2 both rest on each source-text guard
//! being *independently deletable*; a shared helper makes deleting one of them a
//! cross-crate change, which is exactly the friction that keeps a guard alive
//! past its usefulness. `bounds_strictness_guard.rs` carries the same note about
//! `rng_seeding_guards.rs`; this is the third copy and that is fine.
//!
//! # Known limits
//!
//! - It reads **source text**. `\bfn\s+name\b` inside a `/* … */` block comment
//!   reads as code (safe direction: a false *pass* of a row, never a false
//!   accusation), and hand-formatting that breaks `fn name` across lines defeats
//!   it. CI enforces `cargo fmt --all --check`, which is what makes the
//!   assumption hold.
//! - It proves a **name exists**, not that it panics, not that it panics for the
//!   stated Condition, and not that the Condition column is true at all. The
//!   Condition cells are never read.
//! - Module tier proves the name is somewhere in the module subtree, not in a
//!   particular file (see above — this is the deliberate cost).
//! - `macro_rules!`-generated methods are invisible; a contract on one would need
//!   an [`UNRESOLVABLE_SITES`] row.
//!
//! # Armed-ness
//!
//! Every parse failure **panics**; there is no silent skip anywhere in this
//! file. On top of that, [`the_guard_is_armed`] asserts the guard is still
//! *doing* something: row and spec floors, a live resolution through each tier a
//! row can reach, and a floor on the raw `fn` count under
//! `rlevo-evolution/src/ops` so a scanner that has stopped parsing anything
//! cannot pass by finding nothing to check.
//! [`the_wrapped_impl_header_is_recognized`] covers the one tier no row reaches.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// `docs/rules.md`, resolved from this crate's root and read **at run time**.
///
/// Never `include_str!` — see the module docs: the `../../` escapes the package
/// root and breaks `cargo package`'s verify build for a published crate.
const RULES_MD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/rules.md");

/// The workspace's `crates/` directory. Same run-time-read reasoning.
const CRATES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

/// The heading that opens the table.
const TABLE_HEADING: &str = "### Documented Panic Contracts";

/// The header row the parser expects, normalized to single spaces.
const EXPECTED_HEADER: &str = "| Site | Condition |";

/// Below this many rows the table has been gutted and the guard is checking
/// nothing worth checking. Thirty rows today.
const MINIMUM_ROWS: usize = 25;

/// Below this many resolvable specs the *parser* has broken, not the table.
/// Thirty-seven specs today.
const MINIMUM_SPECS: usize = 30;

/// Crates a documented panic contract must **not** resolve into.
///
/// `docs/rules.md` §1 fixes the dependency direction: production crates must not
/// depend on the benchmark, example, or test-support layer. A `rules.md` panic
/// contract is a promise to *users of the library*, so a row that resolves only
/// here is a real defect — the contract is unreachable from anything a user
/// depends on. It is a **failure**, not a skip. Zero rows hit this today.
const NON_PRODUCTION_CRATES: &[&str] = &[
    "rlevo-benchmarks",
    "rlevo-benchmarks-report-client",
    "rlevo-examples",
    "rlevo-test-support",
];

/// A Site cell that names **no resolvable item**, with the reason.
///
/// Keyed on the cell's **verbatim** text, backticks and all. That is what makes
/// the list bidirectionally checked: [`exemption_rows_match_a_live_table_row`]
/// fails if the row is deleted or reworded, and
/// [`no_exempted_row_names_an_identifier`] fails if it grows a backticked
/// identifier that could have been resolved instead.
///
/// This is for contracts that genuinely name no item. It is **not** the way to
/// silence a rename.
struct UnresolvableSite {
    /// The Site cell, byte-for-byte as `docs/rules.md` spells it.
    site: &'static str,
    /// Why no item can be named.
    why: &'static str,
}

/// The one contract in the table that no single item owns.
static UNRESOLVABLE_SITES: &[UnresolvableSite] = &[UnresolvableSite {
    site: "Batch rank assertion",
    why: "a const-generic *relation* (`BD == D + 1`), asserted at every batching call \
          site rather than owned by one function. There is no item to name; naming any \
          one of the call sites would make the row wrong at all the others",
}];

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// One parsed `| Site | Condition |` row.
#[derive(Debug)]
struct TableRow {
    /// 1-based line number in `docs/rules.md`, so a failure is jumpable.
    line: usize,
    /// The Site cell, verbatim (backticks included) — the exemption key.
    site: String,
    /// Every backtick span in the Site cell, normalized and in order.
    specs: Vec<Spec>,
}

/// Which resolution path accepted a spec. Tracked only so
/// [`the_guard_is_armed`] can prove each path is still exercised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Tier {
    /// Type tier, Step B: the `fn` sits in the type's declaring file.
    TypeDeclaringFile,
    /// Type tier, Step C: found by widening to `impl … T` / `for T` files.
    TypeImplWiden,
    /// Module tier: the `fn` sits in the named module's subtree.
    Module,
    /// Glob tier: at least two production `fn` names share the prefix.
    Glob,
}

/// One resolvable claim extracted from a Site cell.
#[derive(Debug, Clone)]
struct Spec {
    /// The span exactly as `docs/rules.md` wrote it, for failure messages.
    verbatim: String,
    /// Path segments before the item, after brace expansion and inheritance.
    qualifier: Vec<String>,
    /// The final segment. A trailing `*` marks a glob and is kept here.
    item: String,
}

impl Spec {
    /// `a::b::item`, the normalized form the resolver actually looks up.
    fn path(&self) -> String {
        let mut parts = self.qualifier.clone();
        parts.push(self.item.clone());
        parts.join("::")
    }

    /// Whether the item is a `prefix_*` glob.
    fn is_glob(&self) -> bool {
        self.item.ends_with('*')
    }

    /// The qualifier's last segment — the type name or the innermost module.
    fn owner(&self) -> &str {
        self.qualifier
            .last()
            .expect("a spec with an empty qualifier is rejected during parsing")
    }

    /// Type tier when the owner starts with an uppercase ASCII letter.
    fn is_type_tier(&self) -> bool {
        self.owner()
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase())
    }
}

/// Reads and parses the panic-contract table. Panics on any structural
/// surprise; there is no "table looked odd, skipping" path.
fn table_rows() -> Vec<TableRow> {
    let text = fs::read_to_string(RULES_MD).unwrap_or_else(|err| {
        panic!(
            "cannot read {RULES_MD}: {err}\n\nThis guard reads docs/rules.md from disk at \
             test time on purpose (never `include_str!`, which escapes the package root \
             and breaks `cargo package` for a published crate — see the module docs). If \
             docs/rules.md moved, update RULES_MD."
        )
    });
    let lines: Vec<&str> = text.lines().collect();

    let heading = lines
        .iter()
        .position(|line| line.trim() == TABLE_HEADING)
        .unwrap_or_else(|| {
            panic!(
                "docs/rules.md: heading `{TABLE_HEADING}` not found. The panic-contract \
                 table is what this guard checks; if the heading was reworded, update \
                 TABLE_HEADING. If the table was deleted, delete this file too — but note \
                 prose-only panic contracts have already let two dead rows (naming a module \
                 ADR 0050 deleted) survive undetected for months before this guard existed."
            )
        });

    let header = (heading..lines.len())
        .find(|index| lines[*index].trim_start().starts_with('|'))
        .unwrap_or_else(|| {
            panic!(
                "docs/rules.md:{}: no table follows `{TABLE_HEADING}`.",
                heading + 1
            )
        });

    let normalized = normalize_spaces(lines[header]);
    assert_eq!(
        normalized,
        EXPECTED_HEADER,
        "docs/rules.md:{}: expected the panic-contract table header to be \
         `{EXPECTED_HEADER}`, found `{normalized}`. This guard reads the **Site** column \
         by position (field 1); a reordered or renamed header would silently feed it \
         Condition cells, which contain unresolvable things like `ACTION_COUNT` and \
         `u64::MAX`. Fix the header or update this parser deliberately.",
        header + 1,
    );

    let separator = header + 1;
    assert!(
        lines
            .get(separator)
            .is_some_and(|line| line.trim_start().starts_with('|') && line.contains("---")),
        "docs/rules.md:{}: expected the markdown header separator (`|---|---|`) directly \
         below the table header.",
        separator + 1,
    );

    let mut rows = Vec::new();
    for (offset, line) in lines[separator + 1..].iter().enumerate() {
        if !line.trim_start().starts_with('|') {
            break;
        }
        let number = separator + 2 + offset;
        let site = site_cell(line, number);
        let specs = parse_specs(&site, number);
        rows.push(TableRow {
            line: number,
            site,
            specs,
        });
    }

    assert!(
        rows.len() >= MINIMUM_ROWS,
        "docs/rules.md:{}: parsed only {} panic-contract rows, expected at least \
         {MINIMUM_ROWS}. Either the table shrank drastically or the row parser broke; \
         both are failures, because a guard over an empty table passes trivially.",
        separator + 2,
        rows.len(),
    );

    rows
}

/// Field index 1 of a `|`-split row: the Site cell, trimmed but otherwise
/// verbatim.
fn site_cell(line: &str, number: usize) -> String {
    let fields: Vec<&str> = line.split('|').collect();
    assert!(
        fields.len() >= 3,
        "docs/rules.md:{number}: `{line}` is not a two-column table row.",
    );
    fields[1].trim().to_owned()
}

/// Every backtick span in `site`, normalized into [`Spec`]s.
///
/// Panics rather than skipping on every malformed shape: a surviving `{`, a
/// bare name with nothing to inherit from, an owner that is neither
/// `UpperCamel` nor `snake_case`.
fn parse_specs(site: &str, number: usize) -> Vec<Spec> {
    let mut specs: Vec<Spec> = Vec::new();

    for span in backtick_spans(site) {
        let stripped = strip_arguments(span);

        for expanded in expand_braces(stripped) {
            assert!(
                !expanded.contains('{') && !expanded.contains('}'),
                "docs/rules.md:{number}: `{span}` still contains a brace after expansion \
                 (`{expanded}`). The guard refuses to skip a spec it cannot parse — a \
                 skipped spec is a row nobody checks. Spell the items out, or extend \
                 `expand_braces`.",
            );

            let mut segments: Vec<String> = expanded
                .split("::")
                .map(|part| part.trim().to_owned())
                .collect();
            let item = segments.pop().expect("split always yields one segment");
            assert!(
                !item.is_empty(),
                "docs/rules.md:{number}: `{span}` normalizes to an empty item name.",
            );

            if segments.is_empty() {
                // Bare-name inheritance: `(via `refine_impl`)` is a claim about
                // the module the same cell already named, not an exemption.
                let previous = specs.last().unwrap_or_else(|| {
                    panic!(
                        "docs/rules.md:{number}: `{span}` is a bare name with no earlier \
                         backticked path in the same Site cell to inherit a qualifier \
                         from. Write it as `module::{item}` or `Type::{item}`.",
                    )
                });
                segments.clone_from(&previous.qualifier);
            }

            let spec = Spec {
                verbatim: span.to_owned(),
                qualifier: segments,
                item,
            };

            let owner = spec.owner();
            let first = owner.chars().next().unwrap_or_else(|| {
                panic!("docs/rules.md:{number}: `{span}` has an empty qualifier segment.")
            });
            assert!(
                first.is_ascii_alphabetic() || first == '_',
                "docs/rules.md:{number}: `{span}`'s qualifier segment `{owner}` starts with \
                 `{first}`, which is neither a type nor a module name. The guard tiers by \
                 case (uppercase = type, otherwise module), which every crate's `[lints] \
                 workspace = true` keeps honest via `non_camel_case_types` / \
                 `non_snake_case`.",
            );

            specs.push(spec);
        }
    }

    specs
}

/// The contents of each `` `…` `` pair, in order.
fn backtick_spans(text: &str) -> Vec<&str> {
    text.split('`')
        .enumerate()
        .filter_map(|(index, part)| (index % 2 == 1).then_some(part.trim()))
        .filter(|part| !part.is_empty())
        .collect()
}

/// Drops a trailing argument list: `from_slice(v)` becomes `from_slice`.
fn strip_arguments(span: &str) -> &str {
    match (span.find('('), span.ends_with(')')) {
        (Some(open), true) => span[..open].trim_end(),
        _ => span,
    }
}

/// Expands `prefix::{a, b}` into `prefix::a`, `prefix::b`. Anything without a
/// brace passes through unchanged.
fn expand_braces(span: &str) -> Vec<String> {
    let Some(open) = span.find('{') else {
        return vec![span.to_owned()];
    };
    let Some(close) = span.rfind('}') else {
        return vec![span.to_owned()];
    };
    if close < open {
        return vec![span.to_owned()];
    }

    let prefix = &span[..open];
    span[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| format!("{prefix}{item}"))
        .collect()
}

/// Collapses runs of whitespace to single spaces, for header comparison.
fn normalize_spaces(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// The workspace source index
// ---------------------------------------------------------------------------

/// One `.rs` file under some `crates/*/src`, with its production lines marked.
#[derive(Debug)]
struct SourceFile {
    /// The owning crate's directory name, e.g. `rlevo-evolution`.
    crate_name: String,
    /// Path relative to the crate's `src/`, `/`-separated.
    relative: String,
    /// `crates/<crate>/src/<relative>`, for failure messages.
    display: String,
    /// The file's lines.
    lines: Vec<String>,
    /// Parallel to [`SourceFile::lines`]: production code, or not.
    production: Vec<bool>,
}

impl SourceFile {
    /// Whether some production line declares a `fn` named exactly `item`.
    fn declares_fn(&self, item: &str) -> bool {
        self.production_fn_names().any(|name| name == item)
    }

    /// Production `fn` names in this file, with duplicates.
    fn production_fn_names(&self) -> impl Iterator<Item = &str> {
        self.lines
            .iter()
            .zip(&self.production)
            .filter(|(_, production)| **production)
            .flat_map(|(line, _)| fn_names(line))
    }

    /// Whether a production line looks like an impl block header for `ty`,
    /// including the `for T` continuation of a **wrapped** header (see the
    /// module docs: `History`'s `impl<…> Index<usize>` / `for History<…>`).
    fn mentions_impl_of(&self, ty: &str) -> bool {
        self.lines
            .iter()
            .zip(&self.production)
            .filter(|(_, production)| **production)
            .any(|(line, _)| {
                let trimmed = line.trim_start();
                let impl_header = starts_with_word(trimmed, "impl") && contains_word(trimmed, ty);
                let wrapped_for = strip_word(trimmed, "for")
                    .is_some_and(|rest| rest.trim_start().starts_with(ty))
                    && contains_word(trimmed, ty);
                impl_header || wrapped_for
            })
    }

    /// The declared type name on a production line, if the line declares one.
    fn type_declarations(&self) -> impl Iterator<Item = (usize, String)> + '_ {
        self.lines
            .iter()
            .zip(&self.production)
            .enumerate()
            .filter(|(_, (_, production))| **production)
            .filter_map(|(index, (line, _))| {
                declared_type_name(line).map(|name| (index + 1, name.to_owned()))
            })
    }
}

/// Every `.rs` file under every `crates/*/src`, parsed once per test binary.
fn workspace() -> &'static Vec<SourceFile> {
    static WORKSPACE: OnceLock<Vec<SourceFile>> = OnceLock::new();
    WORKSPACE.get_or_init(|| {
        let crates_dir = Path::new(CRATES_DIR);
        let mut entries: Vec<PathBuf> = fs::read_dir(crates_dir)
            .unwrap_or_else(|err| panic!("cannot read {}: {err}", crates_dir.display()))
            .map(|entry| entry.expect("readable directory entry").path())
            .collect();
        entries.sort();

        let mut files = Vec::new();
        for crate_root in entries {
            let src = crate_root.join("src");
            if !src.is_dir() {
                continue;
            }
            let crate_name = crate_root
                .file_name()
                .expect("a crate directory has a name")
                .to_string_lossy()
                .into_owned();

            for path in rust_sources(&src) {
                let relative = relative_path(&src, &path);
                let display = format!("crates/{crate_name}/src/{relative}");
                let text = fs::read_to_string(&path)
                    .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
                let borrowed: Vec<&str> = text.lines().collect();
                let production = production_flags(&display, &borrowed);
                files.push(SourceFile {
                    crate_name: crate_name.clone(),
                    relative,
                    display,
                    lines: borrowed.iter().map(|line| (*line).to_owned()).collect(),
                    production,
                });
            }
        }

        assert!(
            !files.is_empty(),
            "no crate sources found under {CRATES_DIR}; the workspace scan is broken.",
        );
        files
    })
}

/// Per-line "this is production code" flags: not a comment, not inside a
/// `#[cfg(test)]` region.
fn production_flags(relative: &str, lines: &[&str]) -> Vec<bool> {
    let regions = cfg_test_regions(relative, lines);
    (0..lines.len())
        .map(|index| {
            !lines[index].trim_start().starts_with("//")
                && !regions
                    .iter()
                    .any(|(start, end)| (*start..=*end).contains(&index))
        })
        .collect()
}

/// Inclusive `(first, last)` line index pairs for every `#[cfg(test)]` region.
///
/// Fail-loud by construction: an unterminated region panics rather than
/// swallowing the rest of the file, and the terminating line must sit at the
/// attribute's own indentation and look like an item end (`}` for a braced item
/// such as `mod tests`, `;` for a braceless one). Those two checks are what stop
/// a brace miscount from quietly hiding production code — see the module docs.
///
/// Lifted verbatim from
/// `rlevo-reinforcement-learning/tests/bounds_strictness_guard.rs:811-865`
/// (itself lifted from `rlevo-environments/tests/rng_seeding_guards.rs`), with
/// only the panic messages' subject changed. **Deliberately duplicated rather
/// than shared**: each of these guards must stay independently deletable (ADR
/// 0062 §4, ADR 0068 §Decision 2), and a shared test-support helper would make
/// deleting any one of them a cross-crate change.
fn cfg_test_regions(relative: &str, lines: &[&str]) -> Vec<(usize, usize)> {
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut index = 0;

    while index < lines.len() {
        if lines[index].trim() != "#[cfg(test)]" {
            index += 1;
            continue;
        }

        let attribute_indent = indent_of(lines[index]);
        let mut depth: i64 = 0;
        let mut opened = false;
        let mut end: Option<usize> = None;

        for (offset, line) in lines[index + 1..].iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            depth += count_char(line, '{') - count_char(line, '}');
            opened |= line.contains('{');

            if (opened && depth <= 0) || (!opened && trimmed.ends_with(';')) {
                end = Some(index + 1 + offset);
                break;
            }
        }

        let end = end.unwrap_or_else(|| {
            panic!(
                "{relative}: the `#[cfg(test)]` at line {} never terminates. This guard \
                 refuses to skip to end-of-file, because doing so would hide every \
                 production `fn` below it and let a test-only function satisfy a \
                 documented panic contract — precisely the failure mode this guard file \
                 exists to close.",
                index + 1,
            )
        });

        let last = lines[end].trim();
        assert!(
            indent_of(lines[end]) == attribute_indent && (last == "}" || last.ends_with(';')),
            "{relative}: the `#[cfg(test)]` region opened at line {} appears to end at \
             line {} (`{last}`), which is not an item end at the attribute's indentation. \
             The brace scan is off — probably a `{{` or `}}` inside a string literal — and \
             an over-long region would hide production code from this guard.",
            index + 1,
            end + 1,
        );

        regions.push((index, end));
        index = end + 1;
    }

    regions
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// A successful resolution: which tier accepted it, and where.
#[derive(Debug)]
struct Resolution {
    /// The path that accepted the spec.
    tier: Tier,
    /// Files that contained the match, for the armed-ness report.
    files: Vec<String>,
    /// Crates that contained the match.
    crates: BTreeSet<String>,
}

/// Resolves one spec, or returns a message explaining what was tried and where.
fn resolve(spec: &Spec) -> Result<Resolution, String> {
    if spec.is_type_tier() {
        resolve_type(spec)
    } else {
        resolve_module(spec)
    }
}

/// Module tier: candidates are `src/<qualifier>.rs` and everything under
/// `src/<qualifier>/`, anchored at each crate root.
fn resolve_module(spec: &Spec) -> Result<Resolution, String> {
    let prefix = spec.qualifier.join("/");
    let file_candidate = format!("{prefix}.rs");
    let dir_prefix = format!("{prefix}/");

    let mut by_crate: BTreeMap<String, Vec<&SourceFile>> = BTreeMap::new();
    for file in workspace() {
        if file.relative == file_candidate || file.relative.starts_with(&dir_prefix) {
            by_crate
                .entry(file.crate_name.clone())
                .or_default()
                .push(file);
        }
    }

    if by_crate.is_empty() {
        return Err(format!(
            "no module `{}` exists in the workspace. Looked for `src/{file_candidate}` and \
             `src/{dir_prefix}` (recursively) under every `crates/*/src`.",
            spec.qualifier.join("::"),
        ));
    }
    if by_crate.len() > 1 {
        return Err(format!(
            "module `{}` is ambiguous — it exists in {}. A panic contract must name one \
             item; disambiguate the row.",
            spec.qualifier.join("::"),
            by_crate.keys().cloned().collect::<Vec<_>>().join(" and "),
        ));
    }

    let (crate_name, candidates) = by_crate
        .into_iter()
        .next()
        .expect("exactly one crate remains");

    if spec.is_glob() {
        return resolve_glob(spec, &crate_name, &candidates);
    }

    let hits: Vec<&&SourceFile> = candidates
        .iter()
        .filter(|file| file.declares_fn(&spec.item))
        .collect();

    if hits.is_empty() {
        return Err(format!(
            "module `{}` exists in `{crate_name}`, but no production line in it declares \
             `fn {}`. Searched {} file(s): {}.",
            spec.qualifier.join("::"),
            spec.item,
            candidates.len(),
            joined(candidates.iter().map(|file| file.display.clone())),
        ));
    }

    Ok(Resolution {
        tier: Tier::Module,
        files: hits.iter().map(|file| file.display.clone()).collect(),
        crates: std::iter::once(crate_name).collect(),
    })
}

/// Glob tier: `prefix_*` must match **two or more** production `fn` names. One
/// match means the row should have named the item.
fn resolve_glob(
    spec: &Spec,
    crate_name: &str,
    candidates: &[&SourceFile],
) -> Result<Resolution, String> {
    let prefix = spec.item.trim_end_matches('*');
    let mut matched: BTreeSet<String> = BTreeSet::new();
    let mut files: BTreeSet<String> = BTreeSet::new();

    for file in candidates {
        for name in file.production_fn_names() {
            if name.starts_with(prefix) {
                matched.insert(name.to_owned());
                files.insert(file.display.clone());
            }
        }
    }

    if matched.len() < 2 {
        return Err(format!(
            "glob `{}` matches {} production function(s) in `{}` ({}). A glob promises a \
             family; one match means the row should name that item, and zero means the \
             family is gone. Note that `#[cfg(test)]` functions are excluded on purpose — \
             `tournament_*` matches three test helpers whose presence would make this \
             check vacuous.",
            spec.path(),
            matched.len(),
            spec.qualifier.join("::"),
            if matched.is_empty() {
                "none".to_owned()
            } else {
                joined(matched.iter().cloned())
            },
        ));
    }

    Ok(Resolution {
        tier: Tier::Glob,
        files: files.into_iter().collect(),
        crates: std::iter::once(crate_name.to_owned()).collect(),
    })
}

/// Type tier: locate the unique declaration, then the item's `fn`.
fn resolve_type(spec: &Spec) -> Result<Resolution, String> {
    let ty = spec.owner();

    let mut declarations: Vec<(&SourceFile, usize)> = Vec::new();
    for file in workspace() {
        for (line, name) in file.type_declarations() {
            if name == ty {
                declarations.push((file, line));
            }
        }
    }

    if declarations.is_empty() {
        return Err(format!(
            "no production `struct`/`enum`/`trait`/`union`/`type` named `{ty}` exists \
             anywhere under `crates/*/src`.",
        ));
    }
    if declarations.len() > 1 {
        return Err(format!(
            "`{ty}` is declared in more than one place: {}. A panic contract must resolve \
             to one type; qualify the row.",
            joined(
                declarations
                    .iter()
                    .map(|(file, line)| format!("{}:{line}", file.display)),
            ),
        ));
    }

    let (declaring, line) = declarations[0];

    // Step B: the declaring file — or, for a `mod.rs`, its whole directory.
    let step_b: Vec<&SourceFile> =
        if declaring.relative.ends_with("/mod.rs") || declaring.relative == "mod.rs" {
            let dir = declaring
                .relative
                .strip_suffix("mod.rs")
                .expect("checked above");
            workspace()
                .iter()
                .filter(|file| file.crate_name == declaring.crate_name)
                .filter(|file| file.relative.starts_with(dir))
                .collect()
        } else {
            vec![declaring]
        };

    let hits: Vec<&&SourceFile> = step_b
        .iter()
        .filter(|file| file.declares_fn(&spec.item))
        .collect();
    if !hits.is_empty() {
        return Ok(Resolution {
            tier: Tier::TypeDeclaringFile,
            files: hits.iter().map(|file| file.display.clone()).collect(),
            crates: std::iter::once(declaring.crate_name.clone()).collect(),
        });
    }

    // Step C: widen to any file carrying an `impl … T` or a wrapped `for T`.
    let widened: Vec<&SourceFile> = workspace()
        .iter()
        .filter(|file| file.mentions_impl_of(ty))
        .filter(|file| file.declares_fn(&spec.item))
        .collect();

    if widened.is_empty() {
        return Err(format!(
            "`{ty}` is declared at `{}:{line}`, but no production line declares \
             `fn {}` — neither in {} (step B), nor in any of the {} file(s) carrying an \
             `impl … {ty}` or a `for {ty}` header (step C).",
            declaring.display,
            spec.item,
            joined(step_b.iter().map(|file| file.display.clone())),
            workspace()
                .iter()
                .filter(|file| file.mentions_impl_of(ty))
                .count(),
        ));
    }

    Ok(Resolution {
        tier: Tier::TypeImplWiden,
        files: widened.iter().map(|file| file.display.clone()).collect(),
        crates: widened.iter().map(|file| file.crate_name.clone()).collect(),
    })
}

/// Resolves every spec of every non-exempt row, panicking with a full
/// explanation on the first failure. Returns the tier each spec used.
fn resolve_all(rows: &[TableRow]) -> Vec<(usize, Tier)> {
    let mut tiers = Vec::new();

    for row in rows {
        if is_exempt(&row.site) {
            continue;
        }
        assert!(!row.specs.is_empty(), "{}", missing_identifier_message(row),);

        for spec in &row.specs {
            match resolve(spec) {
                Ok(resolution) => {
                    let production: Vec<&String> = resolution
                        .crates
                        .iter()
                        .filter(|name| !NON_PRODUCTION_CRATES.contains(&name.as_str()))
                        .collect();
                    assert!(
                        !production.is_empty(),
                        "{}",
                        non_production_message(row, spec, &resolution),
                    );
                    tiers.push((row.line, resolution.tier));
                }
                Err(reason) => panic!("{}", failure_message(row, spec, &reason)),
            }
        }
    }

    tiers
}

/// Whether an [`UNRESOLVABLE_SITES`] row covers this Site cell verbatim.
fn is_exempt(site: &str) -> bool {
    UNRESOLVABLE_SITES.iter().any(|row| row.site == site)
}

// ---------------------------------------------------------------------------
// Failure messages
// ---------------------------------------------------------------------------

/// The message every resolution failure ends with.
fn how_to_fix() -> String {
    "\nFix the **table**, not this test:\n\
     - the item was renamed  -> update the Site cell;\n\
     - the panic was removed -> delete the row;\n\
     - the code moved into `rlevo-benchmarks` / `rlevo-examples` / \
     `rlevo-test-support` -> that is a `docs/rules.md` §1 dependency-direction \
     problem, not a docs problem: a documented panic contract must live in a \
     crate users depend on.\n\n\
     Adding an `UNRESOLVABLE_SITES` row is for a contract that names **no item at \
     all** (the one current entry is a const-generic relation asserted at many \
     call sites). It is not a way to silence a rename, and \
     `no_exempted_row_names_an_identifier` will reject an exemption whose cell \
     still names something.\n\n\
     This assertion exists because two rows of this table once named a module \
     that had been deleted, and they stayed there for months because nothing but \
     a human reading carefully could notice. Do not delete this assertion."
        .to_owned()
}

/// A spec that did not resolve.
fn failure_message(row: &TableRow, spec: &Spec, reason: &str) -> String {
    let tier = if spec.is_type_tier() {
        "type tier (locate the type declaration, then the method)"
    } else if spec.is_glob() {
        "glob tier (prefix-match production `fn` names in the module)"
    } else {
        "module tier (locate the module directory/file, then the function)"
    };

    format!(
        "\n\ndocs/rules.md:{}: the panic-contract row\n\n    | {} | …\n\n\
         names `{}`, which does not resolve in this workspace.\n\n\
         Normalized spec: `{}`\nResolution attempted: {tier}\nResult: {reason}\n{}",
        row.line,
        row.site,
        spec.verbatim,
        spec.path(),
        how_to_fix(),
    )
}

/// A spec that resolved only into a non-production crate.
fn non_production_message(row: &TableRow, spec: &Spec, resolution: &Resolution) -> String {
    format!(
        "\n\ndocs/rules.md:{}: the panic-contract row\n\n    | {} | …\n\n\
         names `{}`, which resolves **only** inside {} — a benchmark / example / \
         test-support crate.\n\nMatched: {}\n\n\
         `docs/rules.md` §1 fixes the dependency direction: production crates must not \
         depend on `rlevo-benchmarks` or the visualization layer, so a panic contract \
         that lives there is unreachable from anything a user of the library depends on. \
         This is a §1 violation, not a documentation nit.\n{}",
        row.line,
        row.site,
        spec.verbatim,
        joined(resolution.crates.iter().cloned()),
        joined(resolution.files.iter().cloned()),
        how_to_fix(),
    )
}

/// A non-exempt row whose Site cell names nothing at all.
fn missing_identifier_message(row: &TableRow) -> String {
    format!(
        "\n\ndocs/rules.md:{}: the panic-contract row\n\n    | {} | …\n\n\
         contains no backticked identifier, so there is nothing for this guard to \
         resolve. Either spell the item in backticks (`Type::method` or \
         `module::function`), or — if the contract genuinely names no single item — add \
         an `UNRESOLVABLE_SITES` row keyed on the cell's verbatim text with a written \
         reason.\n{}",
        row.line,
        row.site,
        how_to_fix(),
    )
}

/// `a`, `b` and `c` — a readable list for a failure message.
fn joined<I: IntoIterator<Item = String>>(items: I) -> String {
    items.into_iter().collect::<Vec<_>>().join(", ")
}

// ---------------------------------------------------------------------------
// Source-text primitives
// ---------------------------------------------------------------------------

/// Every `fn` name declared on one line — `pub fn`, `pub(crate) async fn`,
/// `unsafe fn`, a trait's bare `fn foo(…);`, all of them.
///
/// Kind-agnostic on purpose: six of the table's specs are trait-impl methods
/// spelled as module paths (see the module docs).
fn fn_names(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut names = Vec::new();
    let mut cursor = 0;

    while let Some(found) = line[cursor..].find("fn") {
        let at = cursor + found;
        cursor = at + 2;

        let preceded_by_identifier = at > 0 && is_identifier_byte(bytes[at - 1]);
        if preceded_by_identifier {
            continue;
        }

        let after = &line[at + 2..];
        let trimmed = after.trim_start();
        if after.len() == trimmed.len() {
            // No whitespace after `fn` — this is part of a longer word.
            continue;
        }

        let name_len = trimmed
            .bytes()
            .take_while(|byte| is_identifier_byte(*byte))
            .count();
        if name_len > 0 {
            names.push(&trimmed[..name_len]);
        }
    }

    names
}

/// The type name a line declares, if it declares one.
fn declared_type_name(line: &str) -> Option<&str> {
    let mut rest = line.trim_start();

    if let Some(after) = strip_word(rest, "pub") {
        rest = after.trim_start();
        if let Some(inner) = rest.strip_prefix('(') {
            let close = inner.find(')')?;
            rest = inner[close + 1..].trim_start();
        }
    }
    for modifier in ["default", "unsafe", "auto"] {
        if let Some(after) = strip_word(rest, modifier) {
            rest = after.trim_start();
        }
    }

    for keyword in ["struct", "enum", "trait", "union", "type"] {
        if let Some(after) = strip_word(rest, keyword) {
            let name = after.trim_start();
            let len = name
                .bytes()
                .take_while(|byte| is_identifier_byte(*byte))
                .count();
            return (len > 0).then(|| &name[..len]);
        }
    }

    None
}

/// Strips `word` from the front of `text`, but only as a whole word.
fn strip_word<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(word)?;
    let continues = rest.bytes().next().is_some_and(is_identifier_byte);
    (!continues).then_some(rest)
}

/// Whether `text` begins with `word` as a whole word.
fn starts_with_word(text: &str, word: &str) -> bool {
    strip_word(text, word).is_some()
}

/// Whether `word` appears in `text` with non-identifier characters on both
/// sides.
fn contains_word(text: &str, word: &str) -> bool {
    let bytes = text.as_bytes();
    let mut cursor = 0;
    while let Some(found) = text[cursor..].find(word) {
        let at = cursor + found;
        cursor = at + word.len();
        let before_ok = at == 0 || !is_identifier_byte(bytes[at - 1]);
        let after_ok = bytes
            .get(at + word.len())
            .is_none_or(|byte| !is_identifier_byte(*byte));
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// ASCII identifier byte: alphanumeric or `_`.
fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Leading-space count. rustfmt indents with spaces, so this is the nesting
/// depth of a formatted line.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// `char` occurrences in a line, as an `i64` so the two counts can be
/// subtracted.
fn count_char(line: &str, needle: char) -> i64 {
    i64::try_from(line.matches(needle).count()).expect("a source line has few braces")
}

/// Every `.rs` file under `root`, recursively.
fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries =
        fs::read_dir(root).unwrap_or_else(|err| panic!("cannot read {}: {err}", root.display()));

    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.is_dir() {
            files.extend(rust_sources(&path));
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
    }
    files.sort();
    files
}

/// `path` relative to `root`, `/`-separated so paths in messages read the same
/// on every platform.
fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or_else(|_| panic!("{} is not under {}", path.display(), root.display()))
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

// ---------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------

/// **The guard.** Every Site cell in the panic-contract table names something
/// that exists in production source, or carries an [`UNRESOLVABLE_SITES`] row.
///
/// This is the mechanical check the table lacked: without it, a row can go on
/// naming a deleted module indefinitely, since prose alone never fails a build.
#[test]
fn every_panic_contract_row_names_a_live_item() {
    let rows = table_rows();
    resolve_all(&rows);
}

/// The exemption list, checked in reverse: every [`UNRESOLVABLE_SITES`] entry
/// must match a **live** row verbatim.
///
/// Without this, deleting or rewording the exempted row leaves a stale entry
/// that quietly excuses nothing — and an allowlist checked in one direction
/// rots into a list of names nobody verifies (ADR 0068 §Decision 2's rule,
/// applied here).
#[test]
fn exemption_rows_match_a_live_table_row() {
    let rows = table_rows();
    let sites: BTreeSet<&str> = rows.iter().map(|row| row.site.as_str()).collect();

    for exemption in UNRESOLVABLE_SITES {
        assert!(
            sites.contains(exemption.site),
            "\n\nUNRESOLVABLE_SITES names the Site cell `{}`, which no longer appears in \
             the panic-contract table in docs/rules.md.\n\nRecorded reason: {}\n\n\
             The key is the cell's **verbatim** text, so this fires when the row is \
             deleted *or* reworded. If the row is gone, delete the exemption. If it was \
             reworded, update the key — and check whether the new wording names an item, \
             in which case delete the exemption and let it resolve.\n\
             Live Site cells: {}\n\nDo not delete this assertion.",
            exemption.site,
            exemption.why,
            joined(sites.iter().map(|site| format!("`{site}`"))),
        );
    }
}

/// The other reverse check: an exempted row that has **gained** a backticked
/// identifier is no longer unresolvable, and must lose its exemption.
///
/// An exemption is a standing claim that no item can be named. The moment the
/// cell names one, the claim is false and the row should be checked like every
/// other.
#[test]
fn no_exempted_row_names_an_identifier() {
    for row in table_rows() {
        let Some(exemption) = UNRESOLVABLE_SITES
            .iter()
            .find(|entry| entry.site == row.site)
        else {
            continue;
        };
        assert!(
            row.specs.is_empty(),
            "\n\ndocs/rules.md:{}: the exempted row `{}` now names `{}`; delete the \
             exemption row and let it resolve.\n\nThe exemption was recorded because {}. \
             That reason no longer holds: the cell identifies an item, so the guard can \
             and should check it.\n\nDo not delete this assertion.",
            row.line,
            row.site,
            row.specs[0].verbatim,
            exemption.why,
        );
    }
}

/// Pins type-tier **step C**'s wrapped-impl-header handling, which no row in
/// today's table exercises.
///
/// `crates/rlevo-reinforcement-learning/src/experience.rs` is the workspace's
/// live example: **both** its `History` impl headers wrap, so
/// `impl<…> Index<usize>` (line 103) and `impl<…>` (line 124) each sit on a line
/// that does not contain `History` at all, and the type name appears only on the
/// continuation line `for History<D, AD, O, A, R>` (line 104). A step C that
/// matched only `^\s*impl\b.*\bT\b` would find **zero** candidate files for
/// `History` and silently degrade — and because step C is a *widening*, its
/// degradation shows up as a spurious failure much later, in some future diff
/// that moves a method out of its type's declaring file.
///
/// This test therefore asserts both halves: that the file *is* recognised, and
/// that the single-line `impl … History` alternative alone would *not* have
/// recognised it. The second assertion is what makes the first one informative.
#[test]
fn the_wrapped_impl_header_is_recognized() {
    let experience = workspace()
        .iter()
        .find(|file| file.display == "crates/rlevo-reinforcement-learning/src/experience.rs")
        .expect(
            "crates/rlevo-reinforcement-learning/src/experience.rs should exist; it holds \
             `History`, the workspace's only wrapped impl header",
        );

    assert!(
        experience.mentions_impl_of("History"),
        "type-tier step C no longer recognises `crates/rlevo-reinforcement-learning/src/\
         experience.rs` as carrying an impl of `History`. Its impl headers **wrap** — the \
         type name is on the `for History<D, AD, O, A, R>` continuation line, not on the \
         `impl` line — so the `for T` alternative in `mentions_impl_of` is load bearing, \
         not redundant. Do not delete this assertion.",
    );
    assert!(
        experience.declares_fn("index"),
        "`fn index` is no longer a production declaration in experience.rs; step C would \
         have nothing to accept even with the candidate file in hand.",
    );

    let single_line_impl_headers = experience
        .lines
        .iter()
        .zip(&experience.production)
        .filter(|(_, production)| **production)
        .filter(|(line, _)| {
            let trimmed = line.trim_start();
            starts_with_word(trimmed, "impl") && contains_word(trimmed, "History")
        })
        .count();
    assert_eq!(
        single_line_impl_headers, 0,
        "experience.rs now has a single-line `impl … History` header, so this test no \
         longer proves the `for T` alternative is required. Find another wrapped header to \
         pin step C against, or delete the alternative and this test together.",
    );
}

/// **Tripwires against a vacuously green guard.**
///
/// A source-text guard fails silently by finding nothing: an empty table, a
/// parser that extracts zero specs, a resolution tier that has quietly stopped
/// being reachable. Each assertion below is a floor under one of those.
///
/// The tier coverage is the interesting one: a resolution path no row reaches
/// is a path nobody can tell is still correct. Type-tier step C is the
/// deliberate exception — no row reaches it today, so it is pinned by
/// [`the_wrapped_impl_header_is_recognized`] instead. See the module docs.
#[test]
fn the_guard_is_armed() {
    let rows = table_rows();
    assert!(
        rows.len() >= MINIMUM_ROWS,
        "the panic-contract table has {} rows, below the floor of {MINIMUM_ROWS}.",
        rows.len(),
    );

    let specs: usize = rows.iter().map(|row| row.specs.len()).sum();
    assert!(
        specs >= MINIMUM_SPECS,
        "the table yields only {specs} resolvable specs, below the floor of \
         {MINIMUM_SPECS}. The rows are still there, so this is the **spec parser** \
         breaking, not the table shrinking — a guard that resolves nothing passes \
         trivially.",
    );

    let tiers: BTreeSet<Tier> = resolve_all(&rows)
        .into_iter()
        .map(|(_, tier)| tier)
        .collect();
    // `Tier::TypeImplWiden` is **absent from this list on purpose** — no row in
    // today's table reaches it, because every type-tier item's `fn` happens to
    // live in its type's declaring file. Asserting it here would be asserting a
    // fact about the workspace's file layout, which a legitimate refactor may
    // change in either direction. The widening step is pinned instead by
    // [`the_wrapped_impl_header_is_recognized`], which tests the mechanism
    // directly. See the module docs, "Step C is unreached by today's table".
    for tier in [Tier::TypeDeclaringFile, Tier::Module, Tier::Glob] {
        assert!(
            tiers.contains(&tier),
            "no row resolved through {tier:?}. Every resolution path must stay exercised \
             by a real row, or it rots: an unexercised path is one nobody can tell is \
             still correct. Currently reached: {tiers:?}.",
        );
    }

    let ops_fns = workspace()
        .iter()
        .filter(|file| file.crate_name == "rlevo-evolution" && file.relative.starts_with("ops/"))
        .flat_map(SourceFile::production_fn_names)
        .count();
    assert!(
        ops_fns >= 10,
        "found only {ops_fns} production `fn` declarations under \
         crates/rlevo-evolution/src/ops. The `fn` scanner has stopped parsing; every \
         module-tier and glob-tier check above is therefore meaningless.",
    );
}
