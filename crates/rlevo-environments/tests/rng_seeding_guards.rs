//! Crate-wide regression net for issue #282 — an environment that re-seeds its
//! RNG inside `reset()` replays one bit-identical episode forever.
//!
//! ADR 0029 §1 and ADR 0062 §2
//! (`docs/adr/0062-grid-layout-fidelity-and-no-dead-rng.md`)
//! state the rule: an environment seeds **once**, in a constructor, and
//! `reset()` draws from the persistent stream and lets it advance. The explicit
//! replay hatch is the inherent `reset_with_seed(&mut self, seed: u64)` — never
//! `reset()` itself.
//!
//! The failure mode this file exists to catch is a *combination*, which is why
//! no per-environment test finds it (ADR 0062 Context §4): a contributor adds
//! sampling to an environment whose `reset()` still carries
//! `self.rng = StdRng::seed_from_u64(self.config.seed);`. The environment is
//! then provably calling its RNG *and* provably producing an identical layout
//! every episode, and both facts are consistent with a green suite. Nine grid
//! environments sat in exactly that pre-state until ADR 0062; #104 put the same
//! line in `toy_text/`, `classic/` and `locomotion/`. Prose did not stop it
//! twice, so ADR 0062 §4 requires this mechanical guard.
//!
//! # What the guard does
//!
//! [`every_seed_from_u64_sits_in_an_allowlisted_constructor`] walks **the whole
//! crate's `src/`** — not just `grids/` — from disk at test time via
//! `CARGO_MANIFEST_DIR`, finds every `seed_from_u64` occurrence, resolves the
//! **enclosing `fn`**, and requires that function to be listed in
//! [`ALLOWED_SEEDERS`]. Crate-wide scope is deliberate (ADR 0062 §4): a
//! `grids/`-only guard guarantees the next recurrence lands in `locomotion/`.
//!
//! Resolving the enclosing function is the whole mechanism, not an
//! optimisation. `reset_with_seed` contains
//! `self.rng = StdRng::seed_from_u64(seed);` three lines above `self.reset()`,
//! so a naive `rg 'seed_from_u64' | rg reset` flags the very method ADR 0029
//! mandates. Line-level greps cannot express this rule.
//!
//! Like `tests/landscape_dim_guards.rs`, the check runs
//! **both ways**: [`no_allowlist_row_is_stale`] fails on a row that no longer
//! matches anything in the tree, so the allowlist cannot rot into a list of
//! names nobody checks.
//!
//! # `#[cfg(test)]` regions are skipped — the choice and its cost
//!
//! Unit-test bodies seed freely (`blackjack.rs`, `frozen_lake.rs`,
//! `unlock_pickup.rs`, `box2d/bipedal_walker/terrain.rs`,
//! `grids/core/placement.rs`), including `env.rng = StdRng::seed_from_u64(seed)`
//! against an environment the test itself owns. Those are skipped, deliberately,
//! rather than allowlisted by name.
//!
//! Allowlisting them was the alternative and is worse: ~30 test-function names
//! would join [`ALLOWED_SEEDERS`], every new seeded unit test would have to edit
//! it, and — because the allowlist is checked bidirectionally — *renaming a
//! test* would fail the build with a stale row. That trains contributors to edit
//! the allowlist reflexively, which is precisely the habit that would let a
//! production function name slip onto it. The rule ADR 0062 §2 states is about
//! the shipped environment's seeding seam; a `#[cfg(test)]` block is not
//! compiled into the library and cannot re-seed a user's episode.
//!
//! **The cost**, stated plainly: a re-seed written inside a `#[cfg(test)]`
//! helper is invisible here, so a test helper that hides a re-seed behind a
//! `fn` the production code also calls would not be caught. That is a
//! `#[cfg(test)]`-only path by construction, so it cannot ship.
//!
//! **Why the skip cannot silently hide production code**:
//! [`cfg_test_regions`] is fail-loud. It panics rather than skipping to
//! end-of-file when a `#[cfg(test)]` region does not terminate, and it asserts
//! that each region ends on a line at the *same indentation* as the attribute
//! that opened it (`}` for a braced item, `;` for a braceless one). A brace
//! miscount — the only way a region could over-extend into production code —
//! lands on a line that fails that shape assertion and reports the file and both
//! line numbers.
//!
//! # Known limits (ADR 0062 §4 requires these be recorded, not rediscovered)
//!
//! The guard reads **source text**. Therefore:
//!
//! - It is **brittle against reformatting**. Enclosing-`fn` resolution assumes
//!   rustfmt's indentation, which CI enforces (`cargo fmt --all --check`). Code
//!   formatted by hand can defeat it.
//! - It is **defeatable by aliasing**: `use rand::SeedableRng as S;` followed by
//!   `S::seed_from_u64(..)` is invisible, as is any re-seed routed through a
//!   helper the guard sees as a legitimate constructor.
//! - It checks **where the seeding is written, not who calls it**. A constructor
//!   invoked *from* `reset()` re-seeds on every episode and passes this guard.
//! - A `seed_from_u64` inside a multi-line `/* … */` block comment is read as
//!   code. That direction is safe: it produces a loud false positive, never a
//!   silent pass.
//!
//! It catches the accident, not the adversary — and per ADR 0062 §4 the accident
//! is the actual threat model. Deleting this file costs nothing, which is what
//! makes it a cheap, reversible decision rather than a load-bearing one.

use std::fs;
use std::path::{Path, PathBuf};

/// The crate's source tree, resolved from the crate root at compile time. Cargo
/// sets `CARGO_MANIFEST_DIR` for integration tests, so this points at the real
/// source the test binary was built from.
const CRATE_SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

/// The seeding call the guard hunts for. `StdRng::seed_from_u64` is the only
/// form used in this crate; `SeedableRng::from_seed` takes a byte array and does
/// not appear.
const SEED_CALL: &str = "seed_from_u64";

/// Where an allowlisted seeding function is allowed to live.
#[derive(Debug, Clone, Copy)]
enum Scope {
    /// Any module in the crate. Reserved for names that mean "constructor"
    /// unambiguously wherever they appear.
    Anywhere,
    /// Exactly one file, spelled relative to `src/` with `/` separators — for
    /// names that mean something *else* elsewhere in the crate.
    File(&'static str),
}

/// One function permitted to call [`SEED_CALL`], with the reason it is
/// permitted.
///
/// `why` is not decoration: an allowlist without recorded reasons is the
/// "has an RNG but does not use it yet" state ADR 0062 §2 exists to make
/// unrepresentable, one level up.
struct AllowedSeeder {
    /// The `fn` name, exactly as it appears after `fn`.
    function: &'static str,
    /// Which files this row covers.
    scope: Scope,
    /// Why seeding here is correct under ADR 0029 §1 / ADR 0062 §2.
    why: &'static str,
}

/// Every function in `rlevo-environments` that may seed an RNG: constructors,
/// which seed once from `config.seed`, and the explicit replay hatch.
///
/// This list is checked in both directions. A `seed_from_u64` whose enclosing
/// function is missing here fails
/// [`every_seed_from_u64_sits_in_an_allowlisted_constructor`]; a row that
/// matches nothing on disk fails [`no_allowlist_row_is_stale`].
static ALLOWED_SEEDERS: &[AllowedSeeder] = &[
    AllowedSeeder {
        function: "with_config",
        scope: Scope::Anywhere,
        why: "the ADR 0026 construction chokepoint — validates config, then seeds \
              the persistent stream exactly once",
    },
    AllowedSeeder {
        function: "with_config_and_dynamics",
        scope: Scope::Anywhere,
        why: "`Acrobot`'s generic-dynamics constructor; `with_config` delegates to it",
    },
    AllowedSeeder {
        function: "new",
        scope: Scope::Anywhere,
        why: "plain constructor — `LunarLanderCore::new`, the shared core the two \
              lander variants build on",
    },
    AllowedSeeder {
        function: "reset_with_seed",
        scope: Scope::Anywhere,
        why: "the inherent replay hatch ADR 0029 §1 mandates. This row is the reason \
              the guard resolves the enclosing fn: the body re-seeds three lines \
              above `self.reset()`, so a line-level grep flags it",
    },
    AllowedSeeder {
        function: "build",
        scope: Scope::File("box2d/bipedal_walker/env.rs"),
        why: "`BipedalWalker`'s private constructor, shared by `with_config` and \
              `with_terrain`. Scoped to one file on purpose: in `grids/`, `build` is \
              the per-episode layout builder that must *receive* `rng: &mut R` and \
              never seed one — allowlisting the bare name crate-wide would open a \
              hole exactly where issue #282 lives",
    },
];

/// One `seed_from_u64` occurrence in production source.
#[derive(Debug)]
struct SeedingSite {
    /// Path relative to `src/`, `/`-separated.
    file: String,
    /// 1-based line number, for a message a reader can jump to.
    line: usize,
    /// The enclosing `fn`, or `None` when the occurrence is not inside one
    /// (a `static`/`const` initialiser, say) — which is itself a violation,
    /// since a seed drawn outside a constructor has no owner.
    function: Option<String>,
}

impl SeedingSite {
    /// Renders as `src/grids/lava_gap.rs:341 (fn reset_with_seed)`.
    fn describe(&self) -> String {
        let function = match &self.function {
            Some(name) => format!("fn {name}"),
            None => "no enclosing fn".to_owned(),
        };
        format!("src/{}:{} ({})", self.file, self.line, function)
    }

    /// Whether some [`ALLOWED_SEEDERS`] row covers this site.
    fn allowed_by(&self) -> Option<&'static AllowedSeeder> {
        let name = self.function.as_deref()?;
        ALLOWED_SEEDERS
            .iter()
            .find(|row| row.function == name && row.covers(&self.file))
    }
}

impl AllowedSeeder {
    /// Whether this row's [`Scope`] admits a file (relative to `src/`).
    fn covers(&self, file: &str) -> bool {
        match self.scope {
            Scope::Anywhere => true,
            Scope::File(path) => path == file,
        }
    }
}

/// The heart of #282: a seed drawn anywhere but a constructor or the replay
/// hatch — above all, one drawn inside `reset()`.
#[test]
fn every_seed_from_u64_sits_in_an_allowlisted_constructor() {
    let sites = production_seeding_sites();

    let violations: Vec<String> = sites
        .iter()
        .filter(|site| site.allowed_by().is_none())
        .map(SeedingSite::describe)
        .collect();

    assert!(
        violations.is_empty(),
        "`{SEED_CALL}` outside an allowlisted constructor:\n  {}\n\n\
         An environment seeds its RNG **once**, in a constructor, and `reset()` draws \
         from the persistent stream and lets it advance — ADR 0029 §1, ADR 0062 §2, \
         docs/rules.md §8 (\"Host-RNG seeding convention\").\n\
         Re-seeding inside `reset()` replays one bit-identical episode forever, and \
         does so while a test asserting \"the env samples from its RNG\" stays green \
         (ADR 0062 Context §4, issue #282).\n\n\
         If a site above is genuinely a constructor or an explicit replay hatch, add a \
         row to ALLOWED_SEEDERS in tests/rng_seeding_guards.rs stating why.\n\
         If it is inside `reset()`, delete it: the replay hatch is the inherent \
         `reset_with_seed(&mut self, seed: u64)`.\n\
         Do not delete this assertion.",
        violations.join("\n  "),
    );
}

/// The reverse direction, as ADR 0062 §4 requires: an allowlist row that no
/// longer matches anything is a permission nobody is exercising, and the next
/// reader will trust it.
#[test]
fn no_allowlist_row_is_stale() {
    let sites = production_seeding_sites();

    let stale: Vec<String> = ALLOWED_SEEDERS
        .iter()
        .filter(|row| {
            !sites.iter().any(|site| {
                site.function.as_deref() == Some(row.function) && row.covers(&site.file)
            })
        })
        .map(|row| {
            let scope = match row.scope {
                Scope::Anywhere => "anywhere".to_owned(),
                Scope::File(path) => format!("src/{path}"),
            };
            format!("fn {} ({scope}) — claimed: {}", row.function, row.why)
        })
        .collect();

    assert!(
        stale.is_empty(),
        "stale row(s) in ALLOWED_SEEDERS (tests/rng_seeding_guards.rs):\n  {}\n\n\
         No `{SEED_CALL}` in src/ resolves to that function within that scope. Was it \
         renamed, removed, or narrowed to the wrong file? Delete the row — an \
         allowlist entry that matches nothing is a standing permission no one is \
         checking, which is how the next dead RNG hides (ADR 0062 §2).",
        stale.join("\n  "),
    );
}

/// Disarm tripwire, in the spirit of `landscape_dim_guards.rs`'s
/// "found no landscape modules" assertion: a guard that walks the wrong path,
/// or whose `#[cfg(test)]` skipping swallowed the crate, would pass every other
/// test in this file silently.
#[test]
fn the_scan_reaches_the_crate_source() {
    let sources = rust_sources(Path::new(CRATE_SRC_DIR));
    assert!(
        sources.len() > 50,
        "found only {} .rs files under {CRATE_SRC_DIR} — the path this guard walks is \
         wrong, which would silently disarm it",
        sources.len(),
    );

    let sites = production_seeding_sites();
    assert!(
        !sites.is_empty(),
        "found no `{SEED_CALL}` in production source. Either every environment stopped \
         seeding (in which case delete this guard) or the scan is broken",
    );

    let replay_hatches = sites
        .iter()
        .filter(|site| site.function.as_deref() == Some("reset_with_seed"))
        .count();
    assert!(
        replay_hatches > 0,
        "no `reset_with_seed` seeding site survived the scan — that method is the \
         guard's hardest case (a re-seed three lines above `self.reset()`), and if it \
         has vanished the enclosing-fn resolution is no longer being exercised",
    );
}

/// Every production `seed_from_u64` occurrence in the crate, with its enclosing
/// function resolved.
///
/// "Production" excludes `#[cfg(test)]` regions and comment lines — see this
/// file's module docs for both choices and their costs.
fn production_seeding_sites() -> Vec<SeedingSite> {
    let root = Path::new(CRATE_SRC_DIR);
    let mut sites: Vec<SeedingSite> = rust_sources(root)
        .into_iter()
        .flat_map(|path| {
            let relative = relative_path(root, &path);
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
            sites_in_file(&relative, &text)
        })
        .collect();
    sites.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    sites
}

/// The `seed_from_u64` occurrences in one file's text.
fn sites_in_file(relative: &str, text: &str) -> Vec<SeedingSite> {
    let lines: Vec<&str> = text.lines().collect();
    let test_regions = cfg_test_regions(relative, &lines);

    lines
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            line.contains(SEED_CALL)
                // Doc examples (`///`, `//!`) compile as separate doctest crates and
                // commented-out code does not run; neither is an environment's reset path.
                && !line.trim_start().starts_with("//")
                && !test_regions.iter().any(|(start, end)| (*start..=*end).contains(index))
        })
        .map(|(index, _)| SeedingSite {
            file: relative.to_owned(),
            line: index + 1,
            function: enclosing_fn(&lines, index),
        })
        .collect()
}

/// Inclusive `(first, last)` line index pairs for every `#[cfg(test)]` region.
///
/// Fail-loud by construction: an unterminated region panics rather than
/// swallowing the rest of the file, and the terminating line must sit at the
/// attribute's own indentation and look like an item end (`}` for a braced item
/// such as `mod tests`, `;` for a braceless one such as
/// `#[cfg(test)] const F_PER_DIM: f64 = …;`). Those two checks are what stop a
/// brace miscount from quietly hiding production code — see the module docs.
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
                "src/{relative}: the `#[cfg(test)]` at line {} never terminates. This \
                 guard refuses to skip to end-of-file, because doing so would hide every \
                 production `{SEED_CALL}` below it (issue #282).",
                index + 1,
            )
        });

        let last = lines[end].trim();
        assert!(
            indent_of(lines[end]) == attribute_indent && (last == "}" || last.ends_with(';')),
            "src/{relative}: the `#[cfg(test)]` region opened at line {} appears to end at \
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

/// The name of the `fn` enclosing `lines[index]`, if any.
///
/// Walks backwards for the first *less-indented* line that parses as a function
/// signature. rustfmt guarantees a body is indented past its signature, and CI
/// enforces rustfmt, so this is exact for formatted code; multi-line signatures
/// work because the `fn name(` line still carries the signature's indentation.
/// The walk stops at a top-level item header, so an occurrence outside any
/// function reports `None` rather than borrowing the name of the previous one.
fn enclosing_fn(lines: &[&str], index: usize) -> Option<String> {
    /// Top-level items that cannot enclose a `fn` body: reaching one means the
    /// occurrence is not inside a function at all.
    const ITEM_HEADERS: &[&str] = &[
        "impl", "mod", "trait", "enum", "struct", "union", "static", "const", "use", "}",
    ];

    let target_indent = indent_of(lines[index]);

    for line in lines[..index].iter().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || indent_of(line) >= target_indent {
            continue;
        }
        if let Some(name) = fn_name(trimmed) {
            return Some(name.to_owned());
        }
        if indent_of(line) == 0
            && ITEM_HEADERS
                .iter()
                .any(|header| trimmed.starts_with(header))
        {
            return None;
        }
    }
    None
}

/// The name declared by a function signature line, or `None` if the line is not
/// one.
///
/// Strips the qualifier prefixes rustfmt can put ahead of `fn` — `pub`,
/// `pub(crate)`, `const`, `async`, `unsafe`, `extern "C"`, `default` — in any
/// order, matching whole words only so `pub_thing(..)` is not mistaken for
/// `pub`.
fn fn_name(line: &str) -> Option<&str> {
    let mut rest = line.trim_start();

    loop {
        if let Some(after) = strip_word(rest, "pub") {
            let after = after.trim_start();
            rest = match after.strip_prefix('(') {
                // `pub(crate)`, `pub(super)`, `pub(in path)`
                Some(restriction) => restriction[restriction.find(')')? + 1..].trim_start(),
                None => after,
            };
            continue;
        }
        if let Some(after) = strip_word(rest, "extern") {
            let after = after.trim_start();
            rest = match after.strip_prefix('"') {
                Some(abi) => abi[abi.find('"')? + 1..].trim_start(),
                None => after,
            };
            continue;
        }
        match ["const", "async", "unsafe", "default"]
            .iter()
            .find_map(|keyword| strip_word(rest, keyword))
        {
            Some(after) => rest = after.trim_start(),
            None => break,
        }
    }

    let after_fn = strip_word(rest, "fn")?.trim_start();
    let name_len = after_fn
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(after_fn.len());
    (name_len > 0).then(|| &after_fn[..name_len])
}

/// Strips `word` from the front of `text`, but only as a whole word.
fn strip_word<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(word)?;
    let continues_identifier = rest.starts_with(|c: char| c.is_alphanumeric() || c == '_');
    (!continues_identifier).then_some(rest)
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

/// `path` relative to `root`, `/`-separated so [`Scope::File`] rows read the
/// same on every platform.
fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or_else(|_| panic!("{} is not under {}", path.display(), root.display()))
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
