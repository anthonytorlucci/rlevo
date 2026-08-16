//! Crate-scoped regression net: folding a `config::ordered` scalar pair into
//! one [`Bounds`](rlevo_core::bounds::Bounds) field discharges the
//! **ordering** half of the old check and silently drops the **strictness**
//! half.
//!
//! ADR 0068 §Decision 2
//! (`docs/adr/0068-bounds-strictness-enforcement-is-crate-asymmetric.md`)
//! states the rule and mandates this file. `Bounds::try_new` accepts
//! `lo <= hi` and *deliberately* permits the degenerate `lo == hi` (ADR 0027
//! §2: `clamp` is well-defined on a single point, and every search-space
//! sampler is zero-width-safe). `config::ordered`'s strict `<` did not. So a
//! diff that deletes a `config::ordered` line "because the type now guarantees
//! it" reads as a pure refactor while being a semantic loosening.
//!
//! In **this crate** zero width is never legitimate: both `Bounds` fields are a
//! `log σ` clamp range, and pinning it collapses σ to a constant — for PPO
//! permanently, because `log_std` is a single shared `Param` whose gradient
//! freezes at step 0; for SAC into a state-independently deterministic scale,
//! which is precisely the quantity the entropy temperature is tuned against.
//! Neither produces a `NaN`, a panic, or a failed assertion. Both produce a run
//! that trains and reports finite numbers.
//!
//! That is why the scan is **crate-scoped** rather than workspace-wide, and the
//! scope is itself the ADR-recorded decision (ADR 0068 §Context, following ADR
//! 0062 §4's precedent that a source-text guard's scope is architectural). Of
//! the workspace's 32 `Bounds`-typed struct fields, 30 live in
//! `rlevo-evolution` / `rlevo-environments` where zero width is the documented,
//! intended behaviour. A workspace-wide guard would carry ~30 allowlist rows
//! restating ADR 0027 §2 — and a guard whose rows are 94 % "this one is fine"
//! trains its readers to add a row without thinking, which is the state in
//! which it stops catching anything.
//!
//! # What the guard does
//!
//! It walks **this crate's `src/`** from disk at test time via
//! `CARGO_MANIFEST_DIR`, finds every `Bounds`-typed **struct field**, resolves
//! the **owning struct**, and requires each one to be either
//!
//! - covered by a `config::nondegenerate_bounds` (or the equivalent
//!   `config::distinct`) call **naming that field, for that config, in the same
//!   file** — the check ADR 0068 §Decision 1 names and ADR 0054 §3 originally
//!   wrote by hand; or
//! - listed in [`EXEMPT_BOUNDS_FIELDS`] with a written reason.
//!
//! Both spellings are accepted on purpose. `nondegenerate_bounds` delegates to
//! `distinct` and preserves its error semantics byte-for-byte (ADR 0068
//! §Decision 1), so the two are the same check; accepting both means renaming
//! one call site does not break the guard, and the guard does not dictate which
//! of the two spellings a `validate()` uses.
//!
//! Every field must additionally appear in [`STRICT_BOUNDS_FIELDS`], the
//! registry, and — as ADR 0068 §Decision 2 requires — the check runs **both
//! ways**: [`no_registry_or_exemption_row_is_stale`] fails on a row that no
//! longer resolves to a live field, and on an *exemption* row whose field now
//! satisfies the predicate directly. An allowlist that is only checked in one
//! direction rots into a list of names nobody verifies.
//!
//! ## Two resolution details that are the mechanism, not an optimisation
//!
//! Both were found by inspection while writing this guard and are recorded so
//! they are not rediscovered (ADR 0068 §Context states both):
//!
//! - **The field scan is not `pub`-only.** `rlevo-evolution` has two *private*
//!   `bounds` fields (`hill_climbing.rs:50`, `simulated_annealing.rs:53`).
//!   They are outside this scan's scope, but they are proof that a `pub`-only
//!   pattern is the kind of near-miss that reads as complete.
//! - **Matches resolve to struct fields, not to any `: Bounds` occurrence.** A
//!   rustfmt-formatted **function parameter** occupies its own line and ends in
//!   a comma, so it has the exact shape of a field declaration —
//!   `config::nondegenerate_bounds`'s own `b: Bounds,` in
//!   `rlevo-core/src/config.rs` is the live example. [`enclosing_item`] walks
//!   out to the enclosing item and keeps the match only when that item is a
//!   `struct`; a `fn` owner means "parameter" and is dropped.
//!
//! # `#[cfg(test)]` regions are skipped — the choice and its cost
//!
//! Unit-test bodies build `Bounds` fields freely, including the fixtures that
//! assert the degenerate range *is* rejected. Those are skipped rather than
//! allowlisted, for the reason `tests/rng_seeding_guards.rs` gives: a test-only
//! config is not compiled into the library and cannot misconfigure a user's
//! run, and requiring every new fixture to edit a bidirectionally-checked
//! allowlist trains contributors to edit it reflexively — the habit that lets a
//! production row slip on.
//!
//! **The cost**, plainly: a `Bounds` field on a `#[cfg(test)]`-only config is
//! invisible here. That is a test-only path by construction, so it cannot ship.
//!
//! **Why the skip cannot hide production code**: [`cfg_test_regions`] is
//! fail-loud. It panics rather than skipping to end-of-file on an unterminated
//! region, and asserts each region ends on a line at the *same indentation* as
//! the attribute that opened it. A brace miscount — the only way a region could
//! over-extend into production code — lands on a line that fails that shape
//! assertion and reports the file and both line numbers.
//!
//! # Known limits (ADR 0068 §Decision 2 requires these be recorded, not
//! rediscovered)
//!
//! The guard reads **source text**. Therefore:
//!
//! - It is **brittle against hand reformatting.** Field detection assumes one
//!   field per line and owner resolution assumes rustfmt's indentation; call
//!   detection follows a call's arguments across lines, but a call split in an
//!   unanticipated way (a comment between the arguments, the field name built
//!   by a macro, the config name passed as anything but a `&str` literal or a
//!   `const IDENT: &str = "…";` in the same file) is not recognised. CI enforces
//!   `cargo fmt --all --check`, which is what makes the assumption hold; code
//!   formatted by hand can defeat it.
//! - It is **defeatable by a type alias.** `type Range = Bounds;` followed by
//!   `log_std: Range,` is invisible, as is a field whose type is a struct that
//!   merely *contains* a `Bounds`.
//! - It does not see **tuple-struct fields** (`struct Clamp(Bounds);`) or
//!   `Bounds` nested inside another generic (`Vec<Bounds>`,
//!   `HashMap<K, Bounds>`); only `Bounds` and `Option<Bounds>` field types are
//!   matched.
//! - It checks **that the call is written, not that it runs.** A
//!   `nondegenerate_bounds` line placed after an early `return Ok(())`, or in a
//!   `validate()` no chokepoint calls, satisfies this guard. ADR 0026's
//!   construction chokepoint and ADR 0054 §3's per-config rejection tests are
//!   what cover that; this guard covers the line's *existence*.
//! - A match inside a multi-line `/* … */` block comment is read as code. That
//!   direction is safe: a loud false positive, never a silent pass.
//!
//! It catches the accident, not the adversary — and per ADR 0068 the accident
//! is the actual threat model: a contributor adds a continuous-action Gaussian
//! head by copying one of the two existing ones and brings the `Bounds` field
//! without the `validate()` line. Deleting this file costs nothing, which is
//! what makes it a cheap, reversible decision rather than a load-bearing one.

use std::fs;
use std::path::{Path, PathBuf};

/// The crate's source tree, resolved from the crate root at compile time. Cargo
/// sets `CARGO_MANIFEST_DIR` for integration tests, so this points at the real
/// source the test binary was built from.
const CRATE_SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

/// The field types the guard recognises, spelled exactly as rustfmt writes
/// them. Anything else — an alias, `Vec<Bounds>`, a tuple-struct field — is a
/// stated limit, not an oversight.
const BOUNDS_TYPES: &[&str] = &["Bounds", "Option<Bounds>"];

/// The `config::` helpers that assert strictness on a [`Bounds`] field.
///
/// Both are accepted because they are the *same* predicate:
/// `nondegenerate_bounds` contains no comparison of its own and delegates to
/// `distinct`, preserving `ConstraintKind::DegenerateInterval` and the `field`
/// byte-for-byte (ADR 0068 §Decision 1). Requiring only the newer spelling
/// would make a rename a test failure, for no gain in what is actually checked.
const STRICTNESS_CALLS: &[&str] = &["nondegenerate_bounds", "distinct"];

/// One `Bounds` field in this crate that **must** carry a strictness check.
///
/// This registry is the "every field is accounted for" half of the
/// bidirectional check (ADR 0068 §Decision 2). `why` is not decoration: a row
/// without a recorded reason is indistinguishable from a row someone added to
/// silence the test.
struct StrictField {
    /// The owning struct, exactly as it appears after `struct`.
    config: &'static str,
    /// The field name, exactly as it appears before the `:`.
    field: &'static str,
    /// Why zero width is a misconfiguration for this field.
    why: &'static str,
}

/// A `Bounds` field for which zero width is **legitimate**, and which is
/// therefore excused from the strictness check.
///
/// **Empty, and that emptiness is a property worth having explicitly** (ADR
/// 0068 §Decision 2): it means "every `Bounds` field in this crate needs a
/// strictness check" is a crate-level invariant rather than a per-field
/// judgement, so the guard's question to a future author is a yes/no with a
/// default, not an essay.
///
/// Before adding a row here, read ADR 0068 §Context: if zero width really is
/// meaningful for your field, the likelier answer is that the field belongs in
/// a crate where that is the documented default (`rlevo-environments`'s
/// `action_clip`, `rlevo-evolution`'s search `bounds`), and the *third* row
/// here is one of ADR 0068's reopen triggers.
struct ExemptField {
    /// The owning struct.
    config: &'static str,
    /// The field name.
    field: &'static str,
    /// Why `lo == hi` is a meaningful setting for this field.
    why: &'static str,
}

/// Every `Bounds` field in `rlevo-reinforcement-learning` that must re-assert
/// strictness. Two rows, both the `log σ` clamp of a continuous-action Gaussian
/// head (ADR 0068 §Context: the entire strict population in the workspace).
static STRICT_BOUNDS_FIELDS: &[StrictField] = &[
    StrictField {
        config: "TanhGaussianPolicyHeadConfig",
        field: "log_std",
        why: "PPO's `log_std` is a *single shared* `Param`. A zero-width range pins it \
              to a constant, freezing sigma **and its gradient** from step 0 with no \
              path back (ADR 0049, ADR 0054 §3)",
    },
    StrictField {
        config: "SquashedGaussianPolicyHeadConfig",
        field: "log_std",
        why: "SAC's `log_std` head keeps receiving gradient through the mean path, so \
              nothing freezes permanently — but a zero-width range makes the policy \
              state-independently deterministic in scale, which is exactly the quantity \
              the entropy temperature is tuned against (ADR 0054 §3)",
    },
];

/// Fields excused from the strictness check. **Empty by design** — see
/// [`ExemptField`].
static EXEMPT_BOUNDS_FIELDS: &[ExemptField] = &[];

/// One `Bounds`-typed struct field found in production source.
#[derive(Debug)]
struct BoundsFieldSite {
    /// Path relative to `src/`, `/`-separated.
    file: String,
    /// 1-based line number, for a message a reader can jump to.
    line: usize,
    /// The owning struct's name.
    config: String,
    /// The field's name.
    field: String,
    /// Whether a strictness call in the same file names this config and field.
    checked: bool,
}

impl BoundsFieldSite {
    /// Renders as
    /// `src/algorithms/sac/sac_policy.rs:57 (SquashedGaussianPolicyHeadConfig::log_std)`.
    fn describe(&self) -> String {
        format!(
            "src/{}:{} ({}::{})",
            self.file, self.line, self.config, self.field,
        )
    }

    /// Whether an [`EXEMPT_BOUNDS_FIELDS`] row covers this field.
    fn exemption(&self) -> Option<&'static ExemptField> {
        EXEMPT_BOUNDS_FIELDS
            .iter()
            .find(|row| row.config == self.config && row.field == self.field)
    }
}

/// One strictness call found in production source, resolved to the config and
/// field it names.
#[derive(Debug)]
struct StrictnessCallSite {
    /// The config name — the call's first argument, resolved through a
    /// `const IDENT: &str = "…";` when it is an identifier.
    config: String,
    /// The field name — the call's second argument, a `&str` literal.
    field: String,
}

/// The core failure mode this guard catches: a `Bounds` field whose config
/// never re-asserts the strictness the migration dropped.
#[test]
fn every_bounds_field_asserts_strictness() {
    let sites = bounds_field_sites();

    let violations: Vec<String> = sites
        .iter()
        .filter(|site| !site.checked && site.exemption().is_none())
        .map(BoundsFieldSite::describe)
        .collect();

    assert!(
        violations.is_empty(),
        "`Bounds` field(s) with no strictness check in the owning config's `validate()`:\n  \
         {}\n\n\
         `Bounds::try_new` accepts `lo <= hi` and deliberately permits the degenerate \
         `lo == hi` (ADR 0027 §2), so a `Bounds` field discharges only the *ordering* half \
         of the `config::ordered` check it replaced. The strictness half must be written \
         back explicitly — see ADR 0068 §Decision 1.\n\n\
         Add to that config's `validate()`:\n    \
         config::nondegenerate_bounds(C, \"<field>\", self.<field>)?;\n\n\
         In this crate a zero-width range is never a usable setting: it collapses `log σ` \
         to a constant, which trains and reports finite numbers while silently disabling \
         the policy's scale (ADR 0068 §Context). If you believe zero width *is* meaningful \
         for your field, do not widen this check — either the field belongs in a crate \
         where that is the documented default (ADR 0027 §2), or it needs an \
         EXEMPT_BOUNDS_FIELDS row in tests/bounds_strictness_guard.rs stating why, which \
         is one of ADR 0068's reopen triggers.\n\
         Do not delete this assertion.",
        violations.join("\n  "),
    );
}

/// The registry half: a `Bounds` field that appears in neither list is a field
/// nobody has classified.
#[test]
fn registry_covers_every_bounds_field() {
    let sites = bounds_field_sites();

    let unregistered: Vec<String> = sites
        .iter()
        .filter(|site| {
            !STRICT_BOUNDS_FIELDS
                .iter()
                .any(|row| row.config == site.config && row.field == site.field)
                && site.exemption().is_none()
        })
        .map(BoundsFieldSite::describe)
        .collect();

    assert!(
        unregistered.is_empty(),
        "unclassified `Bounds` field(s) in src/: \n  {}\n\n\
         Every `Bounds` field in this crate must be classified in \
         tests/bounds_strictness_guard.rs (ADR 0068 §Decision 2):\n\
         - zero width is a misconfiguration (the default, and the case for every field \
         here so far): add a STRICT_BOUNDS_FIELDS row and a \
         `config::nondegenerate_bounds` line to the config's `validate()`;\n\
         - zero width is genuinely meaningful: add an EXEMPT_BOUNDS_FIELDS row with the \
         reason — and read ADR 0068 §Context first, because that row is a reopen trigger \
         for the ADR.\n\
         Do not delete this assertion.",
        unregistered.join("\n  "),
    );
}

/// The reverse direction, as ADR 0068 §Decision 2 requires: a row that no
/// longer matches anything is a claim nobody is checking, and the next reader
/// will trust it.
#[test]
fn no_registry_or_exemption_row_is_stale() {
    let sites = bounds_field_sites();
    let mut stale: Vec<String> = Vec::new();

    for row in STRICT_BOUNDS_FIELDS {
        if !sites
            .iter()
            .any(|site| site.config == row.config && site.field == row.field)
        {
            stale.push(format!(
                "STRICT_BOUNDS_FIELDS: {}::{} — no such `Bounds` field in src/. Renamed, \
                 removed, or retyped? Claimed: {}",
                row.config, row.field, row.why,
            ));
        }
    }

    for row in EXEMPT_BOUNDS_FIELDS {
        match sites
            .iter()
            .find(|site| site.config == row.config && site.field == row.field)
        {
            None => stale.push(format!(
                "EXEMPT_BOUNDS_FIELDS: {}::{} — no such `Bounds` field in src/. Renamed, \
                 removed, or retyped? Claimed: {}",
                row.config, row.field, row.why,
            )),
            // An exemption whose field now carries the check is the more
            // interesting staleness: the permission is no longer being
            // exercised, and leaving it standing means the next author reads
            // "zero width is fine here" about a field that rejects it.
            Some(site) if site.checked => stale.push(format!(
                "EXEMPT_BOUNDS_FIELDS: {}::{} is exempted, but {} now calls a strictness \
                 helper on it. Delete the exemption row — it is a standing permission \
                 nobody exercises. Claimed: {}",
                row.config,
                row.field,
                site.describe(),
                row.why,
            )),
            Some(_) => {}
        }
    }

    assert!(
        stale.is_empty(),
        "stale row(s) in tests/bounds_strictness_guard.rs:\n  {}\n\n\
         A bidirectional allowlist is what keeps this guard from rotting into a \
         permanently-green no-op (ADR 0068 §Decision 2, ADR 0062 §4).",
        stale.join("\n  "),
    );
}

/// Disarm tripwire, in the spirit of `rng_seeding_guards.rs`'s
/// `the_scan_reaches_the_crate_source`: a guard that walks the wrong path, or
/// whose parser stopped recognising struct fields, would pass every other test
/// in this file silently. A vacuously green guard is worse than no guard.
#[test]
fn the_scan_reaches_the_crate_source() {
    let sources = rust_sources(Path::new(CRATE_SRC_DIR));
    assert!(
        sources.len() > 40,
        "found only {} .rs files under {CRATE_SRC_DIR} — the path this guard walks is \
         wrong, which would silently disarm it",
        sources.len(),
    );

    let structs = production_struct_count();
    assert!(
        structs > 50,
        "parsed only {structs} `struct` declarations across src/ — this crate has many \
         more, so owner resolution is broken and every `Bounds` field would be \
         misclassified (or dropped) without any test failing",
    );

    let sites = bounds_field_sites();
    assert!(
        !sites.is_empty(),
        "found no `Bounds`-typed struct field in src/. Either the crate genuinely has \
         none — in which case delete this guard rather than leave it vacuously green — \
         or the field parser is broken",
    );

    assert!(
        sites.iter().any(|site| site.checked),
        "no `Bounds` field resolved to a strictness call. One of {STRICTNESS_CALLS:?} \
         paired with a config name and a field name is the guard's whole predicate; if \
         nothing matches it, call resolution is broken and this guard is asserting \
         nothing",
    );
}

/// Every `Bounds`-typed struct field in production source, with its owning
/// struct resolved and the strictness predicate evaluated.
///
/// "Production" excludes `#[cfg(test)]` regions and comment lines — see this
/// file's module docs for both choices and their costs.
fn bounds_field_sites() -> Vec<BoundsFieldSite> {
    let root = Path::new(CRATE_SRC_DIR);
    let mut sites: Vec<BoundsFieldSite> = rust_sources(root)
        .into_iter()
        .flat_map(|path| {
            let relative = relative_path(root, &path);
            let text = read_source(&path);
            sites_in_file(&relative, &text)
        })
        .collect();
    sites.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    sites
}

/// The `Bounds` fields declared in one file, each paired with whether that
/// file's strictness calls cover it.
fn sites_in_file(relative: &str, text: &str) -> Vec<BoundsFieldSite> {
    let lines: Vec<&str> = text.lines().collect();
    let production = production_lines(relative, &lines);
    let calls = strictness_calls(&lines, &production);

    production
        .iter()
        .filter_map(|&index| {
            let field = bounds_field_name(lines[index])?;
            // The resolution that separates a struct field from a
            // rustfmt-formatted function parameter, which has the same shape.
            let config = match enclosing_item(&lines, index) {
                Owner::Struct(name) => name,
                Owner::Function => return None,
                Owner::Unresolved => panic!(
                    "src/{relative}:{}: cannot resolve the item owning `{}`. This guard \
                     refuses to guess: a `Bounds` field whose owner it cannot name is a \
                     field it cannot check, and dropping it silently re-opens the \
                     strictness gap this guard exists to close. Is the file \
                     rustfmt-formatted?",
                    index + 1,
                    lines[index].trim(),
                ),
            };
            let checked = calls
                .iter()
                .any(|call| call.config == config && call.field == field);
            Some(BoundsFieldSite {
                file: relative.to_owned(),
                line: index + 1,
                config,
                field: field.to_owned(),
                checked,
            })
        })
        .collect()
}

/// The field name declared by a `Bounds`-typed field line, or `None` when the
/// line is not one.
///
/// Deliberately **not** `pub`-only: a private `Bounds` field needs the same
/// check, and two of them exist in `rlevo-evolution` today (module docs). The
/// type must be spelled exactly as one of [`BOUNDS_TYPES`], which is what keeps
/// `log_std: Bounds::new(-5.0, 2.0),` — a struct *literal*, not a declaration —
/// out.
fn bounds_field_name(line: &str) -> Option<&str> {
    let mut rest = line.trim();

    // Visibility, in any of its spellings.
    if let Some(after) = strip_word(rest, "pub") {
        let after = after.trim_start();
        rest = match after.strip_prefix('(') {
            // `pub(crate)`, `pub(super)`, `pub(in path)`
            Some(restriction) => restriction[restriction.find(')')? + 1..].trim_start(),
            None => after,
        };
    }

    let (name, ty) = rest.split_once(':')?;
    let name = name.trim();
    // A plain identifier only. A raw identifier (`r#type`) as a field name
    // appears nowhere in the workspace and would be invisible here — one more
    // entry in the module docs' "known limits", in the same class as a type
    // alias for `Bounds`.
    if name.is_empty()
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.starts_with(|c: char| c.is_ascii_digit())
    {
        return None;
    }

    let ty = ty.trim().trim_end_matches(',').trim();
    BOUNDS_TYPES.contains(&ty).then_some(name)
}

/// What owns a line: the nearest enclosing item.
#[derive(Debug)]
enum Owner {
    /// A `struct Name { .. }` — the line is a field declaration.
    Struct(String),
    /// A `fn name(..)` — the line is a parameter, not a field. The name is not
    /// carried because nothing reports it: a parameter is dropped silently, by
    /// design (module docs, `b: Bounds,` in `rlevo-core::config`).
    Function,
    /// Neither could be established. Treated as fail-loud, never as "skip".
    Unresolved,
}

/// The item enclosing `lines[index]`.
///
/// Walks backwards for the first *less-indented* line that parses as a `struct`
/// or `fn` header. rustfmt guarantees a body is indented past its header, and CI
/// enforces rustfmt, so this is exact for formatted code; a multi-line generic
/// list or `where` clause is stepped over by widening the search outward one
/// nesting level at a time.
///
/// This is the whole reason the guard does not flag
/// `config::nondegenerate_bounds`'s own `b: Bounds,` parameter (module docs).
fn enclosing_item(lines: &[&str], index: usize) -> Owner {
    /// Items that own a scope but are neither a struct nor a fn: reaching one
    /// means the line is not a field of a struct.
    const OTHER_HEADERS: &[&str] = &["impl", "trait", "enum", "union", "mod"];

    let mut target_indent = indent_of(lines[index]);

    for line in lines[..index].iter().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("//")
            || trimmed.starts_with('#')
            || indent_of(line) >= target_indent
        {
            continue;
        }
        if let Some(name) = struct_name(trimmed) {
            return Owner::Struct(name.to_owned());
        }
        if fn_name(trimmed).is_some() {
            return Owner::Function;
        }
        if OTHER_HEADERS
            .iter()
            .any(|header| strip_word(trimmed, header).is_some())
        {
            return Owner::Unresolved;
        }
        // A continuation line — a multi-line generic list's `>` or a `where`
        // clause. Widen by one level and keep walking outward rather than
        // guessing.
        target_indent = indent_of(line);
    }
    Owner::Unresolved
}

/// The name declared by a `struct` header line, or `None` if the line is not
/// one. Strips a `pub` / `pub(..)` prefix.
fn struct_name(line: &str) -> Option<&str> {
    let rest = strip_visibility(line.trim_start());
    identifier_after(strip_word(rest, "struct")?)
}

/// The name declared by a `fn` signature line, or `None` if the line is not one.
///
/// Strips the qualifier prefixes rustfmt can put ahead of `fn` — `pub`,
/// `pub(crate)`, `const`, `async`, `unsafe`, `extern "C"`, `default` — in any
/// order, matching whole words only so `pub_thing(..)` is not mistaken for
/// `pub`.
fn fn_name(line: &str) -> Option<&str> {
    let mut rest = line.trim_start();

    loop {
        let stripped = strip_visibility(rest);
        if stripped.len() != rest.len() {
            rest = stripped;
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

    identifier_after(strip_word(rest, "fn")?)
}

/// The leading identifier of `text` after trimming, or `None` when it does not
/// start with one.
fn identifier_after(text: &str) -> Option<&str> {
    let text = text.trim_start();
    let len = text
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(text.len());
    (len > 0).then(|| &text[..len])
}

/// Strips `pub` / `pub(crate)` / `pub(super)` / `pub(in path)` if present,
/// otherwise returns `text` unchanged.
fn strip_visibility(text: &str) -> &str {
    let Some(after) = strip_word(text, "pub") else {
        return text;
    };
    let after = after.trim_start();
    match after.strip_prefix('(') {
        Some(restriction) => match restriction.find(')') {
            Some(close) => restriction[close + 1..].trim_start(),
            None => after,
        },
        None => after,
    }
}

/// Every strictness call in the production lines of one file, resolved to the
/// `(config, field)` pair it names.
///
/// A call's arguments are collected across lines to the matching `)`, so
/// rustfmt breaking a long call is fine; anything more exotic is a stated limit.
fn strictness_calls(lines: &[&str], production: &[usize]) -> Vec<StrictnessCallSite> {
    let mut calls = Vec::new();

    for &index in production {
        for helper in STRICTNESS_CALLS {
            let needle = format!("{helper}(");
            let Some(column) = lines[index].find(&needle) else {
                continue;
            };
            // `find` is enough: the same helper twice on one line would need a
            // nested call, which no `validate()` writes.
            let open = column + needle.len() - 1;
            let Some(arguments) = argument_text(lines, index, open) else {
                continue;
            };
            let parts = top_level_arguments(&arguments);
            let [config, field, ..] = parts.as_slice() else {
                continue;
            };
            let Some(config) = resolve_config_name(lines, index, config) else {
                continue;
            };
            let Some(field) = string_literal(field) else {
                continue;
            };
            calls.push(StrictnessCallSite {
                config,
                field: field.to_owned(),
            });
        }
    }

    calls
}

/// The text between the `(` at `lines[start][open]` and its matching `)`,
/// joined with spaces across lines. `None` if the parentheses never balance.
fn argument_text(lines: &[&str], start: usize, open: usize) -> Option<String> {
    let mut depth = 0i32;
    let mut collected = String::new();

    for (offset, line) in lines[start..].iter().enumerate() {
        // `open` is the byte index of the opening `(`, so this slice lands on a
        // char boundary.
        let tail = if offset == 0 { &line[open..] } else { line };
        for ch in tail.chars() {
            match ch {
                '(' => {
                    depth += 1;
                    if depth == 1 {
                        // The opening paren itself is not part of the arguments.
                        continue;
                    }
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(collected);
                    }
                }
                _ => {}
            }
            collected.push(ch);
        }
        collected.push(' ');
    }
    None
}

/// Splits an argument list on commas that are not nested inside `(`, `[`, `<`
/// or a string literal.
fn top_level_arguments(arguments: &str) -> Vec<String> {
    let mut parts = vec![String::new()];
    let mut depth = 0i32;
    let mut in_string = false;

    for ch in arguments.chars() {
        if in_string {
            parts.last_mut().expect("one part always exists").push(ch);
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(String::new());
                continue;
            }
            _ => {}
        }
        parts.last_mut().expect("one part always exists").push(ch);
    }

    parts
        .into_iter()
        .map(|part| part.trim().to_owned())
        .filter(|part| !part.is_empty())
        .collect()
}

/// The config name a strictness call's first argument denotes.
///
/// Two forms are understood, which are the two the workspace writes: a `&str`
/// literal, and an identifier bound by a `const IDENT: &str = "…";` — the
/// `const C: &str = "TanhGaussianPolicyHeadConfig";` at the top of every
/// `validate()` (ADR 0026). The nearest such binding above the call wins, so
/// two `validate()` bodies in one file resolve independently — which is what
/// makes the guard notice a *second* Gaussian head copied into an existing
/// file without its own check.
fn resolve_config_name(lines: &[&str], call_line: usize, argument: &str) -> Option<String> {
    if let Some(literal) = string_literal(argument) {
        return Some(literal.to_owned());
    }

    let binding = format!("const {argument}: &str = ");
    lines[..=call_line]
        .iter()
        .rev()
        .find_map(|line| string_literal(line.trim().strip_prefix(&binding)?))
        .map(str::to_owned)
}

/// The contents of a `&str` literal, or `None` when `text` is not one. Trailing
/// punctuation (`;`, `,`) is tolerated so a `const` binding can be parsed as-is.
fn string_literal(text: &str) -> Option<&str> {
    let text = text.trim().trim_end_matches([';', ',']).trim();
    let inner = text.strip_prefix('"')?.strip_suffix('"')?;
    (!inner.contains('"')).then_some(inner)
}

/// The line indices of one file that are production code: not a comment, not
/// inside a `#[cfg(test)]` region.
fn production_lines(relative: &str, lines: &[&str]) -> Vec<usize> {
    let test_regions = cfg_test_regions(relative, lines);
    (0..lines.len())
        .filter(|index| {
            // Doc comments (`///`, `//!`) compile as separate doctest crates and
            // commented-out code does not run; neither is a shipped config.
            !lines[*index].trim_start().starts_with("//")
                && !test_regions
                    .iter()
                    .any(|(start, end)| (*start..=*end).contains(index))
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
/// Lifted from `rlevo-environments/tests/rng_seeding_guards.rs`, deliberately
/// duplicated rather than shared: the two guards are independently deletable
/// (ADR 0062 §4, ADR 0068 §Decision 2), and a shared test-support crate would
/// make deleting either one a cross-crate change.
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
                 production `Bounds` field below it, undoing the check this file exists \
                 to run.",
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

/// How many `struct` declarations the parser finds in production source.
///
/// Only used by [`the_scan_reaches_the_crate_source`]: if this collapses, owner
/// resolution has stopped working and every field would be dropped or
/// misclassified without any other test failing.
fn production_struct_count() -> usize {
    let root = Path::new(CRATE_SRC_DIR);
    rust_sources(root)
        .into_iter()
        .map(|path| {
            let relative = relative_path(root, &path);
            let text = read_source(&path);
            let lines: Vec<&str> = text.lines().collect();
            production_lines(&relative, &lines)
                .into_iter()
                .filter(|&index| struct_name(lines[index].trim()).is_some())
                .count()
        })
        .sum()
}

/// Reads a source file, failing loudly. A guard that skips an unreadable file
/// passes vacuously, which is worse than no guard.
fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()))
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
