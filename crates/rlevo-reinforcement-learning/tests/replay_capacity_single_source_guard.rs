//! Crate-scoped guard: the replay-buffer capacity predicate is single-sourced
//! in [`UniformReplayConfig`], and no agent may re-derive it or route around it.
//!
//! # What went wrong
//!
//! Six agent configs — TD3, SAC, DDPG, DQN, C51, QR-DQN — each carried a
//! *hand-written copy* of the same two lines:
//!
//! ```ignore
//! config::nonzero(C, "replay_buffer_capacity", self.replay_buffer_capacity)?;
//! config::at_most(C, "replay_buffer_capacity", self.replay_buffer_capacity, MAX_BUFFER_CAPACITY)?;
//! ```
//!
//! and each agent's `new` then handed that field to the **infallible**
//! `UniformReplay::new` / `ReplayKind::uniform`, both of which `assert!`. Six
//! copies of one predicate is six chances to omit a line, and the infallible
//! constructor means an omission is not a wrong answer — it is an `assert!`
//! (or, past the ceiling, a `VecDeque` allocation abort the caller cannot
//! catch) in the middle of a training run. Correctness rested on
//! `config.validate()?` happening to run before the buffer was built.
//!
//! Both halves are now fixed: each `validate()` delegates to
//! `UniformReplayConfig::validate` and relabels the error to name its own
//! field, and each `new` takes `UniformReplay::from_config` /
//! `ReplayKind::uniform_from_config`, which return `Result`. This guard exists
//! because the *seventh* agent will be written by copying one of the six, and a
//! copy that predates the fix reads exactly as plausible as one that follows it.
//!
//! # What the guard does
//!
//! It walks `src/algorithms/` from disk at test time via `CARGO_MANIFEST_DIR`
//! and enforces two rules on **production** lines:
//!
//! 1. [`no_agent_config_rederives_the_capacity_ceiling`] — no `config::at_most`
//!    call takes `MAX_BUFFER_CAPACITY` as its bound, except for the rows in
//!    [`CEILING_EXEMPTIONS`].
//! 2. [`no_agent_builds_a_replay_buffer_infallibly`] — no call to
//!    `UniformReplay::new` or `ReplayKind::uniform`, the two `assert!`-ing
//!    constructors. An agent's capacity always comes from a config field, never
//!    a literal, so the fallible `*_from_config` form is the only correct one
//!    here. (`new` remains blessed *outside* `src/algorithms/` — for a literal
//!    or a `const`, where a bad value sits at the call site the panic names.)
//!
//! Rule 2 is not redundant with rule 1. Rule 1 alone would be satisfied by an
//! agent that simply *forgot* to check the capacity at all and then aborted
//! inside `VecDeque::with_capacity`, which is the worse of the two failures.
//!
//! [`ceiling_exemptions_are_live`] runs the allowlist the other way, per ADR
//! 0068 §Decision 2's rule for this kind of guard: a row that no longer
//! resolves to a real call is a claim nobody is checking.
//!
//! # `#[cfg(test)]` regions are skipped
//!
//! Test modules legitimately name `MAX_BUFFER_CAPACITY` (the six
//! `rejects_replay_buffer_capacity_above_ceiling` tests compute the expected
//! [`ConstraintKind::TooLarge`](rlevo_core::config::ConstraintKind) from it) and
//! legitimately build buffers with literal capacities. Skipping them rather
//! than allowlisting them follows `tests/bounds_strictness_guard.rs`: a
//! test-only call is not compiled into the library and cannot misconfigure a
//! user's run.
//!
//! **The cost**: a violation written inside a `#[cfg(test)]` module is
//! invisible here. It also cannot ship. [`cfg_test_regions`] is fail-loud about
//! the only way a skip could over-reach — a brace miscount — so the skip cannot
//! silently swallow production code.
//!
//! # Known limits
//!
//! It reads source text, so the same limits as `bounds_strictness_guard.rs`
//! apply: it assumes rustfmt formatting (which CI enforces), it is defeated by
//! aliasing (`use crate::MAX_BUFFER_CAPACITY as CAP;`, `type Buf =
//! UniformReplay<T>;`), and a match inside a `/* … */` block comment reads as
//! code — a loud false positive, never a silent pass. It catches the accident,
//! which is the actual threat model: a contributor copies an existing agent.
//!
//! Deleting this file costs nothing at runtime, which is what makes it a cheap,
//! reversible decision rather than a load-bearing one.

use std::fs;
use std::path::{Path, PathBuf};

/// The agent tree, resolved from the crate root at compile time.
const ALGORITHMS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/algorithms");

/// The ceiling constant, in every spelling an agent config could write it.
const CEILING_NAMES: &[&str] = &["MAX_BUFFER_CAPACITY", "crate::MAX_BUFFER_CAPACITY"];

/// The `assert!`-ing replay constructors. An agent capacity is always a config
/// field, so these are always the wrong choice inside `src/algorithms/`.
///
/// The trailing `(` matters: without it, `ReplayKind::uniform` would also match
/// `ReplayKind::uniform_from_config`, which is the *correct* call, and this
/// guard would forbid the fix it exists to protect.
const INFALLIBLE_CONSTRUCTORS: &[&str] = &["UniformReplay::new(", "ReplayKind::uniform("];

/// The fallible replacements. Only used by [`the_scan_reaches_the_agent_source`]
/// to prove the needle machinery works against the very files under test.
const FALLIBLE_CONSTRUCTORS: &[&str] = &[
    "UniformReplay::from_config(",
    "ReplayKind::uniform_from_config(",
];

/// A `config::at_most(.., MAX_BUFFER_CAPACITY)` call that is **not** the
/// duplicated replay predicate, and is therefore allowed to stand.
struct CeilingExemption {
    /// Path relative to `src/algorithms/`, `/`-separated.
    file: &'static str,
    /// The call's `field` argument, exactly as written.
    field: &'static str,
    /// Why this bound is not the replay predicate.
    why: &'static str,
}

/// The one legitimate `MAX_BUFFER_CAPACITY` bound left under `src/algorithms/`.
///
/// One row, and it is a genuinely different quantity — not a "this one is
/// fine" concession. If a second row ever wants adding, read it twice: two
/// exemptions is the state in which a guard stops catching anything.
static CEILING_EXEMPTIONS: &[CeilingExemption] = &[CeilingExemption {
    file: "ppo/ppo_config.rs",
    field: "\"num_steps\"",
    why: "PPO has no replay buffer. This bounds `num_envs * num_steps`, the capacity handed \
          to `RolloutBuffer::new` — a different buffer, with no `Validate` config of its own \
          to delegate to. It shares only the constant, because it is the same allocation \
          ceiling",
}];

/// One `config::at_most(.., MAX_BUFFER_CAPACITY)` call found in production
/// source.
#[derive(Debug)]
struct CeilingCall {
    /// Path relative to `src/algorithms/`.
    file: String,
    /// 1-based line, for a message a reader can jump to.
    line: usize,
    /// The call's second argument — the field it names.
    field: String,
}

impl CeilingCall {
    fn describe(&self) -> String {
        format!(
            "src/algorithms/{}:{} (at_most on {})",
            self.file, self.line, self.field,
        )
    }

    fn exemption(&self) -> Option<&'static CeilingExemption> {
        CEILING_EXEMPTIONS
            .iter()
            .find(|row| row.file == self.file && row.field == self.field)
    }
}

/// One call to an infallible replay constructor found in production source.
#[derive(Debug)]
struct ConstructorCall {
    file: String,
    line: usize,
    /// The needle that matched, e.g. `UniformReplay::new(`.
    call: String,
}

impl ConstructorCall {
    /// Renders as `src/algorithms/td3/td3_agent.rs:374 (UniformReplay::new)`.
    /// The needle's trailing `(` is dropped here — it is a matching detail, not
    /// something a reader should have to parse out of the message.
    fn describe(&self) -> String {
        format!(
            "src/algorithms/{}:{} ({})",
            self.file,
            self.line,
            self.call.trim_end_matches('('),
        )
    }
}

/// Half one: the duplicated predicate must not come back.
#[test]
fn no_agent_config_rederives_the_capacity_ceiling() {
    let violations: Vec<String> = ceiling_calls()
        .iter()
        .filter(|call| call.exemption().is_none())
        .map(CeilingCall::describe)
        .collect();

    assert!(
        violations.is_empty(),
        "`config::at_most(.., MAX_BUFFER_CAPACITY)` written by hand under src/algorithms/:\n  \
         {}\n\n\
         The replay-capacity predicate — non-zero, and at most MAX_BUFFER_CAPACITY — is \
         single-sourced in `UniformReplayConfig::validate`. Six agent configs used to carry \
         their own copy of it; a seventh copy is how one of them drifts.\n\n\
         In your config's `validate()`, delegate and relabel:\n\n    \
         UniformReplayConfig {{ capacity: self.replay_buffer_capacity }}\n        \
         .validate()\n        \
         .map_err(|e| ConfigError {{ config: C, field: \"replay_buffer_capacity\", ..e }})?;\n\n\
         `kind` passes through untouched, so the error your callers see is byte-identical to \
         the hand-written pair.\n\n\
         If your bound is genuinely a *different* quantity that merely shares the constant \
         (PPO's rollout size is the one such case), add a CEILING_EXEMPTIONS row saying so.\n\
         Do not delete this assertion.",
        violations.join("\n  "),
    );
}

/// Half two: an agent must not reach the `assert!`-ing constructor at all.
///
/// Deleting the `validate()` delegation and *also* dropping the check is a
/// failure rule one cannot see, and it is the worse one: an unchecked capacity
/// reaches `VecDeque::with_capacity`, which aborts the process rather than
/// returning.
#[test]
fn no_agent_builds_a_replay_buffer_infallibly() {
    let violations: Vec<String> = constructor_calls(INFALLIBLE_CONSTRUCTORS)
        .iter()
        .map(ConstructorCall::describe)
        .collect();

    assert!(
        violations.is_empty(),
        "infallible replay constructor(s) called under src/algorithms/:\n  {}\n\n\
         `UniformReplay::new` and `ReplayKind::uniform` **assert** on the capacity. That is \
         the right contract for a literal or a `const`, where a bad value sits at the call \
         site the panic names — but an agent's capacity is always a config field, i.e. user \
         data, and `rules.md` §4 puts user data behind a `Validate` chokepoint returning \
         `ConfigError`.\n\n\
         Every agent `new` returns `Result<Self, ConfigError>`, so this is a `?`:\n\n    \
         UniformReplay::from_config(UniformReplayConfig {{ capacity: config.replay_buffer_capacity }})?\n    \
         ReplayKind::uniform_from_config(UniformReplayConfig {{ capacity: config.replay_buffer_capacity }})?\n\n\
         Do not delete this assertion.",
        violations.join("\n  "),
    );
}

/// The allowlist, checked in reverse: a row that matches nothing is a standing
/// permission nobody exercises, and the next reader will trust it (ADR 0068
/// §Decision 2, ADR 0062 §4).
#[test]
fn ceiling_exemptions_are_live() {
    let calls = ceiling_calls();
    let stale: Vec<String> = CEILING_EXEMPTIONS
        .iter()
        .filter(|row| {
            !calls
                .iter()
                .any(|call| call.file == row.file && call.field == row.field)
        })
        .map(|row| {
            format!(
                "CEILING_EXEMPTIONS: {} / {} — no such `config::at_most(.., \
                 MAX_BUFFER_CAPACITY)` call. Moved, renamed, or already migrated to a \
                 config delegation? Claimed: {}",
                row.file, row.field, row.why,
            )
        })
        .collect();

    assert!(
        stale.is_empty(),
        "stale row(s) in tests/replay_capacity_single_source_guard.rs:\n  {}\n\n\
         Delete the row. A one-directional allowlist rots into a permanently-green no-op.",
        stale.join("\n  "),
    );
}

/// Disarm tripwire. Every assertion above is of the form "found nothing", so a
/// wrong path glob, a broken call parser, or a `#[cfg(test)]` skip that ate the
/// whole file would make all three pass while checking nothing.
#[test]
fn the_scan_reaches_the_agent_source() {
    let root = Path::new(ALGORITHMS_DIR);
    let sources = rust_sources(root);
    assert!(
        sources.len() > 20,
        "found only {} .rs files under {ALGORITHMS_DIR} — the path this guard walks is \
         wrong, which would silently disarm every assertion in this file",
        sources.len(),
    );

    // The parser works end to end — argument splitting, ceiling matching, and
    // field resolution — checked against the one call that is *supposed* to be
    // here. This is a positive control: every other assertion in this file is
    // "found nothing", and a parser that matched nothing at all would satisfy
    // them all. PPO's rollout bound is the one live
    // `at_most(.., MAX_BUFFER_CAPACITY)` call under src/algorithms/, and it must
    // resolve exactly, to the file and the field.
    //
    // `config::at_most` turns out to be rare in this tree — the six migrations
    // removed every other use of it — so a "we parsed N calls" threshold would
    // be measuring the wrong thing. Resolving the one that exists is the check
    // with signal.
    let ceilings = ceiling_calls();
    let resolved: Vec<String> = ceilings.iter().map(CeilingCall::describe).collect();
    assert!(
        ceilings
            .iter()
            .any(|call| call.file == "ppo/ppo_config.rs" && call.field == "\"num_steps\""),
        "the ceiling matcher did not resolve PPO's rollout bound \
         (`at_most(C, \"num_steps\", batch_size, MAX_BUFFER_CAPACITY)` in \
         src/algorithms/ppo/ppo_config.rs). It found: {resolved:?}\n\n\
         Either the parser is broken — in which case \
         `no_agent_config_rederives_the_capacity_ceiling` passes vacuously and this whole \
         file is asserting nothing — or PPO's bound genuinely moved, in which case delete \
         its CEILING_EXEMPTIONS row and this assertion together.",
    );

    // The constructor matcher works, checked against the *fixed* spelling: the
    // six migrated call sites must all be visible. This is what proves
    // `no_agent_builds_a_replay_buffer_infallibly`'s emptiness is a fact about
    // the source and not about the needle.
    let fallible = constructor_calls(FALLIBLE_CONSTRUCTORS);
    assert!(
        fallible.len() >= 6,
        "found only {} fallible replay constructor call(s) under src/algorithms/, expected \
         at least 6 (TD3, SAC, DDPG via `from_config`; DQN, C51, QR-DQN via \
         `uniform_from_config`). The constructor matcher or the production-line filter is \
         broken: {:?}",
        fallible.len(),
        fallible
            .iter()
            .map(ConstructorCall::describe)
            .collect::<Vec<_>>(),
    );

    // The `#[cfg(test)]` skip is doing work rather than eating everything: the
    // six config files name `MAX_BUFFER_CAPACITY` inside their test modules, and
    // that is exactly what rule 1 must not see.
    let skipped_ceiling_mentions: usize = sources
        .iter()
        .map(|path| {
            let relative = relative_path(root, path);
            let text = read_source(path);
            let lines: Vec<&str> = text.lines().collect();
            let production = production_lines(&relative, &lines);
            (0..lines.len())
                .filter(|index| {
                    !production.contains(index)
                        && !lines[*index].trim_start().starts_with("//")
                        && lines[*index].contains("MAX_BUFFER_CAPACITY")
                })
                .count()
        })
        .sum();
    assert!(
        skipped_ceiling_mentions >= 6,
        "only {skipped_ceiling_mentions} `MAX_BUFFER_CAPACITY` mention(s) fell inside a \
         `#[cfg(test)]` region, expected at least 6 (one per agent config's \
         `rejects_replay_buffer_capacity_above_ceiling` test). Either those tests were \
         deleted — which is the thing they exist to prevent — or the region scanner has \
         stopped finding test modules, in which case its skip is not what is making rule 1 \
         pass",
    );
}

/// Every `config::at_most(.., MAX_BUFFER_CAPACITY)` call in production source
/// under `src/algorithms/`.
fn ceiling_calls() -> Vec<CeilingCall> {
    let root = Path::new(ALGORITHMS_DIR);
    let mut calls: Vec<CeilingCall> = rust_sources(root)
        .into_iter()
        .flat_map(|path| {
            let relative = relative_path(root, &path);
            let text = read_source(&path);
            let lines: Vec<&str> = text.lines().collect();
            at_most_calls(&relative, &lines)
                .into_iter()
                .filter(|(_, arguments)| {
                    arguments
                        .iter()
                        .any(|argument| CEILING_NAMES.contains(&argument.as_str()))
                })
                .filter_map(|(line, arguments)| {
                    Some(CeilingCall {
                        file: relative.clone(),
                        line: line + 1,
                        field: arguments.get(1)?.clone(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    calls.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    calls
}

/// Every `config::at_most(` call in one file's production lines, as
/// `(line index, top-level arguments)`.
fn at_most_calls(relative: &str, lines: &[&str]) -> Vec<(usize, Vec<String>)> {
    let mut calls = Vec::new();
    for index in production_lines(relative, lines) {
        let needle = "at_most(";
        let Some(column) = lines[index].find(needle) else {
            continue;
        };
        let open = column + needle.len() - 1;
        let Some(arguments) = argument_text(lines, index, open) else {
            continue;
        };
        calls.push((index, top_level_arguments(&arguments)));
    }
    calls
}

/// Every call to one of `needles` in production source under `src/algorithms/`.
fn constructor_calls(needles: &[&str]) -> Vec<ConstructorCall> {
    let root = Path::new(ALGORITHMS_DIR);
    let mut calls: Vec<ConstructorCall> = rust_sources(root)
        .into_iter()
        .flat_map(|path| {
            let relative = relative_path(root, &path);
            let text = read_source(&path);
            let lines: Vec<&str> = text.lines().collect();
            production_lines(&relative, &lines)
                .into_iter()
                .flat_map(|index| {
                    needles
                        .iter()
                        .filter(|needle| lines[index].contains(**needle))
                        .map(|needle| ConstructorCall {
                            file: relative.clone(),
                            line: index + 1,
                            call: (*needle).to_owned(),
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    calls.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
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
/// or a string literal. String literals keep their quotes, so a `field`
/// argument compares as `"\"replay_buffer_capacity\""` and cannot collide with
/// an identifier of the same spelling.
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

/// The line indices of one file that are production code: not a comment, not
/// inside a `#[cfg(test)]` region.
fn production_lines(relative: &str, lines: &[&str]) -> Vec<usize> {
    let test_regions = cfg_test_regions(relative, lines);
    (0..lines.len())
        .filter(|index| {
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
/// attribute's own indentation and look like an item end. Those two checks are
/// what stop a brace miscount from quietly hiding production code.
///
/// Lifted from `tests/bounds_strictness_guard.rs`, deliberately duplicated
/// rather than shared: the two guards are independently deletable (ADR 0062
/// §4), and a shared test-support crate would make deleting either one a
/// cross-crate change.
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
                "src/algorithms/{relative}: the `#[cfg(test)]` at line {} never terminates. \
                 This guard refuses to skip to end-of-file, because doing so would hide \
                 every production call below it.",
                index + 1,
            )
        });

        let last = lines[end].trim();
        assert!(
            indent_of(lines[end]) == attribute_indent && (last == "}" || last.ends_with(';')),
            "src/algorithms/{relative}: the `#[cfg(test)]` region opened at line {} appears \
             to end at line {} (`{last}`), which is not an item end at the attribute's \
             indentation. The brace scan is off — probably a `{{` or `}}` inside a string \
             literal — and an over-long region would hide production code from this guard.",
            index + 1,
            end + 1,
        );

        regions.push((index, end));
        index = end + 1;
    }

    regions
}

/// Reads a source file, failing loudly. A guard that skips an unreadable file
/// passes vacuously, which is worse than no guard.
fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()))
}

/// Leading-space count. rustfmt indents with spaces.
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
