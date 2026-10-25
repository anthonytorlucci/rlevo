//! A walk: the numbered path one persona takes, from an empty directory to a
//! run of their own code through `rlevo`. [`BYOE1`] is an environment
//! author's.
//!
//! A walk is one step table and two processes. `xtask` owns the steps a
//! `cargo` invocation can observe (the scaffold, the dependency list, a
//! build). The probe binary owns the steps that need its code to run, and
//! reports them by label, never by number, so renumbering touches this file
//! alone. A listing copied from the documentation reports nothing: `xtask`
//! reads its run from its exit status and a line it prints ([`LEARNER1`]).
//!
//! A build failure is reported at the step whose code failed. The probe is
//! one crate, so one build cannot say which step broke; each step's code
//! therefore sits behind a cargo feature that includes the one before it
//! ([`Walk::levels`]). The whole probe is built first. Only if that fails are
//! the levels built in order: the first that fails names the step, and the
//! binary from the last level that built still runs the steps below it.
//! Every step a later level would have built or reported fails with it, so a
//! walk whose levels are not in step order still stops at the step whose
//! code broke.
//!
//! Every walk opens with the same two steps, `scaffold` and `add-deps`,
//! which `xtask` checks itself.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use tracel_xtask::prelude::*;

use super::stage::{self, Resolved};

/// One step of the walk.
#[derive(Debug)]
pub struct Step {
    /// What the probe binary and the step lines call it.
    pub label: &'static str,
    /// Whether a failure stops the walk. A step that is not fatal records a
    /// finding about discovering the path, on a path that works.
    pub fatal: bool,
}

/// A walk: its steps, and the build levels that report into them.
#[derive(Debug)]
pub struct Walk {
    /// What each of its step lines starts with.
    pub prefix: &'static str,
    /// The steps, in order. A step's number is its position here.
    pub steps: &'static [Step],
    /// The build levels, each including the one before it. The first is the
    /// base build, with no feature.
    pub levels: &'static [Level],
}

/// The two steps every walk opens with, which `xtask` checks itself.
const SCAFFOLD: Step = Step {
    label: "scaffold",
    fatal: true,
};
const ADD_DEPS: Step = Step {
    label: "add-deps",
    fatal: false,
};

/// BYOE-1: an environment author's walk, from an empty directory to a harness
/// run of their own environment.
pub const BYOE1: Walk = Walk {
    prefix: "BYOE-STEP",
    steps: &[
        SCAFFOLD,
        ADD_DEPS,
        Step {
            label: "implement-environment",
            fatal: true,
        },
        Step {
            label: "post-terminal",
            fatal: true,
        },
        Step {
            label: "implement-agent",
            fatal: true,
        },
        Step {
            label: "run-suite",
            fatal: true,
        },
    ],
    levels: &[
        Level {
            feature: None,
            step: "implement-environment",
            built: Some(
                "Environment implemented against public API only; the file names nothing outside rlevo::core",
            ),
            reports: &["post-terminal"],
        },
        Level {
            feature: Some("agent"),
            step: "implement-agent",
            built: Some("the policy implements Agent with act alone"),
            reports: &[],
        },
        Level {
            feature: Some("harness"),
            step: "run-suite",
            built: None,
            reports: &["run-suite"],
        },
    ],
};

/// BYOE-2: a landscape author's walk, from an empty directory to a harness
/// run of a shipped optimiser on their own cost surface.
pub const BYOE2: Walk = Walk {
    prefix: "BYOE-STEP",
    steps: &[
        SCAFFOLD,
        ADD_DEPS,
        Step {
            label: "implement-landscape",
            fatal: true,
        },
        Step {
            label: "wrap-landscape",
            fatal: true,
        },
        Step {
            label: "build-loop",
            fatal: true,
        },
        Step {
            label: "run-trials",
            fatal: true,
        },
    ],
    levels: &[
        Level {
            feature: None,
            step: "implement-landscape",
            built: Some(
                "Landscape implemented against public API only; the file names nothing outside rlevo::core",
            ),
            reports: &[],
        },
        Level {
            feature: Some("engine"),
            step: "wrap-landscape",
            built: None,
            reports: &["wrap-landscape"],
        },
        Level {
            feature: Some("strategy"),
            step: "build-loop",
            built: None,
            reports: &["build-loop"],
        },
        Level {
            feature: Some("harness"),
            step: "run-trials",
            built: None,
            reports: &["run-trials"],
        },
    ],
};

/// BYOA-1b: an optimiser author's walk, from an empty directory to a harness
/// run of their own `Strategy`, compared with a strategy that never moves.
pub const BYOA1B: Walk = Walk {
    prefix: "BYOA-STEP",
    steps: &[
        SCAFFOLD,
        ADD_DEPS,
        Step {
            label: "implement-strategy",
            fatal: true,
        },
        Step {
            label: "null-strategy",
            fatal: true,
        },
        Step {
            label: "run-trials",
            fatal: true,
        },
        Step {
            label: "ordering",
            fatal: true,
        },
        Step {
            label: "determinism",
            fatal: true,
        },
    ],
    levels: &[
        Level {
            feature: None,
            step: "implement-strategy",
            built: Some("the hill-climber implements Strategy against public API only"),
            reports: &[],
        },
        Level {
            feature: Some("null"),
            step: "null-strategy",
            built: Some("the null strategy implements Strategy"),
            reports: &[],
        },
        Level {
            feature: Some("harness"),
            step: "run-trials",
            built: None,
            reports: &["run-trials", "ordering", "determinism"],
        },
    ],
};

/// BYOA-1a: an algorithm author's walk, from an empty directory to a learner
/// of their own, trained and measured on a shipped environment against a
/// random baseline.
///
/// Its levels are not in step order: the environment (step 5) is built
/// before the training run (step 4), which needs it.
pub const BYOA1A: Walk = Walk {
    prefix: "BYOA-STEP",
    steps: &[
        SCAFFOLD,
        ADD_DEPS,
        Step {
            label: "implement-algorithm",
            fatal: true,
        },
        Step {
            label: "learning-seam",
            fatal: false,
        },
        Step {
            label: "first-party-env",
            fatal: true,
        },
        Step {
            label: "baseline",
            fatal: true,
        },
    ],
    levels: &[
        Level {
            feature: None,
            step: "implement-algorithm",
            built: Some("the learner implements Agent against public API only"),
            reports: &[],
        },
        Level {
            feature: Some("env"),
            step: "first-party-env",
            built: Some("FrozenLake and TimeLimit reached through rlevo::envs"),
            reports: &[],
        },
        Level {
            feature: Some("learn"),
            step: "learning-seam",
            built: None,
            reports: &["learning-seam"],
        },
        Level {
            feature: Some("harness"),
            step: "baseline",
            built: None,
            reports: &["baseline"],
        },
    ],
};

/// Learner-1: a learner's walk, from an empty directory to a run of a
/// listing in the documentation, copied as written.
pub const LEARNER1: Walk = Walk {
    prefix: "LEARNER-STEP",
    steps: &[
        SCAFFOLD,
        ADD_DEPS,
        Step {
            label: "build-listing",
            fatal: true,
        },
        Step {
            label: "run-listing",
            fatal: true,
        },
    ],
    levels: &[Level {
        feature: None,
        step: "build-listing",
        built: Some("the listing compiles as written"),
        reports: &["run-listing"],
    }],
};

/// A seam of the `rlevo-algo` template: the crate `cargo generate` gives an
/// author, from an empty directory to its first run.
///
/// It runs as a listing does, reporting nothing itself: `xtask` reads its
/// run from its exit status and the table's last row. The lint and test
/// steps are `xtask`'s, run once the crate builds ([`lint`], [`test`]).
pub const TEMPLATE: Walk = Walk {
    prefix: "TEMPLATE-STEP",
    steps: &[
        SCAFFOLD,
        ADD_DEPS,
        Step {
            label: "build-template",
            fatal: true,
        },
        Step {
            label: LINT,
            fatal: true,
        },
        Step {
            label: TEST,
            fatal: true,
        },
        Step {
            label: "run-template",
            fatal: true,
        },
    ],
    levels: &[Level {
        feature: None,
        step: "build-template",
        built: Some("the generated crate builds as rendered"),
        reports: &["run-template"],
    }],
};

/// The step [`lint`] reports at.
pub const LINT: &str = "lint-template";
/// The step [`test`] reports at.
pub const TEST: &str = "test-template";

/// A build level of the probe crate.
#[derive(Debug)]
pub struct Level {
    /// The cargo feature that turns the level on; `None` is the base build.
    pub feature: Option<&'static str>,
    /// The step a build failure at this level is reported at.
    pub step: &'static str,
    /// What the step line says when the level builds and the step has no
    /// code to run. `None` when the binary reports the step itself.
    pub built: Option<&'static str>,
    /// Steps the binary reports once this level is built.
    pub reports: &'static [&'static str],
}

/// The one crate an author asks for. Every other direct dependency of the
/// probe is one they had to add by hand.
const ASKED_FOR: &str = "rlevo";

/// How long the probe binary may run before it is killed and its unreported
/// steps fail.
const RUN_LIMIT: Duration = Duration::from_secs(300);

/// How a step ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The step's assertion held.
    Pass(String),
    /// It did not.
    Fail(String),
}

/// The walk's result: what each step found, by label.
pub type Outcomes = BTreeMap<&'static str, Outcome>;

/// The steps a probe binary reported, by label, in the order it printed them.
type Reported = Vec<(String, Outcome)>;

/// Take `walk` with the probe at `probe_dir`, whose graph has already
/// resolved from the stage.
///
/// `contract`, when given, is the base level's source file, relative to the
/// probe, which must name nothing outside `rlevo::core`. A breach is
/// reported at the base level's step.
///
/// `sentinel`, when given, means the binary is a listing that prints no step
/// lines. The steps its levels report pass when it exits zero having printed
/// a line that starts with `sentinel` on stdout.
///
/// # Errors
///
/// Returns an error if a build or the binary cannot be started, or if the
/// binary reports a step the walk does not have.
pub fn run(
    walk: &Walk,
    probe_dir: &Path,
    target_dir: &Path,
    binary: &str,
    resolved: &Resolved,
    contract: Option<&str>,
    sentinel: Option<&str>,
) -> Result<Outcomes> {
    let mut outcomes = Outcomes::new();
    outcomes.insert(
        "scaffold",
        Outcome::Pass(format!(
            "probe created at {} (outside the workspace)",
            probe_dir.display()
        )),
    );
    outcomes.insert("add-deps", hand_added(resolved));

    if let Some(file) = contract {
        let source = fs::read_to_string(probe_dir.join(file))
            .with_context(|| format!("the template has no {file}"))?;
        let outside = outside_contract(&source);
        if !outside.is_empty() {
            outcomes.insert(
                walk.levels[0].step,
                Outcome::Fail(format!(
                    "the file needs more than the contract crate: {file} names {}",
                    outside.join(", ")
                )),
            );
            return Ok(outcomes);
        }
    }

    let expected = build_levels(walk, &mut outcomes, |feature| {
        build(probe_dir, target_dir, feature)
    })?;

    if !expected.is_empty() {
        let executable = target_dir
            .join("debug")
            .join(format!("{binary}{}", std::env::consts::EXE_SUFFIX));
        let (ended, reported) = run_binary(&executable, probe_dir)?;
        let status = describe(ended);
        if let Some(sentinel) = sentinel {
            let outcome = listing_outcome(
                ended.is_some_and(|ended| ended.success()),
                &status,
                &fs::read_to_string(probe_dir.join("probe-stdout.log"))?,
                &fs::read_to_string(probe_dir.join("probe-stderr.log"))?,
                sentinel,
            );
            for label in expected {
                outcomes.entry(label).or_insert_with(|| outcome.clone());
            }
            return Ok(outcomes);
        }
        for (label, outcome) in reported {
            let Some(step) = walk.steps.iter().find(|step| step.label == label) else {
                bail!("the probe reported a step this walk does not have: `{label}`");
            };
            outcomes.entry(step.label).or_insert(outcome);
        }
        for label in expected {
            outcomes.entry(label).or_insert_with(|| {
                Outcome::Fail(format!("the probe did not report this step ({status})"))
            });
        }
    }
    Ok(outcomes)
}

/// Build the probe's levels with `build`, record what the builds show, and
/// return the steps the binary must then report.
///
/// The whole probe is built first. Only if that fails are the levels built
/// in order, and the first that fails names its step. The levels after it
/// compiled nothing, so every step they would have built or reported fails
/// as well, naming the step whose code did not compile. That matters for a
/// walk whose levels are not in step order: one that builds step 5's code
/// before step 4's must stop at step 5 when that code breaks, not leave step
/// 4 unreached.
fn build_levels(
    walk: &Walk,
    outcomes: &mut Outcomes,
    mut build: impl FnMut(Option<&'static str>) -> Result<std::result::Result<(), String>>,
) -> Result<Vec<&'static str>> {
    let mut expected = Vec::new();
    let top = walk.levels.last().and_then(|level| level.feature);
    if build(top)?.is_ok() {
        for level in walk.levels {
            record_built(outcomes, level);
            expected.extend(level.reports);
        }
        return Ok(expected);
    }
    for (index, level) in walk.levels.iter().enumerate() {
        match build(level.feature)? {
            Ok(()) => {
                record_built(outcomes, level);
                expected.extend(level.reports);
            }
            Err(error) => {
                outcomes.insert(
                    level.step,
                    Outcome::Fail(format!("does not compile: {error}")),
                );
                let failed = step_number(walk, level.step);
                for later in &walk.levels[index + 1..] {
                    for label in std::iter::once(later.step).chain(later.reports.iter().copied()) {
                        outcomes.entry(label).or_insert_with(|| {
                            Outcome::Fail(format!(
                                "not built: the code of step {failed} does not compile"
                            ))
                        });
                    }
                }
                break;
            }
        }
    }
    Ok(expected)
}

/// A step's number: its position in the walk, from 1.
fn step_number(walk: &Walk, label: &str) -> usize {
    walk.steps
        .iter()
        .position(|step| step.label == label)
        .map_or(0, |index| index + 1)
}

fn record_built(outcomes: &mut Outcomes, level: &Level) {
    if let Some(detail) = level.built {
        outcomes
            .entry(level.step)
            .or_insert_with(|| Outcome::Pass(detail.to_string()));
    }
}

/// Step 2: the crates an author adds by hand beyond the one they asked for.
fn hand_added(resolved: &Resolved) -> Outcome {
    if !resolved.direct.contains(ASKED_FOR) {
        return Outcome::Fail(format!("the probe does not depend on {ASKED_FOR}"));
    }
    let extra: Vec<&str> = resolved
        .direct
        .iter()
        .map(String::as_str)
        .filter(|name| *name != ASKED_FOR)
        .collect();
    if extra.is_empty() {
        Outcome::Pass(format!("{ASKED_FOR} alone is sufficient"))
    } else {
        Outcome::Fail(format!(
            "{ASKED_FOR} alone is insufficient; {} additional crate(s) required by hand: {}",
            extra.len(),
            extra.join(", ")
        ))
    }
}

/// Paths in `source` that reach `rlevo` anywhere but `rlevo::core`.
///
/// The base level's file must build against the contract crate alone, so
/// every `rlevo::` path in it is spelled `rlevo::core` in full. A grouped
/// import (`use rlevo::{core::.., envs::..}`) is refused as well: it cannot
/// be respelled mechanically, and it hides what it reaches.
///
/// The scan is textual, so a path in a comment counts.
pub fn outside_contract(source: &str) -> Vec<String> {
    const UMBRELLA: &str = "rlevo::";
    let mut found = Vec::new();
    let mut rest = source;
    let mut previous: Option<char> = None;
    while let Some(at) = rest.find(UMBRELLA) {
        let before = rest[..at].chars().next_back().or(previous);
        let after = &rest[at + UMBRELLA.len()..];
        let part_of_longer_name = before.is_some_and(|c| c.is_alphanumeric() || c == '_');
        if !part_of_longer_name && !names_core(after) {
            let tail: String = if after.starts_with('{') {
                "{".to_string()
            } else {
                after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect()
            };
            let path = format!("{UMBRELLA}{tail}");
            if !found.contains(&path) {
                found.push(path);
            }
        }
        previous = Some(':');
        rest = after;
    }
    found
}

/// Whether the path segment `after` opens with is exactly `core`.
fn names_core(after: &str) -> bool {
    after.strip_prefix("core").is_some_and(|rest| {
        !rest
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

/// Lint the probe as an author's first `cargo clippy` would, at clippy's
/// default lints, every target included, a warning an error.
///
/// # Errors
///
/// Returns an error if clippy cannot be started.
pub fn lint(probe_dir: &Path, target_dir: &Path) -> Result<Outcome> {
    let mut cargo = stage::probe_cargo(probe_dir, target_dir);
    cargo.args([
        "clippy",
        "--all-targets",
        "--message-format=json",
        "--",
        "-D",
        "warnings",
    ]);
    let linted = stage::run(&mut cargo)?;
    if linted.ok {
        return Ok(Outcome::Pass(
            "cargo clippy --all-targets -- -D warnings is clean at clippy's default lints"
                .to_string(),
        ));
    }
    Ok(Outcome::Fail(first_error(&linted.stdout).unwrap_or_else(
        || first_error_line(&linted.stderr, "clippy failed with no error message"),
    )))
}

/// Run the probe's own tests, as an author's first `cargo test` would.
///
/// # Errors
///
/// Returns an error if the tests cannot be started.
pub fn test(probe_dir: &Path, target_dir: &Path) -> Result<Outcome> {
    let mut cargo = stage::probe_cargo(probe_dir, target_dir);
    cargo.args(["test", "--message-format=json"]);
    let tested = stage::run(&mut cargo)?;
    Ok(test_outcome(tested.ok, &tested.stdout, &tested.stderr))
}

/// What a `cargo test --message-format=json` run says: how many tests
/// passed, or which failed, or the error that stopped them building.
fn test_outcome(ok: bool, stdout: &str, stderr: &str) -> Outcome {
    let passed: usize = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("test result: ok. "))
        .filter_map(|rest| rest.split(' ').next()?.parse::<usize>().ok())
        .sum();
    if ok {
        return Outcome::Pass(format!("cargo test passes, {passed} tests"));
    }
    let failed: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("test ")?.strip_suffix(" ... FAILED"))
        .collect();
    if !failed.is_empty() {
        return Outcome::Fail(format!("tests fail: {}", failed.join(", ")));
    }
    Outcome::Fail(
        first_error(stdout)
            .unwrap_or_else(|| first_error_line(stderr, "cargo test failed with no error message")),
    )
}

/// The first line of `stderr` that starts with `error`, or `otherwise`.
fn first_error_line(stderr: &str, otherwise: &str) -> String {
    stderr
        .lines()
        .find(|line| line.starts_with("error"))
        .unwrap_or(otherwise)
        .to_string()
}

/// Build the probe at one level. The inner error is the compiler's first.
fn build(
    probe_dir: &Path,
    target_dir: &Path,
    feature: Option<&str>,
) -> Result<std::result::Result<(), String>> {
    let mut cargo = stage::probe_cargo(probe_dir, target_dir);
    cargo.args(["build", "--message-format=json"]);
    if let Some(feature) = feature {
        cargo.args(["--features", feature]);
    }
    let built = stage::run(&mut cargo)?;
    if built.ok {
        return Ok(Ok(()));
    }
    Ok(Err(first_error(&built.stdout).unwrap_or_else(|| {
        // Not a compiler diagnostic: a manifest or resolution error.
        first_error_line(&built.stderr, "the build failed with no error message")
    })))
}

/// The first error in `cargo build --message-format=json` output: its code,
/// when it has one, and the first line of its message.
fn first_error(messages: &str) -> Option<String> {
    messages
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["reason"] == "compiler-message")
        .find(|record| record["message"]["level"] == "error")
        .map(|record| {
            let message = record["message"]["message"]
                .as_str()
                .and_then(|text| text.lines().next())
                .unwrap_or_default();
            match record["message"]["code"]["code"].as_str() {
                Some(code) => format!("{code} {message}"),
                None => message.to_string(),
            }
        })
}

/// Run the probe binary and collect the steps it reports.
///
/// Its streams go to files beside the probe, which a kept run keeps. The
/// binary is killed at [`RUN_LIMIT`]: a defect in a probe must fail the run,
/// and a hang reads as flake. The exit status is `None` when it was killed.
fn run_binary(executable: &Path, probe_dir: &Path) -> Result<(Option<ExitStatus>, Reported)> {
    let log = probe_dir.join("probe-stderr.log");
    let mut child = Command::new(executable)
        .current_dir(probe_dir)
        .stdout(Stdio::from(fs::File::create(
            probe_dir.join("probe-stdout.log"),
        )?))
        .stderr(Stdio::from(fs::File::create(&log)?))
        .spawn()
        .with_context(|| format!("could not start {}", executable.display()))?;

    let started = Instant::now();
    let ended = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if started.elapsed() > RUN_LIMIT {
            child.kill()?;
            child.wait()?;
            break None;
        }
        thread::sleep(Duration::from_millis(50));
    };
    let reported = fs::read_to_string(&log)?
        .lines()
        .filter_map(parse_step_line)
        .collect();
    Ok((ended, reported))
}

/// How the binary ended, in words.
fn describe(ended: Option<ExitStatus>) -> String {
    ended.map_or_else(
        || format!("it was killed after {} s", RUN_LIMIT.as_secs()),
        |status| format!("it exited with {status}"),
    )
}

/// A listing's run: it must exit zero and print a line that starts with
/// `sentinel`. Exit zero alone would pass a program that printed a usage
/// banner and did nothing. A failure carries the last line on stderr, where
/// a `main` that returns an error prints it.
fn listing_outcome(
    exited_zero: bool,
    status: &str,
    stdout: &str,
    stderr: &str,
    sentinel: &str,
) -> Outcome {
    let printed = stdout.lines().find(|line| line.starts_with(sentinel));
    match (exited_zero, printed) {
        (true, Some(line)) => Outcome::Pass(format!("it exited zero and printed `{line}`")),
        (true, None) => Outcome::Fail(format!(
            "it exited zero and printed no line starting `{sentinel}`"
        )),
        (false, _) => match stderr.lines().rev().find(|line| !line.trim().is_empty()) {
            Some(last) => Outcome::Fail(format!("{status}: {last}")),
            None => Outcome::Fail(status.to_string()),
        },
    }
}

/// Parse `PROBE-STEP <label> <PASS|FAIL> :: <detail>`.
fn parse_step_line(line: &str) -> Option<(String, Outcome)> {
    let rest = line.strip_prefix("PROBE-STEP ")?;
    let (head, detail) = rest.split_once(" :: ")?;
    let (label, verdict) = head.split_once(' ')?;
    let outcome = match verdict {
        "PASS" => Outcome::Pass(detail.to_string()),
        "FAIL" => Outcome::Fail(detail.to_string()),
        _ => return None,
    };
    Some((label.to_string(), outcome))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// What the scan returns for a file that stays inside the contract.
    const NOTHING: [&str; 0] = [];

    #[test]
    fn test_contract_paths_written_in_full_are_accepted() {
        let source =
            "use rlevo::core::base::Observation;\nuse rlevo::core::episode::EpisodeGuard;\n";
        assert_eq!(outside_contract(source), NOTHING);
    }

    #[test]
    fn test_the_contract_crate_named_in_prose_is_accepted() {
        assert_eq!(
            outside_contract("//! Names nothing outside `rlevo::core`.\n"),
            NOTHING
        );
        assert_eq!(outside_contract("use rlevo::core;\n"), NOTHING);
    }

    #[test]
    fn test_a_module_whose_name_only_starts_with_core_is_outside() {
        assert_eq!(
            outside_contract("use rlevo::coreutils::Thing;\n"),
            ["rlevo::coreutils"]
        );
    }

    #[test]
    fn test_a_path_outside_the_contract_is_named() {
        let source = "use rlevo::core::base::Observation;\nuse rlevo::envs::wrappers::TimeLimit;\n";
        assert_eq!(outside_contract(source), ["rlevo::envs"]);
    }

    #[test]
    fn test_a_grouped_import_is_refused_even_when_it_reaches_only_core() {
        assert_eq!(
            outside_contract("use rlevo::{core::base::Observation};\n"),
            ["rlevo::{"]
        );
    }

    #[test]
    fn test_a_longer_crate_name_is_not_the_umbrella() {
        assert_eq!(outside_contract("use my_rlevo::envs::Thing;\n"), NOTHING);
        assert_eq!(outside_contract("use xrlevo::envs::Thing;\n"), NOTHING);
    }

    #[test]
    fn test_each_offending_path_is_reported_once() {
        let source = "rlevo::burn::Tensor; rlevo::burn::Tensor; rlevo::harness::Suite;";
        assert_eq!(outside_contract(source), ["rlevo::burn", "rlevo::harness"]);
    }

    #[test]
    fn test_step_lines_parse_by_label() {
        assert_eq!(
            parse_step_line("PROBE-STEP run-suite PASS :: 2 trials :: none errored"),
            Some((
                "run-suite".to_string(),
                Outcome::Pass("2 trials :: none errored".to_string())
            ))
        );
        assert_eq!(
            parse_step_line("PROBE-STEP post-terminal FAIL :: accepted"),
            Some((
                "post-terminal".to_string(),
                Outcome::Fail("accepted".to_string())
            ))
        );
    }

    #[test]
    fn test_lines_that_are_not_step_lines_are_ignored() {
        assert_eq!(parse_step_line("INFO trial 0 finished"), None);
        assert_eq!(parse_step_line("PROBE-STEP run-suite MAYBE :: x"), None);
        assert_eq!(parse_step_line("PROBE-STEP run-suite PASS"), None);
    }

    #[test]
    fn test_a_listing_that_exits_zero_and_prints_the_sentinel_passes() {
        assert_eq!(
            listing_outcome(
                true,
                "it exited with exit status: 0",
                "start\nEpisode over: Terminated\n",
                "",
                "Episode over: "
            ),
            Outcome::Pass("it exited zero and printed `Episode over: Terminated`".to_string())
        );
    }

    #[test]
    fn test_a_listing_that_exits_zero_without_the_sentinel_fails() {
        assert_eq!(
            listing_outcome(
                true,
                "it exited with exit status: 0",
                "usage: quickstart\n",
                "",
                "Episode over: "
            ),
            Outcome::Fail(
                "it exited zero and printed no line starting `Episode over: `".to_string()
            )
        );
    }

    #[test]
    fn test_a_listing_that_fails_carries_its_last_stderr_line() {
        assert_eq!(
            listing_outcome(
                false,
                "it exited with exit status: 1",
                "Episode over: Terminated\n",
                "warning\nError: bad config\n\n",
                "Episode over: "
            ),
            Outcome::Fail("it exited with exit status: 1: Error: bad config".to_string())
        );
        assert_eq!(
            listing_outcome(false, "it was killed after 300 s", "", "", "Episode over: "),
            Outcome::Fail("it was killed after 300 s".to_string())
        );
    }

    #[test]
    fn test_first_error_skips_warnings_and_keeps_the_code() {
        let messages = concat!(
            r#"{"reason":"compiler-artifact","target":{"name":"probe"}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"warning","message":"unused","code":null}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"error","message":"the trait bound `A: B` is not satisfied\nrequired by","code":{"code":"E0277"}}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"error","message":"aborting","code":null}}"#,
            "\n",
        );
        assert_eq!(
            first_error(messages).as_deref(),
            Some("E0277 the trait bound `A: B` is not satisfied")
        );
    }

    #[test]
    fn test_a_test_run_counts_what_passed_and_names_what_failed() {
        let passed = "running 2 tests\ntest a ... ok\ntest b ... ok\n\
                      test result: ok. 2 passed; 0 failed\n\
                      test result: ok. 0 passed; 0 failed\n";
        assert_eq!(
            test_outcome(true, passed, ""),
            Outcome::Pass("cargo test passes, 2 tests".to_string())
        );
        let failed = "test tests::a ... ok\ntest tests::b ... FAILED\n\
                      test result: FAILED. 1 passed; 1 failed\n";
        assert_eq!(
            test_outcome(
                false,
                failed,
                "error: test failed, to rerun pass `--bin p`\n"
            ),
            Outcome::Fail("tests fail: tests::b".to_string())
        );
        assert_eq!(
            test_outcome(false, "", "error: could not find `Cargo.toml`\n"),
            Outcome::Fail("error: could not find `Cargo.toml`".to_string())
        );
    }

    #[test]
    fn test_no_compiler_error_is_none() {
        assert_eq!(first_error("not json\n"), None);
    }

    /// A walk whose second level is for a later step than its third:
    /// `dependency` is step 5 and is built before `finding`, step 4.
    const OUT_OF_ORDER: Walk = Walk {
        prefix: "TEST-STEP",
        steps: &[
            SCAFFOLD,
            ADD_DEPS,
            Step {
                label: "base",
                fatal: true,
            },
            Step {
                label: "finding",
                fatal: false,
            },
            Step {
                label: "dependency",
                fatal: true,
            },
            Step {
                label: "last",
                fatal: true,
            },
        ],
        levels: &[
            Level {
                feature: None,
                step: "base",
                built: Some("built"),
                reports: &[],
            },
            Level {
                feature: Some("dependency"),
                step: "dependency",
                built: Some("built"),
                reports: &[],
            },
            Level {
                feature: Some("finding"),
                step: "finding",
                built: None,
                reports: &["finding"],
            },
            Level {
                feature: Some("last"),
                step: "last",
                built: None,
                reports: &["last"],
            },
        ],
    };

    /// A build of [`OUT_OF_ORDER`] in which `failing` and every level after
    /// it do not compile, the whole probe included.
    fn build_failing_from(
        failing: &'static str,
    ) -> impl FnMut(Option<&'static str>) -> Result<std::result::Result<(), String>> {
        let broken = OUT_OF_ORDER
            .levels
            .iter()
            .skip_while(|level| level.feature != Some(failing))
            .filter_map(|level| level.feature)
            .collect::<Vec<_>>();
        move |feature| {
            Ok(match feature {
                Some(feature) if broken.contains(&feature) => {
                    Err(format!("E0432 unresolved import in {feature}"))
                }
                _ => Ok(()),
            })
        }
    }

    /// The outcomes of a walk over [`OUT_OF_ORDER`] whose binary reports
    /// nothing: `xtask`'s two steps pass, the builds decide the rest.
    fn walk_out_of_order(failing: &'static str) -> Outcomes {
        let mut outcomes = Outcomes::new();
        outcomes.insert("scaffold", Outcome::Pass("created".to_string()));
        outcomes.insert("add-deps", Outcome::Pass("rlevo alone".to_string()));
        let expected = build_levels(&OUT_OF_ORDER, &mut outcomes, build_failing_from(failing))
            .expect("the fake build does not error");
        assert!(expected.is_empty(), "no level that reports a step built");
        outcomes
    }

    #[test]
    fn test_a_level_built_before_an_earlier_step_stops_the_walk_at_its_own_step() {
        let outcomes = walk_out_of_order("dependency");
        let (lines, verdict) = super::super::walk_report(&OUT_OF_ORDER, &outcomes);
        assert_eq!(verdict, super::super::Verdict::FailsAt { step: 5, of: 6 });
        assert_eq!(
            lines[3],
            "TEST-STEP 4 FAIL finding :: not built: the code of step 5 does not compile"
        );
        assert_eq!(
            lines[4],
            "TEST-STEP 5 FAIL dependency :: does not compile: E0432 unresolved import in dependency"
        );
    }

    #[test]
    fn test_the_levels_after_a_failed_one_are_not_built() {
        let outcomes = walk_out_of_order("finding");
        assert_eq!(outcomes["dependency"], Outcome::Pass("built".to_string()));
        assert_eq!(
            outcomes["last"],
            Outcome::Fail("not built: the code of step 4 does not compile".to_string())
        );
        let (_, verdict) = super::super::walk_report(&OUT_OF_ORDER, &outcomes);
        assert_eq!(verdict, super::super::Verdict::FailsAt { step: 6, of: 6 });
    }

    #[test]
    fn test_a_whole_build_records_every_level_and_expects_its_reports() {
        let mut outcomes = Outcomes::new();
        let expected = build_levels(&OUT_OF_ORDER, &mut outcomes, |_| Ok(Ok(())))
            .expect("the fake build does not error");
        assert_eq!(expected, ["finding", "last"]);
        assert_eq!(outcomes["base"], Outcome::Pass("built".to_string()));
        assert_eq!(outcomes["dependency"], Outcome::Pass("built".to_string()));
        assert!(!outcomes.contains_key("finding"));
    }

    fn resolved(direct: &[&str]) -> Resolved {
        Resolved {
            rlevo: Vec::new(),
            unreached: Vec::new(),
            direct: direct
                .iter()
                .map(|name| (*name).to_string())
                .collect::<BTreeSet<_>>(),
        }
    }

    #[test]
    fn test_the_umbrella_alone_passes_step_two() {
        assert_eq!(
            hand_added(&resolved(&["rlevo"])),
            Outcome::Pass("rlevo alone is sufficient".to_string())
        );
    }

    #[test]
    fn test_every_hand_added_crate_is_listed() {
        assert_eq!(
            hand_added(&resolved(&["rlevo", "rlevo-harness", "rand"])),
            Outcome::Fail(
                "rlevo alone is insufficient; 2 additional crate(s) required by hand: \
                 rand, rlevo-harness"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_a_probe_that_skips_the_umbrella_fails_step_two() {
        assert_eq!(
            hand_added(&resolved(&["rlevo-core"])),
            Outcome::Fail("the probe does not depend on rlevo".to_string())
        );
    }
}
