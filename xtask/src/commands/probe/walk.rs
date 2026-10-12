//! BYOE-1: an environment author's walk, from an empty directory to a harness
//! run of their own environment.
//!
//! The walk is one step table and two processes. `xtask` owns the steps a
//! `cargo` invocation can observe (the scaffold, the dependency list, a
//! build). The probe binary owns the steps that need its code to run, and
//! reports them by label, never by number, so renumbering touches this file
//! alone.
//!
//! A build failure is reported at the step whose code failed. The probe is
//! one crate, so one build cannot say which step broke; each step's code
//! therefore sits behind a cargo feature that includes the one before it
//! ([`LEVELS`]). The whole probe is built first. Only if that fails are the
//! levels built in order: the first that fails names the step, and the binary
//! from the last level that built still runs the steps below it.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
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

/// The steps, in order. A step's number is its position here.
pub const STEPS: &[Step] = &[
    Step {
        label: "scaffold",
        fatal: true,
    },
    Step {
        label: "add-deps",
        fatal: false,
    },
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
];

/// A build level of the probe crate.
#[derive(Debug)]
struct Level {
    /// The cargo feature that turns the level on; `None` is the base build.
    feature: Option<&'static str>,
    /// The step a build failure at this level is reported at.
    step: &'static str,
    /// What the step line says when the level builds and the step has no
    /// code to run. `None` when the binary reports the step itself.
    built: Option<&'static str>,
    /// Steps the binary reports once this level is built.
    reports: &'static [&'static str],
}

/// The levels, each including the one before it.
const LEVELS: &[Level] = &[
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
        built: Some("the policy implements BenchableAgent with act alone"),
        reports: &[],
    },
    Level {
        feature: Some("harness"),
        step: "run-suite",
        built: None,
        reports: &["run-suite"],
    },
];

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

/// Walk the probe at `probe_dir`, whose graph has already resolved from the
/// stage.
///
/// `environment_file` is the base level's source, relative to the probe.
///
/// # Errors
///
/// Returns an error if a build or the binary cannot be started, or if the
/// binary reports a step this file does not know.
pub fn walk(
    probe_dir: &Path,
    target_dir: &Path,
    binary: &str,
    resolved: &Resolved,
    environment_file: &str,
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

    let source = fs::read_to_string(probe_dir.join(environment_file))
        .with_context(|| format!("the template has no {environment_file}"))?;
    let outside = outside_contract(&source);
    if !outside.is_empty() {
        outcomes.insert(
            "implement-environment",
            Outcome::Fail(format!(
                "the environment needs more than the contract crate: {environment_file} names {}",
                outside.join(", ")
            )),
        );
        return Ok(outcomes);
    }

    let top = LEVELS.last().and_then(|level| level.feature);
    let mut expected: Vec<&'static str> = Vec::new();
    if build(probe_dir, target_dir, top)?.is_ok() {
        for level in LEVELS {
            record_built(&mut outcomes, level);
            expected.extend(level.reports);
        }
    } else {
        for level in LEVELS {
            match build(probe_dir, target_dir, level.feature)? {
                Ok(()) => {
                    record_built(&mut outcomes, level);
                    expected.extend(level.reports);
                }
                Err(error) => {
                    outcomes.insert(
                        level.step,
                        Outcome::Fail(format!("does not compile: {error}")),
                    );
                    break;
                }
            }
        }
    }

    if !expected.is_empty() {
        let executable = target_dir
            .join("debug")
            .join(format!("{binary}{}", std::env::consts::EXE_SUFFIX));
        let (status, reported) = run_binary(&executable, probe_dir)?;
        for (label, outcome) in reported {
            let Some(step) = STEPS.iter().find(|step| step.label == label) else {
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
        built
            .stderr
            .lines()
            .find(|line| line.starts_with("error"))
            .unwrap_or("the build failed with no error message")
            .to_string()
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
/// and a hang reads as flake.
fn run_binary(executable: &Path, probe_dir: &Path) -> Result<(String, Vec<(String, Outcome)>)> {
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
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break format!("it exited with {status}");
        }
        if started.elapsed() > RUN_LIMIT {
            child.kill()?;
            child.wait()?;
            break format!("it was killed after {} s", RUN_LIMIT.as_secs());
        }
        thread::sleep(Duration::from_millis(50));
    };
    let reported = fs::read_to_string(&log)?
        .lines()
        .filter_map(parse_step_line)
        .collect();
    Ok((status, reported))
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
        let source = "use rlevo::core::base::State;\nuse rlevo::core::episode::EpisodeGuard;\n";
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
        let source = "use rlevo::core::base::State;\nuse rlevo::envs::wrappers::TimeLimit;\n";
        assert_eq!(outside_contract(source), ["rlevo::envs"]);
    }

    #[test]
    fn test_a_grouped_import_is_refused_even_when_it_reaches_only_core() {
        assert_eq!(
            outside_contract("use rlevo::{core::base::State};\n"),
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
    fn test_no_compiler_error_is_none() {
        assert_eq!(first_error("not json\n"), None);
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
            hand_added(&resolved(&["rlevo", "rlevo-benchmarks", "rand"])),
            Outcome::Fail(
                "rlevo alone is insufficient; 2 additional crate(s) required by hand: \
                 rand, rlevo-benchmarks"
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

    #[test]
    fn test_every_level_reports_at_a_step_of_the_walk() {
        for level in LEVELS {
            assert!(STEPS.iter().any(|step| step.label == level.step));
            for label in level.reports {
                assert!(STEPS.iter().any(|step| step.label == *label));
            }
        }
    }
}
