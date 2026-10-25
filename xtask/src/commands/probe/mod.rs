//! Outside-in probes: the product path, walked from where a consumer stands.
//!
//! Every test, bench and example in the workspace resolves its siblings by
//! path, sees `publish = false` crates through dev-dependencies and has its
//! `required-features` satisfied for it. None of that is true for someone
//! who runs `cargo add rlevo`. A probe is a small crate built *outside* the
//! workspace against the crates as `cargo package` would ship them, so what
//! it finds is what that person finds: a file missing from a package, an
//! item that is not public, a crate they must add by hand.
//!
//! It is also the only build of the packaged crates against each other.
//! `cargo package --no-verify` checks that each crate packages, never that
//! the package compiles.
//!
//! [`PROBES`] is the source of truth, and `--list` prints it for the workflow
//! matrix, as the test tiers do. There are four kinds:
//!
//! - **A walk** ([`walk`]) is a numbered path one persona takes. A step that
//!   fails names the defect: `BYOE-1/thermostat fails at step 3 of 6`.
//! - **A listing** ([`listing`]) is a walk whose code is a listing in the
//!   documentation, copied as written: what a learner pastes into a new
//!   crate.
//! - **A template** ([`template`]) is a walk whose code is a seam of the
//!   `rlevo-algo` template, rendered as `cargo generate` renders it: the
//!   crate an author starts from, built, linted, tested and run.
//! - **A measurement** ([`measure`]) has no steps. It records what a
//!   dependency list costs: packages, Burn packages, a cold build.
//!
//! # What a run does
//!
//! It packages the workspace into a fresh run directory, extracts the crates,
//! writes the probe beside them with a patch table that points every `rlevo`
//! crate at the extracted copy ([`stage`]), checks that the probe's resolved
//! graph took them from there, then walks or measures. The run directory is
//! under `$RLEVO_PROBE_SCRATCH`, or the system's temporary directory. It is
//! deleted when the run is green and kept, without its build output, when it
//! is not.
//!
//! Step and measurement lines go to stderr. The last line is the verdict:
//!
//! ```text
//! BYOE-STEP 1 PASS scaffold :: probe created at /scratch/.../probe (outside the workspace)
//! BYOE-STEP 2 PASS add-deps :: rlevo alone is sufficient
//! ...
//! BYOE-1/thermostat passes.
//! ```
//!
//! A fatal step that fails, or a run that cannot be staged, exits non-zero.
//! A step that is not fatal records a finding on a path that works, and does
//! not: the finding is a line in the probe's manifest template, reviewed like
//! any other, and a check that is red for a known reason hides the day it
//! goes red for a new one.
//!
//! # Writing a template
//!
//! A probe's template is `xtask/probes/<name>/`: a `Cargo.toml.template` and
//! a `src/` tree. The manifest is not named `Cargo.toml`, so cargo never
//! adopts the directory into the workspace, and nothing here is compiled or
//! linted by a workspace build. The rules:
//!
//! 1. **Public API only**, reached the way the documentation says to reach
//!    it. A path dependency on a workspace crate turns the probe into one
//!    more in-tree test.
//! 2. **Do not copy a shipped environment.** It gets a payload source, a
//!    renderer and a family for free, and each of those is what a third
//!    party is missing.
//! 3. **Friction is the output.** Keep the workaround and mark it `FINDING:`
//!    with what it shows. Smoothing it away deletes the evidence.
//! 4. **Bound every loop over environment steps.** A defect must fail the
//!    probe, not hang it.
//! 5. **Seed everything from a constant.** A probe that fails one run in ten
//!    is blamed on the thing under test.
//! 6. **Each step's code sits behind its level's feature**, and a feature
//!    gates the probe's own modules only, so the levels leave the dependency
//!    list unchanged.
//! 7. **The change that alters the API alters the template**, in the same
//!    pull request. The template's diff is then the measurement of what an
//!    author stops writing.
//!
//! # The `rlevo-algo` template
//!
//! `templates/rlevo-algo/` is not a probe's template but the product: what
//! `cargo generate` gives an author, one seam per thing they implement. Each
//! seam has a row here, so CI builds, lints, tests and runs the crate an
//! author would get. Its rules differ from a probe's:
//!
//! 1. **No friction is kept.** A seam is the path with the friction taken
//!    out; a probe keeps its findings.
//! 2. **It works as it stands.** Its placeholder implementation runs, and
//!    its `main` prints its rows as one Markdown table and nothing else.
//! 3. **The change that alters a stamped seam alters the template**, in the
//!    same pull request: `Agent`, `Environment` and `Strategy`, and the
//!    trial shapes each `main` calls.
//! 4. **Its `rlevo` line names this checkout's release**, as the README's
//!    does ([`listing::check_versions`]): the version bump changes both.
//!
//! # Examples
//!
//! ```ignore
//! cargo xtask probe byoe-1-thermostat
//! cargo xtask probe consumer-core
//! cargo xtask probe --list
//! RLEVO_PROBE_SCRATCH=~/scratch cargo xtask probe consumer-umbrella
//! ```

mod listing;
mod measure;
mod stage;
mod template;
mod walk;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, bail};
use tracel_xtask::prelude::*;

use super::tiers;
use stage::Workspace;
use walk::Outcome;

/// A probe: a template under `xtask/probes/<name>/`, or a seam of the
/// `rlevo-algo` template, and how it is run.
struct Probe {
    /// The template directory, the command-line name and the matrix entry.
    /// A seam's probe is named for the seam ([`template::probe_name`]).
    name: &'static str,
    /// What the verdict line calls it.
    title: &'static str,
    kind: Kind,
    /// Printed before the verdict: what a green run of this probe does not
    /// yet show.
    note: Option<&'static str>,
}

enum Kind {
    /// A walk. `contract`, when given, is the base level's source file,
    /// which must name nothing outside the contract crate.
    Walk {
        walk: &'static walk::Walk,
        contract: Option<&'static str>,
    },
    /// A walk over `listing`, which is the probe's dependencies and its
    /// `src/main.rs`.
    Listing {
        walk: &'static walk::Walk,
        listing: &'static listing::Listing,
    },
    /// The seam `templates/rlevo-algo/<seam>/`, rendered and walked. Its run
    /// passes when it exits zero having printed a line that starts with
    /// `sentinel`: the last row of its table.
    Template {
        seam: &'static str,
        sentinel: &'static str,
    },
    /// Cone, Burn count and a cold build. `contract` makes the probe's one
    /// source file a copy of another probe's contract file, built against
    /// `rlevo-core` alone.
    Measurement { contract: Option<ContractFile> },
}

/// A source file that names nothing outside `rlevo::core`, in the template
/// that owns it.
struct ContractFile {
    probe: &'static str,
    file: &'static str,
}

/// What a green BYOE-1 walk does not yet show.
const BYOE1_NOTE: &str = "BYOE-1 does not end at the harness run: recording the run and \
                          rendering a report are not walked yet";

/// What a green BYOE-2 walk does not yet show.
const BYOE2_NOTE: &str = "BYOE-2 does not end at the harness run: recording the population \
                          and drawing the landscape are not walked yet";

/// What a green BYOA-1a walk does not yet show.
const BYOA1A_NOTE: &str = "BYOA-1a does not end at the baseline: recording the run, the \
                           learner's own metrics in the record, and showing that two \
                           submissions faced one task are not walked yet";

/// What a green BYOA-1b walk does not yet show.
const BYOA1B_NOTE: &str = "BYOA-1b does not end at the determinism check: recording the run \
                           and showing that two submissions faced one task are not walked yet";

/// What a green template walk does not yet show.
const TEMPLATE_NOTE: &str = "the seam is rendered by xtask, not by cargo-generate itself, and its \
                             table is printed, not checked: the release checklist runs cargo \
                             generate once per seam";

/// What a green Learner-1 walk does not yet show.
const LEARNER1_NOTE: &str = "Learner-1 copies the README's Quick Start alone: the umbrella's \
                             examples are not copied until each builds with rlevo alone";

/// The probes. Keep in step with `xtask/probes/`:
/// `assert_probes_are_registered` enforces it.
const PROBES: &[Probe] = &[
    Probe {
        name: "byoe-1-thermostat",
        title: "BYOE-1/thermostat",
        kind: Kind::Walk {
            walk: &walk::BYOE1,
            contract: Some("src/thermostat.rs"),
        },
        note: Some(BYOE1_NOTE),
    },
    // The same walk over what the thermostat does not cover: a rank-2
    // observation, a bounded continuous action, and an
    // episode that an overflow terminates or `TimeLimit` truncates.
    Probe {
        name: "byoe-1-tank-farm",
        title: "BYOE-1/tank-farm",
        kind: Kind::Walk {
            walk: &walk::BYOE1,
            contract: Some("src/tank_farm.rs"),
        },
        note: Some(BYOE1_NOTE),
    },
    // A landscape author: a cost surface of their own, run through a shipped
    // strategy. Like an environment, the landscape needs the contract crate
    // alone.
    Probe {
        name: "byoe-2",
        title: "BYOE-2",
        kind: Kind::Walk {
            walk: &walk::BYOE2,
            contract: Some("src/well_siting.rs"),
        },
        note: Some(BYOE2_NOTE),
    },
    // An algorithm author: a learner of their own, trained and measured
    // through the harness on a shipped environment, against a random agent.
    Probe {
        name: "byoa-1a",
        title: "BYOA-1a",
        kind: Kind::Walk {
            walk: &walk::BYOA1A,
            contract: None,
        },
        note: Some(BYOA1A_NOTE),
    },
    // An optimiser author: a strategy of their own, and a control that never
    // moves, on a shipped landscape.
    Probe {
        name: "byoa-1b",
        title: "BYOA-1b",
        kind: Kind::Walk {
            walk: &walk::BYOA1B,
            contract: None,
        },
        note: Some(BYOA1B_NOTE),
    },
    // A learner: the README's Quick Start, pasted into a new crate. The
    // README's and the crate docs' `rlevo` lines must name this release.
    Probe {
        name: "learner-1",
        title: "Learner-1",
        kind: Kind::Listing {
            walk: &walk::LEARNER1,
            listing: &listing::QUICK_START,
        },
        note: Some(LEARNER1_NOTE),
    },
    // The `rlevo-algo` template: the crate `cargo generate` gives an author,
    // one row per seam. The sentinel is the start of the table's last row.
    Probe {
        name: "rlevo-algo-agent",
        title: "rlevo-algo/agent",
        kind: Kind::Template {
            seam: "agent",
            sentinel: "| DqnAgent |",
        },
        note: Some(TEMPLATE_NOTE),
    },
    Probe {
        name: "rlevo-algo-environment",
        title: "rlevo-algo/environment",
        kind: Kind::Template {
            seam: "environment",
            sentinel: "| DqnAgent |",
        },
        note: Some(TEMPLATE_NOTE),
    },
    Probe {
        name: "rlevo-algo-strategy",
        title: "rlevo-algo/strategy",
        kind: Kind::Template {
            seam: "strategy",
            sentinel: "| CmaEs |",
        },
        note: Some(TEMPLATE_NOTE),
    },
    // An environment author with no learner. An empty crate depending on
    // `rlevo-core` would compile whatever the contract asks of an
    // implementor, so this builds the thermostat's environment file.
    Probe {
        name: "consumer-core",
        title: "consumer-core",
        kind: Kind::Measurement {
            contract: Some(ContractFile {
                probe: "byoe-1-thermostat",
                file: "src/thermostat.rs",
            }),
        },
        note: None,
    },
    Probe {
        name: "consumer-ppo",
        title: "consumer-ppo",
        kind: Kind::Measurement { contract: None },
        note: None,
    },
    Probe {
        name: "consumer-umbrella",
        title: "consumer-umbrella",
        kind: Kind::Measurement { contract: None },
        note: None,
    },
];

#[macros::declare_command_args(None, None)]
pub struct ProbeCmdArgs {
    /// The probe to run.
    #[arg(value_name = "PROBE")]
    pub name: Option<String>,
    /// Print the probe names as a JSON array and exit, running nothing.
    ///
    /// The workflow builds its matrix from this with `fromJSON`, so the
    /// probe table is its single source of truth.
    #[arg(long, default_value_t = false)]
    pub list: bool,
}

/// How a run ended.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Every step of the walk held.
    Passes,
    /// A measurement probe resolved, built and reported its numbers.
    Measured,
    /// The path works; a step that is not fatal recorded a finding.
    Discoverability(usize),
    /// A fatal step failed: the step, and how many steps the walk has.
    FailsAt { step: usize, of: usize },
    /// The run did not reach a verdict of its own.
    Aborted(String),
}

impl Verdict {
    fn is_green(&self) -> bool {
        matches!(
            self,
            Self::Passes | Self::Measured | Self::Discoverability(_)
        )
    }

    fn sentence(&self, title: &str) -> String {
        match self {
            Self::Passes => format!("{title} passes."),
            Self::Measured => format!("{title} measured."),
            Self::Discoverability(step) => {
                format!("{title} passes; discoverability fails at step {step}.")
            }
            Self::FailsAt { step, of } => format!("{title} fails at step {step} of {of}."),
            Self::Aborted(why) => format!("{title} aborted: {why}"),
        }
    }
}

/// What a run printed and concluded.
struct Report {
    /// Step or measurement lines, in order.
    lines: Vec<String>,
    /// What the run rested on: where the crates came from, the toolchain.
    facts: Vec<(&'static str, String)>,
    verdict: Verdict,
}

/// Run one probe, or list them.
///
/// # Errors
///
/// Returns an error if the probe table disagrees with `xtask/probes/` or
/// with `templates/rlevo-algo/`, if no probe or an unknown one is named, or
/// if the run is not green. In the last case the error is the verdict.
pub fn handle_command(args: &ProbeCmdArgs, _env: Environment, _ctx: Context) -> Result<()> {
    let workspace = stage::workspace()?;
    let templates = workspace.root.join("xtask").join("probes");
    let seams = workspace.root.join(template::ROOT);
    assert_probes_are_registered(&templates, &seams)?;

    if args.list {
        tiers::print_json_list(PROBES.iter().map(|probe| probe.name.to_string()));
        return Ok(());
    }

    let known = || {
        PROBES
            .iter()
            .map(|probe| probe.name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let Some(name) = &args.name else {
        bail!("name a probe to run: {}", known());
    };
    let Some(probe) = PROBES.iter().find(|probe| probe.name == name) else {
        bail!("no probe named `{name}`\nAvailable probes: {}", known());
    };
    run_probe(probe, &workspace, &templates)
}

/// Fail if a template or a seam has no row in [`PROBES`], or a row nothing
/// to run, or if `cargo generate` would not offer every seam.
///
/// The test tiers cannot see a probe, which is not a cargo test target. A
/// template that no row runs is the analogue of an `#[ignore]`d test that no
/// tier claims: it stops compiling against the workspace and nothing says so.
fn assert_probes_are_registered(templates: &Path, seams: &Path) -> Result<()> {
    let mut on_disk = BTreeSet::new();
    for entry in fs::read_dir(templates)
        .with_context(|| format!("no probe templates at {}", templates.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            on_disk.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }
    let seam_names = template::seams(seams)?;
    template::check_offered(seams, &seam_names)?;
    on_disk.extend(seam_names.iter().map(|seam| template::probe_name(seam)));
    let registered: BTreeSet<String> = PROBES.iter().map(|probe| probe.name.to_string()).collect();
    let (unrun, missing) = registration_gaps(&on_disk, &registered);
    if unrun.is_empty() && missing.is_empty() {
        return Ok(());
    }
    bail!(
        "the probe table disagrees with xtask/probes/ and {}/\n\
         templates no probe runs: {}\n\
         probes with no template: {}\n\
         Edit PROBES in xtask/src/commands/probe/mod.rs.",
        template::ROOT,
        or_none(&unrun),
        or_none(&missing)
    )
}

/// Templates with no row, and rows with no template.
fn registration_gaps(
    on_disk: &BTreeSet<String>,
    registered: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    (
        on_disk.difference(registered).cloned().collect(),
        registered.difference(on_disk).cloned().collect(),
    )
}

fn or_none(names: &[String]) -> String {
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

fn run_probe(probe: &Probe, workspace: &Workspace, templates: &Path) -> Result<()> {
    group!("Probe: {}", probe.name);
    let run_dir = new_run_dir(probe.name, &workspace.root)?;
    let report = attempt(probe, workspace, templates, &run_dir).unwrap_or_else(|error| Report {
        lines: Vec::new(),
        facts: Vec::new(),
        verdict: Verdict::Aborted(format!("{error:#}")),
    });

    for line in &report.lines {
        eprintln!("{line}");
    }
    for (what, value) in &report.facts {
        eprintln!("{what}: {value}");
    }
    let green = report.verdict.is_green();
    if green {
        // Only the path this run created, never a pattern under the root.
        let _ = fs::remove_dir_all(&run_dir);
    } else {
        // The stage, the probe and its lock are what an inspection needs;
        // the build output is recreated from them.
        let _ = fs::remove_dir_all(run_dir.join("target"));
        eprintln!("run kept at {}", run_dir.display());
    }
    if let Some(note) = probe.note {
        eprintln!("note: {note}");
    }
    let verdict = report.verdict.sentence(probe.title);
    write_summary(probe, &report, &verdict);
    endgroup!();

    if green {
        eprintln!("{verdict}");
        Ok(())
    } else {
        bail!("{verdict}")
    }
}

/// A fresh directory for one run, outside the workspace.
fn new_run_dir(name: &str, workspace_root: &Path) -> Result<PathBuf> {
    let root =
        std::env::var_os("RLEVO_PROBE_SCRATCH").map_or_else(std::env::temp_dir, PathBuf::from);
    fs::create_dir_all(&root)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_dir = root.join(format!("rlevo-probe-{name}-{stamp}-{}", std::process::id()));
    fs::create_dir(&run_dir).with_context(|| format!("could not create {}", run_dir.display()))?;
    let run_dir = fs::canonicalize(run_dir)?;
    if run_dir.starts_with(fs::canonicalize(workspace_root)?) {
        fs::remove_dir_all(&run_dir)?;
        bail!(
            "the scratch root is inside the workspace, where a probe would resolve its \
             siblings by path; point RLEVO_PROBE_SCRATCH elsewhere"
        );
    }
    Ok(run_dir)
}

fn attempt(
    probe: &Probe,
    workspace: &Workspace,
    templates: &Path,
    run_dir: &Path,
) -> Result<Report> {
    // A listing or a template that cannot be copied, or that names another
    // release, is known before anything is packaged.
    if matches!(probe.kind, Kind::Listing { .. } | Kind::Template { .. }) {
        listing::check_versions(&workspace.root, &workspace.version)?;
    }
    let copied = match &probe.kind {
        Kind::Listing { listing, .. } => Some(listing::read(&workspace.root, listing)?),
        _ => None,
    };
    let stage = stage::stage(workspace, run_dir)?;
    let probe_dir = run_dir.join("probe");
    let target_dir = run_dir.join("target");

    if let Kind::Template { seam, .. } = &probe.kind {
        template::render(
            &workspace.root.join(template::ROOT).join(seam),
            &probe_dir,
            probe.name,
            &stage,
            &workspace.version,
        )?;
    } else {
        let sources = match &probe.kind {
            Kind::Measurement {
                contract: Some(contract),
            } => vec![("src/lib.rs", contract_source(templates, contract)?)],
            _ => Vec::new(),
        };
        stage::render(
            &templates.join(probe.name),
            &probe_dir,
            &stage,
            &workspace.version,
            &sources,
        )?;
    }
    if let Some(copied) = &copied {
        listing::write(&probe_dir, copied)?;
    }
    let resolved = stage::resolve(&probe_dir, &target_dir, &stage)?;

    let mut facts = vec![(
        "rlevo crates, all from the stage",
        resolved.rlevo.join(", "),
    )];
    // Worth a line only for a probe that asks for the umbrella: these are
    // the published crates `rlevo` does not bring with it.
    if resolved.direct.contains("rlevo") && !resolved.unreached.is_empty() {
        facts.push((
            "published crates the probe's graph does not reach",
            resolved.unreached.join(", "),
        ));
    }
    facts.push(("toolchain", stage::toolchain(&probe_dir)));

    let (walk, contract, sentinel) = match &probe.kind {
        Kind::Measurement { .. } => {
            return Ok(Report {
                lines: measure::measure(&probe_dir, run_dir, &resolved)?
                    .into_iter()
                    .map(|(what, value)| format!("MEASURE {what} :: {value}"))
                    .collect(),
                facts,
                verdict: Verdict::Measured,
            });
        }
        Kind::Walk { walk, contract } => (*walk, *contract, None),
        Kind::Listing { walk, listing } => (*walk, None, Some(listing.sentinel)),
        Kind::Template { sentinel, .. } => (&walk::TEMPLATE, None, Some(*sentinel)),
    };
    let mut outcomes = walk::run(
        walk,
        &probe_dir,
        &target_dir,
        probe.name,
        &resolved,
        contract,
        sentinel,
    )?;
    // An author's next two commands, once the crate builds.
    let built = walk.levels[0].step;
    if matches!(probe.kind, Kind::Template { .. })
        && matches!(outcomes.get(built), Some(Outcome::Pass(_)))
    {
        outcomes.insert(walk::LINT, walk::lint(&probe_dir, &target_dir)?);
        outcomes.insert(walk::TEST, walk::test(&probe_dir, &target_dir)?);
    }
    let (lines, verdict) = walk_report(walk, &outcomes);
    Ok(Report {
        lines,
        facts,
        verdict,
    })
}

/// Another probe's contract file, respelled for a crate that depends on
/// `rlevo-core` and not on the umbrella.
fn contract_source(templates: &Path, contract: &ContractFile) -> Result<String> {
    let path = templates.join(contract.probe).join(contract.file);
    let source = fs::read_to_string(&path)
        .with_context(|| format!("no contract file at {}", path.display()))?;
    let outside = walk::outside_contract(&source);
    if !outside.is_empty() {
        bail!(
            "the environment needs more than the contract crate: {} names {}",
            contract.file,
            outside.join(", ")
        );
    }
    Ok(source.replace("rlevo::core", "rlevo_core"))
}

/// The step lines and the verdict a walk's outcomes add up to.
///
/// The lines stop at the first fatal failure: later steps were not reached,
/// or their result says nothing while an earlier one is broken.
fn walk_report(walk: &walk::Walk, outcomes: &walk::Outcomes) -> (Vec<String>, Verdict) {
    let prefix = walk.prefix;
    let fails_at = |step| Verdict::FailsAt {
        step,
        of: walk.steps.len(),
    };
    let mut lines = Vec::new();
    let mut finding = None;
    for (index, step) in walk.steps.iter().enumerate() {
        let number = index + 1;
        let Some(outcome) = outcomes.get(step.label) else {
            lines.push(format!(
                "{prefix} {number} FAIL {} :: the walk did not reach this step",
                step.label
            ));
            return (lines, fails_at(number));
        };
        match outcome {
            Outcome::Pass(detail) => {
                lines.push(format!("{prefix} {number} PASS {} :: {detail}", step.label));
            }
            Outcome::Fail(detail) => {
                lines.push(format!("{prefix} {number} FAIL {} :: {detail}", step.label));
                if step.fatal {
                    return (lines, fails_at(number));
                }
                finding.get_or_insert(number);
            }
        }
    }
    (
        lines,
        finding.map_or(Verdict::Passes, Verdict::Discoverability),
    )
}

/// Append the run to the job summary, when there is one.
///
/// The step lines and what the run rested on are the product as much as the
/// exit status is: a red check should say which step without the log open.
fn write_summary(probe: &Probe, report: &Report, verdict: &str) {
    let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") else {
        return;
    };
    let mut text = format!("### `{}`\n\n**{verdict}**\n\n", probe.name);
    if !report.lines.is_empty() {
        text.push_str("```text\n");
        for line in &report.lines {
            text.push_str(line);
            text.push('\n');
        }
        text.push_str("```\n\n");
    }
    for (what, value) in &report.facts {
        let _ = writeln!(text, "- {what}: {value}");
    }
    if let Some(note) = probe.note {
        let _ = writeln!(text, "- note: {note}");
    }
    let written = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(text.as_bytes()));
    if let Err(error) = written {
        log::warn!("could not write the job summary: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn test_a_template_with_no_row_and_a_row_with_no_template_are_both_gaps() {
        let (unrun, missing) = registration_gaps(
            &names(&["byoe-1-thermostat", "orphan"]),
            &names(&["byoe-1-thermostat", "consumer-core"]),
        );
        assert_eq!(unrun, ["orphan"]);
        assert_eq!(missing, ["consumer-core"]);
    }

    #[test]
    fn test_a_template_row_is_named_for_its_seam() {
        for probe in PROBES {
            if let Kind::Template { seam, .. } = &probe.kind {
                assert_eq!(probe.name, template::probe_name(seam));
            }
        }
    }

    #[test]
    fn test_a_contract_file_belongs_to_a_registered_probe() {
        for probe in PROBES {
            if let Kind::Measurement {
                contract: Some(contract),
            } = &probe.kind
            {
                assert!(PROBES.iter().any(|other| other.name == contract.probe));
            }
        }
    }

    fn outcomes(pairs: &[(&'static str, Outcome)]) -> walk::Outcomes {
        pairs.iter().cloned().collect()
    }

    fn pass() -> Outcome {
        Outcome::Pass("ok".to_string())
    }

    fn fail() -> Outcome {
        Outcome::Fail("no".to_string())
    }

    fn all_pass() -> Vec<(&'static str, Outcome)> {
        walk::BYOE1
            .steps
            .iter()
            .map(|step| (step.label, pass()))
            .collect()
    }

    #[test]
    fn test_every_step_passing_is_a_pass() {
        let (lines, verdict) = walk_report(&walk::BYOE1, &outcomes(&all_pass()));
        assert_eq!(verdict, Verdict::Passes);
        assert_eq!(lines.len(), walk::BYOE1.steps.len());
        assert_eq!(lines[0], "BYOE-STEP 1 PASS scaffold :: ok");
    }

    #[test]
    fn test_a_finding_does_not_stop_the_walk() {
        let mut steps = all_pass();
        steps[1].1 = fail();
        let (lines, verdict) = walk_report(&walk::BYOE1, &outcomes(&steps));
        assert_eq!(verdict, Verdict::Discoverability(2));
        assert_eq!(lines.len(), walk::BYOE1.steps.len());
        assert!(verdict.is_green());
    }

    #[test]
    fn test_a_fatal_failure_stops_the_walk_and_outranks_a_finding() {
        let mut steps = all_pass();
        steps[1].1 = fail();
        steps[3].1 = fail();
        let (lines, verdict) = walk_report(&walk::BYOE1, &outcomes(&steps));
        assert_eq!(verdict, Verdict::FailsAt { step: 4, of: 6 });
        assert_eq!(lines.len(), 4);
        assert!(!verdict.is_green());
    }

    #[test]
    fn test_a_step_with_no_outcome_fails_where_the_walk_stopped() {
        let mut steps = all_pass();
        steps.truncate(4);
        let (lines, verdict) = walk_report(&walk::BYOE1, &outcomes(&steps));
        assert_eq!(verdict, Verdict::FailsAt { step: 5, of: 6 });
        assert_eq!(
            lines.last().map(String::as_str),
            Some("BYOE-STEP 5 FAIL implement-agent :: the walk did not reach this step")
        );
    }

    #[test]
    fn test_only_a_fatal_failure_or_an_abort_is_red() {
        assert!(Verdict::Passes.is_green());
        assert!(Verdict::Measured.is_green());
        assert!(!Verdict::Aborted("staging failed".to_string()).is_green());
        assert_eq!(
            Verdict::FailsAt { step: 3, of: 6 }.sentence("BYOE-1/thermostat"),
            "BYOE-1/thermostat fails at step 3 of 6."
        );
    }

    /// Every walk a probe takes opens with `xtask`'s two steps, builds its
    /// base level with no feature, and reports only at its own steps.
    #[test]
    fn test_every_walk_is_well_formed() {
        for probe in PROBES {
            let walk = match &probe.kind {
                Kind::Walk { walk, .. } | Kind::Listing { walk, .. } => *walk,
                Kind::Template { .. } => &walk::TEMPLATE,
                Kind::Measurement { .. } => continue,
            };
            let labels: Vec<&str> = walk.steps.iter().map(|step| step.label).collect();
            assert_eq!(labels[..2], ["scaffold", "add-deps"], "{}", probe.name);
            assert_eq!(walk.levels[0].feature, None, "{}", probe.name);
            for level in walk.levels {
                assert!(labels.contains(&level.step), "{}", probe.name);
                for label in level.reports {
                    assert!(labels.contains(label), "{}", probe.name);
                }
            }
        }
    }
}
