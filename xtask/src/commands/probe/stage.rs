//! Staging: the publish boundary without a publish.
//!
//! A probe must see what a `cargo add rlevo` user sees: the files `cargo
//! package` ships, the public API, and no `publish = false` crate. Three
//! moves get there with no registry and no manifest rewrite:
//!
//! 1. One `cargo package --workspace --no-verify` call packages every
//!    publishable crate. Packaging them one at a time resolves each against
//!    crates.io and fails on any feature or item the published sibling lacks.
//! 2. Each `.crate` is extracted into the stage.
//! 3. The probe gets a `.cargo/config.toml` holding one `[patch.crates-io]`
//!    entry per extracted crate. A patch applies to the whole graph, so a
//!    sibling that a staged manifest names by bare version resolves to the
//!    stage as well.
//!
//! A patch that does not apply is only a warning: cargo then takes that crate
//! from crates.io and the probe would test the last release. So nothing here
//! trusts the warning. [`resolve`] reads the probe's resolved graph and fails
//! unless every `rlevo` crate in it comes from the stage.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use tracel_xtask::prelude::*;

/// What a probe needs to know about the checkout it is run from.
#[derive(Debug)]
pub struct Workspace {
    /// The workspace root.
    pub root: PathBuf,
    /// The version every published crate shares.
    pub version: String,
    /// Members `cargo package` must skip: those marked `publish = false`.
    pub unpublished: Vec<String>,
    /// Members that are packaged and staged.
    pub published: Vec<String>,
}

/// The extracted crates a probe is patched onto.
#[derive(Debug)]
pub struct Stage {
    /// Directory holding one `<name>-<version>` directory per staged crate.
    pub crates_dir: PathBuf,
    /// Names of the staged crates.
    pub names: Vec<String>,
}

/// A probe's resolved dependency graph, as far as a probe reads it.
#[derive(Debug)]
pub struct Resolved {
    /// Every `rlevo` crate in the graph, as `name version`.
    pub rlevo: Vec<String>,
    /// Staged crates the probe's graph does not reach.
    pub unreached: Vec<String>,
    /// The probe's own `[dependencies]`, by name.
    pub direct: BTreeSet<String>,
}

/// A finished child process.
#[derive(Debug)]
pub struct Finished {
    /// Whether it exited with status zero.
    pub ok: bool,
    /// Its standard output.
    pub stdout: String,
    /// Its standard error.
    pub stderr: String,
}

/// Run `command` to completion, capturing both streams.
///
/// # Errors
///
/// Returns an error if the process cannot be started.
pub fn run(command: &mut Command) -> Result<Finished> {
    let output = command
        .output()
        .with_context(|| format!("could not start {}", command.get_program().display()))?;
    Ok(Finished {
        ok: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// A `cargo` invocation inside a probe crate.
///
/// The build goes to `target_dir`, never to a directory the caller's
/// environment names. The caller's `RUSTFLAGS` are cleared: staged crates are
/// path dependencies, which cargo does not cap lints for, so a caller's
/// `-D warnings` would fail a probe on a warning no consumer sees.
pub fn probe_cargo(probe_dir: &Path, target_dir: &Path) -> Command {
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(probe_dir)
        .env("CARGO_TARGET_DIR", target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_BUILD_RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS");
    cargo
}

/// Read the workspace from `cargo metadata`.
///
/// # Errors
///
/// Returns an error if `cargo metadata` fails or the workspace has no `rlevo`
/// member to take the version from.
pub fn workspace() -> Result<Workspace> {
    let metadata =
        run(Command::new("cargo").args(["metadata", "--no-deps", "--format-version", "1"]))?;
    if !metadata.ok {
        bail!("cargo metadata failed:\n{}", metadata.stderr);
    }
    let json: Value = serde_json::from_str(&metadata.stdout)?;
    let root = PathBuf::from(
        json["workspace_root"]
            .as_str()
            .context("cargo metadata reported no workspace root")?,
    );

    let mut version = None;
    let mut unpublished = Vec::new();
    let mut published = Vec::new();
    for package in json["packages"].as_array().into_iter().flatten() {
        let name = package["name"].as_str().unwrap_or_default().to_string();
        // `publish = false` is reported as an empty registry list.
        let withheld = package["publish"]
            .as_array()
            .is_some_and(std::vec::Vec::is_empty);
        if name == "rlevo" {
            version = package["version"].as_str().map(str::to_string);
        }
        if withheld {
            unpublished.push(name);
        } else {
            published.push(name);
        }
    }
    Ok(Workspace {
        root,
        version: version.context("the workspace has no `rlevo` member")?,
        unpublished,
        published,
    })
}

/// Package the workspace's publishable crates and extract them under
/// `run_dir/stage/crates`.
///
/// # Errors
///
/// Returns an error if packaging or extraction fails. Either is a finding:
/// the workspace cannot be released as it stands.
pub fn stage(workspace: &Workspace, run_dir: &Path) -> Result<Stage> {
    let package_target = run_dir.join("stage").join("package-target");
    let crates_dir = run_dir.join("stage").join("crates");
    fs::create_dir_all(&crates_dir)?;

    let mut package = Command::new("cargo");
    package
        .current_dir(&workspace.root)
        .args(["package", "--workspace", "--no-verify", "--allow-dirty"])
        .arg("--target-dir")
        .arg(&package_target);
    for name in &workspace.unpublished {
        package.args(["--exclude", name]);
    }
    let packaged = run(&mut package)?;
    if !packaged.ok {
        bail!("cargo package failed:\n{}", packaged.stderr);
    }

    for name in &workspace.published {
        let stem = format!("{name}-{}", workspace.version);
        let tarball = package_target.join("package").join(format!("{stem}.crate"));
        let extracted = run(Command::new("tar")
            .arg("-xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(&crates_dir))?;
        if !extracted.ok || !crates_dir.join(&stem).join("Cargo.toml").is_file() {
            bail!(
                "could not extract {}:\n{}",
                tarball.display(),
                extracted.stderr
            );
        }
    }
    // Only the extracted crates are read from here on.
    fs::remove_dir_all(&package_target)?;

    Ok(Stage {
        crates_dir,
        names: workspace.published.clone(),
    })
}

/// The `[patch.crates-io]` table that points every staged crate at the stage.
pub fn patch_table(stage: &Stage, version: &str) -> String {
    let mut table = String::from("[patch.crates-io]\n");
    for name in &stage.names {
        let dir = stage.crates_dir.join(format!("{name}-{version}"));
        // A TOML literal string takes the path as written, backslashes and all.
        let _ = writeln!(table, "{name} = {{ path = '{}' }}", dir.display());
    }
    table
}

/// Write the probe crate: its manifest from the template, its sources, and
/// the patch table.
///
/// `sources` are `(path under the probe, content)` pairs written after the
/// template's own `src/` tree is copied, for a probe whose source is derived
/// from another's.
///
/// # Errors
///
/// Returns an error if the template is unreadable or the probe cannot be
/// written.
pub fn render(
    template_dir: &Path,
    probe_dir: &Path,
    stage: &Stage,
    version: &str,
    sources: &[(&str, String)],
) -> Result<()> {
    fs::create_dir_all(probe_dir.join(".cargo"))?;
    let template = template_dir.join("Cargo.toml.template");
    let manifest = fs::read_to_string(&template)
        .with_context(|| format!("no manifest template at {}", template.display()))?;
    fs::write(
        probe_dir.join("Cargo.toml"),
        manifest.replace("{{rlevo_version}}", version),
    )?;
    fs::write(
        probe_dir.join(".cargo").join("config.toml"),
        patch_table(stage, version),
    )?;
    if template_dir.join("src").is_dir() {
        copy_tree(&template_dir.join("src"), &probe_dir.join("src"))?;
    }
    for (path, content) in sources {
        let target = probe_dir.join(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(target, content)?;
    }
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Resolve the probe's graph and check where its `rlevo` crates come from.
///
/// # Errors
///
/// Returns an error if the graph does not resolve, or if any `rlevo` crate in
/// it comes from anywhere but the stage. The second means a patch did not
/// apply and the probe would have tested crates.io's copy.
pub fn resolve(probe_dir: &Path, target_dir: &Path, stage: &Stage) -> Result<Resolved> {
    let metadata =
        run(probe_cargo(probe_dir, target_dir).args(["metadata", "--format-version", "1"]))?;
    if !metadata.ok {
        bail!(
            "the probe's dependencies do not resolve:\n{}",
            metadata.stderr
        );
    }
    let json: Value = serde_json::from_str(&metadata.stdout)?;
    let staged_root = fs::canonicalize(&stage.crates_dir)?;
    let root_id = json["resolve"]["root"].as_str().unwrap_or_default();

    let mut rlevo = Vec::new();
    let mut strays = Vec::new();
    let mut direct = BTreeSet::new();
    for package in json["packages"].as_array().into_iter().flatten() {
        let name = package["name"].as_str().unwrap_or_default();
        if package["id"].as_str() == Some(root_id) {
            direct = package["dependencies"]
                .as_array()
                .into_iter()
                .flatten()
                // `kind` is null for a normal dependency.
                .filter(|dependency| dependency["kind"].is_null())
                .filter_map(|dependency| dependency["name"].as_str())
                .map(str::to_string)
                .collect();
        }
        if !is_rlevo_crate(name) {
            continue;
        }
        let manifest = PathBuf::from(package["manifest_path"].as_str().unwrap_or_default());
        let from_stage =
            fs::canonicalize(&manifest).is_ok_and(|path| path.starts_with(&staged_root));
        let version = package["version"].as_str().unwrap_or_default();
        if from_stage {
            rlevo.push(format!("{name} {version}"));
        } else {
            strays.push(format!("{name} {version} ({})", manifest.display()));
        }
    }
    if !strays.is_empty() {
        bail!(
            "these crates did not come from the stage, so the probe would test \
             something other than this checkout: {}",
            strays.join(", ")
        );
    }
    rlevo.sort();

    let mut unreached: Vec<String> = stage
        .names
        .iter()
        .filter(|name| {
            !rlevo
                .iter()
                .any(|entry| entry.split(' ').next() == Some(name))
        })
        .cloned()
        .collect();
    unreached.sort();
    Ok(Resolved {
        rlevo,
        unreached,
        direct,
    })
}

/// Whether `name` is the umbrella or one of its member crates.
fn is_rlevo_crate(name: &str) -> bool {
    name == "rlevo" || name.starts_with("rlevo-")
}

/// Unique package names in `cargo tree -e normal`, less the root.
///
/// `package` restricts the tree to the subtree under one crate, counted in
/// the probe's own graph so the features are the ones the consumer unified.
/// Normal edges only: dev and build edges are not a consumer's cost.
///
/// # Errors
///
/// Returns an error if `cargo tree` fails.
pub fn cone(probe_dir: &Path, target_dir: &Path, package: Option<&str>) -> Result<Vec<String>> {
    let mut tree = probe_cargo(probe_dir, target_dir);
    tree.args(["tree", "-e", "normal", "--prefix", "none"]);
    if let Some(package) = package {
        tree.args(["-p", package]);
    }
    let listed = run(&mut tree)?;
    if !listed.ok {
        bail!("cargo tree failed:\n{}", listed.stderr);
    }
    Ok(cone_from_tree(&listed.stdout))
}

/// The cone a `cargo tree --prefix none` listing describes.
///
/// The first line is the root, which is not part of its own cone.
fn cone_from_tree(listing: &str) -> Vec<String> {
    let mut lines = listing.lines().filter(|line| !line.trim().is_empty());
    let root = lines.next().and_then(|line| line.split(' ').next());
    let names: BTreeSet<&str> = lines
        .filter_map(|line| line.split(' ').next())
        .filter(|name| Some(*name) != root)
        .collect();
    names.into_iter().map(str::to_string).collect()
}

/// The toolchain a probe's builds run on.
///
/// A directory outside the checkout does not see `rust-toolchain.toml`. The
/// probes still build on the pinned toolchain, because `cargo xtask` runs
/// under it and rustup hands it to every child through the environment.
pub fn toolchain(probe_dir: &Path) -> String {
    run(Command::new("rustc")
        .arg("--version")
        .current_dir(probe_dir))
    .ok()
    .filter(|rustc| rustc.ok)
    .map_or_else(
        || "unknown".to_string(),
        |rustc| rustc.stdout.trim().to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_counts_unique_names_and_not_the_root() {
        let listing = "probe v0.0.0 (/scratch/probe)\n\
                       rlevo v0.5.0 (/scratch/stage/rlevo-0.5.0)\n\
                       burn v0.21.0\n\
                       rand v0.10.3\n\
                       rand v0.9.2\n\
                       burn v0.21.0 (*)\n";
        assert_eq!(cone_from_tree(listing), ["burn", "rand", "rlevo"]);
    }

    #[test]
    fn test_cone_of_a_crate_with_no_dependencies_is_empty() {
        assert_eq!(
            cone_from_tree("rlevo-metrics-registry v0.5.0 (/stage)\n"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn test_patch_table_names_every_staged_crate() {
        let stage = Stage {
            crates_dir: PathBuf::from("/scratch/stage/crates"),
            names: vec!["rlevo".to_string(), "rlevo-core".to_string()],
        };
        assert_eq!(
            patch_table(&stage, "0.5.0"),
            "[patch.crates-io]\n\
             rlevo = { path = '/scratch/stage/crates/rlevo-0.5.0' }\n\
             rlevo-core = { path = '/scratch/stage/crates/rlevo-core-0.5.0' }\n"
        );
    }

    #[test]
    fn test_only_the_umbrella_and_its_members_are_rlevo_crates() {
        assert!(is_rlevo_crate("rlevo"));
        assert!(is_rlevo_crate("rlevo-core"));
        assert!(!is_rlevo_crate("rlevolution"));
        assert!(!is_rlevo_crate("burn"));
    }
}
