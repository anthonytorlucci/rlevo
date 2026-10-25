//! A template: the crate `cargo generate` gives an author, rendered here as
//! `cargo generate` would render it.
//!
//! `templates/rlevo-algo/` holds one `cargo-generate` template per seam an
//! author implements: `agent`, `environment` and `strategy`. Rendering a seam
//! here, rather than running `cargo-generate`, keeps a third-party binary
//! and the network out of every pull request. The rendering is exact because
//! a seam asks for as little of `cargo-generate` as it can:
//!
//! - its manifest is `Cargo.toml.liquid`, so cargo never adopts the
//!   directory, and `cargo-generate` strips the suffix;
//! - its `cargo-generate.toml` lists [`RENDERED`] as the only files Liquid
//!   reads, so every other file, the Rust sources included, is copied as
//!   written;
//! - those files use the two built-in placeholders, `{{project-name}}` and
//!   `{{crate_name}}`, and no other Liquid.
//!
//! A seam that asks for more is refused, not approximated. The release
//! checklist runs `cargo generate` once per seam and compares its output
//! with this rendering.
//!
//! The rendered crate is walked like a probe ([`super::walk::TEMPLATE`]):
//! built, linted, tested and run, as an author's first commands would.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use tracel_xtask::prelude::*;

use super::stage::{self, Stage};

/// The template directory, from the workspace root.
pub const ROOT: &str = "templates/rlevo-algo";

/// The `cargo-generate` configuration file, in the root and in each seam.
const CONFIG: &str = "cargo-generate.toml";

/// The files `cargo-generate` renders through Liquid; the rest it copies.
pub const RENDERED: [&str; 2] = ["Cargo.toml", "README.md"];

/// The suffix `cargo-generate` strips from a file's name.
const LIQUID: &str = ".liquid";

/// The probe that walks `seam`.
pub fn probe_name(seam: &str) -> String {
    format!("rlevo-algo-{seam}")
}

/// The seams under `root`: its subdirectories.
///
/// # Errors
///
/// Returns an error if `root` cannot be read.
pub fn seams(root: &Path) -> Result<BTreeSet<String>> {
    let mut seams = BTreeSet::new();
    for entry in
        fs::read_dir(root).with_context(|| format!("no templates at {}", root.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            seams.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }
    Ok(seams)
}

/// Each seam's manifest, from the workspace root: the `rlevo` line an
/// author's crate is generated with.
///
/// # Errors
///
/// Returns an error if the template directory cannot be read.
pub fn manifests(workspace_root: &Path) -> Result<Vec<String>> {
    Ok(seams(&workspace_root.join(ROOT))?
        .into_iter()
        .map(|seam| format!("{ROOT}/{seam}/Cargo.toml{LIQUID}"))
        .collect())
}

/// Fail unless the root's `cargo-generate.toml` offers exactly `seams`.
///
/// `cargo generate` pointed at the root offers the seams that list names. A
/// seam missing from it is one an author is never offered.
///
/// # Errors
///
/// Returns an error if the file is unreadable or offers other seams.
pub fn check_offered(root: &Path, seams: &BTreeSet<String>) -> Result<()> {
    let path = root.join(CONFIG);
    let text =
        fs::read_to_string(&path).with_context(|| format!("no {CONFIG} at {}", root.display()))?;
    let offered = sub_templates(&text)
        .with_context(|| format!("{} lists no sub_templates", path.display()))?;
    if &offered != seams {
        bail!(
            "{} offers {} and the seams on disk are {}; list each seam once",
            path.display(),
            join(&offered),
            join(seams)
        );
    }
    Ok(())
}

/// The names in a `sub_templates = ["a", "b"]` line, written on one line.
fn sub_templates(text: &str) -> Option<BTreeSet<String>> {
    let value = text
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("sub_templates"))?
        .trim_start()
        .strip_prefix('=')?
        .trim()
        .strip_prefix('[')?
        .strip_suffix(']')?;
    Some(
        value
            .split(',')
            .map(|name| name.trim().trim_matches('"').to_string())
            .filter(|name| !name.is_empty())
            .collect(),
    )
}

fn join(names: &BTreeSet<String>) -> String {
    names.iter().cloned().collect::<Vec<_>>().join(", ")
}

/// The `cargo-generate.toml` a seam must have, comments and blank lines
/// aside: Liquid reads [`RENDERED`] and nothing else.
fn expected_config() -> Vec<String> {
    let files = RENDERED.map(|file| format!("\"{file}\"")).join(", ");
    vec!["[template]".to_string(), format!("include = [{files}]")]
}

/// The lines of a TOML file that say something.
fn meaningful_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| line.split('#').next().unwrap_or_default().trim())
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// Render the seam at `seam_dir` into `probe_dir` as the crate `name`,
/// patched onto `stage`, whose crates are at `version`.
///
/// # Errors
///
/// Returns an error if the seam asks `cargo-generate` for more than this
/// rendering does, or if it cannot be read or written.
pub fn render(
    seam_dir: &Path,
    probe_dir: &Path,
    name: &str,
    stage: &Stage,
    version: &str,
) -> Result<()> {
    let config = fs::read_to_string(seam_dir.join(CONFIG))
        .with_context(|| format!("no {CONFIG} in {}", seam_dir.display()))?;
    if meaningful_lines(&config) != expected_config() {
        bail!(
            "{} asks cargo-generate for more than xtask renders; it must say only:\n{}",
            seam_dir.join(CONFIG).display(),
            expected_config().join("\n")
        );
    }
    copy_stripped(seam_dir, probe_dir, true)?;

    for file in RENDERED {
        let path = probe_dir.join(file);
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let rendered = substitute(&text, name);
        if rendered.contains("{{") || rendered.contains("{%") {
            bail!(
                "{} uses Liquid beyond {{{{project-name}}}} and {{{{crate_name}}}}, which xtask \
                 does not render",
                seam_dir.join(file).display()
            );
        }
        fs::write(&path, rendered)?;
    }

    let manifest = probe_dir.join("Cargo.toml");
    if !manifest.is_file() {
        bail!("{} has no Cargo.toml{LIQUID}", seam_dir.display());
    }
    // Keeps cargo from adopting the crate into a workspace above the scratch
    // root. An author's crate has no use for it.
    let mut file = fs::OpenOptions::new().append(true).open(&manifest)?;
    write!(file, "\n[workspace]\n")?;

    fs::create_dir_all(probe_dir.join(".cargo"))?;
    fs::write(
        probe_dir.join(".cargo").join("config.toml"),
        stage::patch_table(stage, version),
    )?;
    Ok(())
}

/// `{{project-name}}` and `{{crate_name}}`, as `cargo-generate` fills them
/// for the crate `name`.
fn substitute(text: &str, name: &str) -> String {
    text.replace("{{project-name}}", name)
        .replace("{{crate_name}}", &name.replace('-', "_"))
}

/// Copy `from` into `to`, stripping a `.liquid` suffix from each file name,
/// as `cargo-generate` does. `top` leaves out the seam's own configuration.
fn copy_stripped(from: &Path, to: &Path, top: bool) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let file_name = entry.file_name().to_string_lossy().into_owned();
        if top && file_name == CONFIG {
            continue;
        }
        if entry.file_type()?.is_dir() {
            copy_stripped(&entry.path(), &to.join(&file_name), false)?;
            continue;
        }
        let target = match file_name.strip_suffix(LIQUID) {
            Some(stripped) => {
                if from.join(stripped).exists() {
                    bail!(
                        "{} ships both {stripped} and {file_name}; keep one",
                        from.display()
                    );
                }
                to.join(stripped)
            }
            None => to.join(&file_name),
        };
        fs::copy(entry.path(), target)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// A fresh directory under the system's temporary directory.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rlevo-xtask-template-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    /// A seam with a manifest, a README, a source file and `config`.
    fn seam(root: &Path, config: &str, readme: &str) -> PathBuf {
        let seam = root.join("seam");
        fs::create_dir_all(seam.join("src")).expect("the seam's src");
        fs::write(seam.join(CONFIG), config).expect("config");
        fs::write(
            seam.join("Cargo.toml.liquid"),
            "[package]\nname = \"{{project-name}}\"\n",
        )
        .expect("manifest");
        fs::write(seam.join("README.md"), readme).expect("readme");
        fs::write(
            seam.join("src").join("main.rs"),
            "fn main() { println!(\"{{}}\", 1); }\n",
        )
        .expect("main");
        seam
    }

    /// A stage holding one crate, `rlevo` 0.5.0.
    fn stage(root: &Path) -> Stage {
        Stage {
            crates_dir: root.join("crates"),
            names: vec!["rlevo".to_string()],
        }
    }

    const CONFIG_TEXT: &str =
        "# Only these.\n[template]\ninclude = [\"Cargo.toml\", \"README.md\"]\n";

    #[test]
    fn test_a_seam_renders_as_cargo_generate_would() {
        let root = scratch("renders");
        let seam = seam(&root, CONFIG_TEXT, "# {{project-name}} ({{crate_name}})\n");
        let probe = root.join("probe");
        render(&seam, &probe, "rlevo-algo-seam", &stage(&root), "0.5.0").expect("renders");

        let manifest = fs::read_to_string(probe.join("Cargo.toml")).expect("manifest");
        assert!(manifest.starts_with("[package]\nname = \"rlevo-algo-seam\"\n"));
        assert!(manifest.ends_with("\n[workspace]\n"));
        assert_eq!(
            fs::read_to_string(probe.join("README.md")).expect("readme"),
            "# rlevo-algo-seam (rlevo_algo_seam)\n"
        );
        // Not a rendered file, so its braces are left alone.
        assert_eq!(
            fs::read_to_string(probe.join("src").join("main.rs")).expect("main"),
            "fn main() { println!(\"{{}}\", 1); }\n"
        );
        assert!(!probe.join(CONFIG).exists());
        assert!(!probe.join("Cargo.toml.liquid").exists());
        let patch = fs::read_to_string(probe.join(".cargo").join("config.toml")).expect("patch");
        assert!(patch.contains("rlevo-0.5.0"), "{patch}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_liquid_beyond_the_two_placeholders_is_refused() {
        let root = scratch("liquid");
        let seam = seam(&root, CONFIG_TEXT, "# {{ authors }}\n");
        let error = render(&seam, &root.join("probe"), "p", &stage(&root), "0.5.0")
            .expect_err("an unrendered tag");
        assert!(error.to_string().contains("uses Liquid beyond"), "{error}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_a_configuration_that_asks_for_more_is_refused() {
        let root = scratch("config");
        let config = "[template]\ninclude = [\"Cargo.toml\", \"README.md\"]\n\
                      [placeholders]\nalgo = { type = \"string\", prompt = \"?\" }\n";
        let seam = seam(&root, config, "# readme\n");
        let error = render(&seam, &root.join("probe"), "p", &stage(&root), "0.5.0")
            .expect_err("more config");
        assert!(
            error.to_string().contains("asks cargo-generate for more"),
            "{error}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_the_offered_list_is_read_from_one_line() {
        let text = "# Offered.\n[template]\nsub_templates = [\"agent\", \"environment\"]\n";
        let expected: BTreeSet<String> = ["agent", "environment"].map(String::from).into();
        assert_eq!(sub_templates(text), Some(expected));
        assert_eq!(sub_templates("[template]\n"), None);
    }

    #[test]
    fn test_a_seam_the_root_does_not_offer_is_refused() {
        let root = scratch("offered");
        fs::write(
            root.join(CONFIG),
            "[template]\nsub_templates = [\"agent\"]\n",
        )
        .expect("config");
        let on_disk: BTreeSet<String> = ["agent", "strategy"].map(String::from).into();
        let error = check_offered(&root, &on_disk).expect_err("strategy is not offered");
        assert!(error.to_string().contains("offers agent"), "{error}");
        let offered: BTreeSet<String> = ["agent"].map(String::from).into();
        check_offered(&root, &offered).expect("the same seams");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_the_rendered_files_are_the_ones_the_seams_configure() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the workspace root")
            .join(ROOT);
        for seam in seams(&root).expect("the seams") {
            let config = fs::read_to_string(root.join(&seam).join(CONFIG)).expect("config");
            assert_eq!(meaningful_lines(&config), expected_config(), "{seam}");
        }
    }
}
