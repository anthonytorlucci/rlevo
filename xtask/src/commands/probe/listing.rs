//! A listing: code in the documentation that a learner copies and runs.
//!
//! Nothing compiles a README. A listing that stops compiling, or a
//! dependency line that names a release other than the one the
//! documentation describes, is found by the first person who copies it. A
//! listing probe copies it as written: the listing's `toml` block becomes
//! the probe's `[dependencies]`, its `rust` block the probe's `src/main.rs`.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use tracel_xtask::prelude::*;

/// A listing in the documentation.
#[derive(Debug)]
pub struct Listing {
    /// The file it is in, from the workspace root.
    pub document: &'static str,
    /// The heading of its section, as written. The section's first `toml`
    /// block and first `rust` block are the listing.
    pub heading: &'static str,
    /// The start of a line the listing prints on stdout once its work is
    /// done.
    pub sentinel: &'static str,
}

/// The README's Quick Start.
pub const QUICK_START: Listing = Listing {
    document: "README.md",
    heading: "## Quick Start",
    sentinel: "Episode over: ",
};

/// The files whose `rlevo` dependency lines a learner copies first. Each line
/// must name the release this checkout builds.
const VERSIONED: &[&str] = &["README.md", "crates/rlevo/src/lib.rs"];

/// A listing as copied.
#[derive(Debug)]
pub struct Copied {
    /// Its `toml` block: the dependencies a learner writes.
    dependencies: String,
    /// Its `rust` block: the program.
    program: String,
}

/// Read `listing` from the checkout at `root`, built as `version`.
///
/// # Errors
///
/// Returns an error if a file in [`VERSIONED`] names no `rlevo` version, or
/// one that `version` is not, or if the listing's section or either of its
/// blocks is missing. A learner who copies it would get another release, or
/// nothing to copy.
pub fn read(root: &Path, listing: &Listing, version: &str) -> Result<Copied> {
    for file in VERSIONED {
        let text = fs::read_to_string(root.join(file))
            .with_context(|| format!("could not read {file}"))?;
        let asked = asked_versions(&text);
        if asked.is_empty() {
            bail!("{file} names no `rlevo` version for a learner to copy");
        }
        for (line, requirement) in asked {
            if !names_release(version, &requirement) {
                bail!(
                    "{file}:{line} asks for rlevo {requirement:?}, which is not this \
                     checkout's {version}: a learner who copies it builds another release"
                );
            }
        }
    }

    let text = fs::read_to_string(root.join(listing.document))
        .with_context(|| format!("could not read {}", listing.document))?;
    let blocks = section_blocks(&text, listing.heading)
        .with_context(|| format!("{} has no `{}` section", listing.document, listing.heading))?;
    let block = |language: &str| {
        blocks
            .iter()
            .find(|(info, _)| info == language)
            .map(|(_, body)| body.clone())
            .with_context(|| {
                format!(
                    "the `{}` section of {} has no ```{language} block",
                    listing.heading, listing.document
                )
            })
    };
    Ok(Copied {
        dependencies: block("toml")?,
        program: block("rust")?,
    })
}

/// Write `copied` into the rendered probe: its dependencies after the
/// template's manifest, its program as `src/main.rs`.
///
/// # Errors
///
/// Returns an error if the probe cannot be written.
pub fn write(probe_dir: &Path, copied: &Copied) -> Result<()> {
    let mut manifest = fs::OpenOptions::new()
        .append(true)
        .open(probe_dir.join("Cargo.toml"))?;
    write!(manifest, "\n{}", copied.dependencies)?;
    let src = probe_dir.join("src");
    fs::create_dir_all(&src)?;
    fs::write(src.join("main.rs"), &copied.program)?;
    Ok(())
}

/// The fenced blocks of the section that opens with the line `heading`, as
/// `(language, body)`, up to the next heading of the same or a higher level.
/// `None` if no line is `heading`.
///
/// A line inside a fence is never a heading: a shell comment starts with `#`.
fn section_blocks(text: &str, heading: &str) -> Option<Vec<(String, String)>> {
    let level = heading.chars().take_while(|c| *c == '#').count();
    let mut lines = text.lines().skip_while(|line| *line != heading);
    lines.next()?;
    let mut blocks = Vec::new();
    let mut open: Option<(String, String)> = None;
    for line in lines {
        if let Some((info, mut body)) = open.take() {
            if line.trim_start().starts_with("```") {
                blocks.push((info, body));
            } else {
                body.push_str(line);
                body.push('\n');
                open = Some((info, body));
            }
        } else if let Some(info) = line.trim_start().strip_prefix("```") {
            open = Some((info.trim().to_string(), String::new()));
        } else if heading_level(line).is_some_and(|other| other <= level) {
            break;
        }
    }
    Some(blocks)
}

/// The level of a Markdown heading line: the number of `#` before a space.
fn heading_level(line: &str) -> Option<usize> {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    (hashes > 0 && line[hashes..].starts_with(' ')).then_some(hashes)
}

/// The line number and version requirement of every `rlevo` dependency
/// line in `text`: `rlevo = "0.5"`, or `rlevo = { version = "0.5", .. }`.
fn asked_versions(text: &str) -> Vec<(usize, String)> {
    const NAME: &str = "rlevo";
    let mut asked = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(at) = rest.find(NAME) {
            let before = rest[..at].chars().next_back();
            let after = &rest[at + NAME.len()..];
            rest = after;
            if before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                continue;
            }
            let Some(value) = after.trim_start().strip_prefix('=') else {
                continue;
            };
            let value = value.trim_start();
            let quoted = match value.strip_prefix('{') {
                Some(table) => table
                    .split_once("version")
                    .and_then(|(_, field)| field.trim_start().strip_prefix('='))
                    .map(str::trim_start),
                None => Some(value),
            };
            if let Some((requirement, _)) = quoted
                .and_then(|value| value.strip_prefix('"'))
                .and_then(|value| value.split_once('"'))
            {
                asked.push((index + 1, requirement.to_string()));
            }
        }
    }
    asked
}

/// Whether `requirement`, written as a version's leading components, names
/// the release `version`: `0.5` names 0.5.0 and 0.5.3, and not 0.6.0.
fn names_release(version: &str, requirement: &str) -> bool {
    let asked: Vec<&str> = requirement.split('.').collect();
    let have: Vec<&str> = version.split('.').collect();
    asked.len() <= have.len()
        && asked
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        && asked.iter().zip(&have).all(|(asked, have)| asked == have)
}

#[cfg(test)]
mod tests {
    use super::*;

    const README: &str = "\
# rlevo

```toml
rlevo = \"0.1\"
```

## Quick Start

```toml
[dependencies]
rlevo = \"0.5\"
```

```rust
fn main() {}
```

### Backends

```bash
# a comment, not a heading
cargo add rlevo
```

```rust
fn later() {}
```

## Development Status

```toml
other = \"1\"
```
";

    #[test]
    fn test_the_listing_is_the_first_block_of_each_language_in_its_section() {
        let blocks = section_blocks(README, "## Quick Start").expect("section");
        let languages: Vec<&str> = blocks.iter().map(|(info, _)| info.as_str()).collect();
        assert_eq!(languages, ["toml", "rust", "bash", "rust"]);
        assert_eq!(blocks[0].1, "[dependencies]\nrlevo = \"0.5\"\n");
        assert_eq!(blocks[1].1, "fn main() {}\n");
    }

    #[test]
    fn test_a_missing_section_is_none() {
        assert_eq!(section_blocks(README, "## Install"), None);
    }

    #[test]
    fn test_a_subheading_does_not_end_the_section_and_a_peer_does() {
        let blocks = section_blocks(README, "## Quick Start").expect("section");
        assert!(blocks.iter().all(|(_, body)| !body.contains("other")));
    }

    #[test]
    fn test_every_rlevo_dependency_line_is_found() {
        let text = "\
rlevo = \"0.5\"
//! rlevo = \"0.5\"
rlevo = { version = \"0.6\", features = [\"wgpu\"] }
rlevo-core = \"0.4\"
my-rlevo = \"0.3\"
use rlevo::prelude::*;
";
        let expected = [(1, "0.5"), (2, "0.5"), (3, "0.6")]
            .map(|(line, requirement)| (line, requirement.to_string()));
        assert_eq!(asked_versions(text), expected);
    }

    #[test]
    fn test_a_requirement_names_the_release_it_leads() {
        assert!(names_release("0.5.0", "0.5"));
        assert!(names_release("0.5.3", "0.5"));
        assert!(names_release("0.5.0", "0.5.0"));
        assert!(!names_release("0.6.0", "0.5"));
        assert!(!names_release("0.5.0", "0.50"));
        assert!(!names_release("0.5.0", "0.5.0.1"));
        assert!(!names_release("0.5.0", ""));
        assert!(!names_release("0.5.0", "0.x"));
    }
}
