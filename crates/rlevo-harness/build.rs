//! Build-time provenance for `report::Provenance` (and the `record`
//! feature's run manifest).
//!
//! Emits three `cargo:rustc-env` values read with `option_env!`:
//! `GIT_COMMIT`, `GIT_DIRTY` (`"1"`/`"0"`/`""`) and `RUSTC_VERSION`. The
//! commit comes from `.cargo_vcs_info.json` in a packaged crate, or from git
//! only in rlevo's own checkout, never from another repository that encloses
//! the build (`src/build_provenance.rs` states the rules). Everything here is
//! **non-fatal**: when a value is unavailable it is emitted as an empty
//! string and the field resolves to `None`. A run must never fail to build
//! because provenance was unavailable.
//!
//! Every `rerun-if-changed` path printed here exists: Cargo treats a missing
//! one as changed on every build, which would rebuild the harness and every
//! crate above it each time.

use std::path::Path;
use std::process::Command;

#[path = "src/build_provenance.rs"]
mod build_provenance;

/// Runs `cmd args...`, returning trimmed stdout on success, or an empty
/// string on any failure (missing binary, non-zero exit, non-UTF-8).
fn capture(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Asks Cargo to rerun this script when `path` changes, if it exists.
fn watch(path: &str) {
    if !path.is_empty() && Path::new(path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}

/// The `0`/`1` that `GIT_DIRTY` carries.
fn flag(dirty: bool) -> String {
    if dirty { "1" } else { "0" }.to_string()
}

/// The commit and dirty flag of rlevo's own checkout, and the git files whose
/// change means they may have changed: `HEAD`, the branch's ref (or
/// `packed-refs`, where a packed ref lives) and the index.
///
/// The dirty flag is the tree's state when this script last ran. Staging or
/// committing reruns it (the index or a ref changes); an edit that is not
/// staged does not, until something else rebuilds the harness. `git status`
/// runs with `--no-optional-locks`, so it never rewrites the index it
/// watches, which would rerun the script on every build.
fn checkout() -> (String, String) {
    watch(&capture("git", &["rev-parse", "--git-path", "HEAD"]));
    watch(&capture("git", &["rev-parse", "--git-path", "index"]));
    watch(&capture("git", &["rev-parse", "--git-path", "packed-refs"]));
    let branch = capture("git", &["symbolic-ref", "-q", "HEAD"]);
    if !branch.is_empty() {
        watch(&capture("git", &["rev-parse", "--git-path", &branch]));
    }

    let sha = capture("git", &["rev-parse", "HEAD"]);
    if sha.is_empty() {
        return (String::new(), String::new());
    }
    // `--porcelain` prints one line per change; empty output means clean.
    let dirty = !capture("git", &["--no-optional-locks", "status", "--porcelain"]).is_empty();
    (sha, flag(dirty))
}

/// The commit and dirty flag, as the strings `GIT_COMMIT` and `GIT_DIRTY`
/// carry; both empty when neither source applies.
fn commit() -> (String, String) {
    let vcs_info = ".cargo_vcs_info.json";
    if let Ok(text) = std::fs::read_to_string(vcs_info) {
        // A packaged crate: its record, and no git. A record that names no
        // commit leaves the commit unknown rather than asking git.
        watch(vcs_info);
        return build_provenance::parse_vcs_info(&text)
            .map_or_else(Default::default, |(sha, dirty)| (sha, flag(dirty)));
    }
    if build_provenance::is_own_checkout(&capture("git", &["rev-parse", "--show-prefix"])) {
        return checkout();
    }
    (String::new(), String::new())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/build_provenance.rs");

    let (git_commit, git_dirty) = commit();
    println!("cargo:rustc-env=GIT_COMMIT={git_commit}");
    println!("cargo:rustc-env=GIT_DIRTY={git_dirty}");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = capture(&rustc, &["-V"]);
    println!("cargo:rustc-env=RUSTC_VERSION={rustc_version}");
}
