//! Repository automation entry point — `cargo xtask <task>`.
//!
//! This binary is a **dispatcher**, not an implementation. Each task's logic
//! lives in a shell script beside its assets, so the same file runs from a
//! developer's terminal and from a CI runner with nothing installed but
//! `bash`. Reimplementing a task in Rust here would give CI a second code
//! path to disagree with.
//!
//! `xtask` is not in `default-members`: a plain `cargo build` at the workspace
//! root never compiles it.
//!
//! # Tasks
//!
//! - `byoe` — run the BYOE-1 outside-in acceptance test. Exits with the number
//!   of the first failing step. See `xtask/byoe/README.md`.

use std::path::PathBuf;
use std::process::{Command, ExitCode};

/// Absolute path to the workspace root, derived from this crate's manifest
/// directory at compile time so the task works from any working directory.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask/ always has a parent")
        .to_path_buf()
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: cargo xtask <task> [args...]\n\
         \n\
         tasks:\n\
         \x20 byoe    BYOE-1 outside-in acceptance test (exit code = first failing step)\n"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(task) = args.next() else {
        return usage();
    };

    let script = match task.as_str() {
        "byoe" => "xtask/byoe/run.sh",
        "-h" | "--help" | "help" => return usage(),
        other => {
            eprintln!("unknown task: {other}");
            return usage();
        }
    };

    let root = repo_root();
    let status = Command::new("bash")
        .arg(root.join(script))
        .args(args)
        .current_dir(&root)
        .status();

    match status {
        // Propagate the script's exit code verbatim — for `byoe` it *is* the
        // result (the first failing step), not merely success or failure.
        Ok(s) => match s.code() {
            Some(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
            None => {
                eprintln!("{script} terminated by signal");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("could not run {script}: {e}");
            ExitCode::FAILURE
        }
    }
}
