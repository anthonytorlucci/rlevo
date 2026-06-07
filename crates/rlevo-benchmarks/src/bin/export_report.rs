//! CLI front-end for [`rlevo_benchmarks::report::emit_static_html`].
//!
//! Reads a previously recorded benchmark run from `<run-dir>` and emits a
//! self-contained static HTML file at `<out.html>` that renders per-episode
//! metrics and environment playback without a running server.
//!
//! Requires the `report` feature:
//! ```text
//! cargo run -p rlevo-benchmarks --features report --bin export-report -- <run-dir> <out.html>
//! ```
//!
//! # Arguments
//!
//! - `<run-dir>` — path to a directory produced by a previous benchmark run
//!   (must contain a `manifest.json` and one or more episode record files).
//! - `<out.html>` — destination path for the generated HTML file.
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0    | Report written successfully. |
//! | 1    | `<run-dir>` could not be opened, or HTML emission failed. |
//! | 2    | Wrong number of arguments. |

use std::path::PathBuf;
use std::process::ExitCode;

use rlevo_benchmarks::report::{EmitConfig, RecordedRun, emit_static_html};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "usage: {} <run-dir> <out.html>",
            args.first().map_or("export-report", String::as_str)
        );
        return ExitCode::from(2);
    }
    let run_dir = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);

    let run = match RecordedRun::open(&run_dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", run_dir.display());
            return ExitCode::FAILURE;
        }
    };
    for w in run.warnings() {
        eprintln!("warning: {w:?}");
    }
    let outcome = match emit_static_html(&run, &out, &EmitConfig::default()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("emit failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "wrote {} ({} episodes, {} bytes)",
        out.display(),
        outcome.episode_count,
        outcome.bytes_written
    );
    ExitCode::SUCCESS
}
