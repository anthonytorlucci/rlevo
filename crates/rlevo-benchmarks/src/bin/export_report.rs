//! CLI front-end for [`rlevo_benchmarks::report::emit_static_html`].
//!
//! Usage:
//! ```text
//! cargo run -p rlevo-benchmarks --features report --bin export-report -- <run-dir> <out.html>
//! ```

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
