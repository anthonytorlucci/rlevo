//! Prints the wire [`FORMAT_VERSION`] the client was compiled against, so
//! `trunk build` can stamp it into the `dist/` bundle (see the `post_build`
//! hook in `Trunk.toml`).
//!
//! Because this bin links the same crate source `trunk` just compiled into
//! the wasm, the printed number is guaranteed equal to the version baked
//! into the bundle — that is the whole point of stamping it rather than
//! hand-typing or grepping it. The native emitter
//! (`rlevo_benchmarks::report::ClientAssets::from_trunk_dist`) reads the
//! stamp and refuses a stale bundle before writing any report HTML.
//!
//! [`FORMAT_VERSION`]: rlevo_benchmarks_report_client::wire::FORMAT_VERSION

// Native-only: `trunk` may compile the package's bins for the wasm target
// during its build, but the stamp is produced by a host-target `cargo run`
// in the post_build hook. Keep the wasm build of this bin a trivial no-op.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    println!("{}", rlevo_benchmarks_report_client::wire::FORMAT_VERSION);
}

#[cfg(target_arch = "wasm32")]
fn main() {}
