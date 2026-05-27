//! Leptos/WASM client for the rlevo static-HTML report (Milestone 5.1).
//!
//! On load the client reads four `<script>` blocks embedded by the
//! `rlevo-benchmarks::report::html` emitter:
//!
//! * `rlevo-manifest` — `application/json` `RunManifest`.
//! * `rlevo-warnings` — `application/json` `[OpenWarning]`.
//! * `rlevo-episode-index` — `application/json` `[EpisodeMeta]`.
//! * `rlevo-episode-NNNNNN` — `application/octet-stream` base64-encoded
//!   raw `.rec` bytes (16-byte preamble + bincode header + length-prefixed
//!   `RecordChunk`s, all `bincode::config::standard()`).
//!
//! M5.1 ships the data-decode + minimal-render path. Per-family
//! playback adapters and convergence plots land in M6+.

pub mod app;
pub mod inline_data;
pub mod wire;

use wasm_bindgen::prelude::*;

/// WASM entry point invoked once at module load via `wasm-bindgen`'s
/// `start` attribute. Installs a panic hook so dev-tools see real
/// stack traces, then mounts the Leptos app on `#rlevo-app`.
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    let mount_point: web_sys::HtmlElement = leptos::prelude::document()
        .get_element_by_id("rlevo-app")
        .expect("missing #rlevo-app mount point — emitter omitted the host div")
        .dyn_into()
        .expect("#rlevo-app is not an HtmlElement");
    leptos::mount::mount_to(mount_point, app::App).forget();
}
