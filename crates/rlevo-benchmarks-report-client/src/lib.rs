//! Leptos/WASM client for the rlevo static-HTML report.
//!
//! This crate is the browser-side half of the two-product visualisation
//! described in ADR 0013.  It is compiled to `wasm32-unknown-unknown` and
//! bundled into a single self-contained `index.html` by
//! `rlevo-benchmarks::report::html`.  It has no runtime server dependency —
//! all data is read from `<script>` blocks that the emitter inlines at build
//! time.
//!
//! # Data sources
//!
//! On load the client reads four `<script>` blocks embedded by the emitter:
//!
//! * `rlevo-manifest` — `application/json` [`wire::RunManifest`].
//! * `rlevo-warnings` — `application/json` `[OpenWarning]`.
//! * `rlevo-episode-index` — `application/json` `[EpisodeMeta]`.
//! * `rlevo-episode-NNNNNN` — `application/octet-stream` base64-encoded
//!   raw `.rec` bytes (16-byte preamble + bincode header + length-prefixed
//!   `RecordChunk`s, all `bincode::config::standard()`).
//!
//! The client decodes those blocks, then dispatches per-family playback
//! adapters (interactive scrubber + styled-frame / SVG rendering) and
//! draws convergence + population chart panels.
//!
//! # Module map
//!
//! | Module | Role |
//! |--------|------|
//! | [`wire`] | Bincode-compatible mirror types for `EpisodeRecord`, `RunManifest`, `StyledFrame`, etc. |
//! | [`inline_data`] | Reads and caches the four `<script>` data blocks from the DOM. |
//! | [`app`] | Root Leptos component; page layout, episode-table, reactive selection. |
//! | [`playback`] | Per-episode scrubber + play/pause/speed panel. |
//! | [`adapters`] | Per-family frame renderers dispatched from `playback`. |
//! | [`charts`] | SVG convergence and population panels (pure-Rust, no JS interop). |
//! | [`series`] | Series-extraction helpers that feed `charts`; unit-testable as native code. |
//! | [`styled`] | `StyledFrame` → `<pre class="rlevo-styled">` HTML translation. |

pub mod adapters;
pub mod app;
pub mod charts;
pub mod inline_data;
pub mod playback;
pub mod series;
pub mod styled;
pub mod wire;

use wasm_bindgen::prelude::*;

/// WASM entry point invoked once at module load via `wasm-bindgen`'s
/// `start` attribute. Installs a panic hook so dev-tools see real
/// stack traces, then mounts the Leptos app on `#rlevo-app`.
///
/// # Panics
///
/// Panics if the host page lacks a `#rlevo-app` element or it is not an
/// `HtmlElement` — the report emitter always writes that mount div, so this
/// indicates a malformed host page.
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
