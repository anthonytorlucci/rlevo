//! Grids adapter (Minigrid-style empty/four_rooms/door_key/etc.).
//!
//! The grids library tier renders the environment as a 2-D ASCII grid where
//! each cell maps to one glyph: `< > ^ v` for the agent (heading), `#` for
//! walls, `G` for goal, `L` for lava/hazard, `D` for doors, and `K` for keys
//! (see `grids/core/render.rs`).  Colour is layered on top via the semantic
//! palette.
//!
//! This adapter forwards the styled projection via [`crate::adapters::frame_body`]
//! and appends a `<figcaption>` that labels every glyph in plain text so the
//! report satisfies the hue-redundant accessibility contract — direction,
//! shape, and label all carry the same information independently of colour.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::FrameRecord;

/// Wraps a grids-family frame in a `<figure>` with a full glyph legend.
///
/// Delegates the ASCII/styled projection to [`crate::adapters::frame_body`]
/// and appends a `<figcaption>` listing every glyph (agent heading, wall,
/// goal, lava, key, door) with plain-language labels so screen readers and
/// greyscale users can interpret the grid without relying on colour.
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    view! {
        <figure class="rlevo-family-grids">
            {frame_body(frame)}
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-cyan rlevo-mod-bold">"< > ^ v"</span>
                    " agent (heading)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-darkgray">"#"</span>
                    " wall"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-green rlevo-mod-bold">"G"</span>
                    " goal"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-red rlevo-mod-reversed">"L"</span>
                    " lava (hazard)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph">"K / D"</span>
                    " key / door"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
