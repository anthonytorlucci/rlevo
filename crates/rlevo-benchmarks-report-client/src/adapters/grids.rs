//! Grids adapter (Minigrid-style empty/four_rooms/door_key/etc.).
//!
//! The grids library tier renders the agent as a direction glyph
//! (`< > ^ v`), walls as `#`, goals as `G`, hazards as `L`, doors as
//! `D`, keys as `K` (see `grids/core/render.rs` lines 57–87). Colour
//! is layered via the semantic palette; the adapter forwards the
//! styled projection and explains the glyphs in the legend.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::FrameRecord;

/// Render one grids-family frame.
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
