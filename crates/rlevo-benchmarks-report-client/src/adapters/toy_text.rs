//! Toy-text adapter (FrozenLake, CliffWalking, Taxi, Blackjack).
//!
//! FrozenLake's library tier paints tiles as `@ F H G S`; the agent
//! glyph (`@`) layers over the underlying tile (see `frozen_lake.rs`
//! lines 665–730). Adapter forwards the styled projection and surfaces
//! the tile-meaning legend.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::FrameRecord;

/// Render one toy-text frame.
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    view! {
        <figure class="rlevo-family-toy_text">
            {frame_body(frame)}
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-cyan rlevo-mod-bold">"@"</span>
                    " agent"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph">"F"</span>
                    " frozen tile"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-red rlevo-mod-reversed">"H"</span>
                    " hole (hazard)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-green rlevo-mod-bold">"G"</span>
                    " goal"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-yellow rlevo-mod-bold">"S"</span>
                    " start"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
