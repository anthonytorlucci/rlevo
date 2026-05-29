//! Toy-text adapter (FrozenLake, CliffWalking, Taxi, Blackjack).
//!
//! The toy-text library tier renders environments as a 2-D tile grid where
//! each cell maps to one glyph: `@` for the agent (overlaid on the underlying
//! tile), `F` for frozen, `H` for hole/hazard, `G` for goal, and `S` for
//! start (see `frozen_lake.rs`).  Colour is layered on top via the semantic
//! palette.
//!
//! This adapter forwards the styled projection via [`crate::adapters::frame_body`]
//! and appends a `<figcaption>` that labels every tile glyph in plain text so
//! the report satisfies the hue-redundant accessibility contract — glyph
//! shape, colour modifier (bold / reversed), and label all carry the same
//! information independently of colour.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::FrameRecord;

/// Wraps a toy-text family frame in a `<figure>` with a full tile-glyph legend.
///
/// Delegates the ASCII/styled projection to [`crate::adapters::frame_body`]
/// and appends a `<figcaption>` listing every tile glyph (`@`, `F`, `H`, `G`,
/// `S`) with plain-language labels so screen readers and greyscale users can
/// interpret the grid without relying on colour alone.
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
