//! Classic-control adapter (CartPole, MountainCar, Pendulum, Acrobot).
//!
//! The library tier renders classic environments as a 1-D ASCII track with a
//! styled agent glyph and a suffix carrying angle / step metrics (see
//! `cartpole.rs`).  This adapter wraps that projection in a `<figure>` with a
//! legend so the web report can display it without duplicating render logic.
//!
//! # Accessibility
//!
//! The legend pairs each glyph with a text label so the meaning is not
//! conveyed by colour alone.  The agent glyph is bold cyan `#`; the track
//! uses a dashed `┄` in dark-gray — hue and stroke pattern differ so both
//! channels carry the same information.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::FrameRecord;

/// Wraps a classic-family frame in a `<figure>` with a labelled legend.
///
/// Delegates the actual ASCII projection to [`crate::adapters::frame_body`]
/// and appends a `<figcaption>` describing the agent glyph, track glyph, and
/// suffix metrics so screen readers and colour-blind users can interpret the
/// display without relying on hue alone.
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    view! {
        <figure class="rlevo-family-classic">
            {frame_body(frame)}
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-cyan rlevo-mod-bold">"#"</span>
                    " agent / cart"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-fg-darkgray">"┄"</span>
                    " track"
                </span>
                <span class="rlevo-legend-key">
                    "θ / step metrics in suffix"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
