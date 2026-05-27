//! Classic-control adapter (CartPole, MountainCar, Pendulum, Acrobot).
//!
//! The library tier already renders these as 1-D ASCII tracks plus a
//! suffix carrying angle / step metrics (see `cartpole.rs` lines
//! 523–568). The adapter forwards the styled projection and adds a
//! legend.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::FrameRecord;

/// Render one classic-family frame.
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
