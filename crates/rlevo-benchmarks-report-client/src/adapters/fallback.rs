//! Fallback adapter for families that don't have a bespoke renderer
//! yet (box2d, locomotion, landscapes — landing in M7). Renders the
//! styled projection if available, otherwise plain ASCII, plus a
//! banner explaining the limitation.

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::{EnvFamily, FrameRecord};

/// Render one frame for a family that hasn't shipped its bespoke
/// adapter yet.
#[must_use]
pub fn render(family: EnvFamily, frame: &FrameRecord) -> AnyView {
    let banner = format!("family {family:?} renders via the generic adapter — bespoke adapter lands in M7");
    view! {
        <figure class="rlevo-family-fallback">
            {frame_body(frame)}
            <figcaption class="legend">
                <p class="rlevo-warnings">{banner}</p>
            </figcaption>
        </figure>
    }
    .into_any()
}
