//! Fallback adapter for env families that lack a bespoke renderer.
//!
//! [`render`] is called by other adapters when a frame's [`FamilyPayload`]
//! variant does not match what the adapter expects (e.g. a Box2D adapter
//! receiving a Classic payload).  It delegates to `frame_body`,
//! which emits a styled ASCII projection when one is present and raw text
//! otherwise, then appends a `<figcaption>` banner so users know they are
//! seeing a generic view rather than a purpose-built one.
//!
//! [`FamilyPayload`]: crate::wire::FamilyPayload

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::{EnvFamily, FrameRecord};

/// Renders a frame generically when no bespoke adapter exists for `family`.
///
/// Wraps `frame_body` in a `<figure>` and appends a
/// warning banner in the `<figcaption>` naming the `family` so users can
/// identify which env variant triggered the fallback path.
#[must_use]
pub fn render(family: EnvFamily, frame: &FrameRecord) -> AnyView {
    let banner = format!(
        "family {family:?} renders via the generic adapter — no bespoke adapter is available"
    );
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
