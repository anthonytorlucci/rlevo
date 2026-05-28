//! Per-family playback adapters for the M6 report tier.
//!
//! Each adapter wraps the shared [`crate::styled::styled_frame_view`]
//! with family-specific framing and a glyph/colour legend. The legend
//! pairs colour with glyph + plain-language wording so a B/W screenshot
//! of the report still conveys agent / goal / hazard meaning — per the
//! project's hue-redundant accessibility contract.

pub mod box2d;
pub mod classic;
pub mod fallback;
pub mod grids;
pub mod landscape;
pub mod locomotion;
pub mod toy_text;

use leptos::prelude::*;

use crate::wire::{EnvFamily, FrameRecord};

/// Dispatch one [`FrameRecord`] to the family-specific renderer.
#[must_use]
pub fn render(family: EnvFamily, frame: &FrameRecord) -> AnyView {
    // `EnvFamily` is `#[non_exhaustive]`; the wildcard arm catches future
    // variants the wire mirror grows. Today (post-M7) every known family
    // has a dedicated adapter, so the wildcard is structural future-proofing.
    #[allow(unreachable_patterns)]
    match family {
        EnvFamily::Classic => classic::render(frame),
        EnvFamily::Grids => grids::render(frame),
        EnvFamily::ToyText => toy_text::render(frame),
        EnvFamily::Landscapes => landscape::render(frame),
        EnvFamily::Locomotion => locomotion::render(frame),
        EnvFamily::Box2d => box2d::render(frame),
        _ => fallback::render(family, frame),
    }
}

/// Shared frame-rendering helper: prefer the `StyledFrame` projection
/// when present, fall back to plain ASCII, and finally render an empty
/// placeholder if neither is available.
#[must_use]
pub(crate) fn frame_body(frame: &FrameRecord) -> AnyView {
    if let Some(styled) = &frame.styled {
        crate::styled::styled_frame_view(styled)
    } else if let Some(ascii) = &frame.ascii {
        crate::styled::ascii_frame_view(ascii)
    } else {
        view! { <pre class="rlevo-styled rlevo-empty">"(no rendered frame)"</pre> }.into_any()
    }
}
