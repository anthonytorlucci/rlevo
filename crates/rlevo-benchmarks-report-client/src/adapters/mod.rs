//! Per-family playback adapters for the static-HTML report tier.
//!
//! Each sub-module (`classic`, `box2d`, `grids`, …) exposes a single
//! `render(frame)` or `render(family, frame)` function that wraps the
//! shared [`frame_body`] helper with family-specific `<figure>` framing and
//! a glyph/colour legend.
//!
//! The top-level [`render`] function dispatches a [`FrameRecord`] to the
//! correct adapter based on [`EnvFamily`].  Unknown future variants (the enum
//! is `#[non_exhaustive]`) fall through to [`fallback::render`].
//!
//! # Accessibility contract
//!
//! Every legend must pair hue with at least one other signal (glyph shape,
//! stroke pattern, or plain-language label) so the report remains readable in
//! greyscale and for colour-blind users — see ADR 0008.

pub mod box2d;
pub mod classic;
pub mod fallback;
pub mod grids;
pub mod landscape;
pub mod locomotion;
pub mod toy_text;

use leptos::prelude::*;

use crate::wire::{EnvFamily, FrameRecord};

/// Dispatches one [`FrameRecord`] to the correct family-specific adapter.
///
/// `EnvFamily` is `#[non_exhaustive]`; any variant not yet covered by a
/// dedicated adapter falls through to [`fallback::render`].
#[must_use]
pub fn render(family: EnvFamily, frame: &FrameRecord) -> AnyView {
    // `EnvFamily` is `#[non_exhaustive]`; the wildcard arm catches future
    // variants the wire mirror grows. Today every known family has a
    // dedicated adapter, so the wildcard is structural future-proofing.
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

/// Renders the best available projection from a [`FrameRecord`].
///
/// Priority order: styled (`StyledFrame`) → plain ASCII → empty placeholder
/// `<pre>`.  All family adapters call this rather than accessing
/// `frame.styled` / `frame.ascii` directly so the fallback chain stays in
/// one place.
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
