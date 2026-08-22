//! Generic frame renderer, used when a bespoke view is unavailable.
//!
//! [`render`] delegates to `frame_body`, which emits a styled ASCII
//! projection when one is present and raw text otherwise, then appends a
//! `<figcaption>` banner so users know they are seeing a generic view rather
//! than a purpose-built one.
//!
//! # Two reasons, two banners
//!
//! The banner used to say, for every caller, that the *family* had no bespoke
//! adapter. That is true for exactly one of the two call sites, and false for
//! the six that actually fire in practice:
//!
//! - [`FallbackReason::NoFamilyAdapter`] — [`adapters::render`]'s wildcard arm,
//!   reached when the [`EnvFamily`] variant is newer than this client. The
//!   family genuinely has no adapter. Structurally unreachable today.
//! - [`FallbackReason::UnsupportedPayload`] — a family adapter received a
//!   [`FamilyPayload`] it cannot decode, typically
//!   [`FamilyPayload::Ascii`](crate::wire::FamilyPayload::Ascii). The family
//!   has a perfectly good adapter; it is the *payload* that is generic. For a
//!   bandit run (`family = Classic`, `payload = Ascii`) the old banner blamed
//!   `Classic` for lacking an adapter it has had all along.
//!
//! Passing the reason in, rather than rewording one shared string, is what
//! keeps both messages true — a reword alone would have made the first caller
//! lie instead of the second.
//!
//! [`FamilyPayload`]: crate::wire::FamilyPayload
//! [`adapters::render`]: crate::adapters::render

use leptos::prelude::*;

use crate::adapters::frame_body;
use crate::wire::{EnvFamily, FrameRecord};

/// Why a frame is being rendered generically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// No adapter exists for this family at all — the recording names an
    /// [`EnvFamily`] variant this client does not know. Only
    /// [`adapters::render`](crate::adapters::render)'s wildcard produces this.
    NoFamilyAdapter,
    /// The family's adapter exists but cannot decode this frame's payload.
    /// Every per-family adapter produces this.
    UnsupportedPayload,
}

/// The warning text shown under a generically-rendered frame.
///
/// Split out as a pure function so both messages are unit-testable without
/// mounting a view.
#[must_use]
pub(crate) fn banner_text(family: EnvFamily, reason: FallbackReason) -> String {
    match reason {
        FallbackReason::NoFamilyAdapter => format!(
            "family {family:?} has no bespoke adapter in this report client — \
             rendering the generic view"
        ),
        FallbackReason::UnsupportedPayload => format!(
            "this frame carries no rich payload for the {family:?} adapter to \
             decode — rendering the generic ASCII view"
        ),
    }
}

/// Renders a frame generically, banner-captioned with why.
#[must_use]
pub fn render(family: EnvFamily, reason: FallbackReason, frame: &FrameRecord) -> AnyView {
    let banner = banner_text(family, reason);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect verbatim: a bandit run is `family = Classic` with an ASCII
    /// payload, and `Classic` has had a bespoke adapter throughout.
    #[test]
    fn payload_fallback_does_not_claim_the_family_lacks_an_adapter() {
        let text = banner_text(EnvFamily::Classic, FallbackReason::UnsupportedPayload);
        assert!(
            !text.contains("no bespoke adapter"),
            "payload-mismatch banner must not blame the family for a missing \
             adapter it has: {text}",
        );
        assert!(
            text.contains("payload"),
            "payload-mismatch banner must name the payload as the cause: {text}",
        );
        assert!(
            text.contains("Classic"),
            "banner must still name the family so the frame is identifiable: {text}",
        );
    }

    /// The other caller's message must stay true too — this is what a plain
    /// reword would have broken.
    #[test]
    fn missing_adapter_fallback_still_names_the_missing_adapter() {
        let text = banner_text(EnvFamily::Classic, FallbackReason::NoFamilyAdapter);
        assert!(
            text.contains("no bespoke adapter"),
            "the no-adapter banner must still say the adapter is missing: {text}",
        );
    }

    /// Distinct causes must read distinctly; a shared string is the defect.
    #[test]
    fn the_two_reasons_render_different_text() {
        assert_ne!(
            banner_text(EnvFamily::Classic, FallbackReason::NoFamilyAdapter),
            banner_text(EnvFamily::Classic, FallbackReason::UnsupportedPayload),
        );
    }
}
