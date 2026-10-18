//! Bridge between [`rlevo_core::render`] styling types and `ratatui` style.
//!
//! The live TUI is metrics-only (ADR-0013), so this module no longer
//! translates whole [`StyledFrame`](rlevo_core::render::StyledFrame)s into
//! terminal text. What remains is the colour/style translation the metric
//! panels and [`theme`](crate::tui::theme) use to honour the semantic
//! [`palette`](rlevo_core::render::palette) — mapping
//! [`rlevo_core::render::Color`] / [`Modifier`] / [`SpanStyle`] onto their
//! `ratatui` equivalents.
//!
//! Style translation is exhaustive over every variant of
//! [`rlevo_core::render::Color`] *that exists today*. The enum is marked
//! `#[non_exhaustive]`, so a wildcard arm is mandatory; we map any unknown
//! future variant to `ratatui::Color::Reset` (the terminal default) rather
//! than panicking, so adding a variant in `rlevo-core` keeps the live tier
//! degrading gracefully.

use ratatui::style::{Color as RatColor, Modifier as RatModifier, Style as RatStyle};
use rlevo_core::render::{Color, Modifier, SpanStyle};

/// Map a [`Color`] to its ratatui equivalent.
#[must_use]
#[allow(
    clippy::match_same_arms,
    reason = "Color::Reset and the #[non_exhaustive] wildcard share an output by design"
)]
pub fn color_to_ratatui(c: Color) -> RatColor {
    match c {
        Color::Reset => RatColor::Reset,
        Color::Black => RatColor::Black,
        Color::Red => RatColor::Red,
        Color::Green => RatColor::Green,
        Color::Yellow => RatColor::Yellow,
        Color::Blue => RatColor::Blue,
        Color::Magenta => RatColor::Magenta,
        Color::Cyan => RatColor::Cyan,
        Color::Gray => RatColor::Gray,
        Color::DarkGray => RatColor::DarkGray,
        Color::LightRed => RatColor::LightRed,
        Color::LightGreen => RatColor::LightGreen,
        Color::LightYellow => RatColor::LightYellow,
        Color::LightBlue => RatColor::LightBlue,
        Color::LightMagenta => RatColor::LightMagenta,
        Color::LightCyan => RatColor::LightCyan,
        Color::White => RatColor::White,
        Color::Indexed(i) => RatColor::Indexed(i),
        // Color is #[non_exhaustive]: any variant added in rlevo-core that
        // pre-dates a bump of this crate falls back to the terminal default.
        _ => RatColor::Reset,
    }
}

/// Map a [`Modifier`] bitset to its ratatui equivalent.
#[must_use]
pub fn modifier_to_ratatui(m: Modifier) -> RatModifier {
    let mut out = RatModifier::empty();
    if m.contains(Modifier::BOLD) {
        out |= RatModifier::BOLD;
    }
    if m.contains(Modifier::DIM) {
        out |= RatModifier::DIM;
    }
    if m.contains(Modifier::ITALIC) {
        out |= RatModifier::ITALIC;
    }
    if m.contains(Modifier::UNDERLINED) {
        out |= RatModifier::UNDERLINED;
    }
    if m.contains(Modifier::REVERSED) {
        out |= RatModifier::REVERSED;
    }
    out
}

/// Map a [`SpanStyle`] to a `ratatui` [`Style`](ratatui::style::Style).
#[must_use]
pub fn span_style_to_ratatui(s: SpanStyle) -> RatStyle {
    let mut out = RatStyle::default();
    if let Some(fg) = s.fg {
        out = out.fg(color_to_ratatui(fg));
    }
    if let Some(bg) = s.bg {
        out = out.bg(color_to_ratatui(bg));
    }
    if !s.modifier.is_empty() {
        out = out.add_modifier(modifier_to_ratatui(s.modifier));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::render::palette;

    /// Every `Color` variant declared today must map to a distinct
    /// `ratatui::Color`. Guards against silent collisions if either palette
    /// grows.
    #[test]
    fn color_exhaustive_and_distinct() {
        let pairs = [
            (Color::Reset, RatColor::Reset),
            (Color::Black, RatColor::Black),
            (Color::Red, RatColor::Red),
            (Color::Green, RatColor::Green),
            (Color::Yellow, RatColor::Yellow),
            (Color::Blue, RatColor::Blue),
            (Color::Magenta, RatColor::Magenta),
            (Color::Cyan, RatColor::Cyan),
            (Color::Gray, RatColor::Gray),
            (Color::DarkGray, RatColor::DarkGray),
            (Color::LightRed, RatColor::LightRed),
            (Color::LightGreen, RatColor::LightGreen),
            (Color::LightYellow, RatColor::LightYellow),
            (Color::LightBlue, RatColor::LightBlue),
            (Color::LightMagenta, RatColor::LightMagenta),
            (Color::LightCyan, RatColor::LightCyan),
            (Color::White, RatColor::White),
            (Color::Indexed(42), RatColor::Indexed(42)),
        ];
        for (ours, theirs) in pairs {
            assert_eq!(color_to_ratatui(ours), theirs, "mismatch on {ours:?}");
        }
        let mapped: std::collections::HashSet<_> =
            pairs.iter().map(|(c, _)| color_to_ratatui(*c)).collect();
        assert_eq!(mapped.len(), pairs.len(), "two Color variants collided");
    }

    #[test]
    fn modifier_bit_round_trip() {
        let m = Modifier::BOLD | Modifier::UNDERLINED | Modifier::REVERSED;
        let rm = modifier_to_ratatui(m);
        assert!(rm.contains(RatModifier::BOLD));
        assert!(rm.contains(RatModifier::UNDERLINED));
        assert!(rm.contains(RatModifier::REVERSED));
        assert!(!rm.contains(RatModifier::ITALIC));
        assert!(!rm.contains(RatModifier::DIM));
    }

    #[test]
    fn modifier_empty_is_empty() {
        let rm = modifier_to_ratatui(Modifier::EMPTY);
        assert!(rm.is_empty());
    }

    #[test]
    fn span_style_handles_unset_colors() {
        let style = SpanStyle::default().bold();
        let rs = span_style_to_ratatui(style);
        assert_eq!(rs.fg, None);
        assert_eq!(rs.bg, None);
        assert!(rs.add_modifier.contains(RatModifier::BOLD));
    }

    /// Palette pairings from [`rlevo_core::render::palette`] arrive in the
    /// terminal with their hue-redundant modifiers intact — the
    /// accessibility contract is preserved across the bridge.
    #[test]
    fn palette_preserves_hazard_modifier_pairing() {
        let style = SpanStyle::default()
            .fg(palette::HAZARD_FG)
            .with_modifier(palette::HAZARD_MODIFIER);
        let rs = span_style_to_ratatui(style);
        assert_eq!(rs.fg, Some(RatColor::Red));
        assert!(rs.add_modifier.contains(RatModifier::REVERSED));
    }
}
