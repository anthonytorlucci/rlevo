//! Bridge between [`rlevo_core::render::StyledFrame`] and `ratatui` text.
//!
//! Production crates ship `StyledFrame` (and its sub-types) without any
//! terminal-side dependency. This module owns the one-way translation into
//! `ratatui::text::Text<'static>` — every span's text is cloned into an
//! owned `Cow`, so converted frames carry no borrow on the source frame and
//! can be freely sent across thread boundaries to the render loop.
//!
//! Both [`StyledFrame`] (defined in `rlevo-core`) and `ratatui::text::Text`
//! (defined in `ratatui-core`) are foreign to this crate, so the orphan
//! rule prevents writing `impl From<StyledFrame> for Text<'static>` here.
//! The bridge is therefore exposed as free functions; callers write
//! `frame_to_text(frame)` rather than `frame.into()`.
//!
//! Style translation is exhaustive over every variant of
//! [`rlevo_core::render::Color`] *that exists today*. The enum is marked
//! `#[non_exhaustive]`, so a wildcard arm is mandatory; we map any unknown
//! future variant to `ratatui::Color::Reset` (the terminal default) rather
//! than panicking, so adding a variant in `rlevo-core` keeps the live tier
//! degrading gracefully.

use std::borrow::Cow;

use ratatui::style::{Color as RatColor, Modifier as RatModifier, Style as RatStyle};
use ratatui::text::{Line, Span, Text};
use rlevo_core::render::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};

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

/// Map a [`StyledSpan`] to a `ratatui` [`Span<'static>`].
///
/// Ownership is transferred: the span's `String` lands in a `Cow::Owned`,
/// so the result has no lifetime tied to the input.
#[must_use]
pub fn span_to_ratatui(span: StyledSpan) -> Span<'static> {
    Span {
        style: span_style_to_ratatui(span.style),
        content: Cow::Owned(span.text),
    }
}

/// Map a [`StyledLine`] to a `ratatui` [`Line<'static>`].
#[must_use]
pub fn line_to_ratatui(line: StyledLine) -> Line<'static> {
    let spans: Vec<Span<'static>> = line.spans.into_iter().map(span_to_ratatui).collect();
    Line::from(spans)
}

/// Map a [`StyledFrame`] to a `ratatui` [`Text<'static>`].
///
/// Empty frames map to `Text::default()`. Newlines that were carried
/// structurally by `StyledFrame.lines` survive: each [`StyledLine`] becomes
/// exactly one [`Line<'static>`].
#[must_use]
pub fn frame_to_ratatui(frame: StyledFrame) -> Text<'static> {
    let lines: Vec<Line<'static>> = frame.lines.into_iter().map(line_to_ratatui).collect();
    Text::from(lines)
}

/// Borrowing variant of [`frame_to_ratatui`].
///
/// Panel widgets render once per tick from a shared [`AppState`]; cloning
/// the held frame into the conversion would do ~one malloc per span per
/// tick. This entry point clones internally only as needed to produce
/// `'static` `Cow`s, so the source frame remains usable for the next tick.
///
/// [`AppState`]: crate::tui::state::AppState
#[must_use]
pub fn frame_to_ratatui_ref(frame: &StyledFrame) -> Text<'static> {
    let lines: Vec<Line<'static>> = frame
        .lines
        .iter()
        .map(|line| {
            let spans: Vec<Span<'static>> = line
                .spans
                .iter()
                .map(|span| Span {
                    style: span_style_to_ratatui(span.style),
                    content: Cow::Owned(span.text.clone()),
                })
                .collect();
            Line::from(spans)
        })
        .collect();
    Text::from(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::render::palette;

    /// `StyledFrame::unstyled` → `Text` should round-trip every glyph and
    /// every newline, so the visible content in the TUI matches what the
    /// env emits.
    #[test]
    fn frame_unstyled_roundtrip() {
        let frame = StyledFrame::unstyled(String::from("ab\ncd"));
        let text = frame_to_ratatui(frame);
        assert_eq!(text.lines.len(), 2);
        let joined: String = text
            .lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(joined, "ab\ncd");
    }

    #[test]
    fn frame_preserves_per_span_style() {
        let frame = StyledFrame {
            lines: vec![StyledLine::from_spans([
                StyledSpan::new("agent ", SpanStyle::default().fg(Color::Cyan).bold()),
                StyledSpan::raw("here"),
            ])],
        };
        let text = frame_to_ratatui(frame);
        let line = &text.lines[0];
        assert_eq!(line.spans.len(), 2);

        assert_eq!(line.spans[0].content.as_ref(), "agent ");
        assert_eq!(line.spans[0].style.fg, Some(RatColor::Cyan));
        assert!(line.spans[0].style.add_modifier.contains(RatModifier::BOLD));

        assert_eq!(line.spans[1].content.as_ref(), "here");
        assert_eq!(line.spans[1].style, RatStyle::default());
    }

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

    #[test]
    fn empty_frame_yields_empty_text() {
        let frame = StyledFrame::default();
        let text = frame_to_ratatui(frame);
        assert!(text.lines.is_empty());
    }

    /// Borrow variant produces the same lines/glyphs/style as the owning
    /// variant; only the ownership semantics differ.
    #[test]
    fn frame_borrow_variant_matches_owning_variant() {
        let frame = StyledFrame {
            lines: vec![StyledLine::from_spans([
                StyledSpan::new(
                    "agent",
                    SpanStyle::default().fg(palette::AGENT_FG).bold(),
                ),
                StyledSpan::raw(" idle"),
            ])],
        };

        let owned = frame_to_ratatui(frame.clone());
        let borrowed = frame_to_ratatui_ref(&frame);

        assert_eq!(owned.lines.len(), borrowed.lines.len());
        for (a, b) in owned.lines.iter().zip(borrowed.lines.iter()) {
            assert_eq!(a.spans.len(), b.spans.len());
            for (sa, sb) in a.spans.iter().zip(b.spans.iter()) {
                assert_eq!(sa.content, sb.content);
                assert_eq!(sa.style, sb.style);
            }
        }
        // Source frame is still usable.
        assert_eq!(frame.plain_text(), "agent idle");
    }
}
