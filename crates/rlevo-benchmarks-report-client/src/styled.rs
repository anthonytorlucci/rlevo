//! `StyledFrame` → HTML rendering for the per-family playback adapters.
//!
//! Translates the wire-mirror [`StyledFrame`] / [`StyledLine`] /
//! [`StyledSpan`] types into a single `<pre class="rlevo-styled">` block
//! whose colour and modifier classes are owned by `app.css`.
//!
//! [`StyledLine`]: crate::wire::StyledLine
//!
//! Per the project accessibility contract: every colour class is paired
//! with a hue-redundant signal — BOLD becomes `font-weight: 700`,
//! REVERSED swaps foreground/background. A B/W screenshot of the report
//! stays legible because the family adapters supplement colour with
//! glyph and legend wording.
//!
//! `Color::Indexed` and `Color::Reset` map to no CSS class — the span
//! renders with the terminal's default foreground, matching the
//! `Option<Color>` semantic on the wire.

use leptos::prelude::*;

use crate::wire::{Color, Modifier, SpanStyle, StyledFrame, StyledSpan};

/// Convert one [`StyledFrame`] into a `<pre>` Leptos view. Empty frames
/// render an empty pre block (the caller decides whether to elide).
///
/// Lines are separated by literal `\n` characters between span groups
/// rather than `<div>` wrappers — block-level descendants inside `<pre>`
/// would compound the whitespace and break grid alignment.
#[must_use]
pub fn styled_frame_view(frame: &StyledFrame) -> AnyView {
    let mut nodes: Vec<AnyView> = Vec::new();
    for (idx, line) in frame.lines.iter().enumerate() {
        if idx > 0 {
            nodes.push(view! { {"\n"} }.into_any());
        }
        for span in &line.spans {
            nodes.push(styled_span_view(span));
        }
    }
    view! { <pre class="rlevo-styled">{nodes}</pre> }.into_any()
}

/// Convenience: render a plain ASCII string when no `StyledFrame` is
/// available. Falls back to the same `<pre>` shell so the playback
/// panel CSS still applies.
#[must_use]
pub fn ascii_frame_view(ascii: &str) -> AnyView {
    view! { <pre class="rlevo-styled">{ascii.to_owned()}</pre> }.into_any()
}

/// Renders one [`StyledSpan`] as a `<span>` with the appropriate CSS classes.
///
/// When [`span_classes`] produces an empty string (default style, `Reset`
/// colour, zero modifier) the `class` attribute is omitted entirely to keep
/// the HTML compact.
fn styled_span_view(span: &StyledSpan) -> AnyView {
    let class = span_classes(&span.style);
    let text = span.text.clone();
    if class.is_empty() {
        view! { <span>{text}</span> }.into_any()
    } else {
        view! { <span class=class>{text}</span> }.into_any()
    }
}

/// Compose the CSS class list for one span style. Returned as a single
/// space-joined string because Leptos's `class=...` attr expects that.
#[must_use]
pub fn span_classes(style: &SpanStyle) -> String {
    let mut parts: Vec<&'static str> = Vec::new();
    if let Some(fg) = style.fg {
        if let Some(c) = color_class(fg, true) {
            parts.push(c);
        }
    }
    if let Some(bg) = style.bg {
        if let Some(c) = color_class(bg, false) {
            parts.push(c);
        }
    }
    parts.extend(modifier_classes(style.modifier));
    parts.join(" ")
}

/// Resolve a [`Color`] to a stable foreground or background CSS class.
/// `Reset` and `Indexed(_)` return `None` — the span renders unstyled.
#[must_use]
pub fn color_class(color: Color, foreground: bool) -> Option<&'static str> {
    let table_fg = [
        // (Color, fg class)
        (Color::Black, "rlevo-fg-black"),
        (Color::Red, "rlevo-fg-red"),
        (Color::Green, "rlevo-fg-green"),
        (Color::Yellow, "rlevo-fg-yellow"),
        (Color::Blue, "rlevo-fg-blue"),
        (Color::Magenta, "rlevo-fg-magenta"),
        (Color::Cyan, "rlevo-fg-cyan"),
        (Color::Gray, "rlevo-fg-gray"),
        (Color::DarkGray, "rlevo-fg-darkgray"),
        (Color::LightRed, "rlevo-fg-lightred"),
        (Color::LightGreen, "rlevo-fg-lightgreen"),
        (Color::LightYellow, "rlevo-fg-lightyellow"),
        (Color::LightBlue, "rlevo-fg-lightblue"),
        (Color::LightMagenta, "rlevo-fg-lightmagenta"),
        (Color::LightCyan, "rlevo-fg-lightcyan"),
        (Color::White, "rlevo-fg-white"),
    ];
    let table_bg = [
        (Color::Black, "rlevo-bg-black"),
        (Color::Red, "rlevo-bg-red"),
        (Color::Green, "rlevo-bg-green"),
        (Color::Yellow, "rlevo-bg-yellow"),
        (Color::Blue, "rlevo-bg-blue"),
        (Color::Magenta, "rlevo-bg-magenta"),
        (Color::Cyan, "rlevo-bg-cyan"),
        (Color::Gray, "rlevo-bg-gray"),
        (Color::DarkGray, "rlevo-bg-darkgray"),
        (Color::LightRed, "rlevo-bg-lightred"),
        (Color::LightGreen, "rlevo-bg-lightgreen"),
        (Color::LightYellow, "rlevo-bg-lightyellow"),
        (Color::LightBlue, "rlevo-bg-lightblue"),
        (Color::LightMagenta, "rlevo-bg-lightmagenta"),
        (Color::LightCyan, "rlevo-bg-lightcyan"),
        (Color::White, "rlevo-bg-white"),
    ];
    let table = if foreground { &table_fg[..] } else { &table_bg[..] };
    table.iter().find(|(c, _)| *c == color).map(|(_, cls)| *cls)
}

/// Decode the [`Modifier`] bit-set into one or more CSS classes. Bit
/// layout mirrors `rlevo_core::render::Modifier` (BOLD=1, DIM=2,
/// ITALIC=4, UNDERLINED=8, REVERSED=16). Unknown high bits are ignored.
#[must_use]
pub fn modifier_classes(modifier: Modifier) -> Vec<&'static str> {
    let mut out = Vec::new();
    let bits = modifier.0;
    if bits & 0b0000_0001 != 0 {
        out.push("rlevo-mod-bold");
    }
    if bits & 0b0000_0010 != 0 {
        out.push("rlevo-mod-dim");
    }
    if bits & 0b0000_0100 != 0 {
        out.push("rlevo-mod-italic");
    }
    if bits & 0b0000_1000 != 0 {
        out.push("rlevo-mod-underlined");
    }
    if bits & 0b0001_0000 != 0 {
        out.push("rlevo-mod-reversed");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_class_known_variant_resolves() {
        assert_eq!(color_class(Color::Cyan, true), Some("rlevo-fg-cyan"));
        assert_eq!(color_class(Color::Cyan, false), Some("rlevo-bg-cyan"));
        assert_eq!(
            color_class(Color::DarkGray, true),
            Some("rlevo-fg-darkgray")
        );
    }

    #[test]
    fn color_class_reset_and_indexed_are_none() {
        assert_eq!(color_class(Color::Reset, true), None);
        assert_eq!(color_class(Color::Indexed(42), true), None);
        assert_eq!(color_class(Color::Indexed(42), false), None);
    }

    #[test]
    fn modifier_classes_decode_each_bit() {
        assert_eq!(modifier_classes(Modifier(0)), Vec::<&str>::new());
        assert_eq!(modifier_classes(Modifier(1)), vec!["rlevo-mod-bold"]);
        assert_eq!(modifier_classes(Modifier(16)), vec!["rlevo-mod-reversed"]);
        assert_eq!(
            modifier_classes(Modifier(1 | 16)),
            vec!["rlevo-mod-bold", "rlevo-mod-reversed"]
        );
        assert_eq!(
            modifier_classes(Modifier(0xFF)),
            vec![
                "rlevo-mod-bold",
                "rlevo-mod-dim",
                "rlevo-mod-italic",
                "rlevo-mod-underlined",
                "rlevo-mod-reversed",
            ]
        );
    }

    #[test]
    fn span_classes_combines_fg_modifier() {
        let style = SpanStyle {
            fg: Some(Color::Green),
            bg: None,
            modifier: Modifier(1),
        };
        assert_eq!(span_classes(&style), "rlevo-fg-green rlevo-mod-bold");
    }

    #[test]
    fn span_classes_empty_when_default() {
        let style = SpanStyle::default();
        assert_eq!(span_classes(&style), "");
    }

    #[test]
    fn span_classes_skips_reset_color() {
        let style = SpanStyle {
            fg: Some(Color::Reset),
            bg: Some(Color::Reset),
            modifier: Modifier(0),
        };
        assert_eq!(span_classes(&style), "");
    }
}
