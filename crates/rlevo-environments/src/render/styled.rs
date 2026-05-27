//! Styled-output primitives consumed by the live TUI and report tiers.
//!
//! These types form the second projection of [`AsciiRenderable`](super::AsciiRenderable):
//! `render_ascii` returns a plain `String` for logs, snapshot tests, and
//! `EpisodeRecord.ascii`, while `render_styled` returns a [`StyledFrame`]
//! carrying foreground/background colour and modifier hints. The type set is
//! intentionally a small subset of the ratatui vocabulary so that
//! `rlevo-environments` ships zero terminal-side dependencies — the
//! `From<StyledFrame>` conversion into ratatui types lives in
//! `rlevo-benchmarks::tui` (behind the `tui` feature), and the report tier
//! deserialises [`StyledFrame`] from `EpisodeRecord`.
//!
//! No truecolor: stick to the 16-colour ANSI palette plus indexed 256-colour.
//! See [`super::palette`] for the project-wide semantic constants every
//! environment impl should use rather than reaching for raw [`Color`] values.

use std::ops::{BitOr, BitOrAssign};

use serde::{Deserialize, Serialize};

/// A multi-line styled projection of an environment frame.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledFrame {
    /// Lines in source order. Empty when the frame carries no content.
    pub lines: Vec<StyledLine>,
}

impl StyledFrame {
    /// Construct an unstyled frame from a plain string, splitting on `\n`.
    ///
    /// Every line becomes a single span with the default style. Used by the
    /// default `AsciiRenderable::render_styled` impl so that environments
    /// without bespoke colouring still produce a well-typed frame.
    #[must_use]
    pub fn unstyled(s: String) -> Self {
        if s.is_empty() {
            return Self { lines: Vec::new() };
        }
        let lines = s.split('\n').map(StyledLine::unstyled).collect();
        Self { lines }
    }

    /// `true` when the frame contains no lines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Concatenate every span's text across every line, separated by `\n`.
    ///
    /// Useful in tests to assert that the styled projection carries the same
    /// glyphs as the plain projection (modulo trailing newlines).
    #[must_use]
    pub fn plain_text(&self) -> String {
        let mut out = String::new();
        for (i, line) in self.lines.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            for span in &line.spans {
                out.push_str(&span.text);
            }
        }
        out
    }
}

/// A single line of styled spans.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledLine {
    /// Spans in source order. Concatenating each span's text yields the plain
    /// line content.
    pub spans: Vec<StyledSpan>,
}

impl StyledLine {
    /// Build a line carrying a single unstyled span.
    #[must_use]
    pub fn unstyled(s: impl Into<String>) -> Self {
        Self {
            spans: vec![StyledSpan {
                text: s.into(),
                style: SpanStyle::default(),
            }],
        }
    }

    /// Build a line from any iterable of spans.
    #[must_use]
    pub fn from_spans<I: IntoIterator<Item = StyledSpan>>(spans: I) -> Self {
        Self {
            spans: spans.into_iter().collect(),
        }
    }
}

/// A run of text with a uniform style.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledSpan {
    /// The text content. Never contains `\n` — line breaks are owned by the
    /// enclosing [`StyledFrame`].
    pub text: String,
    /// Style applied to every character in `text`.
    pub style: SpanStyle,
}

impl StyledSpan {
    /// Build a span with the supplied text and style.
    #[must_use]
    pub fn new(text: impl Into<String>, style: SpanStyle) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    /// Build an unstyled span.
    #[must_use]
    pub fn raw(text: impl Into<String>) -> Self {
        Self::new(text, SpanStyle::default())
    }
}

/// Foreground / background colour and modifier bits applied to a span.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanStyle {
    /// Foreground colour. `None` means "use the terminal's default."
    pub fg: Option<Color>,
    /// Background colour. `None` means "use the terminal's default."
    pub bg: Option<Color>,
    /// Modifier bits (bold, italic, etc.) applied on top of the colour.
    pub modifier: Modifier,
}

impl SpanStyle {
    /// Set the foreground colour.
    #[must_use]
    pub const fn fg(mut self, c: Color) -> Self {
        self.fg = Some(c);
        self
    }

    /// Set the background colour.
    #[must_use]
    pub const fn bg(mut self, c: Color) -> Self {
        self.bg = Some(c);
        self
    }

    /// Add the `BOLD` modifier.
    #[must_use]
    pub const fn bold(mut self) -> Self {
        self.modifier = self.modifier.union(Modifier::BOLD);
        self
    }

    /// Add the `DIM` modifier.
    #[must_use]
    pub const fn dim(mut self) -> Self {
        self.modifier = self.modifier.union(Modifier::DIM);
        self
    }

    /// Add the `ITALIC` modifier.
    #[must_use]
    pub const fn italic(mut self) -> Self {
        self.modifier = self.modifier.union(Modifier::ITALIC);
        self
    }

    /// Add the `UNDERLINED` modifier.
    #[must_use]
    pub const fn underlined(mut self) -> Self {
        self.modifier = self.modifier.union(Modifier::UNDERLINED);
        self
    }

    /// Add the `REVERSED` modifier (swap foreground/background — pairs with
    /// `HAZARD_FG` to give a hue-redundant signal for deuteranopic users).
    #[must_use]
    pub const fn reversed(mut self) -> Self {
        self.modifier = self.modifier.union(Modifier::REVERSED);
        self
    }

    /// Union an existing [`Modifier`] into the style.
    #[must_use]
    pub const fn with_modifier(mut self, m: Modifier) -> Self {
        self.modifier = self.modifier.union(m);
        self
    }
}

/// 16-colour ANSI palette plus indexed 256-colour escape.
///
/// Truecolor (24-bit RGB) is intentionally absent — the spec restricts the
/// library tier to the portable ANSI subset so that any compliant terminal
/// renders the live TUI identically.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    /// Reset to the terminal's default colour.
    Reset,
    /// ANSI black (0).
    Black,
    /// ANSI red (1). Reserve for hazards via [`super::palette::HAZARD_FG`].
    Red,
    /// ANSI green (2). Reserve for goals via [`super::palette::GOAL_FG`].
    Green,
    /// ANSI yellow (3).
    Yellow,
    /// ANSI blue (4).
    Blue,
    /// ANSI magenta (5).
    Magenta,
    /// ANSI cyan (6).
    Cyan,
    /// ANSI bright black / gray (8).
    Gray,
    /// ANSI dim gray (the "bright black" alternate on some palettes).
    DarkGray,
    /// ANSI bright red (9).
    LightRed,
    /// ANSI bright green (10).
    LightGreen,
    /// ANSI bright yellow (11).
    LightYellow,
    /// ANSI bright blue (12).
    LightBlue,
    /// ANSI bright magenta (13).
    LightMagenta,
    /// ANSI bright cyan (14).
    LightCyan,
    /// ANSI white (15).
    White,
    /// Indexed 256-colour palette entry.
    Indexed(u8),
}

/// Bitset of style modifiers (bold, italic, etc.).
///
/// Implemented as a plain `u8` rather than `bitflags!` to keep the crate's
/// dependency cone untouched. The bit layout is private; use the named
/// constants and `BitOr` operator to compose values.
#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modifier(u8);

impl Modifier {
    /// No modifiers set.
    pub const EMPTY: Self = Self(0);
    /// Bold text.
    pub const BOLD: Self = Self(1 << 0);
    /// Dim / faint text.
    pub const DIM: Self = Self(1 << 1);
    /// Italic text.
    pub const ITALIC: Self = Self(1 << 2);
    /// Underlined text.
    pub const UNDERLINED: Self = Self(1 << 3);
    /// Reversed foreground/background. Pairs with `HAZARD_FG` to add the
    /// hue-redundant signal required by the project accessibility contract.
    pub const REVERSED: Self = Self(1 << 4);

    /// `true` when every bit of `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Return the union of two modifier sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Return the intersection of two modifier sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Insert `other`'s bits in place.
    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Clear `other`'s bits in place.
    pub const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// `true` when no bits are set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for Modifier {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl BitOrAssign for Modifier {
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}

impl std::fmt::Debug for Modifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<&'static str> = Vec::new();
        if self.contains(Self::BOLD) {
            names.push("BOLD");
        }
        if self.contains(Self::DIM) {
            names.push("DIM");
        }
        if self.contains(Self::ITALIC) {
            names.push("ITALIC");
        }
        if self.contains(Self::UNDERLINED) {
            names.push("UNDERLINED");
        }
        if self.contains(Self::REVERSED) {
            names.push("REVERSED");
        }
        if names.is_empty() {
            write!(f, "Modifier::EMPTY")
        } else {
            write!(f, "Modifier({})", names.join(" | "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styled_frame_unstyled_round_trip() {
        let s = String::from("hello\nworld");
        let frame = StyledFrame::unstyled(s.clone());
        assert_eq!(frame.lines.len(), 2);
        assert_eq!(frame.plain_text(), s);
    }

    #[test]
    fn styled_frame_unstyled_empty() {
        let frame = StyledFrame::unstyled(String::new());
        assert!(frame.is_empty());
        assert_eq!(frame.plain_text(), "");
    }

    #[test]
    fn styled_frame_unstyled_single_line() {
        let frame = StyledFrame::unstyled(String::from("solo"));
        assert_eq!(frame.lines.len(), 1);
        assert_eq!(frame.plain_text(), "solo");
    }

    #[test]
    fn modifier_bitops() {
        let bold_italic = Modifier::BOLD | Modifier::ITALIC;
        assert!(bold_italic.contains(Modifier::BOLD));
        assert!(bold_italic.contains(Modifier::ITALIC));
        assert!(!bold_italic.contains(Modifier::REVERSED));

        let intersection = bold_italic.intersection(Modifier::BOLD);
        assert!(intersection.contains(Modifier::BOLD));
        assert!(!intersection.contains(Modifier::ITALIC));

        let mut m = Modifier::EMPTY;
        m |= Modifier::UNDERLINED;
        assert!(m.contains(Modifier::UNDERLINED));
        m.remove(Modifier::UNDERLINED);
        assert!(m.is_empty());
    }

    #[test]
    fn spanstyle_builder_chain() {
        let style: SpanStyle = SpanStyle::default()
            .fg(Color::Cyan)
            .bg(Color::Black)
            .bold()
            .reversed();
        assert_eq!(style.fg, Some(Color::Cyan));
        assert_eq!(style.bg, Some(Color::Black));
        assert!(style.modifier.contains(Modifier::BOLD));
        assert!(style.modifier.contains(Modifier::REVERSED));
        assert!(!style.modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn modifier_debug_lists_names() {
        let m: Modifier = Modifier::BOLD | Modifier::REVERSED;
        let s = format!("{m:?}");
        assert!(s.contains("BOLD"));
        assert!(s.contains("REVERSED"));
    }

    #[test]
    fn modifier_debug_empty() {
        let m = Modifier::EMPTY;
        assert_eq!(format!("{m:?}"), "Modifier::EMPTY");
    }
}
