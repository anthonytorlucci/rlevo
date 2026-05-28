//! Static placeholder shown when the wrapped env can't render in the
//! library tier (locomotion envs have no `AsciiRenderable` impl).
//!
//! The placeholder is caller-set via
//! [`PanelMode::LocomotionPlaceholder`](crate::tui::state::PanelMode);
//! family auto-detection is left to the caller.

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::Style;
use ratatui::widgets::{Paragraph, Widget, Wrap};

/// Spec §7 placeholder text, line-by-line.
///
/// Joined with `'\n'` and rendered as a centred [`Paragraph`]. The render
/// thread wraps this in a bordered [`Block`](ratatui::widgets::Block);
/// the panel itself stays bare so the parent owns layout/title decisions.
const PLACEHOLDER_LINES: &[&str] = &[
    "Locomotion runs render only",
    "in the static report.",
    "",
    "Enable the report tier:",
    "  --features viz-report",
    "  --features record",
    "",
    "Then open the run's",
    "index.html in a browser.",
];

/// Renders the locomotion placeholder text.
///
/// Stateless: nothing to construct, nothing to borrow. Construct with
/// `LocomotionPlaceholder` and pass directly to `frame.render_widget`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LocomotionPlaceholder;

impl LocomotionPlaceholder {
    /// Body text the panel will render. Exposed so tests in sibling
    /// panels can assert the placeholder line set without re-typing it.
    #[must_use]
    pub const fn lines() -> &'static [&'static str] {
        PLACEHOLDER_LINES
    }
}

impl Widget for LocomotionPlaceholder {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let body = PLACEHOLDER_LINES.join("\n");
        Paragraph::new(body)
            .style(Style::default())
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: false })
            .render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk the buffer and concatenate non-empty cells into a single
    /// string — handy for asserting which words landed regardless of
    /// position.
    fn buffer_text(buf: &Buffer) -> String {
        buf.content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn placeholder_lines_match_spec_text() {
        // First and last lines are the load-bearing assertion: the user
        // must learn (a) that nothing will render here and (b) where to
        // look for the canonical view.
        assert_eq!(LocomotionPlaceholder::lines()[0], "Locomotion runs render only");
        assert!(LocomotionPlaceholder::lines()
            .iter()
            .any(|line| line.contains("--features viz-report")));
    }

    #[test]
    fn placeholder_renders_key_phrases() {
        let area = Rect::new(0, 0, 40, 12);
        let mut buf = Buffer::empty(area);
        LocomotionPlaceholder.render(area, &mut buf);
        let text = buffer_text(&buf);

        assert!(text.contains("Locomotion"), "missing 'Locomotion' in {text:?}");
        assert!(text.contains("static report"), "missing 'static report' in {text:?}");
        assert!(text.contains("viz-report"), "missing feature flag hint in {text:?}");
    }

    /// Narrow viewport (sparkline-sized) still renders without panic,
    /// even if wrapping truncates phrases.
    #[test]
    fn placeholder_survives_narrow_area() {
        let area = Rect::new(0, 0, 12, 8);
        let mut buf = Buffer::empty(area);
        LocomotionPlaceholder.render(area, &mut buf);
        // No assertion on contents; we only care that render didn't
        // panic or write outside the area.
        assert_eq!(buf.area, area);
    }
}
