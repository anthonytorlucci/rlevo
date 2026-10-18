//! Scrolling log panel.
//!
//! Reads from [`AppState::log_ring`] and renders the most recent lines
//! that fit in the available height. Each line is styled by severity via
//! [`log_level_style`]; ERROR carries the hazard pairing (`Red` +
//! `REVERSED`) and WARN carries `BOLD`, so severity is readable under
//! any colour-vision condition.
//!
//! No keyboard scrolling — the panel is read-only and the user always
//! sees the tail of the ring.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier as RatModifier, Style as RatStyle};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget};

use crate::tui::state::{AppState, CapturedLogLine};
use crate::tui::theme::{log_level_label, log_level_style};

/// Renders the tail of the log ring within the supplied area.
#[derive(Debug, Clone, Copy)]
pub struct LogPanel<'a> {
    state: &'a AppState,
}

impl<'a> LogPanel<'a> {
    /// Construct a panel viewing the supplied state.
    #[must_use]
    pub const fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl Widget for LogPanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.state.log_ring.is_empty() {
            let dim = RatStyle::default().add_modifier(RatModifier::DIM);
            Paragraph::new("(no log lines yet)")
                .style(dim)
                .render(area, buf);
            return;
        }

        // Show only the lines that fit: take the tail of the ring sized
        // to `area.height`.
        let visible = area.height as usize;
        let total = self.state.log_ring.len();
        let start = total.saturating_sub(visible);

        let lines: Vec<Line<'static>> = self
            .state
            .log_ring
            .iter()
            .skip(start)
            .map(format_log_line)
            .collect();

        Paragraph::new(Text::from(lines)).render(area, buf);
    }
}

/// Build a single styled line: `"LEVEL target: message"`.
///
/// Public for unit testing the formatting independently of `TestBackend`.
#[doc(hidden)]
#[must_use]
pub fn format_log_line(line: &CapturedLogLine) -> Line<'static> {
    let level_style = log_level_style(line.level);
    Line::from(vec![
        Span::styled(log_level_label(line.level).to_string(), level_style),
        Span::raw(" "),
        Span::raw(line.target.clone()),
        Span::raw(": "),
        Span::styled(line.message.clone(), level_style),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Cell;
    use ratatui::style::Color as RatColor;

    fn buffer_text(buf: &Buffer) -> String {
        buf.content().iter().map(Cell::symbol).collect()
    }

    fn line(level: tracing::Level, target: &str, msg: &str) -> CapturedLogLine {
        CapturedLogLine {
            level,
            target: target.to_string(),
            message: msg.to_string(),
        }
    }

    #[test]
    fn renders_placeholder_when_ring_empty() {
        let state = AppState::default();
        let area = Rect::new(0, 0, 60, 4);
        let mut buf = Buffer::empty(area);
        LogPanel::new(&state).render(area, &mut buf);
        assert!(buffer_text(&buf).contains("no log lines yet"));
    }

    /// Tail-windowing: 10 lines in the ring, 4-row viewport renders the
    /// last 4 lines (indices 6..10).
    #[test]
    fn renders_only_tail_window() {
        let mut state = AppState::default();
        for i in 0..10 {
            state.record_log(line(tracing::Level::INFO, "test", &format!("line-{i}")));
        }
        let area = Rect::new(0, 0, 60, 4);
        let mut buf = Buffer::empty(area);
        LogPanel::new(&state).render(area, &mut buf);

        let text = buffer_text(&buf);
        // The last four lines must all appear.
        assert!(text.contains("line-6"), "missing line-6 in {text:?}");
        assert!(text.contains("line-7"));
        assert!(text.contains("line-8"));
        assert!(text.contains("line-9"));
        // The earliest ones must NOT.
        assert!(!text.contains("line-0"));
        assert!(!text.contains("line-5"));
    }

    /// ERROR lines must carry the hazard pairing (Red + REVERSED) in the
    /// rendered buffer cells.
    #[test]
    fn error_lines_carry_hazard_modifier() {
        let mut state = AppState::default();
        state.record_log(line(tracing::Level::ERROR, "rlevo_rl::ppo", "exploded"));
        let area = Rect::new(0, 0, 60, 1);
        let mut buf = Buffer::empty(area);
        LogPanel::new(&state).render(area, &mut buf);

        // Find the first 'E' cell of "ERROR" and assert on its style.
        let mut found = false;
        for cell in buf.content() {
            if cell.symbol() == "E" {
                assert_eq!(cell.fg, RatColor::Red, "ERROR fg should be Red");
                assert!(
                    cell.modifier.contains(RatModifier::REVERSED),
                    "ERROR cell must carry REVERSED modifier (accessibility)"
                );
                found = true;
                break;
            }
        }
        assert!(found, "no 'E' cell found in rendered ERROR line");
    }

    /// WARN lines carry yellow + BOLD.
    #[test]
    fn warn_lines_carry_bold_modifier() {
        let mut state = AppState::default();
        state.record_log(line(tracing::Level::WARN, "tgt", "careful"));
        let area = Rect::new(0, 0, 60, 1);
        let mut buf = Buffer::empty(area);
        LogPanel::new(&state).render(area, &mut buf);

        let mut found = false;
        for cell in buf.content() {
            if cell.symbol() == "W" {
                assert_eq!(cell.fg, RatColor::Yellow);
                assert!(cell.modifier.contains(RatModifier::BOLD));
                found = true;
                break;
            }
        }
        assert!(found, "no 'W' cell found in rendered WARN line");
    }

    /// `format_log_line` produces a "LEVEL target: message" shape with
    /// the level prefix styled.
    #[test]
    fn format_log_line_has_expected_shape() {
        let l = line(tracing::Level::INFO, "rlevo_rl", "hello world");
        let formatted = format_log_line(&l);
        let joined: String = formatted.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, " INFO rlevo_rl: hello world");
    }

    /// Viewport sized smaller than 1 row degrades gracefully (no panic).
    #[test]
    fn zero_height_does_not_panic() {
        let mut state = AppState::default();
        state.record_log(line(tracing::Level::INFO, "tgt", "anything"));
        let area = Rect::new(0, 0, 60, 0);
        let mut buf = Buffer::empty(area);
        LogPanel::new(&state).render(area, &mut buf);
        assert_eq!(buf.area, area);
    }
}
