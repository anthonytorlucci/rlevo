//! Panel widget for the most recently captured environment frame.
//!
//! Dispatches on [`AppState::panel_mode`]:
//!
//! - [`PanelMode::LocomotionPlaceholder`] — defers to
//!   [`LocomotionPlaceholder`].
//! - [`PanelMode::Auto`] with a frame present — renders the styled frame
//!   via [`frame_to_ratatui_ref`].
//! - [`PanelMode::Auto`] with no frame yet — shows a brief hint so the
//!   user knows the dashboard is alive while the first step happens.
//!
//! [`AppState::panel_mode`]: crate::tui::state::AppState
//! [`PanelMode::LocomotionPlaceholder`]: crate::tui::state::PanelMode
//! [`PanelMode::Auto`]: crate::tui::state::PanelMode

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::{Paragraph, Widget};

use crate::tui::convert::frame_to_ratatui_ref;
use crate::tui::panels::LocomotionPlaceholder;
use crate::tui::state::AppState;

/// Renders the currently captured env frame.
#[derive(Debug, Clone, Copy)]
pub struct EnvPanel<'a> {
    state: &'a AppState,
}

impl<'a> EnvPanel<'a> {
    /// Construct a panel viewing the supplied state.
    #[must_use]
    pub const fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl Widget for EnvPanel<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.state.use_locomotion_placeholder() {
            LocomotionPlaceholder.render(area, buf);
            return;
        }

        match self.state.frame.as_ref() {
            Some(frame) => {
                let text = frame_to_ratatui_ref(frame);
                Paragraph::new(text).render(area, buf);
            }
            None => {
                Paragraph::new("waiting for first frame…").render(area, buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Cell;
    use rlevo_core::render::{StyledFrame, StyledLine, StyledSpan};

    use crate::tui::state::PanelMode;

    fn buffer_text(buf: &Buffer) -> String {
        buf.content().iter().map(Cell::symbol).collect()
    }

    fn frame_with(text: &str) -> StyledFrame {
        StyledFrame {
            lines: vec![StyledLine::from_spans([StyledSpan::raw(text)])],
        }
    }

    #[test]
    fn renders_styled_frame_when_present() {
        let mut state = AppState::default();
        state.push_frame(frame_with("hello-cartpole"));
        let area = Rect::new(0, 0, 40, 3);
        let mut buf = Buffer::empty(area);
        EnvPanel::new(&state).render(area, &mut buf);
        assert!(
            buffer_text(&buf).contains("hello-cartpole"),
            "buffer did not contain rendered frame"
        );
    }

    #[test]
    fn shows_waiting_hint_before_first_frame() {
        let state = AppState::default();
        let area = Rect::new(0, 0, 40, 3);
        let mut buf = Buffer::empty(area);
        EnvPanel::new(&state).render(area, &mut buf);
        assert!(
            buffer_text(&buf).contains("waiting"),
            "expected waiting hint, got {:?}",
            buffer_text(&buf)
        );
    }

    /// Locomotion mode wins regardless of whether a frame is held — useful
    /// if upstream wraps a non-locomotion env but the caller explicitly
    /// requested the placeholder.
    #[test]
    fn locomotion_mode_renders_placeholder_even_with_frame() {
        let mut state = AppState::new(8, PanelMode::LocomotionPlaceholder);
        state.push_frame(frame_with("should-not-appear"));
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        EnvPanel::new(&state).render(area, &mut buf);

        let text = buffer_text(&buf);
        assert!(
            text.contains("Locomotion"),
            "placeholder text missing: {text:?}"
        );
        assert!(
            !text.contains("should-not-appear"),
            "auto-mode frame leaked through placeholder"
        );
    }

    #[test]
    fn render_does_not_mutate_held_frame() {
        let mut state = AppState::default();
        state.push_frame(frame_with("persist-me"));
        let area = Rect::new(0, 0, 40, 3);
        let mut buf = Buffer::empty(area);
        EnvPanel::new(&state).render(area, &mut buf);
        // Frame still readable after rendering — the borrow variant of
        // frame_to_ratatui must not have consumed it.
        assert_eq!(
            state.frame.as_ref().unwrap().plain_text(),
            "persist-me"
        );
    }
}
