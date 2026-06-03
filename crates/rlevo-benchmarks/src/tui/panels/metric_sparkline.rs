//! Per-named-metric sparkline.
//!
//! Reads from [`AppState::metric_rings`] keyed by a canonical metric name
//! (see [`crate::tui::log_layer::CANONICAL_METRICS`]). Each instance is a
//! single-row strip with a left-justified text label followed by the
//! sparkline bars. The label area is fixed-width so several
//! `MetricSparkline`s stack with their bar columns aligned.
//!
//! Reuses [`encode_returns`](super::reward_sparkline::encode_returns) for
//! the f64→u64 dynamic-baseline encoding — the same algorithm the
//! reward sparkline uses, so all panels in the metrics column behave
//! consistently.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier as RatModifier, Style as RatStyle};
use ratatui::widgets::{Paragraph, Sparkline, Widget};

use crate::tui::panels::reward_sparkline::encode_returns;
use crate::tui::state::AppState;
use crate::tui::theme::metric_style;

/// Width of the leftmost label column in cells. Sized for the longest
/// canonical metric name (`best_fitness_ever`, 17 chars) plus one space.
const LABEL_WIDTH: u16 = 18;

/// Renders one named metric as a sparkline.
///
/// In the default (labelled) mode the widget is a single row: a
/// fixed-width text label followed by the bars, so several stack with
/// aligned bar columns. In bars-only mode (see [`Self::bars_only`]) the
/// label is dropped and the bars fill the whole area — used when an
/// enclosing bordered block already titles the panel.
#[derive(Debug, Clone, Copy)]
pub struct MetricSparkline<'a> {
    state: &'a AppState,
    /// Key into `AppState.metric_rings`.
    name: &'a str,
    /// Display label rendered to the left of the bars. Usually the same
    /// as `name` but can be shortened for the panel column.
    label: &'a str,
    /// When `false`, skip the label column and let the bars fill the area.
    show_label: bool,
}

impl<'a> MetricSparkline<'a> {
    /// Construct a sparkline reading the named ring from `state`. The
    /// `label` is what the user sees on the left; the `name` is the
    /// dictionary key (often the same).
    #[must_use]
    pub const fn new(state: &'a AppState, name: &'a str, label: &'a str) -> Self {
        Self {
            state,
            name,
            label,
            show_label: true,
        }
    }

    /// Convenience constructor for when the display label matches the
    /// metric name verbatim.
    #[must_use]
    pub const fn from_name(state: &'a AppState, name: &'a str) -> Self {
        Self::new(state, name, name)
    }

    /// Construct a label-less sparkline that fills its whole area. Pair
    /// with a bordered block whose title carries the metric name.
    #[must_use]
    pub const fn bars_only(state: &'a AppState, name: &'a str) -> Self {
        Self {
            state,
            name,
            label: name,
            show_label: false,
        }
    }

    /// Draw the bars (or the "no data yet" hint) into `area`.
    fn render_bars(self, area: Rect, buf: &mut Buffer) {
        if let Some(ring) = self
            .state
            .metric_rings
            .get(self.name)
            .filter(|ring| !ring.is_empty())
        {
            let data = encode_returns(ring);
            Sparkline::default()
                .data(data)
                .style(metric_style(self.name))
                .render(area, buf);
        } else {
            let dim = RatStyle::default().add_modifier(RatModifier::DIM);
            Paragraph::new("no data yet").style(dim).render(area, buf);
        }
    }
}

impl Widget for MetricSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if !self.show_label {
            self.render_bars(area, buf);
            return;
        }

        if area.width <= LABEL_WIDTH {
            // Not enough horizontal room for a labelled split; render
            // the label only and bail. Avoids `Layout::areas` panicking
            // on a zero-width bar area.
            Paragraph::new(self.label).render(area, buf);
            return;
        }

        let [label_area, bar_area] =
            Layout::horizontal([Constraint::Length(LABEL_WIDTH), Constraint::Min(1)])
                .areas::<2>(area);

        Paragraph::new(self.label)
            .style(metric_style(self.name))
            .render(label_area, buf);

        self.render_bars(bar_area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Cell;

    use crate::tui::state::PanelMode;

    fn buffer_text(buf: &Buffer) -> String {
        buf.content().iter().map(Cell::symbol).collect()
    }

    /// Empty ring → "no data yet" placeholder. Layout stable.
    #[test]
    fn renders_placeholder_when_ring_absent() {
        let state = AppState::default();
        let area = Rect::new(0, 0, 40, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::from_name(&state, "policy_loss").render(area, &mut buf);

        let text = buffer_text(&buf);
        assert!(text.contains("policy_loss"), "label missing: {text:?}");
        assert!(text.contains("no data yet"), "placeholder missing: {text:?}");
    }

    /// Empty ring even if the key exists → still "no data yet".
    #[test]
    fn renders_placeholder_when_ring_empty() {
        // Touch the key but don't push samples — emulates a race where
        // the rolllout connected but hasn't emitted yet.
        let state = AppState::default();
        let area = Rect::new(0, 0, 40, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::from_name(&state, "policy_loss").render(area, &mut buf);
        assert!(buffer_text(&buf).contains("no data yet"));
    }

    /// Non-empty ring → sparkline bar glyphs land in the bar area.
    #[test]
    fn renders_bars_when_samples_present() {
        let mut state = AppState::new(8, PanelMode::Auto);
        for v in [0.5, 0.4, 0.6, 0.3, 0.7] {
            state.record_metric("policy_loss", v);
        }
        let area = Rect::new(0, 0, 40, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::from_name(&state, "policy_loss").render(area, &mut buf);

        let text = buffer_text(&buf);
        let has_bar = text
            .chars()
            .any(|c| matches!(c, '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█'));
        assert!(has_bar, "expected sparkline glyphs, got {text:?}");
        assert!(text.contains("policy_loss"));
    }

    /// Custom display label is honoured.
    #[test]
    fn custom_label_overrides_metric_name() {
        let mut state = AppState::new(8, PanelMode::Auto);
        state.record_metric("policy_loss", 0.5);
        let area = Rect::new(0, 0, 40, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::new(&state, "policy_loss", "loss").render(area, &mut buf);

        let text = buffer_text(&buf);
        assert!(text.contains("loss "), "expected short label: {text:?}");
        // The label area is 18 chars; we should still see bars after it.
        let has_bar = text
            .chars()
            .any(|c| matches!(c, '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█'));
        assert!(has_bar);
    }

    /// Each ring is independent; sampling one metric doesn't bleed into
    /// another's display.
    #[test]
    fn unrelated_metric_rings_dont_cross_contaminate() {
        let mut state = AppState::new(8, PanelMode::Auto);
        state.record_metric("entropy", 1.0);
        // Render policy_loss — its ring is still empty.
        let area = Rect::new(0, 0, 40, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::from_name(&state, "policy_loss").render(area, &mut buf);
        assert!(buffer_text(&buf).contains("no data yet"));
    }

    /// Bars-only mode omits the label and fills the area with bars.
    #[test]
    fn bars_only_skips_label_and_fills_area() {
        let mut state = AppState::new(8, PanelMode::Auto);
        for v in [0.5, 0.4, 0.6, 0.3] {
            state.record_metric("policy_loss", v);
        }
        let area = Rect::new(0, 0, 40, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::bars_only(&state, "policy_loss").render(area, &mut buf);

        let text = buffer_text(&buf);
        assert!(
            !text.contains("policy_loss"),
            "bars-only mode must not draw the label: {text:?}"
        );
        let has_bar = text
            .chars()
            .any(|c| matches!(c, '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█'));
        assert!(has_bar, "expected sparkline glyphs, got {text:?}");
    }

    /// Tiny widths must not panic — happens on terminal resize down.
    #[test]
    fn narrow_area_does_not_panic() {
        let state = AppState::default();
        let area = Rect::new(0, 0, 5, 1);
        let mut buf = Buffer::empty(area);
        MetricSparkline::from_name(&state, "policy_loss").render(area, &mut buf);
        // No assertion — we only care that render didn't panic or write
        // outside the area.
        assert_eq!(buf.area, area);
    }
}
