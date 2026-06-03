//! Reward sparkline panel.
//!
//! Reads the bounded reward ring from
//! [`AppState`](crate::tui::state::AppState) and renders a horizontal
//! sparkline using `ratatui::widgets::Sparkline`. The sparkline widget
//! consumes `u64`, but episode returns are `f64` and can be negative
//! (`MountainCar` emits -1 per step). The conversion in [`encode_returns`]
//! shifts the visible window so the minimum sits at 0 and scales by a
//! fixed integer factor so the sparkline's bar quantization still has
//! useful resolution on small-magnitude data.
//!
//! Storing raw `f64`s in [`AppState`] and converting at render time —
//! rather than pre-encoding into `u64` at push time — lets the baseline
//! adjust dynamically as the ring window slides.

use std::collections::VecDeque;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Paragraph, Sparkline, Widget};

use crate::tui::convert::color_to_ratatui;
use crate::tui::state::AppState;

use rlevo_core::render::palette;

/// Multiplier applied to the (shifted) return before truncating to `u64`.
///
/// Set high enough that quantization to integer bar heights still resolves
/// small differences. Sparkline normalizes by max internally, so absolute
/// magnitude doesn't matter — only the precision floor does.
const REWARD_SCALE: f64 = 1_000.0;

/// Renders the reward-ring as a sparkline.
#[derive(Debug, Clone, Copy)]
pub struct RewardSparkline<'a> {
    state: &'a AppState,
}

impl<'a> RewardSparkline<'a> {
    /// Construct a sparkline viewing the supplied state.
    #[must_use]
    pub const fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl Widget for RewardSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.state.reward_ring.is_empty() {
            Paragraph::new("no episodes yet").render(area, buf);
            return;
        }

        let data = encode_returns(&self.state.reward_ring);
        Sparkline::default()
            .data(data)
            .style(Style::default().fg(color_to_ratatui(palette::AGENT_FG)))
            .render(area, buf);
    }
}

/// Convert episode returns into the `u64` sequence the sparkline widget
/// consumes.
///
/// Behaviour:
///
/// - Empty input → empty output.
/// - All returns equal → uniform mid-range bars (every entry is
///   [`REWARD_SCALE`] as `u64`), so the sparkline reads as a flat band
///   rather than a phantom-empty plot.
/// - Mixed returns → shifted so the smallest is `0` and scaled by
///   [`REWARD_SCALE`]; NaN entries collapse to `0`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "shifted values are clamped non-negative; quantization to u64 is the desired sparkline encoding"
)]
pub fn encode_returns(ring: &VecDeque<f64>) -> Vec<u64> {
    if ring.is_empty() {
        return Vec::new();
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in ring {
        if v.is_nan() {
            continue;
        }
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }

    // No finite samples; render zeroes rather than NaN-poisoning the cast.
    if !min.is_finite() {
        return vec![0u64; ring.len()];
    }

    if (max - min).abs() < f64::EPSILON {
        // Flat history — render as a constant band so the user can tell
        // the panel is alive but the optimiser is stuck.
        return vec![REWARD_SCALE as u64; ring.len()];
    }

    ring.iter()
        .map(|&v| {
            if v.is_nan() {
                0u64
            } else {
                let shifted = (v - min) * REWARD_SCALE;
                // Clamp negatives to zero defensively; `shifted` is
                // already non-negative when min ≤ v.
                shifted.max(0.0) as u64
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tests intentionally compare against the same u64-encoded scale used at render time"
)]
mod tests {
    use super::*;
    use ratatui::buffer::Cell;

    fn buffer_text(buf: &Buffer) -> String {
        buf.content().iter().map(Cell::symbol).collect()
    }

    fn ring_of<I: IntoIterator<Item = f64>>(iter: I) -> VecDeque<f64> {
        iter.into_iter().collect()
    }

    #[test]
    fn encode_returns_empty_yields_empty() {
        assert!(encode_returns(&VecDeque::new()).is_empty());
    }

    #[test]
    fn encode_returns_shifts_to_non_negative() {
        let ring = ring_of([-5.0, 0.0, 5.0]);
        let out = encode_returns(&ring);
        assert_eq!(out.len(), 3);
        // Min becomes 0; max becomes (max - min) * scale.
        assert_eq!(out[0], 0);
        assert_eq!(out[2], (10.0 * REWARD_SCALE) as u64);
        // Monotonicity preserved.
        assert!(out[0] <= out[1] && out[1] <= out[2]);
    }

    #[test]
    fn encode_returns_flat_window_yields_constant_band() {
        let ring = ring_of([4.2; 6]);
        let out = encode_returns(&ring);
        let expected = REWARD_SCALE as u64;
        assert!(out.iter().all(|&v| v == expected));
    }

    #[test]
    fn encode_returns_handles_nan_without_poison() {
        let ring = ring_of([1.0, f64::NAN, 2.0]);
        let out = encode_returns(&ring);
        assert_eq!(out.len(), 3);
        // NaN entry should not propagate.
        assert_eq!(out[1], 0);
        // The other two should be finite & non-NaN.
        assert_ne!(out[0], u64::MAX);
        assert_ne!(out[2], u64::MAX);
    }

    #[test]
    fn renders_hint_before_any_episodes() {
        let state = AppState::default();
        let area = Rect::new(0, 0, 30, 3);
        let mut buf = Buffer::empty(area);
        RewardSparkline::new(&state).render(area, &mut buf);
        assert!(buffer_text(&buf).contains("no episodes yet"));
    }

    /// With a small visible ring, the sparkline writes its bar glyphs
    /// somewhere in the area — we just assert *something* non-blank
    /// landed, not the exact glyph row.
    #[test]
    fn renders_bars_when_returns_present() {
        let mut state = AppState::new(8);
        for v in [1.0, 5.0, 3.0, 8.0] {
            state.record_episode_end(0, v);
        }
        let area = Rect::new(0, 0, 16, 3);
        let mut buf = Buffer::empty(area);
        RewardSparkline::new(&state).render(area, &mut buf);

        let text = buffer_text(&buf);
        let has_bar = text
            .chars()
            .any(|c| matches!(c, '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█'));
        assert!(has_bar, "expected sparkline glyphs, got {text:?}");
    }
}
