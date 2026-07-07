//! Styling helpers for the live TUI panels.
//!
//! Two responsibilities:
//!
//! 1. Bridge the semantic palette in [`rlevo_core::render::palette`] to
//!    ready-made `ratatui::style::Style` values. Panels that need an
//!    "agent" or "hazard" style call into this module rather than mixing
//!    palette constants with conversion functions at every site.
//! 2. Map per-metric and per-log-level names onto distinct colours so the
//!    metric sparklines and the scrolling log panel read at a glance.
//!
//! # Accessibility
//!
//! Every style decision preserves the hue-redundant signalling contract
//! from the project accessibility memory:
//!
//! - Hazard / ERROR levels carry `HAZARD_MODIFIER` (REVERSED) in addition
//!   to the red foreground.
//! - WARN carries `BOLD` in addition to yellow.
//! - The per-metric colour choices are decorative — the panel always
//!   renders a textual label beside the sparkline, so the metric is
//!   identifiable without colour.

use ratatui::style::{Color as RatColor, Modifier as RatModifier, Style as RatStyle};
use rlevo_core::render::palette;

use crate::tui::convert::{color_to_ratatui, modifier_to_ratatui};

/// Style for the controllable agent / primary actor glyph.
#[must_use]
pub fn agent_style() -> RatStyle {
    RatStyle::default()
        .fg(color_to_ratatui(palette::AGENT_FG))
        .add_modifier(modifier_to_ratatui(palette::AGENT_MODIFIER))
}

/// Style for hazards, errors, and failure terminal states. The
/// `REVERSED` modifier is what makes this readable independently of
/// red/green discrimination.
#[must_use]
pub fn hazard_style() -> RatStyle {
    RatStyle::default()
        .fg(color_to_ratatui(palette::HAZARD_FG))
        .add_modifier(modifier_to_ratatui(palette::HAZARD_MODIFIER))
}

/// Default style for the "best-so-far" candidate marker. Mirrors the
/// landscape-render palette so an EA fitness panel reads as consistent
/// with the env-side rendering.
#[must_use]
pub fn best_style() -> RatStyle {
    RatStyle::default()
        .fg(color_to_ratatui(palette::BEST_FG))
        .add_modifier(modifier_to_ratatui(palette::BEST_MODIFIER))
}

/// Per-metric colour picker. Unknown names fall back to
/// [`agent_style`] so a previously-unsupported algorithm still renders
/// readably; the `CANONICAL_METRICS` registry enumerates the names
/// known today.
///
/// Hue redundancy: the metric sparkline always renders a textual label
/// alongside the bars, so this colour is decorative — sole reliance on
/// hue is avoided structurally.
#[must_use]
pub fn metric_style(name: &str) -> RatStyle {
    match name {
        // RL losses — yellow reads as "watch this go down"
        "policy_loss" | "value_loss" | "loss" => RatStyle::default().fg(RatColor::Yellow),
        // Entropy — magenta keeps it distinct from loss
        "entropy" => RatStyle::default().fg(RatColor::Magenta),
        // PPO stability — blue family
        "approx_kl" | "clip_frac" => RatStyle::default().fg(RatColor::LightBlue),
        // EA fitness signals — green family (matches palette::BEST_FG)
        "best_fitness" | "best_fitness_ever" => best_style(),
        "mean_fitness" => RatStyle::default().fg(RatColor::LightGreen),
        "worst_fitness" => RatStyle::default().fg(RatColor::DarkGray),
        // "reward" is intentionally not enumerated — it falls through
        // here and shares the agent style of the reward sparkline.
        // Unknown metric names get the same readable default.
        _ => agent_style(),
    }
}

/// Style applied to a captured log line based on its severity.
///
/// - ERROR → red + REVERSED (the hazard pairing — visible under any
///   colour-vision deficiency).
/// - WARN → yellow + BOLD (BOLD is the redundant signal).
/// - INFO → terminal default.
/// - DEBUG / TRACE → dim gray (low-priority, fade into the background).
#[must_use]
pub fn log_level_style(level: tracing::Level) -> RatStyle {
    match level {
        tracing::Level::ERROR => hazard_style(),
        tracing::Level::WARN => RatStyle::default()
            .fg(RatColor::Yellow)
            .add_modifier(RatModifier::BOLD),
        tracing::Level::INFO => RatStyle::default(),
        tracing::Level::DEBUG | tracing::Level::TRACE => RatStyle::default().fg(RatColor::DarkGray),
    }
}

/// Compact 5-character level label for the log panel (left-padded so
/// the message column starts at the same offset regardless of level).
#[must_use]
pub const fn log_level_label(level: tracing::Level) -> &'static str {
    match level {
        tracing::Level::ERROR => "ERROR",
        tracing::Level::WARN => " WARN",
        tracing::Level::INFO => " INFO",
        tracing::Level::DEBUG => "DEBUG",
        tracing::Level::TRACE => "TRACE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hazard pairing must survive into the ratatui Style — the
    /// REVERSED modifier is the accessibility-critical bit.
    #[test]
    fn hazard_style_carries_reversed_modifier() {
        let style = hazard_style();
        assert_eq!(style.fg, Some(RatColor::Red));
        assert!(style.add_modifier.contains(RatModifier::REVERSED));
    }

    #[test]
    fn agent_style_carries_bold_modifier() {
        let style = agent_style();
        assert_eq!(style.fg, Some(RatColor::Cyan));
        assert!(style.add_modifier.contains(RatModifier::BOLD));
    }

    /// Known metric names map to distinct foreground colours.
    #[test]
    fn metric_style_distinguishes_known_names() {
        let loss = metric_style("policy_loss");
        let entropy = metric_style("entropy");
        let kl = metric_style("approx_kl");
        assert_ne!(loss.fg, entropy.fg);
        assert_ne!(entropy.fg, kl.fg);
        assert_ne!(loss.fg, kl.fg);
    }

    /// Unknown names fall back gracefully — they get the agent style
    /// rather than panicking or producing an "unset" style.
    #[test]
    fn metric_style_unknown_falls_back_to_agent() {
        let unknown = metric_style("custom_metric");
        assert_eq!(unknown.fg, agent_style().fg);
        assert_eq!(unknown.add_modifier, agent_style().add_modifier);
    }

    #[test]
    fn log_level_error_carries_hazard_pairing() {
        let style = log_level_style(tracing::Level::ERROR);
        assert_eq!(style.fg, Some(RatColor::Red));
        assert!(style.add_modifier.contains(RatModifier::REVERSED));
    }

    #[test]
    fn log_level_warn_carries_bold() {
        let style = log_level_style(tracing::Level::WARN);
        assert_eq!(style.fg, Some(RatColor::Yellow));
        assert!(style.add_modifier.contains(RatModifier::BOLD));
    }

    #[test]
    fn log_level_info_is_default() {
        let style = log_level_style(tracing::Level::INFO);
        assert_eq!(style, RatStyle::default());
    }

    #[test]
    fn log_level_debug_is_dim_gray() {
        let style = log_level_style(tracing::Level::DEBUG);
        assert_eq!(style.fg, Some(RatColor::DarkGray));
    }

    /// All level labels are exactly five characters so the log panel's
    /// message column aligns regardless of severity.
    #[test]
    fn log_level_labels_are_uniform_width() {
        for level in [
            tracing::Level::ERROR,
            tracing::Level::WARN,
            tracing::Level::INFO,
            tracing::Level::DEBUG,
            tracing::Level::TRACE,
        ] {
            assert_eq!(log_level_label(level).len(), 5, "level: {level:?}");
        }
    }
}
