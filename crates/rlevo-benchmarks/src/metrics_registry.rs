//! Canonical metric field-name registry shared by all tracing layers.
//!
//! Both the live-TUI layer (`TuiCaptureLayer`) and
//! the on-disk recording layer (`RecordingLayer`)
//! extract the same named fields from `tracing` events. Keeping the registry
//! here means a single edit covers both surfaces — the two layers just
//! import [`CANONICAL_METRICS`] instead of duplicating it.

/// Field names recognised as chartable metrics across all tracing layers.
///
/// Any `tracing` event field whose name appears in this list is treated as
/// a numeric metric sample. Everything else is captured only as a log line
/// (TUI) or ignored (record layer).
///
/// # Extending the registry
///
/// Add the field name here when a new algorithm starts emitting a metric.
/// That single edit makes the field visible in both the live TUI sparklines
/// and the on-disk recording stream.
pub const CANONICAL_METRICS: &[&str] = &[
    // RL training stats
    "policy_loss",
    "value_loss",
    "loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    // RL value/policy diagnostics (added in FORMAT_VERSION = 6).
    "explained_variance",
    "old_approx_kl",
    // Per-iteration training stats (v6).
    "episode_return_mean",
    "episode_return_std",
    "episode_return_min",
    "episode_return_max",
    "episode_length_mean",
    "env_steps_sampled",
    "steps_per_sec",
    "learning_rate",
    // Per-episode terminal triple emitted by the record writer (v6).
    "episode_return",
    "episode_length",
    "episode_wall_clock_secs",
    // DQN family (v6).
    "td_loss",
    "q_values",
    // SAC family (v6).
    "qf1_loss",
    "qf2_loss",
    "actor_loss",
    "alpha",
    "alpha_loss",
    // Schedules (v6).
    "clip_range",
    "n_updates",
    // Evolution training stats emitted by `EvolutionaryHarness`.
    "best_fitness",
    "mean_fitness",
    "worst_fitness",
    "best_fitness_ever",
];

/// `true` if `name` is a recognised metric field name.
#[must_use]
pub fn is_canonical_metric(name: &str) -> bool {
    CANONICAL_METRICS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_are_recognised() {
        for name in CANONICAL_METRICS {
            assert!(is_canonical_metric(name), "{name} should be canonical");
        }
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert!(!is_canonical_metric("batch_size"));
        assert!(!is_canonical_metric("not_a_metric"));
        assert!(!is_canonical_metric(""));
    }

    #[test]
    fn v6_additions_are_recognised() {
        for name in [
            "explained_variance",
            "old_approx_kl",
            "episode_return_mean",
            "env_steps_sampled",
            "steps_per_sec",
            "learning_rate",
            "episode_return",
            "episode_length",
            "episode_wall_clock_secs",
            "td_loss",
            "q_values",
            "qf1_loss",
            "qf2_loss",
            "actor_loss",
            "alpha",
            "alpha_loss",
            "clip_range",
            "n_updates",
        ] {
            assert!(is_canonical_metric(name), "{name} should be canonical");
        }
    }
}
