//! Agent-level opt-in for prioritized replay: [`PrioritizedReplaySettings`].
//!
//! An off-policy value-based agent config holds `Option<PrioritizedReplaySettings>`:
//! `None` selects [`UniformReplay`](super::UniformReplay), `Some` selects
//! [`PrioritizedReplay`](super::PrioritizedReplay). PER is therefore a
//! **config-level** opt-in (ADR 0051 §1), never a type-level one — a
//! deserialized config field can turn it on.
//!
//! # What lives here vs. on the buffer
//!
//! Schaul's α (priority exponent) and ε (priority floor) configure the buffer
//! and are forwarded into a [`PrioritizedReplayConfig`]. The importance-sampling
//! exponent β and its annealing **schedule** do *not* live on the buffer: the
//! buffer has no step counter, and giving it one duplicates the agent's — the
//! second-source-of-truth shape `rules.md` §10 forbids, and outright wrong the
//! moment two learners share one buffer (ADR 0050 §11). So the schedule endpoints
//! live here and the agent passes the *evaluated* β into
//! [`sample`](super::ReplayStrategy::sample) each step via [`beta`](PrioritizedReplaySettings::beta).

use rlevo_core::config::{self, ConfigError, Validate};
use serde::{Deserialize, Serialize};

use super::config::{DEFAULT_PRIORITY_EPSILON, DEFAULT_PRIORITY_EXPONENT, PrioritizedReplayConfig};
use super::importance_exponent::ImportanceExponent;

/// Schaul et al. (2016) Table 3's β start for the proportional variant.
pub const DEFAULT_BETA_START: f32 = 0.4;

/// Schaul et al. (2016) Table 3's annealed β endpoint (`β → 1`).
pub const DEFAULT_BETA_END: f32 = 1.0;

/// Default number of env steps over which β anneals from
/// [`DEFAULT_BETA_START`] to [`DEFAULT_BETA_END`].
///
/// **Not a paper value.** Schaul anneals β "linearly to 1" over the run but
/// gives no fixed step count (it is training-length dependent). `100_000` is a
/// pragmatic default sized to the small discrete-control runs this library
/// ships defaults for; a longer run should scale it up.
pub const DEFAULT_BETA_ANNEAL_STEPS: usize = 100_000;

/// The agent-side prioritized-replay hyperparameters: Schaul's α/ε plus the β
/// importance-sampling schedule the buffer cannot own.
///
/// # Naming
///
/// The Greek letters are spelled out (ADR 0050 §12): `alpha` already means SAC's
/// entropy temperature, and the collision is in the reader's head even where the
/// compiler is untroubled.
///
/// | Schaul symbol | Field |
/// |---|---|
/// | α (priority exponent) | [`priority_exponent`](Self::priority_exponent) |
/// | ε (priority floor) | [`priority_epsilon`](Self::priority_epsilon) |
/// | β₀ (IS exponent start) | [`beta_start`](Self::beta_start) |
/// | β (IS exponent end) | [`beta_end`](Self::beta_end) |
///
/// # β validation is not optional
///
/// [`beta_anneal_steps`](Self::beta_anneal_steps) is `nonzero`-validated. A zero
/// there makes the evaluated progress fraction `0/0 = NaN`, and a `NaN` β
/// poisons the importance weights, then the loss, then every gradient the
/// optimizer touches next — the release-build defect ADR 0051 §3 exists to
/// close. Validation kills the `0/0` at its source; [`beta`](Self::beta) clamps
/// so an in-range value is produced by construction; and
/// [`ImportanceExponent`] is the loud-panic backstop for any residual bug.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrioritizedReplaySettings {
    /// Schaul Eq. 1's α. Must lie in `[0, 1]`; `0.0` is the uniform case.
    /// Defaults to [`DEFAULT_PRIORITY_EXPONENT`] (Table 3, proportional row).
    pub priority_exponent: f32,

    /// Schaul §3.3's ε in `p_i = |δ_i| + ε`. Must be finite and strictly
    /// positive. Defaults to [`DEFAULT_PRIORITY_EPSILON`].
    pub priority_epsilon: f32,

    /// Schaul §3.4's β at step 0. Must lie in `[0, 1]`. Defaults to
    /// [`DEFAULT_BETA_START`] (`0.4`).
    pub beta_start: f32,

    /// Schaul §3.4's annealed β endpoint. Must lie in `[0, 1]`. Defaults to
    /// [`DEFAULT_BETA_END`] (`1.0`) — the paper's "unbiased near convergence".
    pub beta_end: f32,

    /// Number of env steps over which β interpolates linearly from
    /// `beta_start` to `beta_end`.
    ///
    /// Must be non-zero: a zero makes the schedule's progress fraction `0/0`
    /// (ADR 0051 §3). Defaults to [`DEFAULT_BETA_ANNEAL_STEPS`].
    pub beta_anneal_steps: usize,
}

impl Default for PrioritizedReplaySettings {
    fn default() -> Self {
        Self {
            priority_exponent: DEFAULT_PRIORITY_EXPONENT,
            priority_epsilon: DEFAULT_PRIORITY_EPSILON,
            beta_start: DEFAULT_BETA_START,
            beta_end: DEFAULT_BETA_END,
            beta_anneal_steps: DEFAULT_BETA_ANNEAL_STEPS,
        }
    }
}

impl PrioritizedReplaySettings {
    /// Builds the buffer-side [`PrioritizedReplayConfig`] for a replay of
    /// `capacity` transitions, applying this settings' α and ε.
    ///
    /// Capacity is threaded from the agent config's `replay_buffer_capacity`
    /// rather than duplicated here, so there is exactly one capacity knob.
    #[must_use]
    pub fn buffer_config(&self, capacity: usize) -> PrioritizedReplayConfig {
        PrioritizedReplayConfig {
            capacity,
            priority_exponent: self.priority_exponent,
            priority_epsilon: self.priority_epsilon,
        }
    }

    /// Evaluates Schaul §3.4's β schedule at `step`, linearly interpolating from
    /// `beta_start` to `beta_end` over `beta_anneal_steps`.
    ///
    /// The progress fraction is clamped to `[0, 1]`, and the interpolated value
    /// is clamped to `[beta_start, 1.0]` **before** constructing the
    /// [`ImportanceExponent`], so an in-range β is produced by construction
    /// rather than by luck of a limiter's IEEE-754 behaviour (ADR 0051 §3). With
    /// `beta_anneal_steps` non-zero (enforced by [`validate`](Self::validate))
    /// there is no `0/0`, so this is total on any validated settings.
    ///
    /// # Panics
    ///
    /// Panics only if the clamped value is non-finite — reachable solely by
    /// bypassing [`validate`](Self::validate) with a `NaN` endpoint (e.g. a
    /// struct literal), in which case [`ImportanceExponent::new`] panics at this
    /// named site rather than letting a `NaN` reach `powf` twenty steps later.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // step / anneal_steps: exact for the sizes agents reach
    pub fn beta(&self, step: usize) -> ImportanceExponent {
        let frac = (step as f32 / self.beta_anneal_steps as f32).clamp(0.0, 1.0);
        let raw = self.beta_start + (self.beta_end - self.beta_start) * frac;
        ImportanceExponent::new(raw.clamp(self.beta_start, 1.0))
    }
}

impl Validate for PrioritizedReplaySettings {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "PrioritizedReplaySettings";
        config::in_range(
            C,
            "priority_exponent",
            0.0,
            1.0,
            f64::from(self.priority_exponent),
        )?;
        config::positive(C, "priority_epsilon", f64::from(self.priority_epsilon))?;
        config::in_range(C, "beta_start", 0.0, 1.0, f64::from(self.beta_start))?;
        config::in_range(C, "beta_end", 0.0, 1.0, f64::from(self.beta_end))?;
        config::nonzero(C, "beta_anneal_steps", self.beta_anneal_steps)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_BETA_END, DEFAULT_BETA_START, DEFAULT_PRIORITY_EXPONENT, PrioritizedReplaySettings,
    };
    use rlevo_core::config::Validate;

    #[test]
    fn test_settings_default_validates_and_matches_schaul_table_3() {
        let s = PrioritizedReplaySettings::default();
        assert!(s.validate().is_ok(), "a library default must be valid");
        assert!(
            (s.priority_exponent - DEFAULT_PRIORITY_EXPONENT).abs() < 1e-9,
            "Table 3's proportional row gives alpha = 0.6"
        );
        assert!(
            (s.beta_start - DEFAULT_BETA_START).abs() < 1e-9 && (s.beta_start - 0.4).abs() < 1e-9,
            "Table 3 anneals beta from 0.4"
        );
        assert!(
            (s.beta_end - DEFAULT_BETA_END).abs() < 1e-9,
            "Table 3 anneals beta to 1.0"
        );
    }

    #[test]
    fn test_settings_rejects_zero_anneal_steps() {
        let s = PrioritizedReplaySettings {
            beta_anneal_steps: 0,
            ..PrioritizedReplaySettings::default()
        };
        assert_eq!(
            s.validate().unwrap_err().field,
            "beta_anneal_steps",
            "a zero anneal length makes the schedule's progress fraction 0/0 (ADR 0051 §3)"
        );
    }

    #[test]
    fn test_settings_rejects_out_of_range_exponent_and_beta() {
        let bad_exponent = PrioritizedReplaySettings {
            priority_exponent: 1.5,
            ..PrioritizedReplaySettings::default()
        };
        assert_eq!(
            bad_exponent.validate().unwrap_err().field,
            "priority_exponent",
            "an alpha above 1 must be rejected"
        );

        let bad_start = PrioritizedReplaySettings {
            beta_start: -0.1,
            ..PrioritizedReplaySettings::default()
        };
        assert_eq!(
            bad_start.validate().unwrap_err().field,
            "beta_start",
            "a negative beta_start must be rejected"
        );

        let bad_end = PrioritizedReplaySettings {
            beta_end: 2.0,
            ..PrioritizedReplaySettings::default()
        };
        assert_eq!(
            bad_end.validate().unwrap_err().field,
            "beta_end",
            "a beta_end above 1 must be rejected"
        );
    }

    #[test]
    fn test_beta_starts_at_beta_start_and_anneals_to_one() {
        let s = PrioritizedReplaySettings {
            beta_start: 0.4,
            beta_end: 1.0,
            beta_anneal_steps: 1000,
            ..PrioritizedReplaySettings::default()
        };
        assert!(
            (s.beta(0).get() - 0.4).abs() < 1e-6,
            "at step 0 beta must equal beta_start"
        );
        assert!(
            (s.beta(1000).get() - 1.0).abs() < 1e-6,
            "at the anneal horizon beta must reach beta_end"
        );
        assert!(
            (s.beta(500).get() - 0.7).abs() < 1e-6,
            "midway through the schedule beta must be the linear midpoint"
        );
    }

    #[test]
    fn test_beta_clamps_past_the_horizon_and_never_exceeds_one() {
        let s = PrioritizedReplaySettings {
            beta_start: 0.4,
            beta_end: 1.0,
            beta_anneal_steps: 1000,
            ..PrioritizedReplaySettings::default()
        };
        // Past the horizon the progress fraction clamps to 1, so beta pins at
        // beta_end and never overshoots — the value ImportanceExponent::new
        // would otherwise reject.
        for step in [1_000, 5_000, 1_000_000] {
            let b = s.beta(step).get();
            assert!(
                (b - 1.0).abs() < 1e-6,
                "beta must saturate at beta_end past the horizon, got {b} at step {step}"
            );
        }
    }

    #[test]
    fn test_buffer_config_threads_capacity_and_exponents() {
        let s = PrioritizedReplaySettings {
            priority_exponent: 0.7,
            priority_epsilon: 1e-4,
            ..PrioritizedReplaySettings::default()
        };
        let cfg = s.buffer_config(4_096);
        assert_eq!(cfg.capacity, 4_096, "capacity comes from the agent config");
        assert!((cfg.priority_exponent - 0.7).abs() < 1e-9);
        assert!((cfg.priority_epsilon - 1e-4).abs() < 1e-9);
    }
}
