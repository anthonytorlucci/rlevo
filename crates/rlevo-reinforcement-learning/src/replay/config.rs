//! Configuration for the two shipped replay strategies:
//! [`UniformReplay`](super::UniformReplay) and
//! [`PrioritizedReplay`](super::PrioritizedReplay).
//!
//! Both configs are the [`Validate`] chokepoint their buffer's fallible
//! constructor calls — [`UniformReplay::from_config`](super::UniformReplay::from_config)
//! and [`PrioritizedReplay::new`](super::PrioritizedReplay::new) — so a
//! capacity that arrived through `Deserialize` or struct-update syntax is
//! refused with a [`ConfigError`] naming the field, never with a panic
//! (`rules.md` §4).

use rlevo_core::config::{self, ConfigError, Validate};
use serde::{Deserialize, Serialize};

use crate::MAX_BUFFER_CAPACITY;

/// The ε floor of Schaul et al. (2016) §3.3's `$p_i = |\delta_i| + \epsilon$`.
///
/// **Schaul gives no numeric value for ε** — it is described only as "a small
/// positive constant that prevents the edge-case of transitions not being
/// revisited once their error is zero", and it does not appear in the paper's
/// grid search or in Table 3. `1e-6` is therefore **our** choice, not the
/// paper's, and is justified against the TD-error scale this library actually
/// produces rather than by appeal to convention:
///
/// - **It must not reorder real TD errors.** The shipped configs use rewards of
///   order 1 (`CartPole`'s `+1` per step, the classic-control family, the bandit
///   family) with `$\gamma < 1$`, which puts a residual carrying signal at roughly
///   `1e-2 … 1e1`. At `1e-6`, ε sits at least four orders of magnitude below
///   the smallest such residual, so it perturbs no ordering that matters.
/// - **It must survive `f32`.** `1e-6` is a normal `f32` and exceeds the `f32`
///   spacing at `1.0` (≈`1.19e-7`), so `$|\delta| + \epsilon$` is not absorbed across the
///   residual range above. (For `$|\delta| \gtrsim 8$` the addition *is* absorbed —
///   irrelevant, since ε only has work to do when `$|\delta| \approx 0$`.)
/// - **It must leave a converged transition revisitable but strongly
///   deprioritized.** That is its entire purpose. Under the default
///   `priority_exponent = 0.6`, a fully-converged transition carries
///   `(1e-6)^0.6 ≈ 4e-4` of unnormalized mass against a `$|\delta| = 1$` transition's
///   `1.0` — non-zero, so it is never starved, but ~2500× less likely per draw.
/// - **It must not flatten the distribution when nothing has converged yet.** A
///   larger ε (say `1e-2`) raises the floor to `(1e-2)^0.6 ≈ 6e-2`, pulling
///   prioritization toward uniform early in training when residuals are small
///   and roughly equal — the regime where prioritization is supposed to be
///   doing the most work.
pub const DEFAULT_PRIORITY_EPSILON: f32 = 1e-6;

/// Schaul et al. (2016) Table 3's α for the **proportional** variant.
pub const DEFAULT_PRIORITY_EXPONENT: f32 = 0.6;

/// Hyperparameters for [`PrioritizedReplay`](super::PrioritizedReplay).
///
/// # Naming
///
/// The Greek letters are spelled out (ADR 0050 §12): `alpha` is already SAC's
/// entropy temperature, and the collision is in the reader's head even where the
/// compiler is untroubled.
///
/// | Schaul symbol | Field |
/// |---|---|
/// | α (priority exponent) | [`priority_exponent`](Self::priority_exponent) |
/// | ε (priority floor) | [`priority_epsilon`](Self::priority_epsilon) |
///
/// # β is deliberately absent
///
/// The importance-sampling exponent β and its annealing schedule live on the
/// **agent** config, and the agent passes the evaluated β into
/// [`sample`](super::ReplayStrategy::sample) (ADR 0050 §11). The buffer has
/// no step counter, and giving it one would duplicate the agent's — a second
/// source of truth of exactly the shape `rules.md` §10 forbids — and would be
/// wrong outright the moment two learners share one buffer.
///
/// # Examples
///
/// ```
/// use rlevo_core::config::Validate;
/// use rlevo_reinforcement_learning::replay::PrioritizedReplayConfig;
///
/// let config = PrioritizedReplayConfig {
///     capacity: 100_000,
///     ..PrioritizedReplayConfig::default()
/// };
/// assert!(config.validate().is_ok());
///
/// // alpha = 0 recovers Schaul Eq. 1's uniform case and is a valid setting.
/// let uniform = PrioritizedReplayConfig { priority_exponent: 0.0, ..config };
/// assert!(uniform.validate().is_ok());
///
/// // A zero-capacity buffer is rejected, not silently tolerated.
/// let empty = PrioritizedReplayConfig { capacity: 0, ..config };
/// assert_eq!(empty.validate().unwrap_err().field, "capacity");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrioritizedReplayConfig {
    /// The maximum number of transitions held; the oldest is evicted past this.
    ///
    /// Must be non-zero and at most
    /// [`MAX_BUFFER_CAPACITY`](crate::MAX_BUFFER_CAPACITY). The upper bound is
    /// not a taste judgement: the sum-tree sizes itself as
    /// `capacity.next_power_of_two() * 2`, which wraps silently in release
    /// builds for capacities near `usize::MAX` and yields an empty node array.
    /// See that constant for the measured wrap and for why `2^32` is orders of
    /// magnitude above any published replay buffer.
    pub capacity: usize,

    /// Schaul Eq. 1's α in `$P(i) = p_i^\alpha / \sum_k p_k^\alpha$`.
    ///
    /// `0.0` is the uniform case (every stored transition maps to mass `1.0`),
    /// `1.0` is fully greedy prioritization. Must lie in `[0, 1]`. Defaults to
    /// [`DEFAULT_PRIORITY_EXPONENT`].
    pub priority_exponent: f32,

    /// Schaul §3.3's ε in `$p_i = |\delta_i| + \epsilon$`.
    ///
    /// Must be finite and strictly positive. Defaults to
    /// [`DEFAULT_PRIORITY_EPSILON`] — see that constant for why `1e-6`, and for
    /// the explicit note that the value is ours and not the paper's.
    pub priority_epsilon: f32,
}

impl Default for PrioritizedReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 100_000,
            priority_exponent: DEFAULT_PRIORITY_EXPONENT,
            priority_epsilon: DEFAULT_PRIORITY_EPSILON,
        }
    }
}

impl Validate for PrioritizedReplayConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "PrioritizedReplayConfig";
        config::nonzero(C, "capacity", self.capacity)?;
        config::at_most(C, "capacity", self.capacity, MAX_BUFFER_CAPACITY)?;
        config::in_range(
            C,
            "priority_exponent",
            0.0,
            1.0,
            f64::from(self.priority_exponent),
        )?;
        config::positive(C, "priority_epsilon", f64::from(self.priority_epsilon))?;
        Ok(())
    }
}

/// Hyperparameters for [`UniformReplay`](super::UniformReplay).
///
/// One field, because uniform replay has one knob: how much it remembers. The
/// type exists anyway, and is not folded away into the bare `usize`
/// [`UniformReplay::new`](super::UniformReplay::new) already takes, because a
/// capacity that came from a deserialized run manifest is *user data*, and
/// `rules.md` §4 puts user data behind a [`Validate`] chokepoint returning
/// [`ConfigError`] rather than behind an assert. Passing this config to
/// [`UniformReplay::from_config`](super::UniformReplay::from_config) is how a
/// runtime-derived capacity reaches a buffer without a panic in the path.
///
/// # This `Default` is not the agents' default
///
/// [`Default`] gives `capacity: 100_000`, matching
/// [`PrioritizedReplayConfig::default`] so that flipping `prioritized: true` on
/// an otherwise-default setup does not silently resize the buffer. **No shipped
/// agent uses it.** Each carries its own `replay_buffer_capacity` — `1_000_000`
/// for TD3/SAC/DDPG (Fujimoto et al. 2018, Haarnoja et al. 2018), `10_000` for
/// DQN/C51/QR-DQN — and composes a `UniformReplayConfig` from that field at
/// construction time. This default therefore serves standalone use and
/// [`Deserialize`] of a manifest that omits the field; it is not a claim about
/// what any algorithm should want.
///
/// # Examples
///
/// ```
/// use rlevo_core::config::Validate;
/// use rlevo_reinforcement_learning::replay::UniformReplayConfig;
///
/// let config = UniformReplayConfig { capacity: 10_000 };
/// assert!(config.validate().is_ok());
///
/// // The library default is itself valid, as every default must be.
/// assert!(UniformReplayConfig::default().validate().is_ok());
///
/// // A zero-capacity buffer is rejected, not silently tolerated.
/// let empty = UniformReplayConfig { capacity: 0 };
/// assert_eq!(empty.validate().unwrap_err().field, "capacity");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UniformReplayConfig {
    /// The maximum number of transitions held; the oldest is evicted past this.
    ///
    /// Must be non-zero and at most
    /// [`MAX_BUFFER_CAPACITY`](crate::MAX_BUFFER_CAPACITY). Neither bound is a
    /// taste judgement. A zero-capacity buffer cannot hold the transition it
    /// was just handed, so every push would evict what it just stored. Above
    /// the ceiling, `VecDeque::with_capacity` **aborts** the process on an
    /// allocation it cannot serve — an outcome no caller can catch or
    /// diagnose — so the capacity is refused here, where the offending field
    /// still has a name. The ceiling is shared with
    /// [`PrioritizedReplayConfig::capacity`] so that toggling prioritization
    /// cannot move the boundary between an accepted and a rejected config.
    pub capacity: usize,
}

impl Default for UniformReplayConfig {
    fn default() -> Self {
        Self { capacity: 100_000 }
    }
}

impl Validate for UniformReplayConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "UniformReplayConfig";
        config::nonzero(C, "capacity", self.capacity)?;
        config::at_most(C, "capacity", self.capacity, MAX_BUFFER_CAPACITY)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_PRIORITY_EPSILON, DEFAULT_PRIORITY_EXPONENT, MAX_BUFFER_CAPACITY,
        PrioritizedReplayConfig, UniformReplayConfig,
    };
    use rlevo_core::config::{ConstraintKind, Validate};

    #[test]
    fn test_prioritized_replay_config_default_validates() {
        assert!(
            PrioritizedReplayConfig::default().validate().is_ok(),
            "a library default must itself be valid (ADR 0026)"
        );
    }

    #[test]
    fn test_prioritized_replay_config_default_matches_schaul_table_3() {
        let c = PrioritizedReplayConfig::default();
        assert!(
            (c.priority_exponent - 0.6).abs() < 1e-9,
            "Table 3's proportional row gives alpha = 0.6"
        );
        assert!(
            (DEFAULT_PRIORITY_EXPONENT - 0.6).abs() < 1e-9,
            "the exported constant must agree with the default"
        );
        assert!(
            (c.priority_epsilon - DEFAULT_PRIORITY_EPSILON).abs() < 1e-12,
            "the default epsilon must be the documented constant"
        );
    }

    #[test]
    fn test_prioritized_replay_config_rejects_zero_capacity() {
        let c = PrioritizedReplayConfig {
            capacity: 0,
            ..PrioritizedReplayConfig::default()
        };
        assert_eq!(
            c.validate().unwrap_err().field,
            "capacity",
            "a zero-capacity buffer must be named as the offending field"
        );
    }

    /// The ceiling is inclusive: exactly [`MAX_BUFFER_CAPACITY`] is a legal
    /// (if absurd) request, and the sum-tree arithmetic for it — `2^33` nodes —
    /// is exact. Only strictly above it is rejected.
    #[test]
    fn test_prioritized_replay_config_accepts_capacity_at_ceiling() {
        let c = PrioritizedReplayConfig {
            capacity: MAX_BUFFER_CAPACITY,
            ..PrioritizedReplayConfig::default()
        };
        assert!(
            c.validate().is_ok(),
            "MAX_BUFFER_CAPACITY is an inclusive bound"
        );
    }

    /// Above the ceiling the sum-tree's `next_power_of_two() * 2` is on the road
    /// to a release-mode wrap, so the config must refuse it — naming `capacity`
    /// and reporting `TooLarge`, not some downstream symptom.
    #[test]
    fn test_prioritized_replay_config_rejects_capacity_above_ceiling() {
        for capacity in [MAX_BUFFER_CAPACITY + 1, usize::MAX] {
            let err = PrioritizedReplayConfig {
                capacity,
                ..PrioritizedReplayConfig::default()
            }
            .validate()
            .unwrap_err();
            assert_eq!(err.field, "capacity", "capacity {capacity} is the offender");
            assert_eq!(
                err.kind,
                ConstraintKind::TooLarge {
                    max: MAX_BUFFER_CAPACITY as u64,
                    got: capacity as u64,
                },
                "capacity {capacity} must be rejected as TooLarge, carrying both bounds"
            );
        }
    }

    #[test]
    fn test_prioritized_replay_config_rejects_out_of_range_exponent() {
        for bad in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            let c = PrioritizedReplayConfig {
                priority_exponent: bad,
                ..PrioritizedReplayConfig::default()
            };
            assert_eq!(
                c.validate().unwrap_err().field,
                "priority_exponent",
                "alpha = {bad} is outside [0, 1] and must be rejected"
            );
        }
    }

    #[test]
    fn test_prioritized_replay_config_accepts_alpha_endpoints() {
        for good in [0.0, 1.0] {
            let c = PrioritizedReplayConfig {
                priority_exponent: good,
                ..PrioritizedReplayConfig::default()
            };
            assert!(
                c.validate().is_ok(),
                "alpha = {good} is an inclusive endpoint of [0, 1]"
            );
        }
    }

    #[test]
    fn test_prioritized_replay_config_rejects_non_positive_epsilon() {
        for bad in [0.0, -1e-6, f32::NAN] {
            let c = PrioritizedReplayConfig {
                priority_epsilon: bad,
                ..PrioritizedReplayConfig::default()
            };
            assert_eq!(
                c.validate().unwrap_err().field,
                "priority_epsilon",
                "epsilon = {bad} is not strictly positive and must be rejected"
            );
        }
    }

    #[test]
    fn test_uniform_replay_config_default_validates() {
        assert!(
            UniformReplayConfig::default().validate().is_ok(),
            "a library default must itself be valid (ADR 0026)"
        );
    }

    /// The two configs agree on the default capacity, so turning prioritization
    /// on for an otherwise-default setup does not silently resize the buffer.
    #[test]
    fn test_uniform_replay_config_default_capacity_matches_prioritized() {
        assert_eq!(
            UniformReplayConfig::default().capacity,
            PrioritizedReplayConfig::default().capacity,
            "the two replay configs must share a default capacity"
        );
    }

    #[test]
    fn test_uniform_replay_config_rejects_zero_capacity() {
        let err = UniformReplayConfig { capacity: 0 }.validate().unwrap_err();
        assert_eq!(
            err.field, "capacity",
            "a zero-capacity buffer must be named as the offending field"
        );
    }

    /// The ceiling is inclusive: exactly [`MAX_BUFFER_CAPACITY`] is a legal (if
    /// absurd) request, matching the prioritized config so that the accept/reject
    /// boundary does not move when prioritization is toggled.
    #[test]
    fn test_uniform_replay_config_accepts_capacity_at_ceiling() {
        assert!(
            UniformReplayConfig {
                capacity: MAX_BUFFER_CAPACITY
            }
            .validate()
            .is_ok(),
            "MAX_BUFFER_CAPACITY is an inclusive bound"
        );
    }

    /// Above the ceiling, `VecDeque::with_capacity` aborts the process rather
    /// than unwinding, so the config must refuse it — naming `capacity` and
    /// reporting `TooLarge` with both bounds, not some downstream symptom.
    #[test]
    fn test_uniform_replay_config_rejects_capacity_above_ceiling() {
        for capacity in [MAX_BUFFER_CAPACITY + 1, usize::MAX] {
            let err = UniformReplayConfig { capacity }.validate().unwrap_err();
            assert_eq!(err.field, "capacity", "capacity {capacity} is the offender");
            assert_eq!(
                err.kind,
                ConstraintKind::TooLarge {
                    max: MAX_BUFFER_CAPACITY as u64,
                    got: capacity as u64,
                },
                "capacity {capacity} must be rejected as TooLarge, carrying both bounds"
            );
        }
    }
}
