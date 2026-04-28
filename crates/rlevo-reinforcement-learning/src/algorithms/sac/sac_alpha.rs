//! Learnable temperature `α = exp(log α)` for SAC's maximum-entropy objective.
//!
//! SAC's auto-tuning dual is a one-parameter optimisation:
//!
//! ```text
//! L(log α) = −(log α · (log π(a|s) + H̄)).mean()
//! ∂L/∂(log α) = −(log π.mean() + H̄)
//! ```
//!
//! We don't need Burn's autodiff to take this gradient — it's a closed-form
//! scalar — and threading a single-Param Burn `Module` through the shared
//! autodiff server turns out to conflict with Burn 0.20's graph memory
//! manager: an orphan `Param` registered with `require_grad` can cause the
//! server to prune shared leaf nodes during the critic's first backward,
//! panicking the second critic's backward with "Node should have a step
//! registered". Keeping `log α` as a plain `f32` with hand-rolled Adam
//! sidesteps the interaction entirely.

/// Stateful `log α` with its own Adam first/second-moment estimates.
#[derive(Debug, Clone)]
pub struct LogAlpha {
    log_alpha: f32,
    /// Adam first-moment estimate `m`.
    m: f32,
    /// Adam second-moment estimate `v`.
    v: f32,
    /// Number of Adam updates taken (needed for bias correction).
    t: u32,
}

impl LogAlpha {
    /// Constructs with `log α = init_log_alpha`. Pass
    /// `initial_alpha.max(f32::MIN_POSITIVE).ln()` when you want to seed
    /// from a target initial α.
    pub fn new(init_log_alpha: f32) -> Self {
        Self {
            log_alpha: init_log_alpha,
            m: 0.0,
            v: 0.0,
            t: 0,
        }
    }

    /// Current `log α`.
    pub fn log_alpha(&self) -> f32 {
        self.log_alpha
    }

    /// Current `α = exp(log α)`.
    pub fn alpha(&self) -> f32 {
        self.log_alpha.exp()
    }

    /// Applies one Adam step with closed-form gradient
    /// `g = −(log_prob_mean + target_entropy)`.
    ///
    /// This is the scalar Adam update with CleanRL's default β₁/β₂/ε. The
    /// learning rate is passed per-step so callers can reuse a
    /// [`SacTrainingConfig`](super::sac_config::SacTrainingConfig)
    /// schedule.
    pub fn adam_step(&mut self, log_prob_mean: f32, target_entropy: f32, lr: f32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS: f32 = 1e-8;

        let grad = -(log_prob_mean + target_entropy);
        self.t = self.t.saturating_add(1);
        self.m = BETA1 * self.m + (1.0 - BETA1) * grad;
        self.v = BETA2 * self.v + (1.0 - BETA2) * grad * grad;
        let bc1 = 1.0 - BETA1.powi(self.t as i32);
        let bc2 = 1.0 - BETA2.powi(self.t as i32);
        let m_hat = self.m / bc1;
        let v_hat = self.v / bc2;
        self.log_alpha -= lr * m_hat / (v_hat.sqrt() + EPS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With `log π` well below `target_entropy` (i.e. `log π + H̄ < 0`), the
    /// closed-form gradient is positive so Adam pushes `log α` down.
    #[test]
    fn auto_alpha_decreases_when_logp_is_below_target_entropy() {
        let mut la = LogAlpha::new(0.0);
        let before = la.log_alpha();
        la.adam_step(-5.0, -2.0, 1e-1);
        let after = la.log_alpha();
        assert!(
            after < before,
            "expected log_alpha to decrease: before={before}, after={after}"
        );
    }

    #[test]
    fn auto_alpha_increases_when_logp_is_above_target_entropy() {
        let mut la = LogAlpha::new(0.0);
        let before = la.log_alpha();
        la.adam_step(3.0, -2.0, 1e-1);
        let after = la.log_alpha();
        assert!(
            after > before,
            "expected log_alpha to increase: before={before}, after={after}"
        );
    }

    #[test]
    fn alpha_matches_exp_of_log_alpha() {
        let la = LogAlpha::new(0.5);
        let got = la.alpha();
        let expected = 0.5_f32.exp();
        assert!((got - expected).abs() < 1e-6);
    }
}
