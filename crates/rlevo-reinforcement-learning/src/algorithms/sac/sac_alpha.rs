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
//!
//! # Relation to the published objective
//!
//! The canonical dual is Haarnoja et al., *Soft Actor-Critic Algorithms and
//! Applications* (arXiv:1812.05905), Eq. 18:
//!
//! ```text
//! J(α) = E_{a∼π}[ −α log π(a|s) − α H̄ ]
//! ```
//!
//! with the target entropy `H̄ = −dim(A)`, the default heuristic from that
//! paper's Appendix D / Table 1.
//!
//! Note that the paper writes the dual in terms of **α**, not `log α`.
//! Optimising `log α` unconstrained — as this module does — is an
//! *implementation convention* shared by softlearning (the authors' own
//! code), rlkit, CleanRL and Stable-Baselines3. Its only effect is to enforce
//! `α ≥ 0` by construction; it carries no other guarantee from the paper.
//!
//! # Deliberate deviations from every reference implementation
//!
//! This module applies two hardenings that appear in **no** reference SAC:
//! softlearning, rlkit, CleanRL and SB3 all leave `log α` unbounded, and none
//! of them guards the α optimiser against a non-finite gradient. Both are
//! `rlevo` deviations, justified as defensive engineering rather than as
//! standard practice. They fix **different** problems and are separable.
//!
//! ## 1. Skipping the update on a non-finite gradient
//!
//! The gradient `g = −(log π.mean() + H̄)` is host-side `f32`. A collapsed
//! squashed-Gaussian policy legitimately produces `NaN`/`±Inf` `log π` on
//! out-of-distribution actions, and a diverging critic can feed the same
//! through reparameterisation.
//!
//! Because Adam's moments are exponential moving averages
//! (`m ← β₁·m + (1−β₁)·g`), folding one non-finite `g` into `m`/`v` poisons
//! both buffers **permanently**: every subsequent `log α` is `NaN` no matter
//! how healthy later gradients are. This is unlike the actor/critic
//! optimisers, which are rebuilt from fresh gradients each step and therefore
//! self-heal. [`LogAlpha::adam_step`] therefore returns without touching any
//! state when `g` is not finite.
//!
//! A finite `g` is not sufficient, so the same skip applies to the *derived*
//! moments. `(1 − β₂)·g·g` is left-associative, so `((1 − β₂)·g)·g` overflows
//! to `+inf` from roughly `|g| ≳ 10²¹` while `g` itself is still an ordinary
//! finite float. `v = +inf` is absorbing under the moving average, so
//! `v_hat.sqrt()` is `inf` and every later step size is exactly `0`: the
//! controller freezes **silently**, with no `NaN` and no unusual `log α` to
//! betray it. That is a worse failure than the `NaN` one — nothing observable
//! goes wrong except that the entropy constraint quietly stops being
//! enforced. Both moments are therefore computed into locals and committed
//! only once known finite.
//!
//! Finite raw moments are in turn not sufficient either, so the check extends
//! once more to the **bias-corrected** `m_hat`/`v_hat`. The divisor
//! `bc₂ = 1 − β₂^t` is only ~10⁻³ at `t = 1`, so `v_hat = v/bc₂` overflows for
//! `g` in roughly `(1.8·10¹⁹, 5.8·10²⁰)` at `t = 1` — a band that clears both
//! checks above, since `v` itself is still finite there. The lower edge drifts
//! upward as `bc₂` grows (`2.6·10¹⁹` at `t = 2`, `3.7·10¹⁹` at `t = 4`); the
//! upper edge is fixed, being where `v` itself overflows. The effect is the same
//! silent `0` step, but **bounded rather than permanent**: `bc₂ → 1` as `t`
//! grows, so it self-limits. Measured on this arithmetic, the freeze would
//! last from 0 steps at the bottom of the band to ~686 at the top.
//!
//! All three checks roll the step back completely. For the bias-corrected
//! case that is a measured choice, not a stylistic one: committing the (valid,
//! finite) moments and skipping only the parameter update reproduces the
//! frozen-step counts exactly, because the freeze is caused by the committed
//! large `v` rather than by the skipped subtraction.
//!
//! This path is reachable without adversarial inputs: the
//! [SAC policy head](super::sac_policy) clamps `log_std`, but the Gaussian
//! **mean** is an unclamped `Linear` output, so a diverging mean against a
//! near-floor `std` makes `((a − μ)/σ)²` enormous yet finite, and
//! `log π` follows.
//!
//! The idiom is precedented outside RL by PyTorch AMP's `GradScaler`, which
//! skips `optimizer.step()` when the unscaled gradients contain `inf`/`NaN`.
//! Because `g` is already a host `f32`, the check costs no device sync — the
//! objection raised against the tensor-side guard debated in #173 does not
//! apply here.
//!
//! ## 2. Clamping `log α` as a backstop
//!
//! `log α` is confined to `[-88, 88]` so that `α = exp(log α)` is finite
//! (`exp(88.7) ≈ f32::MAX`). SAC's legitimate α range is roughly `[0, 10]`,
//! i.e. `log α ≤ 2.3`, so the bound is provably non-binding in any healthy
//! run and no converging run's numbers change.
//!
//! This is **not** a fix for the poisoned-moment bug above: once `NaN` is in
//! the EMA it stays there whatever is done to the parameter. It is an
//! independent guard against `α` overflowing to `inf` and is also unrelated
//! to the `log σ` clamp on the policy heads, which bounds a different
//! parameter for a different reason.

/// Stateful `log α` with its own scalar Adam first/second-moment estimates.
///
/// The Adam hyperparameters are fixed at CleanRL's defaults (β₁ = 0.9,
/// β₂ = 0.999, ε = 1 × 10⁻⁸) and are not exposed as configuration. The
/// learning rate is passed per-step via [`LogAlpha::adam_step`] so callers
/// can derive it from [`SacTrainingConfig::alpha_lr`](super::sac_config::SacTrainingConfig).
#[derive(Debug, Clone)]
pub struct LogAlpha {
    log_alpha: f32,
    /// Adam first-moment estimate `m`.
    m: f32,
    /// Adam second-moment estimate `v`.
    v: f32,
    /// Number of Adam updates taken (needed for bias correction).
    t: u32,
    /// One-shot latch for the non-finite-gradient warning.
    ///
    /// A plain `bool` rather than an `AtomicBool` (the mechanism used by
    /// [the PPO Gaussian head][gauss]): [`adam_step`](LogAlpha::adam_step)
    /// takes `&mut self`, so no interior mutability is needed, and an
    /// `AtomicBool` would cost this struct its `Clone` derive.
    ///
    /// [gauss]: crate::algorithms::ppo::policies::gaussian
    nonfinite_grad_warned: bool,
    /// One-shot latch for the moment-overflow warning.
    ///
    /// Deliberately *separate* from [`Self::nonfinite_grad_warned`]: the two
    /// conditions have different causes and different remedies (a NaN source
    /// versus a diverging actor mean), a single run can hit both, and sharing
    /// one latch would let whichever fired first silence the other. The
    /// overflow case is the one that most needs surfacing, since it is
    /// otherwise entirely silent.
    moment_overflow_warned: bool,
    /// One-shot latch for the bias-corrected-moment overflow warning.
    ///
    /// A third latch for the same reason the second one exists: this condition
    /// overflows at a different stage (bias correction, not the raw square),
    /// occupies a disjoint `grad` band, and — unlike the other two — is
    /// bounded and self-limiting, so its message is materially different and
    /// deliberately less alarming. Sharing a latch would let whichever fired
    /// first suppress the other and misreport which failure actually occurred.
    bias_corrected_overflow_warned: bool,
}

/// Lower bound on `log α`, chosen so `exp` cannot underflow to a subnormal.
///
/// See the [module docs](self) for why this is non-binding in healthy runs.
const LOG_ALPHA_MIN: f32 = -88.0;

/// Upper bound on `log α`, chosen so `exp(log α)` stays finite in `f32`
/// (`exp(88.7) ≈ f32::MAX`).
const LOG_ALPHA_MAX: f32 = 88.0;

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
            nonfinite_grad_warned: false,
            moment_overflow_warned: false,
            bias_corrected_overflow_warned: false,
        }
    }

    /// Current `log α`.
    pub fn log_alpha(&self) -> f32 {
        self.log_alpha
    }

    /// Current `α = exp(log α)`.
    ///
    /// `log α` is clamped to `[LOG_ALPHA_MIN, LOG_ALPHA_MAX]` before
    /// exponentiating, so the returned α is always finite. See the
    /// [module docs](self) for why the bound never binds in a healthy run.
    pub fn alpha(&self) -> f32 {
        self.log_alpha.clamp(LOG_ALPHA_MIN, LOG_ALPHA_MAX).exp()
    }

    /// Applies one Adam step with closed-form gradient
    /// `g = −(log_prob_mean + target_entropy)`.
    ///
    /// This is the scalar Adam update with CleanRL's default β₁/β₂/ε. The
    /// learning rate is passed per-step so callers can reuse a
    /// [`SacTrainingConfig`](super::sac_config::SacTrainingConfig)
    /// schedule.
    ///
    /// # Skipped updates
    ///
    /// The step is **skipped in full** — `m`, `v`, `t` and `log α` all left
    /// untouched — in either of two cases, each with its own one-shot
    /// `tracing::warn!`:
    ///
    /// 1. `g` or `lr` is not finite.
    /// 2. `g` is finite but the *derived* moments `m`/`v` are not, which
    ///    happens when `g²` overflows `f32` (from `|g| ≳ 10²¹`).
    ///
    /// Either would poison the moment estimates permanently, since they are
    /// exponential moving averages — see the [module docs](self).
    ///
    /// # Arguments
    ///
    /// * `log_prob_mean` — batch mean of `log π(a|s)` under the current
    ///   policy.
    /// * `target_entropy` — the constraint `H̄`, conventionally `−dim(A)`.
    /// * `lr` — learning rate for this step.
    pub fn adam_step(&mut self, log_prob_mean: f32, target_entropy: f32, lr: f32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS: f32 = 1e-8;

        let grad = -(log_prob_mean + target_entropy);

        // Bail out *before* any state is mutated. `m` and `v` are EMAs, so a
        // single non-finite `grad` folded in here would make every future
        // `log α` NaN regardless of how healthy later gradients are.
        //
        // `lr` is checked here too: a non-finite `lr` leaves `m`/`v` clean but
        // makes the committed `log α` `inf` (or `NaN`, when `grad == 0` turns
        // the step into `inf · 0`), and `NaN` defeats the clamp below because
        // `NaN.clamp(..)` propagates.
        if !grad.is_finite() || !lr.is_finite() {
            if !self.nonfinite_grad_warned {
                self.nonfinite_grad_warned = true;
                tracing::warn!(
                    log_prob_mean = log_prob_mean,
                    target_entropy = target_entropy,
                    grad = grad,
                    lr = lr,
                    log_alpha = self.log_alpha,
                    "SAC alpha received a non-finite gradient or learning rate \
                     (log_prob_mean {log_prob_mean}, target_entropy \
                     {target_entropy}, so grad {grad}; lr {lr}) and the update \
                     was SKIPPED. Adam's moment estimates \
                     are exponential moving averages, so folding this in would \
                     make every later alpha NaN permanently; skipping keeps the \
                     temperature controller usable. For this step alpha is frozen \
                     at its last good value, exp({}) — the entropy constraint is \
                     simply not enforced until a finite gradient arrives. One \
                     isolated occurrence is usually harmless. A persistent source \
                     is almost always a diverging critic or a badly scaled reward \
                     feeding NaN log-probs back through the reparameterised \
                     actor: check critic loss and Q magnitudes, and lower the \
                     critic learning rate or rescale rewards if they are growing \
                     without bound.",
                    self.log_alpha,
                );
            }
            return;
        }

        // A finite `grad` does not imply finite moments. `(1 - β₂) * grad *
        // grad` is left-associative, so `((1 - β₂) * grad) * grad` overflows to
        // `+inf` from around `|grad| ≳ 1e21` while `grad` itself is still a
        // perfectly ordinary finite float. `v = +inf` is absorbing
        // (`β₂·inf + finite = inf`), which makes `v_hat.sqrt() = inf` and every
        // later step size exactly `0` — the controller freezes silently, with
        // no NaN and no odd-looking `log α` to give it away. So compute both
        // moments into locals and commit only once they are known finite.
        let m_next = BETA1 * self.m + (1.0 - BETA1) * grad;
        let v_next = BETA2 * self.v + (1.0 - BETA2) * grad * grad;
        if !m_next.is_finite() || !v_next.is_finite() {
            if !self.moment_overflow_warned {
                self.moment_overflow_warned = true;
                tracing::warn!(
                    grad = grad,
                    m_next = m_next,
                    v_next = v_next,
                    log_alpha = self.log_alpha,
                    "SAC alpha gradient was finite ({grad}) but ENORMOUS, and \
                     squaring it overflowed f32 while updating Adam's second \
                     moment (m {m_next}, v {v_next}); the update was SKIPPED. \
                     Had it been committed, v would be +inf permanently — inf \
                     is absorbing under the moving average — so every later \
                     step size would be exactly 0 and the temperature would \
                     freeze silently, with no NaN anywhere to reveal it. This \
                     is NOT a NaN source: it points at a diverging actor mean. \
                     The policy's log_std is clamped but its Gaussian mean is \
                     an unclamped Linear output, so a mean that has run away \
                     against a near-floor std makes ((a - mu)/sigma)^2 huge but \
                     finite, and log_prob follows. Check the actor's mean \
                     magnitudes and critic Q values, and lower the actor \
                     learning rate or rescale rewards.",
                );
            }
            return;
        }

        // Finite raw moments still do not imply finite *bias-corrected* ones.
        // `bc2 = 1 − β₂^t` is only ~1e-3 at `t = 1`, so dividing by it inflates
        // `v_next` by ~1000×: `v_hat` overflows to `+inf` for `grad` in roughly
        // `(1.8e19, 5.8e20)` — a band that clears both guards above, since
        // `v_next` itself is still finite there.
        //
        // The consequence is milder than the case above but the same shape:
        // `v_hat.sqrt() = inf` makes this step exactly `0`. It is bounded
        // rather than permanent, because `bc2 → 1` as `t` grows and the
        // committed `v` decays. Measured on this exact arithmetic, the freeze
        // lasts from 0 steps at the bottom of the band to ~686 at the top.
        //
        // Roll the whole step back rather than committing the moments and
        // skipping only the subtraction. That choice is empirical, not
        // stylistic: the freeze is caused by the committed finite-but-huge `v`,
        // NOT by the one skipped subtraction, so keeping the moment update
        // reproduces the frozen-step counts above exactly (549 at `grad = 5e20`
        // either way) and buys nothing. Discarding the poisoned gradient
        // entirely drops it to 0, and it keeps all three guards consistent:
        // a skipped step never leaves a trace in `t`, `m`, `v` or `log α`.
        let t_next = self.t.saturating_add(1);
        let bc1 = 1.0 - BETA1.powi(t_next as i32);
        let bc2 = 1.0 - BETA2.powi(t_next as i32);
        let m_hat = m_next / bc1;
        let v_hat = v_next / bc2;
        if !m_hat.is_finite() || !v_hat.is_finite() {
            if !self.bias_corrected_overflow_warned {
                self.bias_corrected_overflow_warned = true;
                tracing::warn!(
                    grad = grad,
                    m_hat = m_hat,
                    v_hat = v_hat,
                    t = t_next,
                    log_alpha = self.log_alpha,
                    "SAC alpha gradient was finite ({grad}) and so were Adam's \
                     raw moments, but bias correction overflowed them (m_hat \
                     {m_hat}, v_hat {v_hat} at t {t_next}); the update was \
                     SKIPPED. Early in training the correction divisor 1 - \
                     beta2^t is only about 1e-3, which inflates v by ~1000x. \
                     This is the MILDEST of the three skip conditions and is \
                     self-limiting: the divisor approaches 1 as t grows, so it \
                     stops recurring on its own. Nothing is committed, so the \
                     temperature controller is unaffected beyond losing this \
                     one step. The cause is the same as an enormous-gradient \
                     skip: a diverging actor mean, since log_std is clamped but \
                     the Gaussian mean is an unclamped Linear output. Worth \
                     checking actor mean magnitudes and reward scale, but a few \
                     isolated occurrences early in a run are not alarming.",
                );
            }
            return;
        }

        self.t = t_next;
        self.m = m_next;
        self.v = v_next;
        self.log_alpha -= lr * m_hat / (v_hat.sqrt() + EPS);

        // Backstop, independent of the guard above: keep `exp(log α)` finite.
        self.log_alpha = self.log_alpha.clamp(LOG_ALPHA_MIN, LOG_ALPHA_MAX);
    }

    /// Whether the one-shot non-finite-gradient warning has already fired for
    /// this instance.
    ///
    /// Exposed for tests: the crate has no `tracing` capture dependency, so
    /// the once-only latch is asserted directly rather than by scraping log
    /// output — the same approach as the PPO Gaussian head's
    /// `clamp_warning_fired`.
    #[cfg(test)]
    pub(crate) fn nonfinite_grad_warning_fired(&self) -> bool {
        self.nonfinite_grad_warned
    }

    /// Whether the one-shot moment-overflow warning has already fired for this
    /// instance. Exposed for tests; see
    /// [`nonfinite_grad_warning_fired`](Self::nonfinite_grad_warning_fired).
    #[cfg(test)]
    pub(crate) fn moment_overflow_warning_fired(&self) -> bool {
        self.moment_overflow_warned
    }

    /// Whether the one-shot bias-corrected-overflow warning has already fired
    /// for this instance. Exposed for tests; see
    /// [`nonfinite_grad_warning_fired`](Self::nonfinite_grad_warning_fired).
    #[cfg(test)]
    pub(crate) fn bias_corrected_overflow_warning_fired(&self) -> bool {
        self.bias_corrected_overflow_warned
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

    // -----------------------------------------------------------------
    // Non-finite gradient guard
    // -----------------------------------------------------------------

    /// Asserts that no field of the Adam state moved.
    fn assert_state_unchanged(got: &LogAlpha, expected: &LogAlpha, case: &str) {
        assert_eq!(got.t, expected.t, "{case}: t must not advance");
        assert!(
            got.m.to_bits() == expected.m.to_bits(),
            "{case}: m must not move (got {}, want {})",
            got.m,
            expected.m
        );
        assert!(
            got.v.to_bits() == expected.v.to_bits(),
            "{case}: v must not move (got {}, want {})",
            got.v,
            expected.v
        );
        assert!(
            got.log_alpha.to_bits() == expected.log_alpha.to_bits(),
            "{case}: log_alpha must not move (got {}, want {})",
            got.log_alpha,
            expected.log_alpha
        );
    }

    /// A `NaN` `log_prob_mean` makes `grad` `NaN`; folding that into the EMAs
    /// would be unrecoverable, so the whole step must be a no-op.
    #[test]
    fn nan_log_prob_leaves_adam_state_untouched() {
        let mut la = LogAlpha::new(0.0);
        // Take one healthy step first so the state under test is non-trivial.
        la.adam_step(3.0, -2.0, 1e-2);
        let before = la.clone();

        la.adam_step(f32::NAN, -2.0, 1e-2);

        assert_state_unchanged(&la, &before, "NaN log_prob_mean");
    }

    /// Both infinities take the same path: `grad` is `±inf`, and `v` would
    /// become `inf` (and `m_hat/v_hat` `NaN`) if the step were applied.
    #[test]
    fn infinite_log_prob_leaves_adam_state_untouched() {
        for (case, lp) in [("+inf", f32::INFINITY), ("-inf", f32::NEG_INFINITY)] {
            let mut la = LogAlpha::new(0.0);
            la.adam_step(3.0, -2.0, 1e-2);
            let before = la.clone();

            la.adam_step(lp, -2.0, 1e-2);

            assert_state_unchanged(&la, &before, case);
        }
    }

    /// The regression test this fix exists for: a poisoned step must not
    /// disable the controller for the rest of the run.
    ///
    /// Without the guard, `m` and `v` are `NaN` after the first call and every
    /// later `log_alpha` is `NaN` — so the finiteness assertion *and* the
    /// direction assertion both fail.
    #[test]
    fn optimizer_recovers_after_a_non_finite_gradient() {
        let mut la = LogAlpha::new(0.0);
        let before = la.log_alpha();

        la.adam_step(f32::NAN, -2.0, 1e-1);

        // `log π + H̄ = 1 > 0`, so grad is negative and `log α` must rise.
        for _ in 0..8 {
            la.adam_step(3.0, -2.0, 1e-1);
        }

        let after = la.log_alpha();
        assert!(
            after.is_finite(),
            "log_alpha must stay finite after a poisoned step, got {after}"
        );
        assert!(
            after > before,
            "the healthy steps must still move log_alpha up: before={before}, after={after}"
        );
        assert!(
            la.alpha().is_finite(),
            "alpha must stay finite, got {}",
            la.alpha()
        );
    }

    /// The warning is a one-shot latch, so a run that emits `NaN` on every
    /// update logs once rather than flooding.
    #[test]
    fn nonfinite_grad_warning_latches_after_first_occurrence() {
        let mut la = LogAlpha::new(0.0);
        assert!(
            !la.nonfinite_grad_warning_fired(),
            "a fresh LogAlpha must not have warned"
        );

        la.adam_step(3.0, -2.0, 1e-2);
        assert!(
            !la.nonfinite_grad_warning_fired(),
            "a finite gradient must not arm the latch"
        );

        la.adam_step(f32::NAN, -2.0, 1e-2);
        assert!(
            la.nonfinite_grad_warning_fired(),
            "the first non-finite gradient must latch the warning"
        );

        // Once latched it stays latched: nothing can un-set it, so `warn!` is
        // unreachable for the rest of this instance's life.
        for lp in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3.0] {
            la.adam_step(lp, -2.0, 1e-2);
            assert!(la.nonfinite_grad_warning_fired());
        }
    }

    // -----------------------------------------------------------------
    // Overflow of the derived Adam moments from a finite gradient
    // -----------------------------------------------------------------

    /// `grad` around `1e21` is finite, but `(1 - β₂) * grad * grad` overflows
    /// to `+inf`. The guard on `grad` alone does not catch this, so the moments
    /// must be checked after they are computed.
    #[test]
    fn overflowing_but_finite_gradient_leaves_adam_state_untouched() {
        // grad = -(log_prob_mean + target_entropy), so this is grad = -1e21.
        let log_prob_mean = 1e21_f32;
        assert!(
            log_prob_mean.is_finite(),
            "precondition: the input itself is finite"
        );

        let mut la = LogAlpha::new(0.0);
        la.adam_step(3.0, -2.0, 1e-2);
        let before = la.clone();

        la.adam_step(log_prob_mean, -2.0, 1e-2);

        assert_state_unchanged(&la, &before, "overflowing finite gradient");
        assert!(
            la.moment_overflow_warning_fired(),
            "the overflow must be reported"
        );
        assert!(
            !la.nonfinite_grad_warning_fired(),
            "this is not a non-finite-input failure and must not be reported as one"
        );
    }

    /// The recovery property for the overflow path. Without the moment check
    /// `v` is `+inf` forever, so `v_hat.sqrt()` is `inf` and every step size is
    /// exactly `0` — `log_alpha` stays finite but never moves again. Asserting
    /// finiteness alone would not catch that, so assert **movement**.
    #[test]
    fn optimizer_recovers_after_an_overflowing_gradient() {
        let mut la = LogAlpha::new(0.0);
        let before = la.log_alpha();

        la.adam_step(1e22, -2.0, 1e-1);

        for _ in 0..8 {
            la.adam_step(3.0, -2.0, 1e-1);
        }

        let after = la.log_alpha();
        assert!(
            after.is_finite(),
            "log_alpha must stay finite after an overflowing step, got {after}"
        );
        assert!(
            after > before,
            "the controller must not be frozen: log_alpha should still rise, \
             before={before}, after={after}"
        );
    }

    /// The two failure modes have independent latches, so an earlier
    /// non-finite-gradient warning cannot suppress the (silent, and therefore
    /// more important) overflow warning.
    #[test]
    fn moment_overflow_warning_latches_independently_and_once() {
        let mut la = LogAlpha::new(0.0);
        la.adam_step(f32::NAN, -2.0, 1e-2);
        assert!(la.nonfinite_grad_warning_fired());
        assert!(
            !la.moment_overflow_warning_fired(),
            "a NaN gradient must not arm the overflow latch"
        );

        la.adam_step(1e22, -2.0, 1e-2);
        assert!(
            la.moment_overflow_warning_fired(),
            "the overflow latch must arm despite the earlier NaN warning"
        );

        for _ in 0..16 {
            la.adam_step(1e22, -2.0, 1e-2);
            assert!(la.moment_overflow_warning_fired());
        }
    }

    // -----------------------------------------------------------------
    // Overflow of the *bias-corrected* moments
    // -----------------------------------------------------------------

    /// The overflow band clears both earlier guards — `grad`, `m` and `v` are
    /// all finite — yet `v_hat = v / bc2` overflows because `bc2` is tiny early
    /// in training.
    ///
    /// The band's **lower edge is `t`-dependent**, since `bc2 = 1 - beta2^t`
    /// grows with `t`: measured, it is `1.84e19` at `t = 1`, `2.61e19` at
    /// `t = 2`, `3.69e19` at `t = 4`. Its upper edge is `t`-independent at
    /// `5.83e20`, where the raw-moment guard takes over. The poison step below
    /// lands at `t = 2` because of the warm-up, so the magnitudes start above
    /// that step's edge rather than the `t = 1` one.
    ///
    /// Asserts the chosen semantics exactly: option (a), full rollback, so
    /// nothing at all is committed.
    #[test]
    fn bias_correction_overflow_rolls_back_the_entire_step() {
        for grad_mag in [3e19_f32, 5e19, 1e20, 3e20, 5e20] {
            // Confirm the precondition: this band really does clear the guards
            // above, otherwise the test would be exercising the wrong branch.
            let m_next = 0.1 * grad_mag;
            let v_next = 0.001 * grad_mag * grad_mag;
            assert!(
                grad_mag.is_finite() && m_next.is_finite() && v_next.is_finite(),
                "precondition for grad={grad_mag}: grad and raw moments are finite"
            );

            let mut la = LogAlpha::new(0.0);
            la.adam_step(3.0, -2.0, 1e-3);
            let before = la.clone();

            // grad = -(log_prob_mean + target_entropy) = -grad_mag
            la.adam_step(grad_mag, -2.0, 1e-3);

            assert_state_unchanged(&la, &before, "bias-corrected overflow");
            assert!(
                la.bias_corrected_overflow_warning_fired(),
                "grad={grad_mag} must report a bias-correction overflow"
            );
            assert!(
                !la.moment_overflow_warning_fired(),
                "grad={grad_mag} must not be misreported as a raw-moment overflow"
            );
            assert!(
                !la.nonfinite_grad_warning_fired(),
                "grad={grad_mag} must not be misreported as a non-finite input"
            );
        }
    }

    /// Recovery for this path. Without the guard the committed `v` is huge but
    /// finite, so `log_alpha` stays finite while the step size is exactly `0`
    /// for hundreds of updates — assert **movement**, which is what breaks.
    #[test]
    fn optimizer_recovers_after_a_bias_correction_overflow() {
        let mut la = LogAlpha::new(0.0);
        let before = la.log_alpha();

        la.adam_step(5e20, -2.0, 1e-3);

        // Measured: the unguarded version is frozen for 549 steps at this
        // magnitude, so a margin well under that must already show movement.
        for _ in 0..64 {
            la.adam_step(3.0, -2.0, 1e-3);
        }

        let after = la.log_alpha();
        assert!(after.is_finite(), "log_alpha must stay finite, got {after}");
        assert!(
            after > before,
            "the controller must not be frozen: log_alpha should still rise, \
             before={before}, after={after}"
        );
    }

    /// Just below the band nothing overflows, so no guard may fire and the
    /// step must be applied normally. Guards that trip early would silently
    /// discard legitimate gradients.
    #[test]
    fn gradients_below_the_overflow_band_are_applied_normally() {
        let mut la = LogAlpha::new(0.0);
        let before = la.log_alpha();

        // 1e19 < 1.845e19 lower edge: v_hat ≈ 1e38, still inside f32.
        la.adam_step(1e19, -2.0, 1e-3);

        assert_eq!(la.t, 1, "a representable gradient must be applied");
        assert!(la.log_alpha() != before, "log_alpha must move");
        assert!(
            !la.bias_corrected_overflow_warning_fired()
                && !la.moment_overflow_warning_fired()
                && !la.nonfinite_grad_warning_fired(),
            "no guard may fire below the overflow band"
        );
    }

    // -----------------------------------------------------------------
    // log_alpha clamp backstop
    // -----------------------------------------------------------------

    /// `exp` overflows around `log α ≈ 88.7`, so an absurd `log α` — reachable
    /// only via a pathological gradient sequence — must still yield a finite α.
    #[test]
    fn alpha_is_finite_for_absurd_log_alpha_magnitudes() {
        for init in [1e3_f32, 1e30, -1e3, -1e30, f32::MAX, f32::MIN] {
            let la = LogAlpha::new(init);
            let alpha = la.alpha();
            assert!(
                alpha.is_finite(),
                "alpha must be finite for log_alpha={init}, got {alpha}"
            );
            assert!(alpha >= 0.0, "alpha must be non-negative, got {alpha}");
        }
    }

    /// A non-finite `lr` must be handled by the input guard, not left to the
    /// `±88` clamp.
    ///
    /// The clamp rescues `lr = inf` only when `grad != 0`: there the step is
    /// `±inf` and clamping pins it to a bound. At `grad == 0` the step is
    /// `inf * 0 = NaN`, and `NaN.clamp(..)` propagates `NaN` — so the clamp is
    /// not a sufficient backstop and `lr` is checked up front instead.
    #[test]
    fn non_finite_lr_is_rejected_rather_than_clamped() {
        for lr in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            // grad != 0 (the case the clamp would have caught anyway) and
            // grad == 0 (the case it would not).
            for log_prob_mean in [3.0_f32, -2.0] {
                let mut la = LogAlpha::new(0.0);
                let before = la.clone();

                la.adam_step(log_prob_mean, -2.0, lr);

                assert_state_unchanged(&la, &before, "non-finite lr");
                assert!(
                    la.alpha().is_finite(),
                    "alpha must stay finite for lr={lr}, got {}",
                    la.alpha()
                );
            }
        }
    }

    /// Boundary documentation: `LogAlpha::new` enforces no invariant on its
    /// argument, and `NaN.clamp(..)` propagates, so a `NaN` seed yields a `NaN`
    /// alpha permanently.
    ///
    /// This is recorded rather than sanitized: the sole production call site
    /// passes `initial_alpha.max(f32::MIN_POSITIVE).ln()`, and `f32::max`
    /// returns the non-NaN operand, so a `NaN` cannot reach the constructor
    /// there. Silently rewriting a caller's `NaN` to some default would hide a
    /// configuration bug rather than surface it. If a future caller can supply
    /// `NaN`, validate at that boundary.
    #[test]
    fn new_does_not_sanitize_a_nan_seed() {
        let la = LogAlpha::new(f32::NAN);
        assert!(
            la.log_alpha().is_nan(),
            "the seed is stored verbatim, not sanitized"
        );
        assert!(la.alpha().is_nan(), "and NaN survives the clamp in alpha()");

        // The production construction path cannot produce that seed.
        let seeded = LogAlpha::new(f32::NAN.max(f32::MIN_POSITIVE).ln());
        assert!(
            seeded.alpha().is_finite(),
            "the real call site is safe: f32::max discards the NaN operand"
        );
    }

    /// Repeated steps must drive `log α` well inside the clamp: the bound is a
    /// backstop for pathological runs, not part of the normal control loop.
    #[test]
    fn clamp_does_not_bind_in_a_realistic_run() {
        // target_entropy = -dim(A) for a 6-DoF action space, with log-probs in
        // the range a converging squashed-Gaussian actually produces.
        let target_entropy = -6.0;
        let mut la = LogAlpha::new(0.0);

        for i in 0..2_000 {
            let log_prob_mean = if i % 2 == 0 { -8.0 } else { -4.0 };
            la.adam_step(log_prob_mean, target_entropy, 3e-4);
        }

        let log_alpha = la.log_alpha();
        assert!(
            log_alpha.is_finite(),
            "log_alpha must remain finite, got {log_alpha}"
        );
        assert!(
            log_alpha.abs() < 10.0,
            "a realistic run must stay far inside the ±88 clamp, got {log_alpha}"
        );
        assert!(
            la.alpha().is_finite(),
            "alpha must remain finite, got {}",
            la.alpha()
        );
    }
}
