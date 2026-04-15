//! Reward helpers shared by every grid environment.

/// Minigrid's canonical success reward: `1 - 0.9 * (step / max_steps)`.
///
/// Reaching the goal on step `0` returns `1.0`; reaching it on the very
/// last legal step returns `0.1`. Returns `0.0` if `max_steps` is zero so
/// that the formula is total.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn success_reward(step: usize, max_steps: usize) -> f32 {
    if max_steps == 0 {
        return 0.0;
    }
    let ratio = step as f32 / max_steps as f32;
    1.0 - 0.9 * ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_zero_is_best() {
        assert!((success_reward(0, 100) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn full_budget_is_worst_nonzero() {
        assert!((success_reward(100, 100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn half_budget_is_between() {
        let r = success_reward(50, 100);
        assert!((r - 0.55).abs() < 1e-6);
    }

    #[test]
    fn zero_max_steps_returns_zero() {
        assert_eq!(success_reward(5, 0), 0.0);
    }

    #[test]
    fn step_over_max_still_defined() {
        // Beyond budget the formula goes negative; we don't clamp because
        // no env should call it past termination, but it shouldn't panic.
        let r = success_reward(200, 100);
        assert!(r < 0.0);
    }
}
