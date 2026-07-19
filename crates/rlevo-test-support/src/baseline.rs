//! Measured random-policy baselines for the learning tests.
//!
//! Rather than hard-coding an expected random-policy score (the old
//! `assert_beats_baseline(avg, -1.0, -6.67)` pattern, where `-6.67` was a
//! comment rather than a measurement), these helpers roll a uniform-random
//! policy over the *same* environment the agent trains on and return its mean
//! episode return. A learning test then asserts the trained agent beats that
//! measured baseline by a margin, so the bar tracks the environment instead of
//! a stale magic number.
//!
//! Two action samplers cover every fixture in the suite:
//! - [`uniform_discrete`] for [`DiscreteAction`] spaces (`CartPole`).
//! - [`uniform_bounded`] for [`BoundedAction`] spaces (`LinearEnv`, Pendulum).
//!
//! The rollout is fully seeded (environment seed + action-RNG seed), so the
//! baseline is deterministic per seed — the resulting threshold is fixed, not
//! flaky.

use rand::RngExt;
use rand::rngs::StdRng;

use rlevo_core::action::{BoundedAction, DiscreteAction};
use rlevo_core::environment::{Environment, Snapshot};

/// Samples a uniformly-random action from a discrete action space.
///
/// Pass as the `sample` argument to [`random_return`] for environments whose
/// `ActionType` is a [`DiscreteAction`] (e.g. `CartPole`).
pub fn uniform_discrete<const AR: usize, A: DiscreteAction<AR>>(rng: &mut StdRng) -> A {
    A::from_index(rng.random_range(0..A::ACTION_COUNT))
}

/// Samples a uniformly-random action from a bounded continuous action space.
///
/// Each component is drawn from `U(low[i], high[i])`. Pass as the `sample`
/// argument to [`random_return`] for environments whose `ActionType` is a
/// [`BoundedAction`] (e.g. `LinearEnv`, Pendulum).
pub fn uniform_bounded<const AR: usize, A: BoundedAction<AR>>(rng: &mut StdRng) -> A {
    let low = A::low();
    let high = A::high();
    // Keyed on COMPONENTS (flattened scalar count), not the rank `AR` — see
    // ADR 0053. `from_slice` asserts on `COMPONENTS`, so an `AR`-length sample
    // would panic for any multi-component rank-1 action.
    let values: Vec<f32> = (0..A::COMPONENTS)
        .map(|i| rng.random_range(low[i]..high[i]))
        .collect();
    A::from_slice(&values)
}

/// Rolls a random policy over `env` for `episodes` episodes and returns the mean
/// episode return (per-episode summed reward, averaged over episodes).
///
/// `sample` produces one random action per step (see [`uniform_discrete`] /
/// [`uniform_bounded`]); `max_steps` caps each episode so a non-terminating
/// environment (e.g. a `Pendulum` with no [`TimeLimit`]) cannot hang the
/// rollout. The reward is read through [`Snapshot::reward`] and converted to
/// `f32` via the `Reward: Into<f32>` supertrait, so the helper is agnostic to
/// the concrete reward type.
///
/// [`TimeLimit`]: rlevo_environments::wrappers::TimeLimit
///
/// # Panics
///
/// Panics if `episodes` is zero, or if `env.reset()` / `env.step()` errors.
pub fn random_return<const R: usize, const SR: usize, const AR: usize, E>(
    env: &mut E,
    episodes: usize,
    max_steps: usize,
    rng: &mut StdRng,
    mut sample: impl FnMut(&mut StdRng) -> E::ActionType,
) -> f32
where
    E: Environment<R, SR, AR>,
{
    assert!(episodes > 0, "episodes must be non-zero");
    let mut total = 0.0_f32;
    for _ in 0..episodes {
        env.reset().expect("reset");
        let mut steps = 0_usize;
        loop {
            let action = sample(rng);
            let snap = env.step(action).expect("step");
            total += snap.reward().clone().into();
            steps += 1;
            if snap.is_done() || steps >= max_steps {
                break;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = total / episodes as f32;
    mean
}
