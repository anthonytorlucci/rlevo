//! End-to-end training loop for QR-DQN.
//!
//! The public surface is a single function, [`train`], which drives a
//! [`QrDqnAgent`] against any [`Environment`] for a fixed number of
//! environment steps. The loop follows the standard off-policy DQN cadence:
//!
//! 1. Select an action with ε-greedy exploration ([`QrDqnAgent::act`]).
//! 2. Step the environment and store the transition in the replay buffer.
//! 3. Every `train_frequency` steps, run one gradient update
//!    ([`QrDqnAgent::learn_step`]).
//! 4. Call [`QrDqnAgent::sync_target`] every step. This is the **hard**-sync
//!    path only, and it is gated internally: it is a no-op unless
//!    `tau == 0.0`. When `tau > 0` the target network is instead maintained by
//!    the Polyak soft update inside `learn_step`, so calling `sync_target`
//!    unconditionally here cannot clobber that lag.
//! 5. On episode termination, record [`QrDqnMetrics`] (including
//!    `quantile_spread`) and reset the environment.
//!
//! The only behavioural difference from [`crate::algorithms::c51::train`] is
//! the metrics type emitted per completed episode — [`QrDqnMetrics`] carries
//! a `quantile_spread` diagnostic in place of C51's `entropy`.

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::qrdqn::qrdqn_agent::{QrDqnAgent, QrDqnAgentError, QrDqnMetrics};
use crate::algorithms::qrdqn::qrdqn_model::QrDqnModel;

/// Drives QR-DQN training for `total_steps` environment steps.
///
/// The loop is step-oriented, not episode-oriented: exactly `total_steps`
/// environment steps are taken regardless of how many episodes finish.
/// The final episode is intentionally left open — if the last step does not
/// land on a terminal transition, no final reset is issued, which prevents a
/// phantom episode record that would cause an `EpisodeCountMismatch` in
/// recording environments.
///
/// # Arguments
///
/// - `agent` — mutable reference to the [`QrDqnAgent`] being trained.
/// - `env` — mutable reference to the environment; must implement
///   [`Environment`] with matching type parameters.
/// - `rng` — entropy source for ε-greedy exploration and replay sampling.
/// - `total_steps` — number of environment steps to run.
/// - `log_every` — emit a `tracing::info!` progress line every this many
///   steps. Pass `0` to disable all logging.
///
/// # Errors
///
/// Returns [`QrDqnAgentError::InvalidAction`] if either [`Environment::reset`]
/// or [`Environment::step`] returns an `EnvironmentError`.
pub fn train<B, M, E, O, A, R, const DO: usize, const SD: usize, const DB: usize>(
    agent: &mut QrDqnAgent<B, M, O, A, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_steps: usize,
    log_every: usize,
) -> Result<(), QrDqnAgentError>
where
    B: AutodiffBackend,
    M: QrDqnModel<B, DB>,
    E: Environment<DO, SD, 1, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
    R: Reward + Copy,
{
    let mut snapshot = env.reset().map_err(io_from_env)?;
    let mut episode_reward = 0.0_f32;
    let mut episode_steps = 0_usize;
    let mut last_loss = 0.0_f32;
    let mut last_q_mean = 0.0_f32;
    let mut last_spread = 0.0_f32;

    for step in 0..total_steps {
        let obs_current = snapshot.observation().clone();
        let action = agent.act(&obs_current, rng);

        let next_snapshot = env.step(action.clone()).map_err(io_from_env)?;
        let reward_f32: f32 = (*next_snapshot.reward()).into();
        // Two distinct masks — do NOT collapse these back into one.
        //
        // `done` drives episode bookkeeping (metrics, `env.reset()`): the
        // episode is over either way.
        //
        // `terminated` is the Bellman bootstrap mask and is true only for an
        // *environmental* termination. On a truncation (time-limit cutoff) the
        // MDP has not ended, so `next_obs` is a real continuation state and
        // `γ · V(next_obs)` must survive in the target. Masking on `done` here
        // would zero the bootstrap at every timeout and bias Q downward on any
        // time-limited env (Pardo et al., "Time Limits in Reinforcement
        // Learning", ICML 2018, Eq. 6 — partial-episode bootstrapping).
        let done = next_snapshot.is_done();
        let terminated = next_snapshot.is_terminated();
        let next_obs = next_snapshot.observation().clone();

        agent.remember(
            obs_current,
            &action,
            reward_f32,
            next_obs.clone(),
            terminated,
        );
        agent.on_env_step();

        episode_reward += reward_f32;
        episode_steps += 1;

        if agent.should_train()
            && let Some(outcome) = agent.learn_step(rng)
        {
            last_loss = outcome.loss;
            last_q_mean = outcome.q_mean;
            last_spread = outcome.quantile_spread;
        }
        agent.sync_target();
        agent.decay_exploration();

        if done {
            let metrics = QrDqnMetrics {
                reward: episode_reward,
                steps: episode_steps,
                policy_loss: last_loss,
                epsilon: agent.epsilon() as f32,
                q_mean: last_q_mean,
                quantile_spread: last_spread,
            };
            agent.record_episode(metrics);
            episode_reward = 0.0;
            episode_steps = 0;
            // Skip the reset on the final step: that phantom episode never
            // reaches a terminal `step`, so a recording env writes an episode
            // file the manifest's count omits (`EpisodeCountMismatch`).
            if step + 1 < total_steps {
                snapshot = env.reset().map_err(io_from_env)?;
            }
        } else {
            snapshot = next_snapshot;
        }

        if log_every > 0 && (step + 1) % log_every == 0 {
            let avg = agent
                .stats()
                .avg_score()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string());
            tracing::info!(
                step = step + 1,
                total_steps,
                avg_reward = %avg,
                epsilon = agent.epsilon(),
                quantile_spread = last_spread,
                buffer = agent.buffer_len(),
                "qr-dqn training progress"
            );
        }
    }
    Ok(())
}

fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> QrDqnAgentError {
    QrDqnAgentError::InvalidAction(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::{Autodiff, Flex};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::environment::EpisodeStatus;

    type TestBackend = Autodiff<Flex>;

    /// Episodes end every `PERIOD` steps; `STEPS` spans three whole episodes.
    const PERIOD: usize = 3;
    const STEPS: usize = 9;

    use crate::algorithms::bootstrap_mask::{
        AtomNet, DiscreteMaskEnv, MaskDiscreteAction, MaskObservation,
    };
    use crate::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfig;

    /// Drives `train` against a clock env whose episodes all end with
    /// `end_status`, and returns the bootstrap masks the loop recorded.
    ///
    /// `learning_starts` is pushed past the step budget so no gradient step
    /// ever runs — this guard is about *which predicate reaches `remember`*,
    /// not about learning.
    fn recorded_masks(end_status: EpisodeStatus) -> Vec<bool> {
        let device = Default::default();
        let config = QrDqnTrainingConfig {
            learning_starts: 1_000_000,
            replay_buffer_capacity: 64,
            ..QrDqnTrainingConfig::default()
        };
        let net = AtomNet::<TestBackend>::new(config.num_quantiles, &device);
        let mut agent = QrDqnAgent::<
            TestBackend,
            AtomNet<TestBackend>,
            MaskObservation,
            MaskDiscreteAction,
            1,
            2,
        >::new(net, config, device)
        .expect("default config is valid");
        let mut env = DiscreteMaskEnv::new(PERIOD, end_status);
        let mut rng = StdRng::seed_from_u64(7);
        train(&mut agent, &mut env, &mut rng, STEPS, 0).expect("training loop succeeds");
        agent.replay_terminated_flags()
    }

    /// The episode boundaries this fixture produces: steps 3, 6 and 9.
    fn episode_end_steps() -> Vec<bool> {
        (1..=STEPS).map(|s| s.is_multiple_of(PERIOD)).collect()
    }

    /// A genuine environmental termination *must* zero the bootstrap.
    ///
    /// This also pins the fixture's episode cadence, which is what makes the
    /// truncation test below non-vacuous: it proves three of these nine steps
    /// really do end an episode.
    #[test]
    fn test_train_termination_sets_bootstrap_mask() {
        assert_eq!(
            recorded_masks(EpisodeStatus::Terminated),
            episode_end_steps(),
            "an environmental termination must zero the Bellman bootstrap"
        );
    }

    /// A truncation must **not** zero the bootstrap.
    ///
    /// Same env, same cadence, same `is_done()` pattern as the test above —
    /// only `is_terminated()` differs. Masking on `is_done()` here would
    /// reproduce `episode_end_steps()` and bias every Q-value downward on any
    /// time-limited env (Pardo et al., ICML 2018, Eq. 6).
    #[test]
    fn test_train_truncation_leaves_bootstrap_mask_clear() {
        let flags = recorded_masks(EpisodeStatus::Truncated);
        assert_eq!(
            flags.len(),
            STEPS,
            "every step must reach the replay buffer"
        );
        assert!(
            flags.iter().all(|f| !f),
            "truncation must not zero the bootstrap; got {flags:?}"
        );
    }
}
