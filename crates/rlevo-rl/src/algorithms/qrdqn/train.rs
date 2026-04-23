//! End-to-end training loop for QR-DQN.
//!
//! Mirrors [`crate::algorithms::c51::train`]. The only behavioural difference
//! is the metrics type emitted per completed episode — [`QrDqnMetrics`]
//! carries a `quantile_spread` diagnostic in place of C51's `entropy`.

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::qrdqn::qrdqn_agent::{QrDqnAgent, QrDqnAgentError, QrDqnMetrics};
use crate::algorithms::qrdqn::qrdqn_model::QrDqnModel;

/// Drives QR-DQN training for `total_steps` environment steps.
///
/// Pass `log_every > 0` to enable periodic progress logging via `tracing`;
/// `log_every == 0` disables it.
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
        let done = next_snapshot.is_done();
        let next_obs = next_snapshot.observation().clone();

        agent.remember(obs_current, &action, reward_f32, next_obs.clone(), done);
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
            snapshot = env.reset().map_err(io_from_env)?;
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
