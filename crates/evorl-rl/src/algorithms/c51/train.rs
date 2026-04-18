//! End-to-end training loop for C51.
//!
//! Mirrors [`crate::algorithms::dqn::train`]. The only behavioural difference
//! is the metrics type emitted per completed episode — [`C51Metrics`] carries
//! an additional `entropy` field in place of DQN's `value_loss` mirror.

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use evorl_core::action::DiscreteAction;
use evorl_core::base::{Observation, Reward, TensorConvertible};
use evorl_core::environment::{Environment, Snapshot};

use crate::algorithms::c51::c51_agent::{C51Agent, C51AgentError, C51Metrics};
use crate::algorithms::c51::c51_model::C51Model;

/// Drives C51 training for `total_steps` environment steps.
///
/// Pass `log_every > 0` to enable periodic progress logging via `tracing`;
/// `log_every == 0` disables it.
pub fn train<B, M, E, O, A, R, const DO: usize, const SD: usize, const DB: usize>(
    agent: &mut C51Agent<B, M, O, A, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_steps: usize,
    log_every: usize,
) -> Result<(), C51AgentError>
where
    B: AutodiffBackend,
    M: C51Model<B, DB>,
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
    let mut last_entropy = 0.0_f32;

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

        if agent.should_train() {
            if let Some(outcome) = agent.learn_step(rng) {
                last_loss = outcome.loss;
                last_q_mean = outcome.q_mean;
                last_entropy = outcome.entropy;
            }
        }
        agent.sync_target();
        agent.decay_exploration();

        if done {
            let metrics = C51Metrics {
                reward: episode_reward,
                steps: episode_steps,
                policy_loss: last_loss,
                epsilon: agent.epsilon() as f32,
                q_mean: last_q_mean,
                entropy: last_entropy,
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
                entropy = last_entropy,
                buffer = agent.buffer_len(),
                "c51 training progress"
            );
        }
    }
    Ok(())
}

fn io_from_env(err: evorl_core::environment::EnvironmentError) -> C51AgentError {
    C51AgentError::InvalidAction(err.to_string())
}
