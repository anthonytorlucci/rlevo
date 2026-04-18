//! End-to-end training loop for DQN.
//!
//! [`train`] drives the collect-learn-sync cycle described in the DQN spec:
//! reset the environment, step with ε-greedy actions, push each transition
//! into the agent's replay buffer, call [`DqnAgent::learn_step`] on the
//! configured cadence, and sync the target network. Per-episode metrics are
//! accumulated into the agent's [`AgentStats`].
//!
//! [`AgentStats`]: evorl_core::metrics::AgentStats

use rand::Rng;

use evorl_core::action::DiscreteAction;
use evorl_core::base::{Observation, Reward, TensorConvertible};
use evorl_core::environment::{Environment, Snapshot};
use burn::tensor::backend::AutodiffBackend;

use crate::algorithms::dqn::dqn_agent::{DqnAgent, DqnAgentError, DqnMetrics};
use crate::algorithms::dqn::dqn_model::DqnModel;

/// Drives DQN training for `total_steps` environment steps.
///
/// The loop is intentionally unparameterised over logging: pass a `log_every`
/// value to print a progress line; pass `0` to disable.
pub fn train<B, M, E, O, A, R, const DO: usize, const SD: usize, const DB: usize>(
    agent: &mut DqnAgent<B, M, O, A, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_steps: usize,
    log_every: usize,
) -> Result<(), DqnAgentError>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
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
            }
        }
        agent.sync_target();
        agent.decay_exploration();

        if done {
            let metrics = DqnMetrics {
                reward: episode_reward,
                steps: episode_steps,
                policy_loss: last_loss,
                value_loss: last_loss,
                epsilon: agent.epsilon() as f32,
                q_mean: last_q_mean,
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
                buffer = agent.buffer_len(),
                "dqn training progress"
            );
        }
    }
    Ok(())
}

fn io_from_env(err: evorl_core::environment::EnvironmentError) -> DqnAgentError {
    DqnAgentError::InvalidAction(err.to_string())
}
