//! End-to-end training loop for DDPG.
//!
//! [`train`] drives the collect-learn cycle: reset the environment, act via
//! [`DdpgAgent::act`] (uniform random during
//! warm-up, noisy actor afterwards), push each transition into the replay
//! buffer, and invoke [`DdpgAgent::learn_step`] once the warm-up is complete.

use burn::tensor::backend::AutodiffBackend;
use rand::Rng;

use evorl_core::action::BoundedAction;
use evorl_core::base::{Observation, Reward, TensorConvertible};
use evorl_core::environment::{Environment, Snapshot};

use crate::algorithms::ddpg::ddpg_agent::{DdpgAgent, DdpgAgentError, DdpgMetrics};
use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};

/// Drives DDPG training for `total_steps` environment steps.
///
/// Pass `log_every = 0` to disable progress logging.
#[allow(clippy::too_many_arguments)]
pub fn train<
    B,
    Actor,
    Critic,
    E,
    O,
    A,
    R,
    const DO: usize,
    const SD: usize,
    const DB: usize,
    const DA: usize,
    const DAB: usize,
>(
    agent: &mut DdpgAgent<B, Actor, Critic, O, A, DO, DB, DA, DAB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_steps: usize,
    log_every: usize,
) -> Result<(), DdpgAgentError>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    E: Environment<DO, SD, DA, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend> + Clone,
    A: BoundedAction<DA>,
    R: Reward + Copy,
{
    let mut snapshot = env.reset().map_err(env_to_err)?;
    let mut episode_reward = 0.0_f32;
    let mut episode_steps = 0_usize;
    let mut last_critic_loss = 0.0_f32;
    let mut last_q_mean = 0.0_f32;

    for step in 0..total_steps {
        let obs_current = snapshot.observation().clone();
        let action = agent.act(&obs_current, true, rng);

        let next_snapshot = env.step(action.clone()).map_err(env_to_err)?;
        let reward_f32: f32 = (*next_snapshot.reward()).into();
        let done = next_snapshot.is_done();
        let next_obs = next_snapshot.observation().clone();

        agent.remember(obs_current, &action, reward_f32, next_obs, done);
        agent.on_env_step();

        episode_reward += reward_f32;
        episode_steps += 1;

        if let Some(outcome) = agent.learn_step(rng) {
            last_critic_loss = outcome.critic_loss;
            last_q_mean = outcome.q_mean;
        }

        if done {
            let metrics = DdpgMetrics {
                reward: episode_reward,
                steps: episode_steps,
                critic_loss: last_critic_loss,
                actor_loss: agent.last_actor_loss(),
                q_mean: last_q_mean,
            };
            agent.record_episode(metrics);
            episode_reward = 0.0;
            episode_steps = 0;
            snapshot = env.reset().map_err(env_to_err)?;
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
                critic_loss = last_critic_loss,
                q_mean = last_q_mean,
                buffer = agent.buffer_len(),
                "ddpg training progress"
            );
        }
    }
    Ok(())
}

fn env_to_err(err: evorl_core::environment::EnvironmentError) -> DdpgAgentError {
    DdpgAgentError::InvalidAction(err.to_string())
}
