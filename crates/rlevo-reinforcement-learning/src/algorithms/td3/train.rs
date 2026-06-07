//! End-to-end training loop for TD3.
//!
//! [`train`] drives the same collect-learn cycle as DDPG: reset the
//! environment, act via [`Td3Agent::act`] (uniform random during warm-up,
//! noisy actor afterwards), push each transition into the replay buffer, and
//! invoke [`Td3Agent::learn_step`] once warm-up is complete. The only
//! TD3-specific behaviour (twin critics, target smoothing, delayed policy
//! updates) is wholly contained inside `learn_step`.

use burn::tensor::backend::AutodiffBackend;
use rand::Rng;

use rlevo_core::action::BoundedAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::td3::td3_agent::{Td3Agent, Td3AgentError, Td3Metrics};
use crate::algorithms::td3::td3_model::{ContinuousQ, DeterministicPolicy};

/// Drives TD3 training for `total_steps` environment steps.
///
/// The loop runs the standard collect-then-learn cycle:
///
/// 1. Observe the current state and call [`Td3Agent::act`] (uniform random
///    during warm-up, noisy actor afterwards).
/// 2. Step the environment and store the transition with
///    [`Td3Agent::remember`].
/// 3. Call [`Td3Agent::learn_step`]; all TD3-specific logic (twin critics,
///    target-policy smoothing, delayed actor updates) happens inside that
///    method.
/// 4. On episode termination, record metrics via
///    [`Td3Agent::record_episode`] and reset the environment — except on
///    the final step, where the reset is skipped to avoid writing a
///    partial episode to any recording sink.
///
/// Progress is logged via [`tracing::info!`] every `log_every` steps.
/// Pass `log_every = 0` to disable all progress logging.
///
/// # Errors
///
/// Returns [`Td3AgentError::InvalidAction`] if `env.reset()` or `env.step()`
/// returns an [`rlevo_core::environment::EnvironmentError`]. The agent's
/// internal state (buffer, step counter, statistics) reflects all transitions
/// completed before the error.
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
    agent: &mut Td3Agent<B, Actor, Critic, O, A, DO, DB, DA, DAB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_steps: usize,
    log_every: usize,
) -> Result<(), Td3AgentError>
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
            let metrics = Td3Metrics {
                reward: episode_reward,
                steps: episode_steps,
                critic_loss: last_critic_loss,
                actor_loss: agent.last_actor_loss(),
                q_mean: last_q_mean,
            };
            agent.record_episode(metrics);
            episode_reward = 0.0;
            episode_steps = 0;
            // Skip the reset on the final step: that phantom episode never
            // reaches a terminal `step`, so a recording env writes an episode
            // file the manifest's count omits (`EpisodeCountMismatch`).
            if step + 1 < total_steps {
                snapshot = env.reset().map_err(env_to_err)?;
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
                critic_loss = last_critic_loss,
                q_mean = last_q_mean,
                buffer = agent.buffer_len(),
                "td3 training progress"
            );
        }
    }
    Ok(())
}

/// Converts an [`rlevo_core::environment::EnvironmentError`] into a
/// [`Td3AgentError`] for use as a `map_err` adapter in [`train`].
fn env_to_err(err: rlevo_core::environment::EnvironmentError) -> Td3AgentError {
    Td3AgentError::InvalidAction(err.to_string())
}
