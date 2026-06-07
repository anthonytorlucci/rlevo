//! End-to-end training loop for DDPG.
//!
//! [`train`] drives the standard collect-learn cycle:
//!
//! 1. Reset the environment at the start of each episode.
//! 2. Act via [`DdpgAgent::act`] — uniform random during warm-up, noisy actor
//!    output afterwards.
//! 3. Push each `(obs, action, reward, next_obs, done)` transition into the
//!    replay buffer via [`DdpgAgent::remember`].
//! 4. Invoke [`DdpgAgent::learn_step`] — a no-op during warm-up, one critic
//!    update plus a conditional actor + Polyak update thereafter.
//! 5. Log progress via [`tracing::info!`] every `log_every` steps when
//!    `log_every > 0`.
//!
//! The function returns when `total_steps` environment interactions have been
//! collected, or immediately on the first environment error.

use burn::tensor::backend::AutodiffBackend;
use rand::Rng;

use rlevo_core::action::BoundedAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::ddpg::ddpg_agent::{DdpgAgent, DdpgAgentError, DdpgMetrics};
use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};

/// Drives DDPG training for `total_steps` environment steps.
///
/// On every done step the episode metrics are committed to the agent's
/// [`AgentStats`](crate::metrics::AgentStats) via
/// [`DdpgAgent::record_episode`]. The environment is not reset after the
/// final done step to avoid creating a phantom episode that the record
/// manifest would not account for.
///
/// # Arguments
///
/// - `agent` — mutable reference to the DDPG agent.
/// - `env` — mutable reference to an environment compatible with the agent's
///   observation and action types.
/// - `rng` — any [`Rng`] implementation; must remain consistent between calls
///   if reproducibility is required.
/// - `total_steps` — number of environment interactions to run.
/// - `log_every` — emit a [`tracing::info!`] progress line every this many
///   steps; pass `0` to disable logging.
///
/// # Errors
///
/// Returns [`DdpgAgentError::InvalidAction`] (wrapping the underlying
/// [`EnvironmentError`](rlevo_core::environment::EnvironmentError) message)
/// if either [`Environment::reset`]
/// or [`Environment::step`] fails.
///
/// # Const generics
///
/// The same const-generic parameters as [`DdpgAgent`]: `DO`/`DB` are the
/// single and batched observation ranks, `DA`/`DAB` the single and batched
/// action ranks. `SD` is the state rank required by the `Environment` bound
/// and is not used directly by this function.
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
                "ddpg training progress"
            );
        }
    }
    Ok(())
}

fn env_to_err(err: rlevo_core::environment::EnvironmentError) -> DdpgAgentError {
    DdpgAgentError::InvalidAction(err.to_string())
}
