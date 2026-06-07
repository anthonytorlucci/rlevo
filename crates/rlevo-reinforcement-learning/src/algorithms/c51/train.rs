//! End-to-end training loop for C51 (Categorical DQN).
//!
//! Provides a single entry-point, [`train`], that drives a [`C51Agent`]
//! against any [`Environment`] for a fixed number of environment steps. The
//! loop handles ε-greedy action selection, replay-buffer ingestion, periodic
//! gradient updates, target-network synchronisation, episode boundary resets,
//! and optional `tracing` progress logging.
//!
//! The structure mirrors [`crate::algorithms::dqn::train`]; the only
//! behavioural difference is the metrics type emitted per completed episode —
//! [`C51Metrics`] carries an additional `entropy` field that tracks the
//! predicted return-distribution entropy and is absent from DQN's metrics.

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::c51::c51_agent::{C51Agent, C51AgentError, C51Metrics};
use crate::algorithms::c51::c51_model::C51Model;

/// Drives C51 training for `total_steps` environment steps.
///
/// The loop runs one environment step per iteration. After collecting each
/// transition, it calls [`C51Agent::learn_step`] whenever
/// [`C51Agent::should_train`] returns `true`, synchronises the target network
/// via [`C51Agent::sync_target`], and decays ε via
/// [`C51Agent::decay_exploration`]. On episode termination the loop records
/// per-episode [`C51Metrics`] into the agent's rolling statistics window and
/// calls [`Environment::reset`] to begin a new episode, except on the very
/// last step (to avoid recording a phantom episode in recording environments).
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor (e.g. `1` for flat vector
///   observations).
/// - `SD` — rank of the state tensor (unused by the loop itself; forwarded to
///   the environment bound).
/// - `DB` — rank of a batched observation tensor (`DO + 1`).
///
/// # Arguments
///
/// - `agent` — mutable reference to the agent being trained.
/// - `env` — mutable reference to the environment.
/// - `rng` — random number generator used for ε-greedy exploration and
///   uniform replay-buffer sampling.
/// - `total_steps` — number of environment steps to execute.
/// - `log_every` — emit a `tracing::info!` progress line every this many
///   steps. Pass `0` to disable logging.
///
/// # Errors
///
/// Returns [`C51AgentError::InvalidAction`] (wrapping the underlying
/// [`rlevo_core::environment::EnvironmentError`]) if the environment's
/// `reset` or `step` call fails.
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

        if agent.should_train()
            && let Some(outcome) = agent.learn_step(rng)
        {
            last_loss = outcome.loss;
            last_q_mean = outcome.q_mean;
            last_entropy = outcome.entropy;
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
                entropy = last_entropy,
                buffer = agent.buffer_len(),
                "c51 training progress"
            );
        }
    }
    Ok(())
}

fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> C51AgentError {
    C51AgentError::InvalidAction(err.to_string())
}
