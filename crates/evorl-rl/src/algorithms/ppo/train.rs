//! End-to-end training loop for PPO.
//!
//! Iterates: **collect rollout** (sequential `num_steps` env steps) →
//! **finalise** (compute GAE) → **update** (`update_epochs × num_minibatches`
//! gradient updates) → **record episode metrics**. The loop owns the
//! env/agent/rng references and threads them through; it is *not* a trait
//! impl so the entire sequence reads top-to-bottom in one function, matching
//! CleanRL's pedagogical style.
//!
//! Two entry points:
//! - [`train_discrete`] — for `DiscreteAction<1>` envs paired with a
//!   categorical policy head.
//! - [`train_continuous`] — for `ContinuousAction<AD>` envs paired with a
//!   tanh-Gaussian policy head (or any policy whose `raw_to_env_row` produces
//!   `ContinuousAction::from_slice`-compatible values).

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use evorl_core::action::{ContinuousAction, DiscreteAction};
use evorl_core::base::{Observation, Reward, TensorConvertible};
use evorl_core::environment::{Environment, Snapshot};

use crate::algorithms::ppo::ppo_agent::{PpoAgent, PpoAgentError, PpoMetrics, PpoUpdateStats};
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;

/// PPO training loop against a **discrete** action environment.
///
/// Pass `log_every > 0` to enable periodic `tracing::info!` progress; set
/// `log_every == 0` to suppress logging.
pub fn train_discrete<
    B,
    P,
    V,
    E,
    O,
    A,
    R,
    const DO: usize,
    const SD: usize,
    const DB: usize,
>(
    agent: &mut PpoAgent<B, P, V, O, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_timesteps: usize,
    log_every: usize,
) -> Result<(), PpoAgentError>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    E: Environment<DO, SD, 1, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B>,
    A: DiscreteAction<1>,
    R: Reward + Copy,
{
    run_loop(agent, env, rng, total_timesteps, log_every, |row| {
        assert_eq!(row.len(), 1, "discrete action row must have length 1");
        A::from_index(row[0] as usize)
    })
}

/// PPO training loop against a **continuous** action environment.
pub fn train_continuous<
    B,
    P,
    V,
    E,
    O,
    A,
    R,
    const DO: usize,
    const SD: usize,
    const AD: usize,
    const DB: usize,
>(
    agent: &mut PpoAgent<B, P, V, O, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_timesteps: usize,
    log_every: usize,
) -> Result<(), PpoAgentError>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    E: Environment<DO, SD, AD, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B>,
    A: ContinuousAction<AD>,
    R: Reward + Copy,
{
    run_loop(agent, env, rng, total_timesteps, log_every, A::from_slice)
}

fn run_loop<B, P, V, E, O, A, R, F, const DO: usize, const SD: usize, const AD: usize, const DB: usize>(
    agent: &mut PpoAgent<B, P, V, O, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_timesteps: usize,
    log_every: usize,
    action_from_row: F,
) -> Result<(), PpoAgentError>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    E: Environment<DO, SD, AD, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B>,
    A: evorl_core::base::Action<AD> + Clone,
    R: Reward + Copy,
    F: Fn(&[f32]) -> A,
{
    let rollout_len = agent.config().num_steps;

    let mut snapshot = env.reset().map_err(io_from_env)?;
    let mut episode_reward = 0.0_f32;
    let mut episode_steps = 0_usize;
    let mut last_update_stats = PpoUpdateStats {
        policy_loss: 0.0,
        value_loss: 0.0,
        entropy: 0.0,
        approx_kl: 0.0,
        clip_frac: 0.0,
        epochs_run: 0,
    };
    let mut last_done_in_rollout = false;
    let mut global_step = 0_usize;

    while global_step < total_timesteps {
        // -------- Collect rollout --------
        for _ in 0..rollout_len {
            let obs_now = snapshot.observation().clone();
            let act = agent.act(&obs_now, rng);

            let typed_action = action_from_row(&act.env_row);
            let next_snapshot = env.step(typed_action).map_err(io_from_env)?;
            let reward_f32: f32 = (*next_snapshot.reward()).into();
            let status = next_snapshot.status();
            let done = status.is_done();

            agent.record_step(obs_now, &act, reward_f32, status);
            global_step += 1;

            episode_reward += reward_f32;
            episode_steps += 1;

            if done {
                last_done_in_rollout = true;
                let metrics = PpoMetrics {
                    reward: episode_reward,
                    steps: episode_steps,
                    policy_loss: last_update_stats.policy_loss,
                    value_loss: last_update_stats.value_loss,
                    entropy: last_update_stats.entropy,
                    approx_kl: last_update_stats.approx_kl,
                    clip_frac: last_update_stats.clip_frac,
                    learning_rate: agent.current_learning_rate() as f32,
                };
                agent.record_episode(metrics);
                episode_reward = 0.0;
                episode_steps = 0;
                snapshot = env.reset().map_err(io_from_env)?;
            } else {
                last_done_in_rollout = false;
                snapshot = next_snapshot;
            }

            if global_step >= total_timesteps {
                break;
            }
        }

        // -------- Update --------
        let last_obs_for_bootstrap = snapshot.observation().clone();
        agent.finalize_rollout(&last_obs_for_bootstrap, last_done_in_rollout);
        last_update_stats = agent.update(rng);

        if log_every > 0 && global_step.is_multiple_of(log_every) {
            let avg = agent
                .stats()
                .avg_score()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string());
            tracing::info!(
                step = global_step,
                total_steps = total_timesteps,
                iteration = agent.iteration(),
                avg_reward = %avg,
                policy_loss = last_update_stats.policy_loss,
                value_loss = last_update_stats.value_loss,
                entropy = last_update_stats.entropy,
                approx_kl = last_update_stats.approx_kl,
                clip_frac = last_update_stats.clip_frac,
                "ppo training progress"
            );
        }
    }

    Ok(())
}

fn io_from_env(err: evorl_core::environment::EnvironmentError) -> PpoAgentError {
    PpoAgentError::Environment(err.to_string())
}
