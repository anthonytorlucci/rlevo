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

use std::time::Instant;

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use rlevo_core::action::{ContinuousAction, DiscreteAction};
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::ppo::ppo_agent::{PpoAgent, PpoAgentError, PpoMetrics, PpoUpdateStats};
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;

/// PPO training loop against a **discrete** action environment.
///
/// Pass `log_every > 0` to enable periodic `tracing::info!` progress; set
/// `log_every == 0` to suppress logging.
pub fn train_discrete<B, P, V, E, O, A, R, const DO: usize, const SD: usize, const DB: usize>(
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
///
/// Identical control flow to [`train_discrete`] but expects the environment's
/// action type to implement [`ContinuousAction<AD>`] and reconstructs each
/// action from the policy's `env_row` (the tanh-squashed, scaled output of
/// [`PpoPolicy::raw_to_env_row`]) via [`ContinuousAction::from_slice`].
///
/// Pass `log_every > 0` to enable periodic `tracing::info!` progress; set
/// `log_every == 0` to suppress logging.
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

fn run_loop<
    B,
    P,
    V,
    E,
    O,
    A,
    R,
    F,
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
    action_from_row: F,
) -> Result<(), PpoAgentError>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    E: Environment<DO, SD, AD, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B>,
    A: rlevo_core::base::Action<AD> + Clone,
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
        old_approx_kl: 0.0,
        clip_frac: 0.0,
        explained_variance: 0.0,
        epochs_run: 0,
    };
    let mut global_step = 0_usize;
    let loop_start = Instant::now();

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
                // Only re-open an episode if training will continue. Resetting
                // on the *final* done opens a fresh episode that we never step
                // to completion — for recording envs that leaves a phantom
                // episode file the manifest's count omits, tripping
                // `EpisodeCountMismatch`. Leaving `snapshot` stale is safe:
                // the rollout's final step is recorded as done, so
                // `finalize_rollout` never reads the observation at all.
                if global_step < total_timesteps {
                    snapshot = env.reset().map_err(io_from_env)?;
                }
            } else {
                snapshot = next_snapshot;
            }

            if global_step >= total_timesteps {
                break;
            }
        }

        // -------- Update --------
        // Borrowed, not cloned: `finalize_rollout` only reads this when the
        // rollout's final step left the episode `Running`, which is exactly
        // when `snapshot` is the genuine continuation state.
        agent.finalize_rollout(snapshot.observation());
        last_update_stats = agent.update(rng);

        if log_every > 0 && global_step.is_multiple_of(log_every) {
            let avg = agent
                .stats()
                .avg_score()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "n/a".to_string());
            // Per-iteration episode-return statistics over the recent-episode
            // window. Cheap: a handful of arithmetic ops on a bounded VecDeque,
            // gated behind the periodic-log guard so the hot path is untouched.
            let ret_stats = EpisodeReturnStats::from_window(&agent.stats().recent_history);
            let elapsed = loop_start.elapsed().as_secs_f64();
            let steps_per_sec = if elapsed > 0.0 {
                global_step as f64 / elapsed
            } else {
                0.0
            };
            tracing::info!(
                step = global_step,
                total_steps = total_timesteps,
                iteration = agent.iteration(),
                avg_reward = %avg,
                policy_loss = last_update_stats.policy_loss,
                value_loss = last_update_stats.value_loss,
                entropy = last_update_stats.entropy,
                approx_kl = last_update_stats.approx_kl,
                old_approx_kl = last_update_stats.old_approx_kl,
                clip_frac = last_update_stats.clip_frac,
                explained_variance = last_update_stats.explained_variance,
                episode_return_mean = ret_stats.mean,
                episode_return_std = ret_stats.std,
                episode_return_min = ret_stats.min,
                episode_return_max = ret_stats.max,
                episode_length_mean = ret_stats.length_mean,
                env_steps_sampled = global_step,
                steps_per_sec = steps_per_sec,
                learning_rate = agent.current_learning_rate(),
                "ppo training progress"
            );
        }
    }

    Ok(())
}

/// Summary statistics over a window of recently completed episodes.
///
/// Used only to populate the periodic `tracing` progress event; computed off
/// the agent's bounded recent-episode window, never per step.
struct EpisodeReturnStats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    length_mean: f32,
}

impl EpisodeReturnStats {
    /// Folds a window of [`PerformanceRecord`]s into return/length summaries.
    ///
    /// An empty window yields all-zero stats (the value the report suppresses).
    fn from_window<'a, T, I>(window: I) -> Self
    where
        T: crate::metrics::PerformanceRecord + 'a,
        I: IntoIterator<Item = &'a T>,
    {
        let mut n = 0_usize;
        let mut sum = 0.0_f32;
        let mut sum_len = 0.0_f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut scores: Vec<f32> = Vec::new();
        for rec in window {
            let s = rec.score();
            scores.push(s);
            sum += s;
            sum_len += rec.duration() as f32;
            min = min.min(s);
            max = max.max(s);
            n += 1;
        }
        if n == 0 {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                length_mean: 0.0,
            };
        }
        let n_f = n as f32;
        let mean = sum / n_f;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n_f;
        Self {
            mean,
            std: var.sqrt(),
            min,
            max,
            length_mean: sum_len / n_f,
        }
    }
}

fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> PpoAgentError {
    PpoAgentError::Environment(err.to_string())
}
