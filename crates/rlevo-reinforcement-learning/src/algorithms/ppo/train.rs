//! End-to-end training loop for PPO.
//!
//! Iterates: **collect rollout** (sequential `num_steps` env steps) →
//! **finalise** (compute GAE) → **update** (`update_epochs × num_minibatches`
//! gradient updates) → **record episode metrics**. The loop owns the
//! env/agent/rng references and threads them through; it is *not* a trait
//! impl so the entire sequence reads top-to-bottom in one function, matching
//! `CleanRL`'s pedagogical style.
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
use crate::algorithms::shared::LogWatermark;

/// PPO training loop against a **discrete** action environment.
///
/// Pass `log_every > 0` to enable periodic `tracing::info!` progress; set
/// `log_every == 0` to suppress logging. Progress can only be emitted at a
/// rollout boundary (the payload reads the update statistics), so a line lands
/// at the first boundary at or after each `log_every` steps have elapsed —
/// never less often, and at most once per rollout.
/// # Errors
///
/// Returns [`PpoAgentError`] if the environment's `reset` or `step` fails, or
/// if a rollout update cannot be applied.
///
/// # Panics
///
/// Panics if the policy emits an action row whose length is not 1, which means
/// the head was built for a different action arity than the environment's.
// Action indices only. `argmax` yields a non-negative index below
// `A::ACTION_COUNT`, so the i64 -> usize narrowing can neither wrap nor lose a
// sign; where an index round-trips through f32 it stays far below the 2^24
// exact-integer limit. `from_index` bounds-checks on the way back.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
/// `log_every == 0` to suppress logging. The cadence is the rollout-boundary
/// one described on [`train_discrete`].
/// # Errors
///
/// Returns [`PpoAgentError`] if the environment's `reset` or `step` fails, or
/// if a rollout update cannot be applied.
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

// Config knobs are stored as f64 for ergonomics; every tensor in this crate is
// f32. This is the intended narrowing point, and the values are hyperparameters
// (rates, discounts, epsilons) where f32 has far more precision than the
// schedules that produce them.
#[allow(clippy::cast_possible_truncation)]
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
        min_log_std: None,
    };
    let mut global_step = 0_usize;
    // Watermark, not `global_step % log_every`: the step counter is only
    // sampled at rollout boundaries (multiples of `num_steps`), so a
    // divisibility test would fire on `lcm(num_steps, log_every)` — see
    // `LogWatermark` and issue #321.
    let mut log_watermark = LogWatermark::new(log_every);
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

            // `next_snapshot.observation()` is the continuation state, read
            // *before* the `env.reset()` below — on a truncation the agent
            // bootstraps V from it (ADR 0048), so a post-reset observation
            // here would silently poison the advantage.
            agent.record_step(
                obs_now,
                &act,
                reward_f32,
                next_snapshot.observation(),
                status,
            );
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

        if log_watermark.should_log(global_step) {
            emit_progress(
                agent,
                &last_update_stats,
                global_step,
                total_timesteps,
                loop_start,
            );
        }
    }

    // The final rollout is usually partial, so the last boundary can fall short
    // of `log_every` past the watermark and never fire — which would drop the
    // run's final update stats. Report the terminal step unless it was already
    // logged above.
    if log_watermark.should_log_final(global_step) {
        emit_progress(
            agent,
            &last_update_stats,
            global_step,
            total_timesteps,
            loop_start,
        );
    }

    Ok(())
}

/// Emits one PPO progress event.
///
/// Factored out so the periodic in-loop line and the terminal line are
/// literally the same event: with ~18 fields, two copies of the
/// `tracing::info!` would drift the moment either is edited. Kept as a free
/// function rather than a closure because the loop needs `agent` mutably
/// between calls, which an `agent`-capturing closure would forbid.
// Divisor/normalizer derived from a count -- batch size, minibatch count,
// history length, iteration number. All are bounded by configured sizes far
// below f32's 2^24 (f64's 2^53) exact-integer limit.
#[allow(clippy::cast_precision_loss)]
fn emit_progress<B, P, V, O, const DO: usize, const DB: usize>(
    agent: &PpoAgent<B, P, V, O, DO, DB>,
    stats: &PpoUpdateStats,
    global_step: usize,
    total_timesteps: usize,
    loop_start: Instant,
) where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    let avg = agent
        .stats()
        .avg_score()
        .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2}"));
    // Per-iteration episode-return statistics over the recent-episode window.
    // Cheap: a handful of arithmetic ops on a bounded VecDeque, and only ever
    // reached behind the periodic-log guard so the hot path is untouched.
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
        policy_loss = stats.policy_loss,
        value_loss = stats.value_loss,
        entropy = stats.entropy,
        approx_kl = stats.approx_kl,
        old_approx_kl = stats.old_approx_kl,
        clip_frac = stats.clip_frac,
        explained_variance = stats.explained_variance,
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
    // Divisor/normalizer derived from a count -- batch size, minibatch count,
    // history length, iteration number. All are bounded by configured sizes far
    // below f32's 2^24 (f64's 2^53) exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
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

// Passed by name to `.map_err(..)`, which hands the closure an owned
// `EnvironmentError`. Taking `&EnvironmentError` to satisfy the lint would force
// a `|e| ..(&e)` closure at every call site for no benefit.
#[allow(clippy::needless_pass_by_value)]
fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> PpoAgentError {
    PpoAgentError::Environment(err.to_string())
}
