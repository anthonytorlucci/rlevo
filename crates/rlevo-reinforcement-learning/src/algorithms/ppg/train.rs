//! End-to-end training loop for PPG (discrete action spaces).
//!
//! One iteration:
//!
//! 1. Collect `num_steps` env steps into the rollout buffer.
//! 2. Finalise GAE.
//! 3. Snapshot `(obs, returns, old_logits)` into the auxiliary buffer.
//! 4. Run the PPO policy-phase update (clears the rollout buffer).
//! 5. If the auxiliary buffer has accumulated `n_iteration` rollouts, run
//!    `e_aux` epochs of auxiliary updates and drain it.
//!
//! v1 is discrete-only; a continuous variant will land alongside a
//! tanh-Gaussian PPG head.

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::ppg::ppg_agent::{AuxPhaseStats, PpgAgent, PpgAgentError, PpgMetrics};
use crate::algorithms::ppg::ppg_policy::PpgAuxValueHead;
use crate::algorithms::ppo::ppo_agent::PpoUpdateStats;
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;
use crate::algorithms::shared::LogWatermark;

/// Run the PPG training loop against a **discrete** action environment.
///
/// Executes the five-step PPG iteration — rollout collection, GAE, auxiliary-
/// buffer snapshot, policy-phase update, and optional auxiliary phase — until
/// `total_timesteps` environment steps have been taken.
///
/// Episode boundaries inside a rollout are handled transparently: the
/// environment is reset immediately after a terminal step and a new episode
/// begins in the same rollout. A reset is deliberately skipped on the *final*
/// terminal step to avoid opening a phantom episode that the recording
/// infrastructure would never close.
///
/// # Parameters
///
/// - `agent` — mutable PPG agent; holds all network weights and buffers.
/// - `env` — mutable environment instance; reset internally on episode end.
/// - `rng` — random number generator; used for action sampling and minibatch
///   shuffling.
/// - `total_timesteps` — total environment steps to collect before returning.
/// - `log_every` — emit a `tracing::info!` log line every this many global
///   steps; pass `0` to disable logging. Progress can only be emitted at a
///   rollout boundary (the payload reads the policy-phase and auxiliary-phase
///   statistics), so a line lands at the first boundary at or after each
///   `log_every` steps have elapsed — never less often, and at most once per
///   rollout.
///
/// # Errors
///
/// Returns `PpgAgentError::Environment` if the environment's `reset` or `step`
/// returns an error.
///
/// # Type parameters
///
/// - `B` — Burn autodiff backend.
/// - `P` — Policy network: must implement `PpoPolicy<B, DB>` and
///   `PpgAuxValueHead<B, DB>`.
/// - `V` — Main value network implementing `PpoValue<B, DB>`.
/// - `E` — Environment with `ObservationType = O`, `ActionType = A`,
///   `RewardType = R`, and an action space of rank `1` (discrete).
/// - `O` — Observation type convertible to a rank-`DO` tensor.
/// - `A` — Discrete action implementing `DiscreteAction<1>`.
/// - `R` — Scalar reward implementing `Reward + Copy`.
/// - `DO` — Observation tensor rank.
/// - `SD` — State tensor rank (consumed by the environment bound only).
/// - `DB` — Batched observation tensor rank (`DO + 1`).
// Config knobs are stored as f64 for ergonomics; every tensor in this crate is
// f32. This is the intended narrowing point, and the values are hyperparameters
// (rates, discounts, epsilons) where f32 has far more precision than the
// schedules that produce them.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn train_discrete<B, P, V, E, O, A, R, const DO: usize, const SD: usize, const DB: usize>(
    agent: &mut PpgAgent<B, P, V, O, DO, DB>,
    env: &mut E,
    rng: &mut impl Rng,
    total_timesteps: usize,
    log_every: usize,
) -> Result<(), PpgAgentError>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    E: Environment<DO, SD, 1, ObservationType = O, ActionType = A, RewardType = R>,
    O: Observation<DO> + TensorConvertible<DO, B>,
    A: DiscreteAction<1>,
    R: Reward + Copy,
{
    let rollout_len = agent.config().ppo.num_steps;

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
    // Hoisted out of the loop so the terminal progress line can report the
    // final iteration's auxiliary phase. Left loop-local it would be out of
    // scope after the loop, and the closing line would claim `aux_ran = false`
    // for a rollout that may well have run one.
    let mut last_aux: Option<AuxPhaseStats> = None;
    // Watermark, not `global_step % log_every`: the step counter is only
    // sampled at rollout boundaries (multiples of `num_steps`), so a
    // divisibility test would fire on `lcm(num_steps, log_every)` — see
    // `LogWatermark` and issue #321.
    let mut log_watermark = LogWatermark::new(log_every);

    while global_step < total_timesteps {
        for _ in 0..rollout_len {
            let obs_now = snapshot.observation().clone();
            let act = agent.act(&obs_now, rng);

            let typed_action = A::from_index(act.env_row[0] as usize);
            let next_snapshot = env.step(typed_action).map_err(io_from_env)?;
            let reward_f32: f32 = (*next_snapshot.reward()).into();
            let status = next_snapshot.status();
            let done = status.is_done();

            // Continuation state, read *before* the `env.reset()` below — on a
            // truncation the agent bootstraps V from it (ADR 0048).
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
                let aux_last = agent.last_aux_phase();
                let metrics = PpgMetrics {
                    reward: episode_reward,
                    steps: episode_steps,
                    policy_loss: last_update_stats.policy_loss,
                    value_loss: last_update_stats.value_loss,
                    entropy: last_update_stats.entropy,
                    approx_kl: last_update_stats.approx_kl,
                    clip_frac: last_update_stats.clip_frac,
                    aux_main_value_loss: aux_last.map_or(0.0, |a| a.main_value_loss),
                    aux_value_loss: aux_last.map_or(0.0, |a| a.aux_value_loss),
                    aux_policy_kl: aux_last.map_or(0.0, |a| a.policy_kl),
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

        // Borrowed, not cloned: `finalize_rollout` only reads this when the
        // rollout's final step left the episode `Running`, which is exactly
        // when `snapshot` is the genuine continuation state.
        agent.finalize_rollout(snapshot.observation());
        agent.snapshot_into_aux_buffer();
        last_update_stats = agent.policy_phase_update(rng);
        last_aux = agent.maybe_aux_phase(rng);

        if log_watermark.should_log(global_step) {
            emit_progress(
                agent,
                &last_update_stats,
                last_aux,
                global_step,
                total_timesteps,
            );
        }
    }

    // The final rollout is usually partial, so the last boundary can fall short
    // of `log_every` past the watermark and never fire — which would drop the
    // run's final policy- and auxiliary-phase stats. Report the terminal step
    // unless it was already logged above.
    if log_watermark.should_log_final(global_step) {
        emit_progress(
            agent,
            &last_update_stats,
            last_aux,
            global_step,
            total_timesteps,
        );
    }

    Ok(())
}

/// Emits one PPG progress event.
///
/// Factored out so the periodic in-loop line and the terminal line are
/// literally the same event rather than two copies that drift apart. Kept as a
/// free function rather than a closure because the loop needs `agent` mutably
/// between calls, which an `agent`-capturing closure would forbid.
///
/// `aux` is the *most recent* iteration's auxiliary-phase result: `None` means
/// the auxiliary buffer was not yet full on that iteration, which is the normal
/// case for `n_iteration - 1` out of every `n_iteration` rollouts.
fn emit_progress<B, P, V, O, const DO: usize, const DB: usize>(
    agent: &PpgAgent<B, P, V, O, DO, DB>,
    stats: &PpoUpdateStats,
    aux: Option<AuxPhaseStats>,
    global_step: usize,
    total_timesteps: usize,
) where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    let avg = agent
        .stats()
        .avg_score()
        .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2}"));
    tracing::info!(
        step = global_step,
        total_steps = total_timesteps,
        iteration = agent.iteration(),
        avg_reward = %avg,
        policy_loss = stats.policy_loss,
        value_loss = stats.value_loss,
        entropy = stats.entropy,
        approx_kl = stats.approx_kl,
        clip_frac = stats.clip_frac,
        aux_ran = aux.is_some(),
        aux_value_loss = aux.map_or(0.0, |a| a.aux_value_loss),
        aux_policy_kl = aux.map_or(0.0, |a| a.policy_kl),
        "ppg training progress"
    );
}

/// Maps an [`EnvironmentError`](rlevo_core::environment::EnvironmentError)
/// into [`PpgAgentError::Environment`] for use with `?` in the training loop.
// Passed by name to `.map_err(..)`, which hands the closure an owned
// `EnvironmentError`. Taking `&EnvironmentError` to satisfy the lint would force
// a `|e| ..(&e)` closure at every call site for no benefit.
#[allow(clippy::needless_pass_by_value)]
fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> PpgAgentError {
    PpgAgentError::Environment(err.to_string())
}
