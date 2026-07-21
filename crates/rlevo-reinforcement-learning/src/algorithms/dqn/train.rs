//! End-to-end training loop for DQN.
//!
//! [`train`] drives the collect-learn-sync cycle: reset the environment,
//! step with ε-greedy actions, push each transition
//! into the agent's replay buffer, call [`DqnAgent::learn_step`] on the
//! configured cadence, and sync the target network. Per-episode metrics are
//! accumulated into the agent's [`AgentStats`].
//!
//! [`AgentStats`]: crate::metrics::AgentStats

use rand::Rng;

use burn::tensor::backend::AutodiffBackend;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, Reward, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};

use crate::algorithms::dqn::dqn_agent::{DqnAgent, DqnAgentError, DqnMetrics};
use crate::algorithms::dqn::dqn_model::DqnModel;

/// Drives DQN training for `total_steps` environment steps.
///
/// Each iteration of the loop:
///
/// 1. Selects an action via [`DqnAgent::act`] (ε-greedy).
/// 2. Steps the environment and pushes the resulting transition into the
///    replay buffer with [`DqnAgent::remember`].
/// 3. Calls [`DqnAgent::learn_step`] when `agent.should_train()` returns
///    `true` (controlled by [`DqnTrainingConfig::train_frequency`]).
/// 4. Calls [`DqnAgent::sync_target`], which performs a **hard** target-network
///    copy every [`DqnTrainingConfig::target_update_frequency`] steps — and
///    only when [`DqnTrainingConfig::tau`] is `0.0`. With `tau > 0.0` (the
///    default) this call is a no-op: the target is instead maintained by the
///    Polyak soft update inside [`DqnAgent::learn_step`].
/// 5. Decays ε with [`DqnAgent::decay_exploration`].
///
/// When an episode ends the collected [`DqnMetrics`] are recorded via
/// [`DqnAgent::record_episode`] and the environment is reset. To avoid
/// writing a partial trailing episode record, the reset is skipped on the
/// very last step.
///
/// # Const generics
///
/// | Parameter | Meaning |
/// |-----------|---------|
/// | `DO`      | Rank of a single observation tensor (e.g. `1` for flat vectors). |
/// | `SD`      | Rank of a single state tensor (passed through to the `Environment` bound). |
/// | `DB`      | Rank of a batched observation tensor (`= DO + 1`). |
///
/// # Arguments
///
/// - `agent` — mutable DQN agent holding network weights and replay buffer.
/// - `env` — environment implementing [`Environment`]; must use the same
///   observation type `O` and action type `A` as the agent.
/// - `rng` — caller-owned RNG used for ε-greedy sampling and batch selection.
/// - `total_steps` — total number of environment steps to run.
/// - `log_every` — emit a [`tracing::info!`] progress line every this many
///   steps. Pass `0` to disable all progress logging.
///
/// # Errors
///
/// Returns [`DqnAgentError::InvalidAction`] (wrapping the underlying
/// [`EnvironmentError`] message) if `env.reset()` or `env.step()` fails.
///
/// [`DqnTrainingConfig::train_frequency`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::train_frequency
/// [`DqnTrainingConfig::tau`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::tau
/// [`DqnTrainingConfig::target_update_frequency`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::target_update_frequency
/// [`EnvironmentError`]: rlevo_core::environment::EnvironmentError
// Config knobs are stored as f64 for ergonomics; every tensor in this crate is
// f32. This is the intended narrowing point, and the values are hyperparameters
// (rates, discounts, epsilons) where f32 has far more precision than the
// schedules that produce them.
#[allow(clippy::cast_possible_truncation)]
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
    let mut n_updates = 0_u64;

    for step in 0..total_steps {
        let obs_current = snapshot.observation().clone();
        let action = agent.act(&obs_current, rng);

        let next_snapshot = env.step(action.clone()).map_err(io_from_env)?;
        let reward_f32: f32 = (*next_snapshot.reward()).into();
        // Two distinct masks — do NOT collapse these back into one.
        //
        // `done` drives episode bookkeeping (metrics, `env.reset()`): the
        // episode is over either way.
        //
        // `terminated` is the Bellman bootstrap mask and is true only for an
        // *environmental* termination. On a truncation (time-limit cutoff) the
        // MDP has not ended, so `next_obs` is a real continuation state and
        // `γ · V(next_obs)` must survive in the target. Masking on `done` here
        // would zero the bootstrap at every timeout and bias Q downward on any
        // time-limited env (Pardo et al., "Time Limits in Reinforcement
        // Learning", ICML 2018, Eq. 6 — partial-episode bootstrapping).
        let done = next_snapshot.is_done();
        let terminated = next_snapshot.is_terminated();
        let next_obs = next_snapshot.observation().clone();

        agent.remember(
            obs_current,
            &action,
            reward_f32,
            next_obs.clone(),
            terminated,
        );
        agent.on_env_step();

        episode_reward += reward_f32;
        episode_steps += 1;

        if agent.should_train()
            && let Some(outcome) = agent.learn_step(rng)?
        {
            last_loss = outcome.loss;
            last_q_mean = outcome.q_mean;
            n_updates += 1;
        }
        agent.sync_target();
        agent.decay_exploration();

        if done {
            let metrics = DqnMetrics {
                reward: episode_reward,
                steps: episode_steps,
                policy_loss: last_loss,
                epsilon: agent.epsilon() as f32,
                q_mean: last_q_mean,
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
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2}"));
            tracing::info!(
                step = step + 1,
                total_steps,
                avg_reward = %avg,
                epsilon = agent.epsilon(),
                buffer = agent.buffer_len(),
                // Canonical metrics consumed by the TUI / record layers.
                td_loss = last_loss,
                q_values = last_q_mean,
                learning_rate = agent.learning_rate(),
                n_updates = n_updates,
                "dqn training progress"
            );
        }
    }
    Ok(())
}

/// Converts an [`EnvironmentError`] into a [`DqnAgentError`] so the training
/// loop returns a single error type.
///
/// The environment error message is preserved as the `InvalidAction` payload,
/// which is a slight semantic mismatch but avoids introducing an additional
/// variant solely for environment I/O failures.
///
/// [`EnvironmentError`]: rlevo_core::environment::EnvironmentError
// Passed by name to `.map_err(..)`, which hands the closure an owned
// `EnvironmentError`. Taking `&EnvironmentError` to satisfy the lint would force
// a `|e| ..(&e)` closure at every call site for no benefit.
#[allow(clippy::needless_pass_by_value)]
fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> DqnAgentError {
    DqnAgentError::InvalidAction(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::{Autodiff, Flex};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::environment::EpisodeStatus;

    use crate::algorithms::bootstrap_mask::{
        ACTIONS, DiscreteMaskEnv, FlatNet, MaskDiscreteAction, MaskObservation,
    };
    use crate::algorithms::dqn::dqn_config::DqnTrainingConfig;

    type TestBackend = Autodiff<Flex>;

    /// Episodes end every `PERIOD` steps; `STEPS` spans three whole episodes.
    const PERIOD: usize = 3;
    const STEPS: usize = 9;

    /// Drives `train` against a clock env whose episodes all end with
    /// `end_status`, and returns the bootstrap masks the loop recorded.
    ///
    /// `learning_starts` is pushed past the step budget so no gradient step
    /// ever runs — this guard is about *which predicate reaches `remember`*,
    /// not about learning.
    fn recorded_masks(end_status: EpisodeStatus) -> Vec<bool> {
        let device = Default::default();
        let config = DqnTrainingConfig {
            learning_starts: 1_000_000,
            replay_buffer_capacity: 64,
            ..DqnTrainingConfig::default()
        };
        let net = FlatNet::<TestBackend>::new(ACTIONS, &device);
        let mut agent = DqnAgent::<
            TestBackend,
            FlatNet<TestBackend>,
            MaskObservation,
            MaskDiscreteAction,
            1,
            2,
        >::new(net, config, device)
        .expect("default config is valid");
        let mut env = DiscreteMaskEnv::new(PERIOD, end_status);
        let mut rng = StdRng::seed_from_u64(7);
        train(&mut agent, &mut env, &mut rng, STEPS, 0).expect("training loop succeeds");
        agent.replay_terminated_flags()
    }

    /// The episode boundaries this fixture produces: steps 3, 6 and 9.
    fn episode_end_steps() -> Vec<bool> {
        (1..=STEPS).map(|s| s.is_multiple_of(PERIOD)).collect()
    }

    /// A genuine environmental termination *must* zero the bootstrap.
    ///
    /// This also pins the fixture's episode cadence, which is what makes the
    /// truncation test below non-vacuous: it proves three of these nine steps
    /// really do end an episode.
    #[test]
    fn test_train_termination_sets_bootstrap_mask() {
        assert_eq!(
            recorded_masks(EpisodeStatus::Terminated),
            episode_end_steps(),
            "an environmental termination must zero the Bellman bootstrap"
        );
    }

    /// A truncation must **not** zero the bootstrap.
    ///
    /// Same env, same cadence, same `is_done()` pattern as the test above —
    /// only `is_terminated()` differs. Masking on `is_done()` here would
    /// reproduce `episode_end_steps()` and bias every Q-value downward on any
    /// time-limited env (Pardo et al., ICML 2018, Eq. 6).
    #[test]
    fn test_train_truncation_leaves_bootstrap_mask_clear() {
        let flags = recorded_masks(EpisodeStatus::Truncated);
        assert_eq!(
            flags.len(),
            STEPS,
            "every step must reach the replay buffer"
        );
        assert!(
            flags.iter().all(|f| !f),
            "truncation must not zero the bootstrap; got {flags:?}"
        );
    }
}
