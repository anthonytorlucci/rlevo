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
/// [`C51Agent::should_train`] returns `true`, calls [`C51Agent::sync_target`],
/// and decays ε via [`C51Agent::decay_exploration`]. On episode termination the loop records
/// per-episode [`C51Metrics`] into the agent's rolling statistics window and
/// calls [`Environment::reset`] to begin a new episode, except on the very
/// last step (to avoid recording a phantom episode in recording environments).
///
/// `sync_target` is called unconditionally and self-gates: it is the **hard
/// sync only**, and does nothing unless `tau == 0`. With the default `tau =
/// 0.005` the target is maintained solely by the Polyak update inside
/// [`C51Agent::learn_step`].
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

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::{Autodiff, Flex};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::environment::EpisodeStatus;

    type TestBackend = Autodiff<Flex>;

    /// Episodes end every `PERIOD` steps; `STEPS` spans three whole episodes.
    const PERIOD: usize = 3;
    const STEPS: usize = 9;

    use crate::algorithms::bootstrap_mask::{
        AtomNet, DiscreteMaskEnv, MaskDiscreteAction, MaskObservation,
    };
    use crate::algorithms::c51::c51_config::C51TrainingConfig;

    /// Drives `train` against a clock env whose episodes all end with
    /// `end_status`, and returns the bootstrap masks the loop recorded.
    ///
    /// `learning_starts` is pushed past the step budget so no gradient step
    /// ever runs — this guard is about *which predicate reaches `remember`*,
    /// not about learning.
    fn recorded_masks(end_status: EpisodeStatus) -> Vec<bool> {
        let device = Default::default();
        let config = C51TrainingConfig {
            learning_starts: 1_000_000,
            replay_buffer_capacity: 64,
            ..C51TrainingConfig::default()
        };
        let net = AtomNet::<TestBackend>::new(config.num_atoms, &device);
        let mut agent = C51Agent::<
            TestBackend,
            AtomNet<TestBackend>,
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
