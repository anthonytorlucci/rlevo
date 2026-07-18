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

        agent.remember(obs_current, &action, reward_f32, next_obs, terminated);
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

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::{Autodiff, Flex};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::environment::EpisodeStatus;

    use crate::algorithms::bootstrap_mask::{
        ContinuousMaskEnv, MaskContinuousAction, MaskObservation, TinyCritic,
    };

    type TestBackend = Autodiff<Flex>;

    /// Episodes end every `PERIOD` steps; `STEPS` spans three whole episodes.
    const PERIOD: usize = 3;
    const STEPS: usize = 9;

    use crate::algorithms::bootstrap_mask::TinyActor;
    use crate::algorithms::td3::td3_config::Td3TrainingConfig;

    /// Drives `train` against a clock env whose episodes all end with
    /// `end_status`, and returns the bootstrap masks the loop recorded.
    ///
    /// `learning_starts` is pushed past the step budget so no gradient step
    /// ever runs — this guard is about *which predicate reaches `remember`*,
    /// not about learning.
    fn recorded_masks(end_status: EpisodeStatus) -> Vec<bool> {
        let device = Default::default();
        let config = Td3TrainingConfig {
            learning_starts: 1_000_000,
            buffer_capacity: 64,
            ..Td3TrainingConfig::default()
        };
        let actor = TinyActor::<TestBackend>::new(&device);
        let critic_1 = TinyCritic::<TestBackend>::new(&device);
        let critic_2 = TinyCritic::<TestBackend>::new(&device);
        let mut agent = Td3Agent::<
            TestBackend,
            TinyActor<TestBackend>,
            TinyCritic<TestBackend>,
            MaskObservation,
            MaskContinuousAction,
            1,
            2,
            1,
            2,
        >::new(actor, critic_1, critic_2, config, device)
        .expect("default config is valid");
        let mut env = ContinuousMaskEnv::new(PERIOD, end_status);
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
