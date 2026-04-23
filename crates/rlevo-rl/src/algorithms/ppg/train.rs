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

use crate::algorithms::ppg::ppg_agent::{PpgAgent, PpgAgentError, PpgMetrics};
use crate::algorithms::ppg::ppg_policy::PpgAuxValueHead;
use crate::algorithms::ppo::ppo_agent::PpoUpdateStats;
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;

/// PPG training loop against a **discrete** action environment.
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
        clip_frac: 0.0,
        epochs_run: 0,
    };
    let mut last_done_in_rollout = false;
    let mut global_step = 0_usize;

    while global_step < total_timesteps {
        for _ in 0..rollout_len {
            let obs_now = snapshot.observation().clone();
            let act = agent.act(&obs_now, rng);

            let typed_action = A::from_index(act.env_row[0] as usize);
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
                let aux_last = agent.last_aux_phase();
                let metrics = PpgMetrics {
                    reward: episode_reward,
                    steps: episode_steps,
                    policy_loss: last_update_stats.policy_loss,
                    value_loss: last_update_stats.value_loss,
                    entropy: last_update_stats.entropy,
                    approx_kl: last_update_stats.approx_kl,
                    clip_frac: last_update_stats.clip_frac,
                    aux_main_value_loss: aux_last.map(|a| a.main_value_loss).unwrap_or(0.0),
                    aux_value_loss: aux_last.map(|a| a.aux_value_loss).unwrap_or(0.0),
                    aux_policy_kl: aux_last.map(|a| a.policy_kl).unwrap_or(0.0),
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

        let last_obs_for_bootstrap = snapshot.observation().clone();
        agent.finalize_rollout(&last_obs_for_bootstrap, last_done_in_rollout);
        agent.snapshot_into_aux_buffer();
        last_update_stats = agent.policy_phase_update(rng);
        let aux = agent.maybe_aux_phase(rng);

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
                aux_ran = aux.is_some(),
                aux_value_loss = aux.map(|a| a.aux_value_loss).unwrap_or(0.0),
                aux_policy_kl = aux.map(|a| a.policy_kl).unwrap_or(0.0),
                "ppg training progress"
            );
        }
    }

    Ok(())
}

fn io_from_env(err: rlevo_core::environment::EnvironmentError) -> PpgAgentError {
    PpgAgentError::Environment(err.to_string())
}
