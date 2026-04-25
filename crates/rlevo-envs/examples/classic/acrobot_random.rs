//! Random-policy rollout over [`Acrobot`].
//!
//! Applies a uniformly random torque action each step. Uses
//! `TimeLimit::new(env, 500)` as the step cap. Prints the environment spec,
//! per-episode aggregate statistics, an action histogram, the distribution of
//! best end-effector heights reached, and the split between terminations
//! (goal reached) and truncations (time-limit hit).
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example acrobot_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{Acrobot, AcrobotAction, AcrobotConfig, BookDynamics};
use rlevo_envs::wrappers::TimeLimit;

const NUM_EPISODES: usize = 100;
const TIME_LIMIT: usize = 500;

fn main() {
    let cfg = AcrobotConfig::default();

    println!("Acrobot random-policy rollout");
    println!("  episodes      = {NUM_EPISODES}");
    println!("  time_limit    = {TIME_LIMIT} steps");
    println!();
    println!("Environment spec:");
    println!("  observation   = [cos θ1, sin θ1, cos θ2, sin θ2, θ̇1, θ̇2]  shape={:?}",
        <rlevo_envs::classic::AcrobotObservation as Observation<1>>::shape());
    println!("  actions       = TorqueNeg(-1) | TorqueZero(0) | TorquePos(+1)  count={}",
        AcrobotAction::ACTION_COUNT);
    println!("  reward        = -1 per step until goal; 0 on termination");
    println!("  goal          = end-effector height > 1.0 m");
    println!("  dynamics      = BookDynamics (Sutton & Barto)");
    println!("  dt            = {} s", cfg.dt);
    println!("  gravity       = {} m/s²", cfg.gravity);
    println!("  link lengths  = ({}, {}) m", cfg.link_length_1, cfg.link_length_2);
    println!("  torque_noise  = {}", cfg.torque_noise_max);

    let env = Acrobot::<BookDynamics>::with_config(cfg.clone());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(21);
    let dist = Uniform::new(0, AcrobotAction::ACTION_COUNT).unwrap();

    let mut episode_lengths = Vec::with_capacity(NUM_EPISODES);
    let mut episode_returns = Vec::with_capacity(NUM_EPISODES);
    let mut best_heights = Vec::with_capacity(NUM_EPISODES);
    let mut action_counts = [0_usize; AcrobotAction::ACTION_COUNT];
    let mut terminations = 0_usize;
    let mut truncations = 0_usize;

    let l1 = cfg.link_length_1;
    let l2 = cfg.link_length_2;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut steps = 0_usize;
        let mut cumulative = 0.0_f32;
        let mut best_height = f32::NEG_INFINITY;

        loop {
            let idx = dist.sample(&mut rng);
            action_counts[idx] += 1;
            let action = AcrobotAction::from_index(idx);
            let snap = timed.step(action).expect("step");
            steps += 1;
            cumulative += snap.reward().0;

            let obs = snap.observation();
            // height = -l1·cos θ1 - l2·cos(θ1 + θ2); use cos/sin sum identity
            let cos_sum = obs.cos_theta1 * obs.cos_theta2 - obs.sin_theta1 * obs.sin_theta2;
            let height = -l1 * obs.cos_theta1 - l2 * cos_sum;
            if height > best_height {
                best_height = height;
            }

            if snap.is_done() {
                if snap.is_terminated() {
                    terminations += 1;
                } else if snap.is_truncated() {
                    truncations += 1;
                }
                break;
            }
        }

        episode_lengths.push(steps as f32);
        episode_returns.push(cumulative);
        best_heights.push(best_height);
    }

    println!();
    println!("Results over {NUM_EPISODES} episodes:");
    print_stats("episode length      ", &episode_lengths);
    print_stats("cumulative reward   ", &episode_returns);
    print_stats("best height reached ", &best_heights);

    println!();
    println!("Terminations: {terminations}/{NUM_EPISODES} (goal reached)");
    println!("Truncations:  {truncations}/{NUM_EPISODES} (time limit hit)");

    let total_actions: usize = action_counts.iter().sum();
    println!();
    println!("Action histogram ({total_actions} steps):");
    for (i, &count) in action_counts.iter().enumerate() {
        let action = AcrobotAction::from_index(i);
        let pct = 100.0 * count as f32 / total_actions as f32;
        println!("  {action:?}: {count:>6} ({pct:>5.2}%)");
    }
}

fn print_stats(label: &str, values: &[f32]) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {label}  mean={mean:>8.2}  std={:>7.2}  min={min:>8.2}  max={max:>8.2}",
        var.sqrt()
    );
}
