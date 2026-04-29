//! Random-policy rollout over [`Pendulum`].
//!
//! Applies a uniformly random torque in `[-2, 2]` each step.
//! Uses `TimeLimit::new(env, 200)` (the Gymnasium default). Prints the
//! environment spec, per-episode aggregate statistics (cumulative reward,
//! best per-step reward as a proxy for "closest-to-upright"), continuous
//! torque distribution stats, and the termination/truncation split (Pendulum
//! never terminates intrinsically, so all episodes truncate).
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example pendulum_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{Pendulum, PendulumAction, PendulumConfig, PendulumObservation};
use rlevo_environments::wrappers::TimeLimit;

const NUM_EPISODES: usize = 50;
const TIME_LIMIT: usize = 200;

fn main() {
    let cfg = PendulumConfig::default();

    println!("Pendulum random-policy rollout");
    println!("  episodes      = {NUM_EPISODES}");
    println!("  time_limit    = {TIME_LIMIT} steps");
    println!();
    println!("Environment spec:");
    println!(
        "  observation   = [cos θ, sin θ, θ̇]  shape={:?}",
        <PendulumObservation as Observation<1>>::shape()
    );
    println!(
        "  action        = continuous torque in [-{0}, {0}]  shape=[1]",
        cfg.max_torque
    );
    println!("  reward        ∈ [≈ -16.27, 0]  (0 when upright with zero velocity/torque)");
    println!("  termination   = never (env runs until TimeLimit truncates)");
    println!("  max_speed     = ±{} rad/s", cfg.max_speed);
    println!("  gravity       = {} m/s²", cfg.g);
    println!("  mass          = {} kg", cfg.m);
    println!("  length        = {} m", cfg.l);
    println!("  dt            = {} s", cfg.dt);

    let env = Pendulum::with_config(cfg);
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(5);
    let dist = Uniform::new_inclusive(-2.0_f32, 2.0_f32).unwrap();

    let mut episode_lengths = Vec::with_capacity(NUM_EPISODES);
    let mut episode_returns = Vec::with_capacity(NUM_EPISODES);
    let mut best_step_rewards = Vec::with_capacity(NUM_EPISODES);
    let mut all_torques: Vec<f32> = Vec::new();
    let mut terminations = 0_usize;
    let mut truncations = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut steps = 0_usize;
        let mut cumulative = 0.0_f32;
        let mut best_step = f32::NEG_INFINITY;

        loop {
            let torque = dist.sample(&mut rng);
            all_torques.push(torque);
            let action = PendulumAction::new(torque).expect("valid torque");
            let snap = timed.step(action).expect("step");
            steps += 1;
            let r = snap.reward().0;
            cumulative += r;
            if r > best_step {
                best_step = r;
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
        best_step_rewards.push(best_step);
    }

    println!();
    println!("Results over {NUM_EPISODES} episodes:");
    print_stats("episode length      ", &episode_lengths);
    print_stats("cumulative reward   ", &episode_returns);
    print_stats("best step reward    ", &best_step_rewards);

    println!();
    println!(
        "Terminations: {terminations}/{NUM_EPISODES} (Pendulum never terminates intrinsically)"
    );
    println!("Truncations:  {truncations}/{NUM_EPISODES} (time limit hit)");

    println!();
    print_stats(
        &format!("applied torque ({} steps)", all_torques.len()),
        &all_torques,
    );
}

fn print_stats(label: &str, values: &[f32]) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {label}  mean={mean:>8.4}  std={:>7.4}  min={min:>8.4}  max={max:>8.4}",
        var.sqrt()
    );
}
