//! Random-policy rollout over [`MountainCarContinuous`].
//!
//! Applies a uniformly random continuous force in `[-1, 1]` each step.
//! Uses `TimeLimit::new(env, 999)` (the Gymnasium default). Prints the
//! environment spec, per-episode aggregate statistics (length, return,
//! best position reached), continuous-action distribution stats, and the
//! split between terminations (goal reached) and truncations (time limit).
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example mountain_car_continuous_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{
    MountainCarContinuous, MountainCarContinuousAction, MountainCarContinuousConfig,
    MountainCarContinuousObservation,
};
use rlevo_environments::wrappers::TimeLimit;

const NUM_EPISODES: usize = 20;
const TIME_LIMIT: usize = 999;

fn main() {
    let cfg = MountainCarContinuousConfig::default();

    println!("MountainCarContinuous random-policy rollout");
    println!("  episodes      = {NUM_EPISODES}");
    println!("  time_limit    = {TIME_LIMIT} steps");
    println!();
    println!("Environment spec:");
    println!(
        "  observation   = [position, velocity]  shape={:?}",
        <MountainCarContinuousObservation as Observation<1>>::shape()
    );
    println!(
        "  action        = continuous force in [{}, {}]  shape=[1]",
        cfg.min_action, cfg.max_action
    );
    println!("  reward        = -0.1 · action²  per step;  +100 on goal");
    println!(
        "  termination   = position ≥ {} m and velocity ≥ {}",
        cfg.goal_position, cfg.goal_velocity
    );
    println!("  position range = [{}, {}] m", cfg.min_pos, cfg.max_pos);
    println!("  max_speed     = ±{} m/s", cfg.max_speed);
    println!("  power         = {}", cfg.power);

    let env = MountainCarContinuous::with_config(cfg.clone());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(13);
    let dist = Uniform::new_inclusive(-1.0_f32, 1.0_f32).unwrap();

    let mut episode_lengths = Vec::with_capacity(NUM_EPISODES);
    let mut episode_returns = Vec::with_capacity(NUM_EPISODES);
    let mut best_positions = Vec::with_capacity(NUM_EPISODES);
    let mut all_forces: Vec<f32> = Vec::new();
    let mut terminations = 0_usize;
    let mut truncations = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut steps = 0_usize;
        let mut cumulative = 0.0_f32;
        let mut best_pos = f32::NEG_INFINITY;

        loop {
            let force = dist.sample(&mut rng);
            all_forces.push(force);
            let action = MountainCarContinuousAction::new(force).expect("valid force");
            let snap = timed.step(action).expect("step");
            steps += 1;
            cumulative += snap.reward().0;

            let pos = snap.observation().position;
            if pos > best_pos {
                best_pos = pos;
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
        best_positions.push(best_pos);
    }

    println!();
    println!("Results over {NUM_EPISODES} episodes:");
    print_stats("episode length      ", &episode_lengths);
    print_stats("cumulative reward   ", &episode_returns);
    print_stats("best position (m)   ", &best_positions);

    println!();
    println!(
        "Terminations: {terminations}/{NUM_EPISODES} (goal reached, position ≥ {})",
        cfg.goal_position
    );
    println!("Truncations:  {truncations}/{NUM_EPISODES} (time limit hit)");

    println!();
    print_stats(
        &format!("applied force ({} steps)", all_forces.len()),
        &all_forces,
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
