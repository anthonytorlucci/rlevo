//! Random-policy rollout over [`MountainCar`].
//!
//! A random policy almost never reaches the goal; the environment is composed
//! with `TimeLimit::new(env, 200)` to cap each episode at 200 steps (the
//! Gymnasium default). Prints the environment spec, per-episode aggregate
//! statistics (length, return, best position reached), an action histogram,
//! and the split between terminations (goal reached) and truncations (time
//! limit hit).
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example mountain_car_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{
    MountainCar, MountainCarAction, MountainCarConfig, MountainCarObservation,
};
use rlevo_envs::wrappers::TimeLimit;

const NUM_EPISODES: usize = 50;
const TIME_LIMIT: usize = 200;

fn main() {
    let cfg = MountainCarConfig::default();

    println!("MountainCar random-policy rollout");
    println!("  episodes      = {NUM_EPISODES}");
    println!("  time_limit    = {TIME_LIMIT} steps");
    println!();
    println!("Environment spec:");
    println!("  observation   = [position, velocity]  shape={:?}",
        <MountainCarObservation as Observation<1>>::shape());
    println!("  actions       = Left | NoAccel | Right  count={}",
        MountainCarAction::ACTION_COUNT);
    println!("  reward        = -1 per step (no goal bonus)");
    println!("  termination   = position ≥ {} m and velocity ≥ {}",
        cfg.goal_position, cfg.goal_velocity);
    println!("  position range = [{}, {}] m", cfg.min_pos, cfg.max_pos);
    println!("  max_speed     = ±{} m/s", cfg.max_speed);
    println!("  force         = {}", cfg.force);
    println!("  gravity       = {} (slope factor)", cfg.gravity);

    let env = MountainCar::with_config(cfg.clone());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(7);
    let dist = Uniform::new(0, MountainCarAction::ACTION_COUNT).unwrap();

    let mut episode_lengths = Vec::with_capacity(NUM_EPISODES);
    let mut episode_returns = Vec::with_capacity(NUM_EPISODES);
    let mut best_positions = Vec::with_capacity(NUM_EPISODES);
    let mut action_counts = [0_usize; MountainCarAction::ACTION_COUNT];
    let mut terminations = 0_usize;
    let mut truncations = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut steps = 0_usize;
        let mut cumulative = 0.0_f32;
        let mut best_pos = f32::NEG_INFINITY;

        loop {
            let idx = dist.sample(&mut rng);
            action_counts[idx] += 1;
            let action = MountainCarAction::from_index(idx);
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
    println!("Terminations: {terminations}/{NUM_EPISODES} (goal reached, position ≥ {})",
        cfg.goal_position);
    println!("Truncations:  {truncations}/{NUM_EPISODES} (time limit hit)");

    let total_actions: usize = action_counts.iter().sum();
    println!();
    println!("Action histogram ({total_actions} steps):");
    for (i, &count) in action_counts.iter().enumerate() {
        let action = MountainCarAction::from_index(i);
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
