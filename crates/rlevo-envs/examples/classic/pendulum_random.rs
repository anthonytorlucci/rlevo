//! Random-policy rollout over [`Pendulum`].
//!
//! Applies a uniformly random torque in `[-2, 2]` each step.
//! Uses `TimeLimit::new(env, 200)` (the Gymnasium default). Prints
//! cumulative reward statistics.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example pendulum_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{Pendulum, PendulumAction, PendulumConfig};
use rlevo_envs::wrappers::TimeLimit;

const NUM_EPISODES: usize = 50;
const TIME_LIMIT: usize = 200;

fn main() {
    println!("Pendulum random-policy rollout | time_limit={TIME_LIMIT} episodes={NUM_EPISODES}");

    let env = Pendulum::with_config(PendulumConfig::default());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(5);
    let dist = Uniform::new_inclusive(-2.0_f32, 2.0_f32).unwrap();

    let mut total_rewards = Vec::with_capacity(NUM_EPISODES);

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut cumulative = 0.0_f32;
        loop {
            let torque = dist.sample(&mut rng);
            let action = PendulumAction::new(torque).expect("valid torque");
            let snap = timed.step(action).expect("step");
            cumulative += snap.reward().0;
            if snap.is_done() {
                break;
            }
        }
        total_rewards.push(cumulative);
    }

    let n = total_rewards.len() as f32;
    let mean = total_rewards.iter().sum::<f32>() / n;
    let var = total_rewards
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f32>()
        / n;
    let min = total_rewards.iter().copied().fold(f32::INFINITY, f32::min);
    let max = total_rewards
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    println!();
    println!("Cumulative reward over {NUM_EPISODES} episodes:");
    println!("  mean = {mean:>9.2}");
    println!("  std  = {:.4}", var.sqrt());
    println!("  min  = {min:>9.2}");
    println!("  max  = {max:>9.2}");
}
