//! Random-policy rollout over [`EmptyEnv`].
//!
//! This example demonstrates the minimal grid-env usage pattern:
//! construct, reset, loop [`Environment::step`] until done. It uses a
//! uniformly random policy via [`GridAction::random`] and prints reward
//! statistics over a batch of episodes.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p evorl-envs --example grid_empty_random
//! ```

use evorl_core::action::DiscreteAction;
use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::grids::core::GridAction;
use evorl_envs::grids::{EmptyConfig, EmptyEnv};

const NUM_EPISODES: usize = 200;

fn main() {
    let cfg = EmptyConfig::default();
    println!(
        "EmptyEnv random-policy rollout | size={} max_steps={} episodes={}",
        cfg.size, cfg.max_steps, NUM_EPISODES
    );

    let mut env = EmptyEnv::with_config(cfg, false);
    let mut rewards = Vec::with_capacity(NUM_EPISODES);
    let mut successes = 0_usize;

    for _ in 0..NUM_EPISODES {
        env.reset().expect("reset");
        let (final_reward, solved) = roll_out(&mut env);
        rewards.push(final_reward);
        if solved {
            successes += 1;
        }
    }

    print_summary(&rewards, successes);
}

/// Run a single episode with a uniformly random policy. Returns the
/// terminal reward and whether the episode ended on the goal (reward > 0).
fn roll_out(env: &mut EmptyEnv) -> (f32, bool) {
    loop {
        let action = GridAction::random();
        let snap = env.step(action).expect("step");
        if snap.is_done() {
            let reward = f32::from(*snap.reward());
            return (reward, reward > 0.0);
        }
    }
}

fn print_summary(rewards: &[f32], successes: usize) {
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = rewards.iter().copied().fold(f32::INFINITY, f32::min);
    let max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let success_rate = successes as f32 / n;

    println!();
    println!("Reward statistics over {} episodes:", rewards.len());
    println!("  mean         = {mean:>7.4}");
    println!("  std          = {std:>7.4}");
    println!("  min          = {min:>7.4}");
    println!("  max          = {max:>7.4}");
    println!("  success rate = {success_rate:>7.2}");
}
