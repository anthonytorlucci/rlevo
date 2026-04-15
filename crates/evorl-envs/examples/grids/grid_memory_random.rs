//! Random-policy rollout over [`MemoryEnv`].
//!
//! This example exists to demonstrate *why* memory matters: a uniformly
//! random policy solves [`MemoryEnv`] only by luck. The reported
//! success rate should be close to zero — any real policy needs to
//! remember the cue it saw at step 0 and pick the matching fork option
//! hundreds of steps later.
//!
//! Accepts a single optional positional argument `swap` (`true`/`false`)
//! that sets [`MemoryConfig::swap_fork`]:
//!
//! ```bash
//! cargo run -p evorl-envs --example grid_memory_random
//! cargo run -p evorl-envs --example grid_memory_random -- swap=true
//! ```

use evorl_core::action::DiscreteAction;
use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::grids::core::GridAction;
use evorl_envs::grids::{MemoryConfig, MemoryEnv};

const NUM_EPISODES: usize = 500;

fn main() {
    let swap_fork = parse_swap_arg().unwrap_or(false);

    let cfg = MemoryConfig::new(140, 0, swap_fork);
    println!(
        "MemoryEnv random-policy rollout | swap_fork={} max_steps={} episodes={}",
        cfg.swap_fork, cfg.max_steps, NUM_EPISODES
    );
    println!("(random policy is expected to almost never win — that is the point.)");

    let mut env = MemoryEnv::with_config(cfg, false);
    let mut rewards = Vec::with_capacity(NUM_EPISODES);
    let mut successes = 0_usize;

    for _ in 0..NUM_EPISODES {
        env.reset().expect("reset");
        let reward = roll_out(&mut env);
        if reward > 0.0 {
            successes += 1;
        }
        rewards.push(reward);
    }

    print_summary(&rewards, successes);
}

/// Uniformly random rollout. Returns the terminal reward (may be `0.0`).
fn roll_out(env: &mut MemoryEnv) -> f32 {
    loop {
        let action = GridAction::random();
        let snap = env.step(action).expect("step");
        if snap.is_done() {
            return f32::from(*snap.reward());
        }
    }
}

/// Parse `swap=true` / `swap=false` from the first CLI argument.
fn parse_swap_arg() -> Option<bool> {
    let arg = std::env::args().nth(1)?;
    let (_, value) = arg.split_once('=')?;
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

fn print_summary(rewards: &[f32], successes: usize) {
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let success_rate = successes as f32 / n;

    println!();
    println!("Reward statistics over {} episodes:", rewards.len());
    println!("  mean         = {mean:>7.4}");
    println!("  max          = {max:>7.4}");
    println!("  successes    = {successes} / {}", rewards.len());
    println!("  success rate = {success_rate:>7.4}");
}
