//! Random-policy rollout over [`CartPole`].
//!
//! Constructs the environment, resets it, and runs a uniformly random
//! left/right policy until the pole falls or the max episode length is reached.
//! Prints cumulative reward statistics over a batch of episodes.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example cartpole_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{CartPole, CartPoleAction, CartPoleConfig};

const NUM_EPISODES: usize = 200;
const MAX_STEPS: usize = 500;

fn main() {
    println!("CartPole random-policy rollout | max_steps={MAX_STEPS} episodes={NUM_EPISODES}");

    let mut env = CartPole::with_config(CartPoleConfig::default());
    let mut rng = StdRng::seed_from_u64(42);
    let dist = Uniform::new(0, CartPoleAction::ACTION_COUNT).unwrap();
    let mut lengths = Vec::with_capacity(NUM_EPISODES);

    for _ in 0..NUM_EPISODES {
        env.reset().expect("reset");
        let mut steps = 0_usize;
        loop {
            let action = CartPoleAction::from_index(dist.sample(&mut rng));
            let snap = env.step(action).expect("step");
            steps += 1;
            if snap.is_done() || steps >= MAX_STEPS {
                break;
            }
        }
        lengths.push(steps as f32);
    }

    print_summary("episode length", &lengths);
}

fn print_summary(label: &str, values: &[f32]) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!();
    println!("{label} statistics over {} episodes:", values.len());
    println!("  mean = {mean:>7.2}");
    println!("  std  = {:.4}", var.sqrt());
    println!("  min  = {min:>7.2}");
    println!("  max  = {max:>7.2}");
}
