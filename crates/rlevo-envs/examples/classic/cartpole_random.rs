//! Random-policy rollout over [`CartPole`].
//!
//! Constructs the environment, resets it, and runs a uniformly random
//! left/right policy until the pole falls or the max episode length is reached.
//! Prints the environment spec, per-episode aggregate statistics, an action
//! histogram, the worst (largest-magnitude) pole angle observed per episode,
//! and the split between terminations (pole fell / cart out of bounds) and
//! manual cap hits.
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
use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};

const NUM_EPISODES: usize = 200;
const MAX_STEPS: usize = 500;

fn main() {
    let cfg = CartPoleConfig::default();

    println!("CartPole random-policy rollout");
    println!("  episodes      = {NUM_EPISODES}");
    println!("  max_steps     = {MAX_STEPS}");
    println!();
    println!("Environment spec:");
    println!("  observation   = [cart_pos, cart_vel, pole_angle, pole_ang_vel]  shape={:?}",
        <CartPoleObservation as Observation<1>>::shape());
    println!("  actions       = Left | Right  count={}", CartPoleAction::ACTION_COUNT);
    println!("  reward        = +1 per step until failure");
    println!("  termination   = |cart_pos| > {} m  or  |pole_angle| > {:.4} rad ({:.1}°)",
        cfg.x_threshold,
        cfg.theta_threshold_radians,
        cfg.theta_threshold_radians.to_degrees());
    println!("  gravity       = {} m/s²", cfg.gravity);
    println!("  masscart      = {} kg", cfg.masscart);
    println!("  masspole      = {} kg", cfg.masspole);
    println!("  pole length   = {} m (half)", cfg.length);
    println!("  force_mag     = {} N", cfg.force_mag);
    println!("  tau           = {} s", cfg.tau);

    let mut env = CartPole::with_config(cfg.clone());
    let mut rng = StdRng::seed_from_u64(42);
    let dist = Uniform::new(0, CartPoleAction::ACTION_COUNT).unwrap();

    let mut episode_lengths = Vec::with_capacity(NUM_EPISODES);
    let mut episode_returns = Vec::with_capacity(NUM_EPISODES);
    let mut worst_abs_angles = Vec::with_capacity(NUM_EPISODES);
    let mut action_counts = [0_usize; CartPoleAction::ACTION_COUNT];
    let mut terminations = 0_usize;
    let mut cap_hits = 0_usize;

    for _ in 0..NUM_EPISODES {
        env.reset().expect("reset");
        let mut steps = 0_usize;
        let mut cumulative = 0.0_f32;
        let mut worst_angle = 0.0_f32;

        loop {
            let idx = dist.sample(&mut rng);
            action_counts[idx] += 1;
            let action = CartPoleAction::from_index(idx);
            let snap = env.step(action).expect("step");
            steps += 1;
            cumulative += snap.reward().0;

            let abs_angle = snap.observation().pole_angle.abs();
            if abs_angle > worst_angle {
                worst_angle = abs_angle;
            }

            if snap.is_done() {
                terminations += 1;
                break;
            }
            if steps >= MAX_STEPS {
                cap_hits += 1;
                break;
            }
        }

        episode_lengths.push(steps as f32);
        episode_returns.push(cumulative);
        worst_abs_angles.push(worst_angle);
    }

    println!();
    println!("Results over {NUM_EPISODES} episodes:");
    print_stats("episode length      ", &episode_lengths);
    print_stats("cumulative reward   ", &episode_returns);
    print_stats("worst |pole_angle|  ", &worst_abs_angles);

    println!();
    println!("Terminations: {terminations}/{NUM_EPISODES} (pole fell / cart left bounds)");
    println!("Cap hits:     {cap_hits}/{NUM_EPISODES} (reached MAX_STEPS={MAX_STEPS})");

    let total_actions: usize = action_counts.iter().sum();
    println!();
    println!("Action histogram ({total_actions} steps):");
    for (i, &count) in action_counts.iter().enumerate() {
        let action = CartPoleAction::from_index(i);
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
