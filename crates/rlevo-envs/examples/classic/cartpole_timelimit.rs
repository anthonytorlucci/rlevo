//! CartPole with [`TimeLimit`] wrapper demonstrating truncation vs termination.
//!
//! Uses `TimeLimit::new(env, 500)` so episodes that survive 500 steps end
//! with `is_truncated() == true`, while episodes where the pole falls end
//! with `is_terminated() == true`. Prints the environment spec, per-episode
//! aggregate statistics, an action histogram, and the termination/truncation
//! split — the core distinction this example demonstrates.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example cartpole_timelimit
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};
use rlevo_envs::wrappers::TimeLimit;

const NUM_EPISODES: usize = 100;
const TIME_LIMIT: usize = 500;

fn main() {
    let cfg = CartPoleConfig::default();

    println!("CartPole + TimeLimit({TIME_LIMIT}) random rollout");
    println!("  episodes      = {NUM_EPISODES}");
    println!("  time_limit    = {TIME_LIMIT} steps");
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
    println!("  truncation    = TimeLimit wrapper at {TIME_LIMIT} steps");
    println!("  gravity       = {} m/s²", cfg.gravity);
    println!("  force_mag     = {} N", cfg.force_mag);
    println!("  tau           = {} s", cfg.tau);

    let env = CartPole::with_config(cfg);
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(99);
    let dist = Uniform::new(0, CartPoleAction::ACTION_COUNT).unwrap();

    let mut episode_lengths = Vec::with_capacity(NUM_EPISODES);
    let mut episode_returns = Vec::with_capacity(NUM_EPISODES);
    let mut worst_abs_angles = Vec::with_capacity(NUM_EPISODES);
    let mut action_counts = [0_usize; CartPoleAction::ACTION_COUNT];
    let mut terminated = 0_usize;
    let mut truncated = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut steps = 0_usize;
        let mut cumulative = 0.0_f32;
        let mut worst_angle = 0.0_f32;

        loop {
            let idx = dist.sample(&mut rng);
            action_counts[idx] += 1;
            let action = CartPoleAction::from_index(idx);
            let snap = timed.step(action).expect("step");
            steps += 1;
            cumulative += snap.reward().0;

            let abs_angle = snap.observation().pole_angle.abs();
            if abs_angle > worst_angle {
                worst_angle = abs_angle;
            }

            if snap.is_done() {
                if snap.is_terminated() {
                    terminated += 1;
                }
                if snap.is_truncated() {
                    truncated += 1;
                }
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
    println!("Terminated: {terminated}/{NUM_EPISODES} (pole fell)");
    println!("Truncated:  {truncated}/{NUM_EPISODES} (TimeLimit hit)");
    assert_eq!(
        terminated + truncated,
        NUM_EPISODES,
        "every episode must end"
    );

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
