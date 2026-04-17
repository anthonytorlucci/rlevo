//! Random-policy rollout over [`MountainCarContinuous`].
//!
//! Applies a uniformly random continuous force in `[-1, 1]` each step.
//! Uses `TimeLimit::new(env, 999)` (the Gymnasium default). Prints
//! cumulative reward and goal-reach rate.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p evorl-envs --example mountain_car_continuous_random
//! ```

use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::classic::{
    MountainCarContinuous, MountainCarContinuousAction, MountainCarContinuousConfig,
};
use evorl_envs::wrappers::TimeLimit;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const NUM_EPISODES: usize = 20;
const TIME_LIMIT: usize = 999;

fn main() {
    println!(
        "MountainCarContinuous random-policy rollout | time_limit={TIME_LIMIT} episodes={NUM_EPISODES}"
    );

    let env = MountainCarContinuous::with_config(MountainCarContinuousConfig::default());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(13);
    let dist = Uniform::new_inclusive(-1.0_f32, 1.0_f32).unwrap();

    let mut total_rewards = Vec::with_capacity(NUM_EPISODES);
    let mut goals = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut cumulative = 0.0_f32;
        loop {
            let force = dist.sample(&mut rng);
            let action = MountainCarContinuousAction::new(force).expect("valid force");
            let snap = timed.step(action).expect("step");
            cumulative += snap.reward().0;
            if snap.is_done() {
                if snap.is_terminated() { goals += 1; }
                break;
            }
        }
        total_rewards.push(cumulative);
    }

    let n = total_rewards.len() as f32;
    let mean = total_rewards.iter().sum::<f32>() / n;
    println!();
    println!("Results over {NUM_EPISODES} episodes:");
    println!("  mean cumulative reward = {mean:.3}");
    println!("  goal reached           = {goals}/{NUM_EPISODES}");
}
