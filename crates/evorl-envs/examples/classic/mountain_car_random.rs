//! Random-policy rollout over [`MountainCar`].
//!
//! A random policy almost never reaches the goal; the environment is composed
//! with `TimeLimit::new(env, 200)` to cap each episode at 200 steps (the
//! Gymnasium default).  Prints cumulative reward and goal-reach rate.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p evorl-envs --example mountain_car_random
//! ```

use evorl_core::action::DiscreteAction;
use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::classic::{MountainCar, MountainCarAction, MountainCarConfig};
use evorl_envs::wrappers::TimeLimit;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const NUM_EPISODES: usize = 50;
const TIME_LIMIT: usize = 200;

fn main() {
    println!(
        "MountainCar random-policy rollout | time_limit={TIME_LIMIT} episodes={NUM_EPISODES}"
    );

    let env = MountainCar::with_config(MountainCarConfig::default());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(7);
    let dist = Uniform::new(0, MountainCarAction::ACTION_COUNT).unwrap();

    let mut total_rewards = Vec::with_capacity(NUM_EPISODES);
    let mut goals = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut cumulative = 0.0_f32;
        loop {
            let action = MountainCarAction::from_index(dist.sample(&mut rng));
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
    println!("  mean cumulative reward = {mean:.1}");
    println!("  goal reached           = {goals}/{NUM_EPISODES}");
}
