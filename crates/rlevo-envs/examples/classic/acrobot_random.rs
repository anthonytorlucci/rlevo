//! Random-policy rollout over [`Acrobot`].
//!
//! Applies a uniformly random torque action each step. Uses
//! `TimeLimit::new(env, 500)` as the step cap. Prints cumulative reward and
//! goal-reach rate.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example acrobot_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{Acrobot, AcrobotAction, AcrobotConfig, BookDynamics};
use rlevo_envs::wrappers::TimeLimit;

const NUM_EPISODES: usize = 100;
const TIME_LIMIT: usize = 500;

fn main() {
    println!("Acrobot random-policy rollout | time_limit={TIME_LIMIT} episodes={NUM_EPISODES}");

    let env = Acrobot::<BookDynamics>::with_config(AcrobotConfig::default());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(21);
    let dist = Uniform::new(0, AcrobotAction::ACTION_COUNT).unwrap();

    let mut total_rewards = Vec::with_capacity(NUM_EPISODES);
    let mut goals = 0_usize;

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut cumulative = 0.0_f32;
        loop {
            let action = AcrobotAction::from_index(dist.sample(&mut rng));
            let snap = timed.step(action).expect("step");
            cumulative += snap.reward().0;
            if snap.is_done() {
                if snap.is_terminated() {
                    goals += 1;
                }
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
