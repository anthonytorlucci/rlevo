//! CartPole with [`TimeLimit`] wrapper demonstrating truncation vs termination.
//!
//! Uses `TimeLimit::new(env, 500)` so episodes that survive 500 steps end
//! with `is_truncated() == true`, while episodes where the pole falls end
//! with `is_terminated() == true`.
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
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::classic::{CartPole, CartPoleAction, CartPoleConfig};
use rlevo_envs::wrappers::TimeLimit;

const NUM_EPISODES: usize = 100;
const TIME_LIMIT: usize = 500;

fn main() {
    println!("CartPole + TimeLimit({TIME_LIMIT}) | episodes={NUM_EPISODES}");

    let env = CartPole::with_config(CartPoleConfig::default());
    let mut timed = TimeLimit::new(env, TIME_LIMIT);
    let mut rng = StdRng::seed_from_u64(99);
    let dist = Uniform::new(0, CartPoleAction::ACTION_COUNT).unwrap();

    let mut terminated = 0_usize;
    let mut truncated = 0_usize;
    let mut lengths = Vec::with_capacity(NUM_EPISODES);

    for _ in 0..NUM_EPISODES {
        timed.reset().expect("reset");
        let mut steps = 0_usize;
        loop {
            let action = CartPoleAction::from_index(dist.sample(&mut rng));
            let snap = timed.step(action).expect("step");
            steps += 1;
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
        lengths.push(steps as f32);
    }

    let n = lengths.len() as f32;
    let mean = lengths.iter().sum::<f32>() / n;
    println!();
    println!("Results over {NUM_EPISODES} episodes:");
    println!("  mean length  = {mean:.1}");
    println!("  terminated   = {terminated}  (pole fell)");
    println!("  truncated    = {truncated}  (step cap reached)");
    assert_eq!(
        terminated + truncated,
        NUM_EPISODES,
        "every episode must end"
    );
}
