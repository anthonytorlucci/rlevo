//! Random-agent smoke test for LunarLander (continuous variant).

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::box2d::lunar_lander::{
    LunarLanderConfig, LunarLanderContinuous, LunarLanderContinuousAction,
};

fn main() {
    let config = LunarLanderConfig::builder().seed(42).build();
    let mut env = LunarLanderContinuous::with_config(config);
    let mut rng = StdRng::seed_from_u64(0);

    let snap = env.reset().expect("reset failed");
    println!("reset done, running={}", !snap.is_done());

    let mut total_reward = 0.0f32;
    for step in 0..500 {
        let action = LunarLanderContinuousAction([
            rng.random_range(-1.0_f32..=1.0),
            rng.random_range(-1.0_f32..=1.0),
        ]);
        match env.step(action) {
            Ok(snap) => {
                let r: f32 = (*snap.reward()).into();
                total_reward += r;
                if snap.is_done() {
                    println!("episode ended at step {step}, total_reward={total_reward:.2}");
                    return;
                }
            }
            Err(e) => {
                eprintln!("step error at {step}: {e}");
                return;
            }
        }
    }
    println!("truncated after 500 steps, total_reward={total_reward:.2}");
}
