//! Random-agent smoke test for CarRacing.

use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::box2d::car_racing::{CarRacing, CarRacingAction, CarRacingConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let config = CarRacingConfig::builder().seed(42).build();
    let mut env = CarRacing::with_config(config);
    let mut rng = StdRng::seed_from_u64(0);

    let snap = env.reset().expect("reset failed");
    println!("reset done, running={}", !snap.is_done());

    let mut total_reward = 0.0f32;
    for step in 0..200 {
        let action = CarRacingAction::random_valid(&mut rng);
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
    println!("truncated after 200 steps, total_reward={total_reward:.2}");
}
