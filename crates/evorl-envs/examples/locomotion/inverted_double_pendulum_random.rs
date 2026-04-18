//! Random-agent smoke test for InvertedDoublePendulum-v5 (Rapier3D-backed).

use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::locomotion::inverted_double_pendulum::{
    InvertedDoublePendulumAction, InvertedDoublePendulumConfig, InvertedDoublePendulumRapier,
    METADATA_KEY_ALIVE, METADATA_KEY_DISTANCE, METADATA_KEY_VELOCITY,
};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let config = InvertedDoublePendulumConfig { seed: 42, ..Default::default() };
    let mut env = InvertedDoublePendulumRapier::with_config(config);
    let mut rng = StdRng::seed_from_u64(0);

    let snap = env.reset().expect("reset failed");
    println!(
        "reset done; initial cart_x = {:.3}",
        snap.observation().cart_position()
    );

    let mut total_reward = 0.0f32;
    let mut total_alive = 0.0f32;
    let mut total_distance = 0.0f32;
    let mut total_velocity = 0.0f32;
    let mut final_step = 0usize;

    for step in 0..1000 {
        final_step = step;
        let a = rng.random_range(-1.0_f32..=1.0);
        match env.step(InvertedDoublePendulumAction::new(a)) {
            Ok(snap) => {
                total_reward += snap.reward().0;
                if let Some(meta) = snap.metadata() {
                    total_alive += meta.components[METADATA_KEY_ALIVE];
                    total_distance += meta.components[METADATA_KEY_DISTANCE];
                    total_velocity += meta.components[METADATA_KEY_VELOCITY];
                }
                if snap.is_done() {
                    println!(
                        "episode ended at step {step}: status={:?}",
                        snap.status()
                    );
                    break;
                }
            }
            Err(e) => {
                eprintln!("step error at {step}: {e}");
                return;
            }
        }
    }

    println!("  steps             = {}", final_step + 1);
    println!("  Σ alive           = {total_alive:.4}");
    println!("  Σ distance        = {total_distance:.4}");
    println!("  Σ velocity        = {total_velocity:.4}");
    println!("  Σ total_reward    = {total_reward:.4}");
}
