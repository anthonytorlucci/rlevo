//! Random-agent smoke test for Swimmer-v5 (Rapier3D-backed).

use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::locomotion::swimmer::{
    METADATA_KEY_CTRL, METADATA_KEY_FORWARD, SwimmerAction, SwimmerConfig, SwimmerRapier,
};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let config = SwimmerConfig { seed: 42, ..Default::default() };
    let mut env = SwimmerRapier::with_config(config);
    let mut rng = StdRng::seed_from_u64(0);

    let _ = env.reset().expect("reset failed");
    println!("reset done; swimmer floating");

    let mut total_reward = 0.0f32;
    let mut total_forward = 0.0f32;
    let mut total_ctrl = 0.0f32;
    let mut steps = 0usize;
    let mut last_vx = 0.0f32;

    for step in 0..1000 {
        let action = SwimmerAction::new(
            rng.random_range(-1.0_f32..=1.0),
            rng.random_range(-1.0_f32..=1.0),
        );
        match env.step(action) {
            Ok(snap) => {
                let r = snap.reward().0;
                total_reward += r;
                if let Some(meta) = snap.metadata() {
                    total_forward += meta.components[METADATA_KEY_FORWARD];
                    total_ctrl += meta.components[METADATA_KEY_CTRL];
                }
                last_vx = snap.observation().vx_com();
                steps = step + 1;
                if snap.is_done() {
                    println!(
                        "episode ended at step {step}: status={:?}, total_reward={total_reward:.4}",
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

    println!("  steps             = {steps}");
    println!("  Σ forward         = {total_forward:.4}");
    println!("  Σ ctrl            = {total_ctrl:.4}");
    println!("  Σ total_reward    = {total_reward:.4}");
    println!("  final vx_com      = {last_vx:.4}");
}
