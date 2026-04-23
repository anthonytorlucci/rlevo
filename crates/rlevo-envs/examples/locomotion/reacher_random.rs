//! Random-agent smoke test for Reacher-v5 (Rapier3D-backed).

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::locomotion::reacher::{
    METADATA_KEY_REWARD_CONTROL, METADATA_KEY_REWARD_DISTANCE, ReacherAction, ReacherConfig,
    ReacherRapier,
};

fn main() {
    let config = ReacherConfig {
        seed: 42,
        ..Default::default()
    };
    let mut env = ReacherRapier::with_config(config);
    let mut rng = StdRng::seed_from_u64(0);

    let snap = env.reset().expect("reset failed");
    let [tx, ty] = snap.observation().target_xy();
    println!("reset done; target = ({tx:.3}, {ty:.3})");

    let mut total_reward = 0.0f32;
    let mut total_distance = 0.0f32;
    let mut total_control = 0.0f32;
    let mut final_distance = 0.0f32;

    for step in 0..50 {
        let action = ReacherAction::new(
            rng.random_range(-1.0_f32..=1.0),
            rng.random_range(-1.0_f32..=1.0),
        );
        match env.step(action) {
            Ok(snap) => {
                let r = snap.reward().0;
                total_reward += r;
                if let Some(meta) = snap.metadata() {
                    total_distance += meta.components[METADATA_KEY_REWARD_DISTANCE];
                    total_control += meta.components[METADATA_KEY_REWARD_CONTROL];
                }
                let [dx, dy] = snap.observation().finger_minus_target_xy();
                final_distance = (dx * dx + dy * dy).sqrt();
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

    println!("  Σ reward_distance = {total_distance:.4}");
    println!("  Σ reward_control  = {total_control:.4}");
    println!("  Σ total_reward    = {total_reward:.4}");
    println!("  final ||finger − target|| = {final_distance:.4}");
}
