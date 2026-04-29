//! Random-policy rollout over the generic [`KArmedBandit`].
//!
//! Demonstrates the const-generic interface by running uniformly random
//! arm selection for `K = 10` arms over a fixed step budget. Prints the
//! true arm means, the per-arm pull histogram, and the cumulative reward.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-environments --example k_armed_bandit_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{KArmedBandit, KArmedBanditAction, KArmedBanditConfig};

const K: usize = 10;
const NUM_STEPS: usize = 1000;
const SEED: u64 = 42;

fn main() {
    let cfg = KArmedBanditConfig {
        max_steps: NUM_STEPS,
        seed: SEED,
    };
    let mut env = KArmedBandit::<K>::with_config(cfg);
    <KArmedBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");

    println!("KArmedBandit<{K}> random-policy rollout (seed={SEED}, steps={NUM_STEPS})");
    println!();
    println!("True arm means q*(a):");
    for (a, mean) in env.arm_means().iter().enumerate() {
        println!("  arm {a:>2}: {mean:+.4}");
    }
    println!();

    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let dist = Uniform::new(0_usize, K).expect("non-empty range");
    let mut counts = [0_usize; K];
    let mut total_reward = 0.0_f32;

    for _ in 0..NUM_STEPS {
        let arm = dist.sample(&mut rng);
        counts[arm] += 1;
        let action = KArmedBanditAction::<K>::from_index(arm);
        let snap = <KArmedBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).expect("step");
        total_reward += f32::from(*snap.reward());
        if snap.is_done() {
            break;
        }
    }

    println!("Per-arm pull counts:");
    for (a, c) in counts.iter().enumerate() {
        let pct = 100.0 * (*c as f32) / NUM_STEPS as f32;
        println!("  arm {a:>2}: {c:>5} ({pct:>5.2}%)");
    }
    println!();
    println!("Total reward over {NUM_STEPS} steps: {total_reward:+.3}");
    println!(
        "Mean reward per step:               {:+.3}",
        total_reward / NUM_STEPS as f32
    );
}
