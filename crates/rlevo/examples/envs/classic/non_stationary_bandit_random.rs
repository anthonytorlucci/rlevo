//! Random-policy rollout over [`NonStationaryBandit`].
//!
//! Demonstrates the random-walk drift from Sutton & Barto §2.5: each arm
//! mean is perturbed by `N(0, sigma_walk^2)` after every step. The example
//! prints the initial and final arm means so the drift is visible.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-environments --example non_stationary_bandit_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{
    KArmedBanditAction, NonStationaryBandit, NonStationaryBanditConfig,
};

const K: usize = 10;
const NUM_STEPS: usize = 1000;
const SEED: u64 = 42;

fn main() {
    let cfg = NonStationaryBanditConfig {
        max_steps: NUM_STEPS,
        seed: SEED,
        sigma_walk: 0.01,
    };
    let mut env = NonStationaryBandit::<K>::with_config(cfg.clone());
    <NonStationaryBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");

    println!(
        "NonStationaryBandit<{K}> random-policy rollout (seed={SEED}, steps={NUM_STEPS}, sigma_walk={})",
        cfg.sigma_walk
    );
    println!();

    let initial_means = *env.arm_means();
    println!("Initial arm means q*(a):");
    for (a, mean) in initial_means.iter().enumerate() {
        println!("  arm {a:>2}: {mean:+.4}");
    }
    println!();

    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let dist = Uniform::new(0_usize, K).expect("non-empty range");
    let mut total_reward = 0.0_f32;

    for _ in 0..NUM_STEPS {
        let arm = dist.sample(&mut rng);
        let action = KArmedBanditAction::<K>::from_index(arm);
        let snap =
            <NonStationaryBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).expect("step");
        total_reward += f32::from(*snap.reward());
        if snap.is_done() {
            break;
        }
    }

    let final_means = *env.arm_means();
    println!("Final arm means q*(a) after {NUM_STEPS} drift steps:");
    for (a, (initial, final_)) in initial_means.iter().zip(final_means.iter()).enumerate() {
        let delta = final_ - initial;
        println!("  arm {a:>2}: {final_:+.4}  (Δ {delta:+.4})");
    }
    println!();
    println!("Total reward: {total_reward:+.3}");
}
