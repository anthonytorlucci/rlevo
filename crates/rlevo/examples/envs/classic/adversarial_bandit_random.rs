//! Random-policy rollout over [`AdversarialBandit`].
//!
//! Runs the EXP3-style oblivious adversarial bandit with a uniform-random
//! policy and prints the per-arm phase offsets that drive the deterministic
//! cosine reward schedule, plus the realised cumulative reward per arm.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example adversarial_bandit_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{AdversarialBandit, AdversarialBanditConfig, KArmedBanditAction};

const K: usize = 10;
const NUM_STEPS: usize = 1000;
const SEED: u64 = 42;

fn main() {
    let cfg = AdversarialBanditConfig {
        max_steps: NUM_STEPS,
        seed: SEED,
        period: 50,
        amplitude: 1.0,
    };
    let mut env = AdversarialBandit::<K>::with_config(cfg.clone());
    <AdversarialBandit<K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");

    println!(
        "AdversarialBandit<{K}> random-policy rollout (seed={SEED}, steps={NUM_STEPS}, period={}, amplitude={})",
        cfg.period, cfg.amplitude
    );
    println!();

    println!("Per-arm phase offsets:");
    for (a, phase) in env.phases().iter().enumerate() {
        println!("  arm {a:>2}: phase = {phase}");
    }
    println!();

    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let dist = Uniform::new(0_usize, K).expect("non-empty range");
    let mut counts = [0_usize; K];
    let mut rewards = [0.0_f32; K];

    for _ in 0..NUM_STEPS {
        let arm = dist.sample(&mut rng);
        counts[arm] += 1;
        let action = KArmedBanditAction::<K>::from_index(arm);
        let snap =
            <AdversarialBandit<K> as Environment<1, 1, 1>>::step(&mut env, action).expect("step");
        rewards[arm] += f32::from(*snap.reward());
        if snap.is_done() {
            break;
        }
    }

    println!("Per-arm pulls and total reward:");
    for a in 0..K {
        let n = counts[a].max(1);
        let mean = rewards[a] / n as f32;
        println!(
            "  arm {a:>2}: pulls={:>5}  total={:+.3}  mean={mean:+.3}",
            counts[a], rewards[a]
        );
    }
    let total: f32 = rewards.iter().sum();
    println!();
    println!("Total reward over {NUM_STEPS} steps: {total:+.3}");
}
