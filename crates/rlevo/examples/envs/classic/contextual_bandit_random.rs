//! Random-policy rollout over [`ContextualBandit`].
//!
//! Runs `C = 4` contexts × `K = 10` arms with a uniform-random policy and
//! prints the per-context per-arm true means, plus the realised reward
//! split by context.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-environments --example contextual_bandit_random
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::{ContextualBandit, ContextualBanditConfig, KArmedBanditAction};

const C: usize = 4;
const K: usize = 10;
const NUM_STEPS: usize = 1000;
const SEED: u64 = 42;

#[allow(clippy::cast_precision_loss)]
fn main() {
    let cfg = ContextualBanditConfig {
        max_steps: NUM_STEPS,
        seed: SEED,
    };
    let mut env = ContextualBandit::<C, K>::with_config(cfg);
    <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");

    println!("ContextualBandit<{C},{K}> random-policy rollout (seed={SEED}, steps={NUM_STEPS})");
    println!();
    println!("True per-context arm means q*(c, a):");
    for (c, row) in env.arm_means().iter().enumerate() {
        let formatted: Vec<String> = row.iter().map(|m| format!("{m:+.2}")).collect();
        println!("  context {c}: [{}]", formatted.join(", "));
    }
    println!();

    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let arm_dist = Uniform::new(0_usize, K).expect("non-empty arm range");
    let mut reward_by_ctx = [0.0_f32; C];
    let mut count_by_ctx = [0_usize; C];

    for _ in 0..NUM_STEPS {
        let ctx = env.current_context();
        let arm = arm_dist.sample(&mut rng);
        let action = KArmedBanditAction::<K>::from_index(arm);
        let snap =
            <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut env, action).expect("step");
        reward_by_ctx[ctx] += f32::from(*snap.reward());
        count_by_ctx[ctx] += 1;
        if snap.is_done() {
            break;
        }
    }

    println!("Per-context realised mean reward (uniform-random policy):");
    for c in 0..C {
        let n = count_by_ctx[c].max(1);
        let mean = reward_by_ctx[c] / n as f32;
        println!(
            "  context {c}: visits={:>4}, mean reward={mean:+.3}",
            count_by_ctx[c]
        );
    }
    let total: f32 = reward_by_ctx.iter().sum();
    println!();
    println!("Total reward over {NUM_STEPS} steps: {total:+.3}");
}
