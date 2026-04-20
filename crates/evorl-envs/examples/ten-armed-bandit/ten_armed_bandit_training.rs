//! Training example for the ten-armed bandit.
//!
//! Drives a sample-average ε-greedy agent against [`TenArmedBandit`] via the
//! standard `Environment::reset` / `Environment::step` loop. The bandit
//! samples each reward from `N(q*(a), 1)` where the arm means `q*(a)` are
//! themselves drawn from `N(0, 1)` at construction; the agent must balance
//! exploration (sampling each arm enough times to estimate its value)
//! against exploitation (pulling the arm with the highest estimate).
//!
//! Based on Chapter 2 of Sutton & Barto, *Reinforcement Learning: An
//! Introduction* (2nd ed.), pp. 25–36.
//!
//! ```bash
//! cargo run -p evorl-envs --example ten_armed_bandit_training
//! ```

use std::fmt::{self, Display, Formatter};

use evorl_core::action::DiscreteAction;
use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::classic::ten_armed_bandit::{
    TenArmedBandit, TenArmedBanditAction, TenArmedBanditConfig,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Number of arms (matches `TenArmedBanditAction::ACTION_COUNT`).
const NUM_ARMS: usize = 10;

/// Number of training steps. Matches the classic S&B figure-2.2 horizon.
const TOTAL_STEPS: usize = 1000;

/// ε for the ε-greedy policy.
const EPSILON: f32 = 0.1;

/// Sample-average ε-greedy agent.
///
/// Maintains per-arm value estimates `Q(a)` and selects actions greedily
/// w.r.t. `Q` with probability `1 - ε`, otherwise uniformly at random.
/// Updates `Q(a)` incrementally: `Q(a) ← Q(a) + (1/n(a)) · [r - Q(a)]`.
struct EpsilonGreedyAgent {
    q_values: [f32; NUM_ARMS],
    action_counts: [usize; NUM_ARMS],
    rng: StdRng,
    epsilon: f32,
}

impl EpsilonGreedyAgent {
    fn new(epsilon: f32, seed: u64) -> Self {
        Self {
            q_values: [0.0; NUM_ARMS],
            action_counts: [0; NUM_ARMS],
            rng: StdRng::seed_from_u64(seed),
            epsilon,
        }
    }

    fn select_action(&mut self) -> usize {
        use rand::RngExt;
        if self.rng.random::<f32>() < self.epsilon {
            self.rng.random_range(0..NUM_ARMS)
        } else {
            argmax(&self.q_values)
        }
    }

    fn update(&mut self, action: usize, reward: f32) {
        self.action_counts[action] += 1;
        let n = self.action_counts[action] as f32;
        self.q_values[action] += (reward - self.q_values[action]) / n;
    }

    fn q_values(&self) -> &[f32; NUM_ARMS] {
        &self.q_values
    }
}

/// Per-episode statistics.
struct BanditMetrics {
    total_reward: f32,
    optimal_action_count: usize,
    optimal_arm: usize,
    steps: usize,
}

impl BanditMetrics {
    fn new(optimal_arm: usize) -> Self {
        Self {
            total_reward: 0.0,
            optimal_action_count: 0,
            optimal_arm,
            steps: 0,
        }
    }

    fn update(&mut self, reward: f32, action: usize) {
        self.total_reward += reward;
        if action == self.optimal_arm {
            self.optimal_action_count += 1;
        }
        self.steps += 1;
    }

    fn average_reward(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_reward / self.steps as f32
        }
    }

    fn optimal_action_percentage(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            100.0 * (self.optimal_action_count as f32) / (self.steps as f32)
        }
    }
}

impl Display for BanditMetrics {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Steps: {:4}, Avg Reward: {:+.3}, Optimal Action: {:.1}% ({}/{})",
            self.steps,
            self.average_reward(),
            self.optimal_action_percentage(),
            self.optimal_action_count,
            self.steps
        )
    }
}

fn argmax(values: &[f32; NUM_ARMS]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("finite q values"))
        .map(|(idx, _)| idx)
        .expect("NUM_ARMS > 0")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ten-Armed Bandit Training Example");
    println!("=================================\n");

    let config = TenArmedBanditConfig {
        max_steps: TOTAL_STEPS,
        seed: 42,
    };
    let mut env = TenArmedBandit::with_config(config);

    // Lift the true arm means — available only because `TenArmedBandit`
    // exposes `arm_means()` for analysis. A real agent would not see this.
    let optimal_arm = argmax(env.arm_means());
    println!("True optimal arm: {optimal_arm}  (for evaluation only)");
    println!("Algorithm: ε-greedy (ε = {EPSILON})");
    println!("Total steps: {TOTAL_STEPS}\n");

    let mut agent = EpsilonGreedyAgent::new(EPSILON, 123);
    let mut stats = BanditMetrics::new(optimal_arm);

    // Standard RL loop: reset, then step until the snapshot reports done.
    let _ = <TenArmedBandit as Environment<1, 1, 1>>::reset(&mut env)?;
    loop {
        let arm = agent.select_action();
        let action = TenArmedBanditAction::from_index(arm);
        let snap = <TenArmedBandit as Environment<1, 1, 1>>::step(&mut env, action)?;
        let reward = f32::from(*snap.reward());

        agent.update(arm, reward);
        stats.update(reward, arm);

        if stats.steps.is_multiple_of(100) {
            println!("  {stats}");
        }
        if snap.is_done() {
            break;
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("Training complete");
    println!("{}", "=".repeat(70));
    println!("\nFinal statistics:");
    println!("  {stats}\n");

    println!("Learned Q-values vs. true arm means:");
    println!("  Arm | Q-estimate | q*(a) true |  Δ      ");
    println!("  ----+------------+------------+---------");
    let q = agent.q_values();
    let means = env.arm_means();
    for i in 0..NUM_ARMS {
        let marker = if i == optimal_arm { " *" } else { "  " };
        println!(
            "  {i:3} |  {:+7.3}   |  {:+7.3}   | {:+7.3}{marker}",
            q[i],
            means[i],
            q[i] - means[i],
        );
    }

    Ok(())
}
