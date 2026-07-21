//! Chapter 3 — Classic control: `CartPole` with DQN.
//!
//! Trains a Deep Q-Network to balance a pole on a cart. Unlike the
//! evolutionary examples in Chapters 1–2 — which optimise a *static* function —
//! this agent acts *sequentially* inside an [`Environment`], learning from a
//! reward it collects one timestep at a time via experience replay and a
//! soft-updated target network.
//!
//! Structurally this mirrors the PPO cartpole scaffolding
//! (`examples/common/ppo_cartpole.rs`) — a Burn MLP, a typed agent, shared
//! hyperparameters, one `train` call — but swaps the on-policy PPO agent for
//! off-policy DQN. The same `Environment` drives both.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p rlevo-examples --example ch03_dqn_cartpole
//! ```
//!
//! A full 30k-step run takes a few minutes on CPU and reaches a 100-episode
//! moving average comfortably above 100 (≈183 on `seed = 42` in development).

// ANCHOR: model
use burn::backend::{Autodiff, Flex};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

/// Two-hidden-layer `ReLU` MLP mapping a 4-D `CartPole` observation to a Q-value
/// per action: `4 → 64 → 64 → 2`. The input/output sizes live in *the model*,
/// not in the training config — DQN is generic over whatever network you bring.
#[derive(Module, Debug)]
struct DqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> DqnMlp<B> {
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, 2).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(observations));
        let x = relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for DqnMlp<B> {
    /// Forward pass on the autodiff backend (used for the gradient step).
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(observations)
    }

    /// Forward pass on the inner (non-autodiff) backend — used by the target
    /// network and for ε-greedy action selection, where no gradient is needed.
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(observations)
    }

    /// Polyak (soft) update of the target network toward the active weights:
    /// `target ← τ·active + (1 − τ)·target`.
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, DqnMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}
// ANCHOR_END: model

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;

/// Deterministic seed shared by the env, the RNG, and backend weight init.
const SEED: u64 = 42;
/// Total environment steps to train for.
const TOTAL_STEPS: usize = 30_000;
/// Emit a `tracing` progress line every this many steps (`0` disables logging).
const LOG_EVERY: usize = 2_000;

/// Autodiff backend the agent trains on. DQN learns by backpropagation, so the
/// backend must be autodiff-capable — `Autodiff<Flex>`, not bare `Flex`.
type Be = Autodiff<Flex>;
/// Concrete agent: a `DqnMlp` policy over `CartPole`'s typed observation/action,
/// with observation rank `1` (flat vector) and batched rank `2`.
type Agent = DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

// ANCHOR: agent
/// Builds a fresh DQN agent with hyperparameters tuned for the 4-observation /
/// 2-action `CartPole` task.
fn build_agent(seed: u64) -> Agent {
    let device = Default::default();

    // Seed Burn's process-global RNG so weight initialisation is reproducible.
    // Full bit-for-bit determinism additionally requires pinning rayon to a
    // single thread — see the contributor book on Flex nondeterminism.
    <Be as Backend>::seed(&device, seed);

    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005) // target-network soft-update rate
        .learning_rate(5e-4)
        .epsilon_start(1.0) // start fully exploratory
        .epsilon_end(0.05) // floor on exploration
        .epsilon_decay(0.9995) // multiplicative per-step decay
        .learning_starts(1_000) // fill the buffer before the first gradient step
        .train_frequency(4) // one gradient step every 4 env steps
        .target_update_frequency(500)
        .replay_buffer_capacity(50_000)
        .double_q(false)
        .build()
        .expect("valid config");

    let model: DqnMlp<Be> = DqnMlp::new(&device);
    DqnAgent::new(model, config, device).expect("valid config")
}
// ANCHOR_END: agent

fn main() {
    // Minimal subscriber so the library's per-`LOG_EVERY` progress lines print.
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::INFO)
        .with_target(false)
        .init();

    // ANCHOR: train
    let mut env = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    })
    .expect("valid config");
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut agent = build_agent(SEED);

    // The whole RL run: act ε-greedily, store transitions, and learn after
    // every few steps. `train` owns the sequential loop — the DQN analogue of
    // the evolutionary harness from Chapter 1.
    train(&mut agent, &mut env, &mut rng, TOTAL_STEPS, LOG_EVERY).expect("training loop");
    // ANCHOR_END: train

    let stats = agent.stats();
    println!("\n=== CartPole DQN — training complete ===");
    println!("episodes     : {}", stats.total_episodes);
    println!("env steps    : {}", stats.total_steps);
    println!("best episode : {:.1}", stats.best_score.unwrap_or(f32::NAN));
    println!(
        "avg (last {:>3}): {:.1}",
        stats.recent_history.len(),
        stats.avg_score().unwrap_or(f32::NAN)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fast smoke test (kept short so `cargo test -p rlevo-examples` stays
    /// quick): a brief run must fill the replay buffer and produce only finite
    /// episode rewards. The full convergence target (avg ≥ 100) is asserted by
    /// the `#[ignore]`-gated `dqn_cart_pole_reaches_100` integration test.
    #[test]
    fn dqn_cartpole_smoke() {
        let seed: u64 = 7;
        let mut env = CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        })
        .expect("valid config");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agent = build_agent(seed);

        train(&mut agent, &mut env, &mut rng, 1_200, 0).expect("training");

        assert!(
            agent.buffer_len() > 0,
            "replay buffer should have transitions"
        );
        for (i, m) in agent.stats().recent_history.iter().enumerate() {
            assert!(m.reward.is_finite(), "non-finite reward at episode {i}");
        }
    }
}
