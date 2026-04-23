//! End-to-end integration tests for the C51 (Categorical DQN) agent.
//!
//! Follows the shape of `dqn_integration.rs`: modest step budgets on
//! CartPole for `cargo test` throughput, with a couple of heavier
//! reproducibility checks gated behind `#[ignore]` because Burn's ndarray
//! backend shares a global RNG.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use evorl_envs::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use evorl_rl::algorithms::c51::c51_agent::C51Agent;
use evorl_rl::algorithms::c51::c51_config::C51TrainingConfigBuilder;
use evorl_rl::algorithms::c51::c51_model::C51Model;
use evorl_rl::algorithms::c51::train::train;

// ---------------------------------------------------------------------------
// Shared MLP and Polyak update helpers (mirror of the c51_cart_pole example).
// ---------------------------------------------------------------------------

const N_ACTIONS: usize = 2;

#[derive(Module, Debug)]
struct C51Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_atoms: usize,
}

impl<B: Backend> C51Mlp<B> {
    fn new(num_atoms: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, N_ACTIONS * num_atoms).init(device),
            num_atoms,
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = observations.dims();
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        let flat = self.l3.forward(x);
        flat.reshape([batch_size, N_ACTIONS, self.num_atoms])
    }
}

impl<B: AutodiffBackend> C51Model<B, 2> for C51Mlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(observations)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(observations)
    }
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, C51Mlp<B::InnerBackend>>(
            active.valid(),
            target,
            tau as f32,
        )
    }
}

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: std::marker::PhantomData<B>,
}
impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}
struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: std::marker::PhantomData<B>,
}
impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        let active = self.active.remove(&id).expect("paired active param");
        let tau = self.tau;
        param.map(move |t| {
            let device = t.device();
            let a = Tensor::<B, D>::from_data(active, &device);
            t.mul_scalar(1.0 - tau) + a.mul_scalar(tau)
        })
    }
}
fn polyak_update<B: Backend, M: Module<B>>(active: M, target: M, tau: f32) -> M {
    let mut c = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: std::marker::PhantomData,
    };
    active.visit(&mut c);
    let mut m = PolyakMapper::<B> {
        active: c.tensors,
        tau,
        _marker: std::marker::PhantomData,
    };
    target.map(&mut m)
}

type Be = Autodiff<NdArray>;
type Agent = C51Agent<Be, C51Mlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

fn fresh_agent(seed: u64) -> Agent {
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let num_atoms = 51;
    let config = C51TrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .tau(0.005)
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .target_update_frequency(500)
        .replay_buffer_capacity(50_000)
        .num_atoms(num_atoms)
        .v_min(0.0)
        .v_max(500.0)
        .build();
    let model: C51Mlp<Be> = C51Mlp::new(num_atoms, &device);
    C51Agent::new(model, config, device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: a short run completes, the buffer populates, and episode
/// rewards are finite. Catches silent regressions (NaN logits, projection
/// numerical blow-up, etc.). Marked `#[ignore]` because running it alongside
/// other training tests perturbs Burn's global ndarray RNG; exercise with
/// `cargo test -p evorl-rl --test c51_integration -- --ignored
/// --test-threads=1`.
#[test]
#[ignore = "perturbs global Burn RNG; run with --test-threads=1"]
fn c51_short_run_produces_finite_rewards() {
    let seed: u64 = 7;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 2_500, 0).expect("training");
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    for (i, m) in agent.stats().recent_history.iter().enumerate() {
        assert!(m.reward.is_finite(), "non-finite reward at episode {i}");
        assert!(m.policy_loss.is_finite(), "non-finite loss at episode {i}");
        assert!(m.q_mean.is_finite(), "non-finite q_mean at episode {i}");
    }
}

/// Reproducibility smoke test: two seeded back-to-back runs produce identical
/// reward sequences. Marked `#[ignore]` for the same reason as the DQN
/// counterpart — Burn's ndarray backend uses a process-global RNG.
#[test]
#[ignore = "requires --test-threads=1 to isolate Burn's global RNG"]
fn c51_reproducibility_ndarray() {
    fn run(seed: u64, total: usize) -> Vec<f32> {
        let mut env = CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        });
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agent = fresh_agent(seed);
        train(&mut agent, &mut env, &mut rng, total, 0).expect("training");
        agent
            .stats()
            .recent_history
            .iter()
            .map(|m| m.reward)
            .collect()
    }

    let a = run(123, 3_000);
    let b = run(123, 3_000);
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "divergence at episode {i}: {x} vs {y}");
    }
}

/// CartPole target: after 30k steps the 100-episode moving average should
/// comfortably clear 100. This is the distributional counterpart to the DQN
/// integration test; the modest budget keeps CI fast while still catching
/// wholesale training failures.
#[test]
fn c51_cart_pole_reaches_100() {
    let seed: u64 = 42;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 30_000, 0).expect("training loop");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(avg >= 100.0, "expected avg reward >= 100, got {avg:.2}");
}
