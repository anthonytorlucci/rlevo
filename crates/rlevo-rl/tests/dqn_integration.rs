//! End-to-end integration tests for the DQN agent.
//!
//! These tests wire [`DqnAgent`] against real `evorl-envs` environments and
//! assert learning behaviour. They intentionally use modest step budgets so
//! `cargo test` remains tractable; the longer macro-bench targets live in
//! `benches/dqn_bench.rs`.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_envs::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_rl::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_rl::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_rl::algorithms::dqn::dqn_model::DqnModel;
use rlevo_rl::algorithms::dqn::train::train;

// ---------------------------------------------------------------------------
// Shared MLP and Polyak update helpers (mirror of the dqn_cart_pole example).
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct DqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> DqnMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, 2).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for DqnMlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(observations)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(observations)
    }

    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, DqnMlp<B::InnerBackend>>(
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
        let active = self
            .active
            .remove(&id)
            .expect("param not collected from active network");
        let tau = self.tau;
        param.map(move |target_tensor| {
            let device = target_tensor.device();
            let active_tensor = Tensor::<B, D>::from_data(active, &device);
            target_tensor.mul_scalar(1.0 - tau) + active_tensor.mul_scalar(tau)
        })
    }
}

fn polyak_update<B: Backend, M: Module<B>>(active: M, target: M, tau: f32) -> M {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: std::marker::PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        tau,
        _marker: std::marker::PhantomData,
    };
    target.map(&mut mapper)
}

type Be = Autodiff<NdArray>;
type Agent = DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

fn fresh_agent(seed: u64) -> Agent {
    use burn::tensor::backend::Backend;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let config = DqnTrainingConfigBuilder::new()
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
        .double_q(false)
        .build();
    let _ = seed; // seeds env + rng externally; agent is deterministic given those inputs.
    let model: DqnMlp<Be> = DqnMlp::new(&device);
    DqnAgent::new(model, config, device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// CartPole target: after 30k steps with a 2x64 MLP, the 100-episode moving
/// average should comfortably exceed 100. The smoke run during development
/// reached ~183 on seed=42; we assert a conservative floor of 100.
#[test]
fn dqn_cart_pole_reaches_100() {
    let seed: u64 = 42;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 30_000, 0).expect("training loop");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    // The acceptance target is >= 100 within 200k steps; this test trains
    // for 30k to keep CI fast. The smoke/reproducibility tests are
    // `#[ignore]` so they don't compete for Burn's global ndarray RNG.
    assert!(avg >= 100.0, "expected avg reward >= 100, got {avg:.2}");
}

/// Reproducibility smoke test: two back-to-back runs from the same seed
/// produce identical reward sequences when run sequentially in the same
/// thread. Marked `#[ignore]` because Burn's ndarray backend keeps a
/// *global* RNG — running alongside other tests via the default
/// multi-threaded runner poisons that global state. Run explicitly with
/// `cargo test -p evorl-rl --test dqn_integration dqn_reproducibility
/// -- --ignored --test-threads=1`.
#[test]
#[ignore = "requires --test-threads=1 to isolate Burn's global RNG"]
fn dqn_reproducibility_ndarray() {
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

/// Smoke test: a short run completes with finite rewards and a populated
/// buffer. Protects against silent regressions that would NaN-out. Marked
/// `#[ignore]` because running it alongside `dqn_cart_pole_reaches_100`
/// perturbs Burn's global ndarray RNG; run with `-- --ignored
/// --test-threads=1` to exercise it.
#[test]
#[ignore = "perturbs global Burn RNG; run with --test-threads=1"]
fn dqn_short_run_produces_finite_rewards() {
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
    }
}
