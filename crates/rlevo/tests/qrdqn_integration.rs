//! End-to-end integration tests for the QR-DQN (Quantile Regression DQN)
//! agent.
//!
//! Mirrors the shape of `c51_integration.rs`: modest step budgets on
//! `CartPole` for `cargo test` throughput, with heavier reproducibility
//! checks gated behind `#[ignore]` because Burn's ndarray backend shares a
//! global RNG.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_agent::QrDqnAgent;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_model::QrDqnModel;
use rlevo_reinforcement_learning::algorithms::qrdqn::train::train;

// ---------------------------------------------------------------------------
// Shared MLP and Polyak update helpers (mirror of the qrdqn_cart_pole example).
// ---------------------------------------------------------------------------

const N_ACTIONS: usize = 2;

#[derive(Module, Debug)]
struct QrDqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_quantiles: usize,
}

impl<B: Backend> QrDqnMlp<B> {
    fn new(num_quantiles: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, N_ACTIONS * num_quantiles).init(device),
            num_quantiles,
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = observations.dims();
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        let flat = self.l3.forward(x);
        flat.reshape([batch_size, N_ACTIONS, self.num_quantiles])
    }
}

impl<B: AutodiffBackend> QrDqnModel<B, 2> for QrDqnMlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(observations)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(observations)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, QrDqnMlp<B::InnerBackend>>(
            &active.valid(),
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
fn polyak_update<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
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
type Agent = QrDqnAgent<Be, QrDqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

fn fresh_agent(seed: u64) -> Agent {
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    // QR-DQN's quantile Huber loss uses a `(batch, N, N)` broadcast — one
    // factor of `N` slower per learn step than C51's categorical loss. Use
    // a smaller quantile count than C51 to keep the integration test's
    // wall-clock on ndarray practical for CI.
    let num_quantiles = 21;
    let config = QrDqnTrainingConfigBuilder::new()
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
        .num_quantiles(num_quantiles)
        .kappa(1.0)
        .build();
    let model: QrDqnMlp<Be> = QrDqnMlp::new(num_quantiles, &device);
    QrDqnAgent::new(model, config, device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: a short run completes, the buffer populates, and episode
/// rewards are finite. Catches silent regressions (NaN quantiles, loss
/// numerical blow-up, etc.). Marked `#[ignore]` because running it alongside
/// other training tests perturbs Burn's global ndarray RNG; exercise with
/// `cargo test -p rlevo-reinforcement-learning --test qrdqn_integration -- --ignored
/// --test-threads=1`.
#[test]
#[ignore = "perturbs global Burn RNG; run with --test-threads=1"]
fn qrdqn_short_run_produces_finite_rewards() {
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
        assert!(
            m.quantile_spread.is_finite(),
            "non-finite quantile_spread at episode {i}"
        );
    }
}

/// Reproducibility smoke test: two seeded back-to-back runs produce identical
/// reward sequences. Marked `#[ignore]` for the same reason as the DQN/C51
/// counterparts — Burn's ndarray backend uses a process-global RNG.
#[test]
#[ignore = "requires --test-threads=1 to isolate Burn's global RNG"]
#[allow(clippy::float_cmp)]
fn qrdqn_reproducibility_ndarray() {
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

/// `CartPole` target: after 20k steps the 100-episode moving average should
/// clear 50 — a rough "is it training?" sanity check tuned for the
/// smaller quantile count used in CI. A stricter 195-at-500k-steps test
/// lives behind `#[ignore]` below for manual validation.
#[test]
fn qrdqn_cart_pole_reaches_50() {
    let seed: u64 = 42;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 20_000, 0).expect("training loop");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(avg >= 50.0, "expected avg reward >= 50, got {avg:.2}");
}

/// Full acceptance target: ≥ 195 on CartPole-v1 in ≤ 500k
/// steps at seed 42. Ignored by default — too expensive for regular CI.
/// Run with:
/// `cargo test -p rlevo-reinforcement-learning --test qrdqn_integration --release --
///      --ignored qrdqn_solves_cart_pole_ndarray_seed_42`.
#[test]
#[ignore = "long-running acceptance target; ~500k steps on ndarray CPU"]
fn qrdqn_solves_cart_pole_ndarray_seed_42() {
    let seed: u64 = 42;
    let mut env = CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 500_000, 0).expect("training loop");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(avg >= 195.0, "expected avg reward >= 195, got {avg:.2}");
}
