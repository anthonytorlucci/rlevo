//! End-to-end integration tests for the DDPG agent.
//!
//! The default-run test exercises DDPG on a minimal synthetic 1-D continuous
//! env so that `cargo test` stays tractable (no physics simulator, small
//! networks, ~50k env steps). The Pendulum macro-smoke is gated behind
//! `#[ignore]` per the project's integration-test budget convention.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_rl::algorithms::ddpg::ddpg_agent::DdpgAgent;
use rlevo_rl::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
use rlevo_rl::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use rlevo_rl::algorithms::ddpg::train::train;

// ---------------------------------------------------------------------------
// Synthetic 1-D continuous environment
// ---------------------------------------------------------------------------
//
// Each step emits an observation `x ∈ [-1, 1]`. The optimal action is `a = x`
// and the reward is `-(a - x)²`, peaking at `0`. Episodes last 20 steps.
// The actor must learn an identity mapping; convergence shows up as the
// moving-average episode reward climbing toward `0` from the U(-1, 1)
// baseline of ≈ `-20 · 1/3 ≈ -6.67`.

#[derive(Debug, Clone, Copy, PartialEq)]
struct LinearState {
    x: f32,
    steps: usize,
}

impl State<1> for LinearState {
    type Observation = LinearObservation;
    fn shape() -> [usize; 1] {
        [1]
    }
    fn numel(&self) -> usize {
        1
    }
    fn is_valid(&self) -> bool {
        self.x.is_finite()
    }
    fn observe(&self) -> LinearObservation {
        LinearObservation { x: self.x }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct LinearObservation {
    x: f32,
}

impl Observation<1> for LinearObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl<B: Backend> TensorConvertible<1, B> for LinearObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_data(TensorData::new(vec![self.x], vec![1]), device)
    }
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let v = tensor.into_data().convert::<f32>();
        Ok(Self {
            x: v.as_slice::<f32>().unwrap()[0],
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct LinearAction(f32);

impl Action<1> for LinearAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0.abs() <= 1.0
    }
}

impl ContinuousAction<1> for LinearAction {
    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }
    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.clamp(min, max))
    }
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1);
        Self(values[0])
    }
}

impl BoundedAction<1> for LinearAction {
    fn low() -> [f32; 1] {
        [-1.0]
    }
    fn high() -> [f32; 1] {
        [1.0]
    }
}

struct LinearEnv {
    state: LinearState,
    rng: StdRng,
    episode_len: usize,
}

impl LinearEnv {
    fn with_seed(seed: u64, episode_len: usize) -> Self {
        Self {
            state: LinearState { x: 0.0, steps: 0 },
            rng: StdRng::seed_from_u64(seed),
            episode_len,
        }
    }

    fn sample_x(rng: &mut StdRng) -> f32 {
        use rand::RngExt;
        rng.random_range(-1.0_f32..=1.0_f32)
    }
}

impl Environment<1, 1, 1> for LinearEnv {
    type StateType = LinearState;
    type ObservationType = LinearObservation;
    type ActionType = LinearAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, LinearObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_seed(0, 20)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = LinearState {
            x: Self::sample_x(&mut self.rng),
            steps: 0,
        };
        Ok(SnapshotBase {
            observation: self.state.observe(),
            reward: ScalarReward::new(0.0),
            status: EpisodeStatus::Running,
        })
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let a = action.0.clamp(-1.0, 1.0);
        let err = a - self.state.x;
        let reward = -(err * err);
        let next_x = Self::sample_x(&mut self.rng);
        self.state = LinearState {
            x: next_x,
            steps: self.state.steps + 1,
        };
        let status = if self.state.steps >= self.episode_len {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };
        Ok(SnapshotBase {
            observation: self.state.observe(),
            reward: ScalarReward::new(reward),
            status,
        })
    }
}

// ---------------------------------------------------------------------------
// Actor & critic: tiny MLPs (shared with Pendulum smoke test).
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Actor<B: Backend> {
    fc1: Linear<B>,
    head: Linear<B>,
    action_scale: f32,
}

impl<B: Backend> Actor<B> {
    fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        scale: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            head: LinearConfig::new(hidden, action_dim).init(device),
            action_scale: scale,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        tanh(self.head.forward(h)).mul_scalar(self.action_scale)
    }
}

impl<B: AutodiffBackend> DeterministicPolicy<B, 2, 2> for Actor<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(obs)
    }
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Actor<B::InnerBackend>>(active.valid(), target, tau as f32)
    }
}

#[derive(Module, Debug)]
struct Critic<B: Backend> {
    fc1: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> Critic<B> {
    fn new(obs_dim: usize, action_dim: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim + action_dim, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }
    fn forward_impl(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, act], 1);
        let h = relu(self.fc1.forward(x));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> ContinuousQ<B, 2, 2> for Critic<B> {
    fn forward(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs, act)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        act: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 1> {
        inner.forward_impl(obs, act)
    }
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Critic<B::InnerBackend>>(
            active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// Polyak averaging (shared copy of the dqn_cart_pole example helper).
// ---------------------------------------------------------------------------

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
        param.map(move |t| {
            let device = t.device();
            let active_tensor = Tensor::<B, D>::from_data(active, &device);
            t.mul_scalar(1.0 - tau) + active_tensor.mul_scalar(tau)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Default-run test: DDPG should solve the tracking task well enough that the
/// 100-episode moving average clears a lax `-1.0` threshold within 30k env
/// steps. Random actions average ≈ `-6.67` per episode, and an optimal policy
/// approaches `0`.
#[test]
fn ddpg_solves_linear_1d_continuous() {
    let seed: u64 = 42;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: Actor<Be> = Actor::new(1, 32, 1, 1.0, &device);
    let critic: Critic<Be> = Critic::new(1, 1, 32, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .buffer_capacity(20_000)
        .batch_size(32)
        .learning_starts(500)
        .actor_lr(1e-3)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.02)
        .exploration_noise(0.2)
        .policy_frequency(2)
        .build();
    let mut agent: DdpgAgent<
        Be,
        Actor<Be>,
        Critic<Be>,
        LinearObservation,
        LinearAction,
        1,
        2,
        1,
        2,
    > = DdpgAgent::new(actor, critic, config, device);

    train::<Be, _, _, _, _, LinearAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, 8_000, 0,
    )
    .expect("training");

    let avg = agent.stats().avg_score().expect("non-empty history");
    assert!(avg.is_finite(), "avg reward must be finite, got {avg}");
    // Random baseline ≈ −6.67. A learned policy should easily beat −1.0.
    assert!(
        avg > -1.0,
        "expected avg reward > -1.0, got {avg:.3} (random baseline ≈ -6.67)"
    );
}

/// Pendulum macro-smoke: 500k steps, checks the moving average is finite and
/// better than the zero-torque baseline. Gated — Burn's ndarray backend
/// shares a global RNG with other tests so this runs at `--test-threads=1`.
#[test]
#[ignore = "macro run (~500k Pendulum steps); --test-threads=1 for isolated Burn RNG"]
fn ddpg_pendulum_smoke() {
    let seed: u64 = 42;
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    });
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor: Actor<Be> = Actor::new(3, 256, 1, 2.0, &device);
    let critic: Critic<Be> = Critic::new(3, 1, 256, &device);
    let config = DdpgTrainingConfigBuilder::new()
        .buffer_capacity(200_000)
        .batch_size(256)
        .learning_starts(10_000)
        .actor_lr(1e-4)
        .critic_lr(1e-3)
        .gamma(0.99)
        .tau(0.005)
        .exploration_noise(0.1)
        .policy_frequency(2)
        .build();
    let mut agent: DdpgAgent<
        Be,
        Actor<Be>,
        Critic<Be>,
        PendulumObservation,
        PendulumAction,
        1,
        2,
        1,
        2,
    > = DdpgAgent::new(actor, critic, config, device);

    train::<Be, _, _, _, _, PendulumAction, _, 1, 1, 2, 1, 2>(
        &mut agent, &mut env, &mut rng, 500_000, 0,
    )
    .expect("training");

    let avg = agent.stats().avg_score().expect("non-empty history");
    assert!(avg.is_finite(), "avg reward must be finite, got {avg}");
    // Zero-torque Pendulum scores ≈ -1200; DDPG should comfortably beat that
    // after 500k steps. A tighter -200 acceptance target stays aspirational
    // and is not gated as a test.
    assert!(avg > -800.0, "expected avg reward > -800, got {avg:.2}");
}
