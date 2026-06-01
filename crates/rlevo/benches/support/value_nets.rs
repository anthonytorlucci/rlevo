//! Shared value networks for the `*_rl` benches.
//!
//! Every classic-control and gridworld bench drives a value-based or
//! actor-critic agent over a small MLP. Those models were identical bench to
//! bench apart from observation width, hidden size, and action count, so they
//! live here once, parameterised, rather than copied per target. Each
//! `[[bench]]` is its own compilation unit, so this module is pulled in via
//! `#[path = "support/value_nets.rs"] mod ...;`; it sits in a subdirectory so
//! Cargo's bench autodiscovery does not treat it as a standalone target.
//!
//! - [`VecMlpDqn`] — DQN Q-network for rank-1 vector observations
//!   (batched `[batch, features]`), e.g. classic-control envs.
//! - [`GridMlpDqn`] — DQN Q-network fronted by a flatten, for rank-3 grid
//!   observations (batched `[batch, h, w, c]`), e.g. the gridworlds.
//! - [`C51Mlp`] — categorical (C51) critic: `(batch, obs)` →
//!   `(batch, actions, atoms)`.
//! - [`QrDqnMlp`] — quantile (QR-DQN) critic: `(batch, obs)` →
//!   `(batch, actions, quantiles)`.
//! - [`ValueMlp`] — scalar PPG critic: `(batch, obs)` → `(batch,)`.
//!
//! The four target-network models share one Polyak update built on Burn's
//! `ModuleVisitor`/`ModuleMapper`.

// Each bench includes this module but uses only the models it needs, so the
// other models (and their helpers) are legitimately unused per compilation unit.
#![allow(dead_code)]

use std::collections::HashMap;
use std::marker::PhantomData;

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData, activation};

use rlevo_reinforcement_learning::algorithms::c51::c51_model::C51Model;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::qrdqn::qrdqn_model::QrDqnModel;

// ---------------------------------------------------------------------------
// DQN vector MLP — rank-1 observations, batched to rank 2.
// ---------------------------------------------------------------------------

/// Two-hidden-layer `ReLU` MLP mapping batched `[batch, in_features]`
/// observations to `[batch, out_features]` Q-values.
#[derive(Module, Debug)]
pub struct VecMlpDqn<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> VecMlpDqn<B> {
    /// Builds an MLP with two `hidden`-wide rectified-linear layers.
    pub fn new(
        in_features: usize,
        hidden: usize,
        out_features: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            l1: LinearConfig::new(in_features, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, out_features).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for VecMlpDqn<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(observations)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(observations)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, VecMlpDqn<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// DQN grid MLP — rank-3 observations flattened, batched to rank 4.
// ---------------------------------------------------------------------------

/// Flatten + two-hidden-layer `ReLU` MLP mapping batched `[batch, h, w, c]`
/// grid observations to `[batch, out_features]` Q-values.
#[derive(Module, Debug)]
pub struct GridMlpDqn<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> GridMlpDqn<B> {
    /// Builds an MLP whose input width is the flattened observation size
    /// (`h * w * c`).
    pub fn new(
        in_features: usize,
        hidden: usize,
        out_features: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            l1: LinearConfig::new(in_features, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, out_features).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, h, w, c] = observations.dims();
        let x = observations.reshape([batch, h * w * c]);
        let x = activation::relu(self.l1.forward(x));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 4> for GridMlpDqn<B> {
    fn forward(&self, observations: Tensor<B, 4>) -> Tensor<B, 2> {
        self.forward_impl(observations)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 4>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(observations)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, GridMlpDqn<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// C51 model — (batch, obs) → (batch, actions, atoms)
// ---------------------------------------------------------------------------

/// Categorical (C51) value network: `(batch, obs)` → `(batch, actions, atoms)`.
///
/// A two-hidden-layer `ReLU` MLP whose final layer is reshaped into a per-action
/// distribution over `num_atoms` support points. Implements [`C51Model`] so the
/// C51 trainer can drive it and Polyak-average a target copy.
#[derive(Module, Debug)]
pub struct C51Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_actions: usize,
    num_atoms: usize,
}

impl<B: Backend> C51Mlp<B> {
    /// Builds a `hidden`-wide two-layer MLP over `obs_features` inputs emitting
    /// `num_actions * num_atoms` logits, reshaped to `(batch, actions, atoms)`.
    pub fn new(
        obs_features: usize,
        hidden: usize,
        num_actions: usize,
        num_atoms: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            l1: LinearConfig::new(obs_features, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, num_actions * num_atoms).init(device),
            num_actions,
            num_atoms,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = obs.dims();
        let x = activation::relu(self.l1.forward(obs));
        let x = activation::relu(self.l2.forward(x));
        self.l3
            .forward(x)
            .reshape([batch, self.num_actions, self.num_atoms])
    }
}

impl<B: AutodiffBackend> C51Model<B, 2> for C51Mlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(obs)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, C51Mlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// QR-DQN model — (batch, obs) → (batch, actions, quantiles)
// ---------------------------------------------------------------------------

/// Quantile (QR-DQN) value network: `(batch, obs)` → `(batch, actions, quantiles)`.
///
/// A two-hidden-layer `ReLU` MLP whose final layer is reshaped into `num_quantiles`
/// quantile estimates per action. Implements [`QrDqnModel`] so the QR-DQN trainer
/// can drive it and Polyak-average a target copy.
#[derive(Module, Debug)]
pub struct QrDqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    num_actions: usize,
    num_quantiles: usize,
}

impl<B: Backend> QrDqnMlp<B> {
    /// Builds a `hidden`-wide two-layer MLP over `obs_features` inputs emitting
    /// `num_actions * num_quantiles` values, reshaped to `(batch, actions, quantiles)`.
    pub fn new(
        obs_features: usize,
        hidden: usize,
        num_actions: usize,
        num_quantiles: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            l1: LinearConfig::new(obs_features, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, num_actions * num_quantiles).init(device),
            num_actions,
            num_quantiles,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = obs.dims();
        let x = activation::relu(self.l1.forward(obs));
        let x = activation::relu(self.l2.forward(x));
        self.l3
            .forward(x)
            .reshape([batch, self.num_actions, self.num_quantiles])
    }
}

impl<B: AutodiffBackend> QrDqnModel<B, 2> for QrDqnMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.forward_impl(obs)
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

// ---------------------------------------------------------------------------
// PPG value network — two-layer tanh MLP → scalar
// ---------------------------------------------------------------------------

/// PPG critic network: `(batch, obs)` → `(batch,)` scalar value estimates.
///
/// A two-hidden-layer `tanh` MLP with a single-unit head, squeezed to one value
/// per observation. Implements [`PpoValue`] to serve as the PPG agent's critic.
#[derive(Module, Debug)]
pub struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    /// Builds a `hidden`-wide two-layer `tanh` MLP over `obs_features` inputs
    /// with a scalar head.
    pub fn new(
        obs_features: usize,
        hidden: usize,
        device: &<B as BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_features, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> PpoValue<B, 2> for ValueMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs)
    }
}

// ---------------------------------------------------------------------------
// Polyak averaging via Burn's ModuleVisitor / ModuleMapper
// ---------------------------------------------------------------------------

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}

struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: PhantomData<B>,
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

/// Polyak-averages `active` into `target`: `target ← (1 - τ)·target + τ·active`.
///
/// Public so sibling support modules (e.g. `pendulum.rs`) can share this one
/// implementation rather than re-deriving it.
pub fn polyak_update<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        tau,
        _marker: PhantomData,
    };
    target.map(&mut mapper)
}
