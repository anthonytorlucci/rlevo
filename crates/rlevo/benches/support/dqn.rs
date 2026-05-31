//! Shared DQN scaffolding for the `*_dqn` random-vs-learned benches.
//!
//! Each `[[bench]]` target is its own compilation unit, so this module is
//! pulled in per-bench via `#[path = "support/dqn.rs"] mod support;`. It lives
//! in a subdirectory so Cargo's bench autodiscovery does not treat it as a
//! standalone target. It provides the two pieces every DQN bench repeats:
//!
//! - [`VecMlpDqn`] — a two-hidden-layer MLP for rank-1 vector observations
//!   (batched `[batch, features]`), e.g. classic-control envs.
//! - [`GridMlpDqn`] — the same MLP fronted by a flatten, for rank-3 grid
//!   observations (batched `[batch, h, w, c]`), e.g. the gridworlds.
//!
//! Both reuse one Polyak target-network update built on Burn's
//! `ModuleVisitor`/`ModuleMapper`, identical to the `dqn_cart_pole` example.

// Each bench includes this module but uses only the model it needs, so the
// other model (and its helpers) are legitimately unused per compilation unit.
#![allow(dead_code)]

use std::collections::HashMap;
use std::marker::PhantomData;

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData, activation};

use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;

// ---------------------------------------------------------------------------
// Vector MLP — rank-1 observations, batched to rank 2.
// ---------------------------------------------------------------------------

/// Two-hidden-layer MLP mapping batched `[batch, in_features]` observations
/// to `[batch, out_features]` Q-values.
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
// Grid MLP — rank-3 observations flattened, batched to rank 4.
// ---------------------------------------------------------------------------

/// Flatten + two-hidden-layer MLP mapping batched `[batch, h, w, c]` grid
/// observations to `[batch, out_features]` Q-values.
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
fn polyak_update<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
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
