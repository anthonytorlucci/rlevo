//! Regression witness for ADR 0053 / issue #253: a **multi-component rank-1**
//! `BoundedAction` must survive the continuous-control agents' action paths.
//!
//! Every `BoundedAction` impl in the workspace happens to satisfy
//! `RANK == COMPONENTS`, which is exactly what kept the rank-vs-component
//! conflation latent: `low()`/`high()` returned `[f32; R]` (one bound for all
//! `C` components) and the agents looped `0..A::RANK` over a `COMPONENTS`-wide
//! actor output.
//!
//! The fixture here is deliberately the shape no existing impl has, and the
//! shape ADR 0053 §7 names as the regression witness: rank 1, `COMPONENTS = 3`,
//! and **asymmetric** per-component bounds `[-1, 0, 0] .. [1, 1, 1]` — the real
//! Gymnasium `CarRacing` action space (steer ∈ [-1,1], gas ∈ [0,1],
//! brake ∈ [0,1]).
//!
//! These tests were verified to **fail** against the pre-fix agent code (the
//! `A::RANK`-keyed loops temporarily restored): three of the four panicked,
//! with `from_slice` rejecting a 1-element slice on the warm-up and greedy
//! paths and `GaussianNoise::apply`'s own length check rejecting a 1-element
//! mean on the exploration path. Each test's docs record its observed failure.
//! (Pre-fix the trait could not even *express* this fixture — `[f32; 1]` holds
//! one of the three bounds — so the trait half of the fix is a compile-time
//! prerequisite for the fixture existing at all.)

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::Action;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_agent::DdpgAgent;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{
    ContinuousQ, DeterministicPolicy,
};
use rlevo_reinforcement_learning::utils::polyak_update;

use rlevo_test_support::env::LinearObservation;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};

// ---------------------------------------------------------------------------
// The fixture: rank 1, three components, asymmetric bounds.
// ---------------------------------------------------------------------------

/// `CarRacing`-shaped action: `steer ∈ [-1, 1]`, `gas ∈ [0, 1]`,
/// `brake ∈ [0, 1]`.
///
/// Rank **1** with **3** components — the combination that makes the rank/
/// component distinction observable.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CarLikeAction([f32; 3]);

impl Action<1> for CarLikeAction {
    fn shape() -> [usize; 1] {
        [3]
    }

    fn is_valid(&self) -> bool {
        let (low, high) = (Self::low(), Self::high());
        self.0
            .iter()
            .enumerate()
            .all(|(i, v)| v.is_finite() && *v >= low[i] && *v <= high[i])
    }
}

impl ContinuousAction<1> for CarLikeAction {
    const COMPONENTS: usize = 3;

    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self([
            self.0[0].clamp(min, max),
            self.0[1].clamp(min, max),
            self.0[2].clamp(min, max),
        ])
    }

    fn from_slice(values: &[f32]) -> Self {
        // This is the assertion that fires against the pre-fix agents: they
        // hand over `RANK == 1` value where `COMPONENTS == 3` are required.
        assert_eq!(
            values.len(),
            Self::COMPONENTS,
            "expected {} components, got {}",
            Self::COMPONENTS,
            values.len()
        );
        Self([values[0], values[1], values[2]])
    }
}

impl BoundedAction<1> for CarLikeAction {
    fn low() -> &'static [f32] {
        &[-1.0, 0.0, 0.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0, 1.0]
    }
}

// ---------------------------------------------------------------------------
// Minimal actor / critic over a 1-D observation and a 3-component action.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Actor<B: Backend> {
    head: Linear<B>,
}

impl<B: Backend> Actor<B> {
    fn new(
        obs_dim: usize,
        action_dim: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            head: LinearConfig::new(obs_dim, action_dim).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        tanh(self.head.forward(obs))
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
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Actor<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

#[derive(Module, Debug)]
struct Critic<B: Backend> {
    head: Linear<B>,
}

impl<B: Backend> Critic<B> {
    fn new(
        obs_dim: usize,
        action_dim: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            head: LinearConfig::new(obs_dim + action_dim, 1).init(device),
        }
    }
    fn forward_impl(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, act], 1);
        relu(self.head.forward(x)).squeeze_dim::<1>(1)
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
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, Critic<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type CarAgent = DdpgAgent<Be, Actor<Be>, Critic<Be>, LinearObservation, CarLikeAction, 1, 2, 1, 2>;

/// Builds a DDPG agent whose action type has 3 components and a warm-up window
/// long enough that `act(.., training = true)` takes the uniform-sample branch.
fn build_agent(learning_starts: usize) -> CarAgent {
    let device = seeded_device::<Be>(0xA53);
    let actor = Actor::<Be>::new(1, CarLikeAction::COMPONENTS, &device);
    let critic = Critic::<Be>::new(1, CarLikeAction::COMPONENTS, &device);
    let config = DdpgTrainingConfigBuilder::default()
        .learning_starts(learning_starts)
        .batch_size(4)
        .replay_buffer_capacity(64)
        .build()
        .expect("valid DDPG config");
    CarAgent::new(actor, critic, config, device).expect("agent construction")
}

/// Asserts every component sits inside its **own** bound, not component 0's.
fn assert_within_per_component_bounds(action: &CarLikeAction) {
    let (low, high) = (CarLikeAction::low(), CarLikeAction::high());
    let values = action.as_slice();
    assert_eq!(
        values.len(),
        CarLikeAction::COMPONENTS,
        "action must carry COMPONENTS values, not RANK"
    );
    for (i, v) in values.iter().enumerate() {
        assert!(
            *v >= low[i] && *v <= high[i],
            "component {i} = {v} outside its own bound [{}, {}]",
            low[i],
            high[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The trait's own contract, on a fixture where `RANK != COMPONENTS`.
///
/// This is the assertion that distinguishes ADR 0053's design from the one it
/// replaced: `low()` has three entries because the action has three
/// *components*, not because it has rank 3 (it has rank 1).
#[test]
fn bounds_are_keyed_on_components_not_rank() {
    assert_eq!(CarLikeAction::RANK, 1);
    assert_eq!(CarLikeAction::COMPONENTS, 3);
    assert_ne!(
        CarLikeAction::RANK,
        CarLikeAction::COMPONENTS,
        "fixture is pointless unless rank and component count disagree"
    );

    assert_eq!(CarLikeAction::low().len(), CarLikeAction::COMPONENTS);
    assert_eq!(CarLikeAction::high().len(), CarLikeAction::COMPONENTS);
    for i in 0..CarLikeAction::COMPONENTS {
        assert!(CarLikeAction::low()[i] < CarLikeAction::high()[i]);
    }

    // The bounds must actually differ per component, or the fixture cannot
    // distinguish a per-component clip from a `low[0]`/`high[0]` collapse.
    let low = CarLikeAction::low();
    assert!(
        low.iter().any(|v| (v - low[0]).abs() > f32::EPSILON),
        "asymmetric bounds are the point of this fixture: {low:?}"
    );
}

/// Warm-up path: `act(.., training = true)` below `learning_starts` draws a
/// uniform sample per component.
///
/// Pre-fix this looped `0..A::RANK` (= 1), producing a 1-element `Vec` that
/// `CarLikeAction::from_slice` rejects. Observed pre-fix failure:
/// `assertion `left == right` failed: expected 3 components, got 1`.
#[test]
fn warmup_sample_produces_all_components_within_bounds() {
    let _guard = flex_guard();
    let agent = build_agent(1_000);
    let mut rng = StdRng::seed_from_u64(0x00C0_FFEE);

    for _ in 0..32 {
        let action = agent.act(&LinearObservation { x: 0.25 }, true, &mut rng);
        assert_within_per_component_bounds(&action);
    }
}

/// Greedy/eval path: `act(.., training = false)` clips the actor output per
/// component.
///
/// The actor emits `tanh(..) ∈ [-1, 1]` on all three outputs, so the `gas` and
/// `brake` components are genuinely out of *their* bounds before clipping —
/// which is what makes this a test of per-component clipping rather than of
/// `low[0]`/`high[0]`.
///
/// Pre-fix this looped `0..A::RANK` and failed identically to the warm-up
/// path (`expected 3 components, got 1`).
#[test]
fn greedy_action_clips_each_component_against_its_own_bound() {
    let _guard = flex_guard();
    let agent = build_agent(0);
    let mut rng = StdRng::seed_from_u64(7);

    let action = agent.act(&LinearObservation { x: -0.5 }, false, &mut rng);
    assert_within_per_component_bounds(&action);
    assert!(
        action.is_valid(),
        "clipped action must satisfy its own validity predicate"
    );

    // Same path through the pre-snapshotted inner actor.
    let net = agent.inference_net();
    let via_snapshot = agent.act_with(&net, &LinearObservation { x: -0.5 });
    assert_within_per_component_bounds(&via_snapshot);
}

/// Exploration path: noisy actions are still clipped per component.
///
/// Pre-fix this failed *earlier* than the other two and in the more telling
/// place — `.take(A::RANK)` handed a 1-element mean to a `GaussianNoise` whose
/// bounds were 3 elements long, so the exploration layer's own length check
/// fired first:
/// `exploration.rs:50 — assertion `left == right` failed: mean/low length
/// mismatch, left: 1, right: 3`. That layer was per-component correct all
/// along (ADR 0053 §Context (a)); only the trait feeding it was wrong.
#[test]
fn noisy_action_stays_within_per_component_bounds() {
    let _guard = flex_guard();
    let agent = build_agent(0);
    let mut rng = StdRng::seed_from_u64(99);

    for _ in 0..32 {
        let action = agent.act(&LinearObservation { x: 1.0 }, true, &mut rng);
        assert_within_per_component_bounds(&action);
    }
}
