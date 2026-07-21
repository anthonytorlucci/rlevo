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

use std::sync::{Arc, Mutex};

use burn::module::{AutodiffModule, Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::Action;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_agent::DdpgAgent;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{
    ContinuousQ, DeterministicPolicy,
};
use rlevo_reinforcement_learning::algorithms::sac::sac_agent::SacAgent;
use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::sac::sac_model::{
    SampleOutput, SquashedGaussianPolicy,
};
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
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

// ===========================================================================
// SAC — the same fixture through the squashed-Gaussian agent.
//
// `sac_agent.rs` has two `COMPONENTS`-keyed action sites: the warm-up uniform
// sampler and the per-component clamp on the policy path. Both were `A::RANK`
// before ADR 0053 §5. Every `BoundedAction` impl reachable from
// `sac_integration.rs` is 1/1, so nothing there can tell the two apart; this
// fixture (rank 1, 3 components, asymmetric bounds) can.
// ===========================================================================

/// Test-local SAC actor with a **controllable saturation direction**.
///
/// `mean = obs · gain` with a large `gain`, so a negative observation drives
/// every pre-squash logit far negative and `tanh` pins the whole row at `≈ -1`
/// — outside the `gas`/`brake` lower bound of `0`, which is what makes the
/// per-component clamp observable rather than incidental.
///
/// The returned `log_prob` is a differentiable placeholder, **not** the true
/// squashed-Gaussian density. That is sound here because these tests exercise
/// only [`SacAgent::act`], which never reads it; `sac_integration.rs` owns the
/// correctness of the real density.
#[derive(Module, Debug)]
struct SquashedActor<B: Backend> {
    gain: Param<Tensor<B, 2>>,
    action_dim: usize,
}

impl<B: Backend> SquashedActor<B> {
    /// σ on the reparameterized sample. Small enough that a `gain`-saturated
    /// mean stays saturated for any plausible ε.
    const SIGMA: f32 = 0.1;

    fn new(
        action_dim: usize,
        gain: f32,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        let data = TensorData::new(vec![gain; action_dim], vec![1, action_dim]);
        Self {
            gain: Param::from_tensor(Tensor::from_data(data, device)),
            action_dim,
        }
    }

    /// `[batch, 1] × [1, action_dim]` broadcasts to `[batch, action_dim]`.
    fn sample_impl(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let z = obs * self.gain.val() + eps.mul_scalar(Self::SIGMA);
        let log_prob = z
            .clone()
            .powi_scalar(2)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .neg();
        (tanh(z), log_prob)
    }
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for SquashedActor<B> {
    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn forward_sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        let (action, log_prob) = self.sample_impl(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        let (action, log_prob) = inner.sample_impl(obs, eps);
        SampleOutput { action, log_prob }
    }

    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        tanh(obs * self.gain.val())
    }
}

type CarSacAgent =
    SacAgent<Be, SquashedActor<Be>, Critic<Be>, LinearObservation, CarLikeAction, 1, 2, 1, 2>;

/// Builds a SAC agent over the 3-component fixture. `gain` sets how hard the
/// policy saturates; the sign of the observation then picks the direction.
fn build_sac_agent(learning_starts: usize, gain: f32) -> CarSacAgent {
    let device = seeded_device::<Be>(0x5AC);
    let actor = SquashedActor::<Be>::new(CarLikeAction::COMPONENTS, gain, &device);
    let critic_1 = Critic::<Be>::new(1, CarLikeAction::COMPONENTS, &device);
    let critic_2 = Critic::<Be>::new(1, CarLikeAction::COMPONENTS, &device);
    let config = SacTrainingConfigBuilder::default()
        .learning_starts(learning_starts)
        .batch_size(4)
        .replay_buffer_capacity(64)
        .build()
        .expect("valid SAC config");
    CarSacAgent::new(actor, critic_1, critic_2, config, device).expect("agent construction")
}

/// SAC warm-up path: `act(.., training = true)` below `learning_starts` draws
/// a uniform sample per component (`sac_agent.rs:378`).
///
/// Reverting that loop to `0..A::RANK` yields a 1-element `Vec` where
/// `CarLikeAction::from_slice` demands 3, so the fix is load-bearing for this
/// test to run at all — and the per-component bound assertion below then pins
/// that each sampled component used *its own* range rather than component 0's.
#[test]
fn sac_warmup_sample_produces_all_components_within_bounds() {
    let _guard = flex_guard();
    let agent = build_sac_agent(1_000, 10.0);
    let mut rng = StdRng::seed_from_u64(0x5AC0_0001);

    for _ in 0..32 {
        let action = agent.act(&LinearObservation { x: 0.25 }, true, &mut rng);
        assert_within_per_component_bounds(&action);
    }
}

/// SAC policy path: `act` clamps the squashed actor output against
/// `low[i]`/`high[i]` (`sac_agent.rs:410`).
///
/// The actor is driven to saturate at `≈ -1` on **every** component, so `gas`
/// and `brake` arrive genuinely below their own lower bound of `0`. The
/// expected row is therefore `[-1, 0, 0]`, which no `low[0]`-keyed clamp can
/// produce, and which the pre-fix `0..A::RANK` loop cannot even reach — it
/// hands 1 value to a `from_slice` that requires 3.
#[test]
fn sac_policy_action_clamps_each_component_against_its_own_bound() {
    let _guard = flex_guard();
    let agent = build_sac_agent(0, 10.0);
    let mut rng = StdRng::seed_from_u64(0x5AC0_0002);

    // `training = false` ⇒ ε = 0 ⇒ the deterministic squashed mean.
    let action = agent.act(&LinearObservation { x: -1.0 }, false, &mut rng);
    assert_within_per_component_bounds(&action);
    let values = action.as_slice();
    for (i, want) in [-1.0_f32, 0.0, 0.0].iter().enumerate() {
        assert!(
            (values[i] - want).abs() < 1e-5,
            "a fully-negative squashed action must clamp to each component's own \
             lower bound: wanted {want} at component {i}, got {} (full row: {values:?})",
            values[i]
        );
    }

    // The noisy branch takes the same clamp.
    for _ in 0..16 {
        let noisy = agent.act(&LinearObservation { x: -1.0 }, true, &mut rng);
        assert_within_per_component_bounds(&noisy);
    }
}

// ===========================================================================
// DDPG `learn_step` — the target-action clip, end to end.
//
// `clip_to_action_bounds` is unit-tested in `shared.rs`, but in isolation:
// nothing there exercises DDPG's *wiring* of `low_t`/`high_t` through
// `action_bound_tensors::<B::InnerBackend, A, DA, DAB>` at construction. A
// swapped pair, or a wrong generic at that call site, would pass every
// existing test. These two tests observe the tensor the target critic is
// actually handed.
// ===========================================================================

/// Actions the **target** critic was evaluated on, one flat `[batch × C]` row
/// per `forward_inner` call.
type RecordedActions = Arc<Mutex<Vec<Vec<f32>>>>;

/// Critic that records every action batch reaching its no-autodiff path.
///
/// In DDPG's `learn_step`, `forward_inner` is called exactly once — on the
/// target critic, with the clipped target action (`ddpg_agent.rs:499-505`).
/// The live critic's `forward` is a different method and is not recorded, so
/// the capture is unambiguous.
///
/// `recorded` is `#[module(skip)]`, so the derive treats it as a plain cloned
/// field rather than a submodule; `AutodiffModule::valid()` clones the `Arc`,
/// which is what lets the target twin write into the same buffer the test
/// reads.
#[derive(Module, Debug)]
struct RecordingCritic<B: Backend> {
    head: Linear<B>,
    #[module(skip)]
    recorded: RecordedActions,
}

impl<B: Backend> RecordingCritic<B> {
    fn new(
        obs_dim: usize,
        action_dim: usize,
        recorded: RecordedActions,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            head: LinearConfig::new(obs_dim + action_dim, 1).init(device),
            recorded,
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, act], 1);
        relu(self.head.forward(x)).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> ContinuousQ<B, 2, 2> for RecordingCritic<B> {
    fn forward(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs, act)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        act: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 1> {
        let row = act
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("target action host read");
        inner.recorded.lock().expect("recorder mutex").push(row);
        inner.forward_impl(obs, act)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, RecordingCritic<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

/// Actor whose output saturates at `±1` on **every** component, with the sign
/// chosen by the sign of the observation.
///
/// The target actor is a `valid()` snapshot taken at construction and is only
/// Polyak-nudged *after* the target computation, so the single `learn_step`
/// below sees exactly this mapping.
#[derive(Module, Debug)]
struct SaturatingActor<B: Backend> {
    gain: Param<Tensor<B, 2>>,
}

impl<B: Backend> SaturatingActor<B> {
    fn new(
        action_dim: usize,
        gain: f32,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        let data = TensorData::new(vec![gain; action_dim], vec![1, action_dim]);
        Self {
            gain: Param::from_tensor(Tensor::from_data(data, device)),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        tanh(obs * self.gain.val())
    }
}

impl<B: AutodiffBackend> DeterministicPolicy<B, 2, 2> for SaturatingActor<B> {
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
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, SaturatingActor<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type SaturatingAgent = DdpgAgent<
    Be,
    SaturatingActor<Be>,
    RecordingCritic<Be>,
    LinearObservation,
    CarLikeAction,
    1,
    2,
    1,
    2,
>;

const LEARN_BATCH: usize = 4;

/// Runs one DDPG `learn_step` whose every `next_obs` is `next_obs_x`, and
/// returns the target-action rows the target critic was handed, one `Vec` of
/// `COMPONENTS` values per batch element.
fn target_actions_after_one_learn_step(next_obs_x: f32) -> Vec<Vec<f32>> {
    let device = seeded_device::<Be>(0xD06);
    let recorded: RecordedActions = Arc::new(Mutex::new(Vec::new()));
    let actor = SaturatingActor::<Be>::new(CarLikeAction::COMPONENTS, 10.0, &device);
    let critic =
        RecordingCritic::<Be>::new(1, CarLikeAction::COMPONENTS, Arc::clone(&recorded), &device);
    let config = DdpgTrainingConfigBuilder::default()
        .learning_starts(0)
        .batch_size(LEARN_BATCH)
        .replay_buffer_capacity(64)
        .build()
        .expect("valid DDPG config");
    let mut agent =
        SaturatingAgent::new(actor, critic, config, device).expect("agent construction");

    let mut rng = StdRng::seed_from_u64(0xD06_0001);
    for _ in 0..(LEARN_BATCH * 2) {
        agent.remember(
            LinearObservation { x: 0.1 },
            &CarLikeAction([0.0, 0.5, 0.5]),
            1.0,
            LinearObservation { x: next_obs_x },
            false,
        );
    }
    agent
        .learn_step(&mut rng)
        .expect("no polyak error")
        .expect("learn_step must run: learning_starts is 0 and the buffer is full");

    let calls = recorded.lock().expect("recorder mutex").clone();
    assert_eq!(
        calls.len(),
        1,
        "DDPG's learn_step evaluates the target critic exactly once"
    );
    calls[0]
        .chunks(CarLikeAction::COMPONENTS)
        .map(<[f32]>::to_vec)
        .collect()
}

/// Asserts every recorded target-action row equals `expected`.
fn assert_every_target_row(rows: &[Vec<f32>], expected: [f32; 3], why: &str) {
    assert_eq!(
        rows.len(),
        LEARN_BATCH,
        "one target action per batch element"
    );
    for (r, row) in rows.iter().enumerate() {
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (row[i] - want).abs() < 1e-5,
                "{why}: row {r} component {i} should be {want}, got {} (full row: {row:?})",
                row[i]
            );
        }
    }
}

/// Guards `d1bdbb5`'s replacement of DDPG's scalar
/// `.clamp(self.low[0], self.high[0])` with the per-component
/// `clip_to_action_bounds(.., self.low_t, self.high_t)`.
///
/// The target actor saturates at `≈ [-1, -1, -1]`. Per component that clips to
/// `[-1, 0, 0]`; the scalar collapse uses `low[0]/high[0]` = `-1`/`1` for all
/// three and yields `[-1, -1, -1]` — negative gas and negative brake, the
/// "values of impossible actions" TD3 Eq. 14's clip exists to suppress.
///
/// Unlike the `shared.rs` unit tests, this observes the tensor DDPG actually
/// builds, so it also covers `action_bound_tensors::<B::InnerBackend, A, DA,
/// DAB>` being called with the right generics at the right place.
#[test]
fn ddpg_learn_step_clips_the_target_action_per_component() {
    let _guard = flex_guard();
    let rows = target_actions_after_one_learn_step(-1.0);
    assert_every_target_row(
        &rows,
        [-1.0, 0.0, 0.0],
        "a saturated-low target action must be floored at each component's own low()",
    );
}

/// The companion direction, which is what makes a swapped `low_t`/`high_t`
/// detectable.
///
/// Swapping the pair computes `max_pair(high).min_pair(low)`. On a
/// saturated-**low** action that happens to reproduce the correct
/// `[-1, 0, 0]`, so the test above cannot see it. On a saturated-**high**
/// action the two disagree maximally: correct gives `[1, 1, 1]`, swapped gives
/// `[-1, 0, 0]`.
#[test]
fn ddpg_learn_step_target_clip_does_not_swap_low_and_high() {
    let _guard = flex_guard();
    let rows = target_actions_after_one_learn_step(1.0);
    assert_every_target_row(
        &rows,
        [1.0, 1.0, 1.0],
        "a saturated-high target action must be capped at high(), not floored at low()",
    );
}
