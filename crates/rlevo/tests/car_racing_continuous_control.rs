//! Acceptance check for ADR 0053: a continuous-control agent driving the real
//! [`CarRacing`] environment.
//!
//! `CarRacingAction` is the **only** action in the workspace whose components
//! disagree on their bounds — Gymnasium's `Box([-1,0,0], [1,1,1])`, where
//! steering is signed but gas and brake are floored at zero. That makes it the
//! sole in-workspace witness for the two halves of the ADR at once:
//!
//! - the trait half (§1): `low()`/`high()` must carry **three** bounds for a
//!   rank-**1** action, which `[f32; R]` could not express;
//! - the agent half (§6): the DDPG/TD3 target-action clip must be applied per
//!   component, not collapsed to `low[0]`/`high[0]`.
//!
//! Unlike `bounded_action_multi_component.rs`, which uses a hand-rolled fixture
//! and a synthetic observation, this drives the shipped environment end to end:
//! the real 96x96x3 pixel observation, the real physics step, and — critically
//! — the real [`CarRacing::step`] validity gate, which returns
//! `Err(InvalidAction)` on a negative gas or brake. An agent whose action path
//! were not per-component correct could not complete a single episode here.
//!
//! The budgets are deliberately tiny (a handful of steps, one learn step, a
//! 4-sample batch). This is a wiring and correctness check, not a learning
//! test — `CarRacing` rasterizes a frame per step, so a learning budget would not
//! fit the default `cargo test` run.

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::Action;
use rlevo_core::environment::{ConstructableEnv, Environment, Snapshot};
use rlevo_environments::box2d::car_racing::{
    CarRacing, CarRacingAction, CarRacingConfig, CarRacingObservation,
};
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{
    ContinuousQ, DeterministicPolicy,
};
use rlevo_reinforcement_learning::algorithms::td3::td3_agent::Td3Agent;
use rlevo_reinforcement_learning::algorithms::td3::td3_config::Td3TrainingConfigBuilder;
use rlevo_reinforcement_learning::utils::polyak_update;

use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};

/// Flattened pixel-observation width: 96 x 96 x 3.
const OBS_NUMEL: usize = 96 * 96 * 3;

// ---------------------------------------------------------------------------
// Actor / critic over a rank-3 pixel observation and a 3-component action.
// ---------------------------------------------------------------------------

/// Single-layer actor: flatten the frame, project to `COMPONENTS`, squash.
///
/// `tanh` deliberately emits `[-1, 1]` on **all three** outputs, including gas
/// and brake, whose true lower bound is `0`. So the raw policy output is out of
/// bounds on two of three components by construction, and every emitted action
/// depends on the per-component clip to become legal. A scalar clip against
/// `low[0] = -1` would leave the negative values in place and `CarRacing::step`
/// would reject them.
#[derive(Module, Debug)]
struct Actor<B: Backend> {
    head: Linear<B>,
}

impl<B: Backend> Actor<B> {
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            head: LinearConfig::new(OBS_NUMEL, CarRacingAction::COMPONENTS).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch = obs.dims()[0];
        tanh(self.head.forward(obs.reshape([batch, OBS_NUMEL])))
    }
}

impl<B: AutodiffBackend> DeterministicPolicy<B, 4, 2> for Actor<B> {
    fn forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 4>,
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
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            head: LinearConfig::new(OBS_NUMEL + CarRacingAction::COMPONENTS, 1).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 4>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let batch = obs.dims()[0];
        let flat = obs.reshape([batch, OBS_NUMEL]);
        let x = Tensor::cat(vec![flat, act], 1);
        relu(self.head.forward(x)).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> ContinuousQ<B, 4, 2> for Critic<B> {
    fn forward(&self, obs: Tensor<B, 4>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs, act)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 4>,
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

type CarAgent =
    Td3Agent<Be, Actor<Be>, Critic<Be>, CarRacingObservation, CarRacingAction, 3, 4, 1, 2>;

/// Builds a TD3 agent over `CarRacing` with `learning_starts` warm-up steps.
///
/// Construction alone exercises ADR 0053 §4's contract assert and §6's
/// `[1, 3]` bound-tensor build: a `CarRacingAction::low()` of the wrong length,
/// or a `shape()` disagreeing with `COMPONENTS`, panics here.
fn build_agent(learning_starts: usize) -> CarAgent {
    let device = seeded_device::<Be>(0xCA5);
    let config = Td3TrainingConfigBuilder::default()
        .learning_starts(learning_starts)
        .batch_size(4)
        .replay_buffer_capacity(32)
        .policy_frequency(1)
        .build()
        .expect("valid TD3 config");
    CarAgent::new(
        Actor::<Be>::new(&device),
        Critic::<Be>::new(&device),
        Critic::<Be>::new(&device),
        config,
        device,
    )
    .expect("agent construction")
}

/// A small `CarRacing` instance: a short episode cap keeps the rasterizer cost
/// bounded without changing the action space under test.
fn build_env() -> CarRacing {
    let config = CarRacingConfig {
        max_steps: 64,
        ..CarRacingConfig::default()
    };
    CarRacing::with_config(config).unwrap_or_else(|_| CarRacing::new(false))
}

/// Asserts an action sits inside its own per-component bound.
fn assert_per_component_valid(action: &CarRacingAction) {
    let (low, high) = (CarRacingAction::low(), CarRacingAction::high());
    let values = action.as_slice();
    assert_eq!(
        values.len(),
        CarRacingAction::COMPONENTS,
        "action must carry COMPONENTS values"
    );
    for (i, v) in values.iter().enumerate() {
        assert!(
            *v >= low[i] && *v <= high[i],
            "component {i} = {v} outside its own bound [{}, {}]",
            low[i],
            high[i]
        );
    }
    assert!(
        action.is_valid(),
        "the environment's own validity gate must accept the agent's action: {action:?}"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The shipped impl matches Gymnasium's `Box([-1,0,0], [1,1,1])` and satisfies
/// the ADR 0053 §4 contract.
#[test]
fn car_racing_bounds_match_the_gymnasium_box() {
    assert_eq!(CarRacingAction::low(), &[-1.0, 0.0, 0.0]);
    assert_eq!(CarRacingAction::high(), &[1.0, 1.0, 1.0]);
    assert_eq!(CarRacingAction::low().len(), CarRacingAction::COMPONENTS);
    assert_eq!(CarRacingAction::high().len(), CarRacingAction::COMPONENTS);
    assert_eq!(CarRacingAction::RANK, 1);
    assert_ne!(
        CarRacingAction::RANK,
        CarRacingAction::COMPONENTS,
        "the witness is only a witness while rank and component count disagree"
    );
    assert!(
        CarRacingAction::low()[1] > CarRacingAction::low()[0],
        "gas must not share steering's lower bound, or this env witnesses nothing"
    );
}

/// TD3 drives real `CarRacing` through both the warm-up and the policy branch,
/// and the environment accepts every action it is handed.
///
/// `CarRacing::step` returns `Err(InvalidAction)` for a negative gas or brake,
/// so this is not a self-referential check against the agent's own bounds: the
/// environment independently rejects an impossible action, and a run that
/// completes is evidence that none was produced.
#[test]
fn td3_steps_car_racing_without_producing_an_impossible_action() {
    let _guard = flex_guard();
    let mut agent = build_agent(4);
    let mut env = build_env();
    let mut rng = StdRng::seed_from_u64(0x0D15_EA5E);

    let mut obs = env.reset().expect("reset").observation().clone();

    // 4 warm-up steps (uniform sample per component) then 4 policy steps
    // (tanh output clipped per component) — both action paths, one run.
    for step in 0..8 {
        let action = agent.act(&obs, true, &mut rng);
        assert_per_component_valid(&action);

        let snapshot = env.step(action.clone()).unwrap_or_else(|e| {
            panic!("CarRacing rejected the agent's action at step {step}: {e}")
        });
        let next_obs = snapshot.observation().clone();

        agent.remember(
            obs,
            &action,
            snapshot.reward().0,
            next_obs.clone(),
            snapshot.is_terminated(),
        );
        agent.on_env_step();

        if snapshot.is_done() {
            obs = env.reset().expect("reset").observation().clone();
        } else {
            obs = next_obs;
        }
    }

    assert_eq!(agent.buffer_len(), 8, "every transition must be stored");
}

/// A TD3 learn step runs against the asymmetric space.
///
/// This is the path ADR 0053 §6 fixes: the target actor's `tanh` output spans
/// `[-1, 1]` on all three components, so the Eq. 14 clip has real work to do on
/// gas and brake. The pre-fix scalar collapse did not panic here — it silently
/// fed the twin critics a target computed at negative gas and negative brake —
/// which is why the value-level assertion lives in the unit tests on
/// `clip_to_action_bounds` and `smoothed_target_action`, and this test's job is
/// to prove the fixed path is actually reachable from the shipped environment.
#[test]
fn td3_learn_step_runs_against_the_asymmetric_action_space() {
    let _guard = flex_guard();
    let mut agent = build_agent(0);
    let mut env = build_env();
    let mut rng = StdRng::seed_from_u64(0xBEEF);

    let mut obs = env.reset().expect("reset").observation().clone();
    for _ in 0..8 {
        let action = agent.act(&obs, true, &mut rng);
        assert_per_component_valid(&action);
        let snapshot = env.step(action.clone()).expect("valid action accepted");
        let next_obs = snapshot.observation().clone();
        agent.remember(
            obs,
            &action,
            snapshot.reward().0,
            next_obs.clone(),
            snapshot.is_terminated(),
        );
        agent.on_env_step();
        obs = next_obs;
    }

    let outcome = agent
        .learn_step(&mut rng)
        .expect("buffer holds >= batch_size transitions, so a learn step must run");
    assert!(
        outcome.critic_loss.is_finite(),
        "critic loss must be finite, got {}",
        outcome.critic_loss
    );
    assert!(
        outcome.q_mean.is_finite(),
        "mean Q must be finite, got {}",
        outcome.q_mean
    );
}
