//! Seam-level tests for the stateful `RolloutFitness` rollout (issues #91/#92).
//!
//! These exercise the `StatefulPolicy` threading and the `ModuleEvalFn`
//! delegation through the **public** `RolloutFitness` API:
//!
//! - hidden state is threaded across the step loop, and a memory policy beats a
//!   reactive one on a fixture where reactivity provably caps the score;
//! - `reset` re-zeros the hidden state per episode (no cross-episode leakage);
//! - `evaluate_batch` delegation preserves shape and `episodes_per_eval`
//!   averaging;
//! - a recurrent GRU policy (`Hidden = Tensor<B, 2>`) is hosted end-to-end on
//!   the production seam against the real `SantaFeAnt` POMDP, and its recurrence
//!   is load-bearing (the #69 `recurrence_carries_state` assertion, on the seam).
//!
//! `AlternatingEnv` is a tiny deterministic fixture; it reuses the canonical
//! `SantaFeAntObservation` / `SantaFeAntAction` types so no bespoke serde-bound
//! observation is introduced.

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

use rlevo_core::action::DiscreteAction;
use rlevo_core::base::State;
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_environments::classic::{SantaFeAnt, SantaFeAntAction, SantaFeAntObservation};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::param_reshaper::ModuleReshaper;
use rlevo_hybrid::{ReactivePolicy, RolloutFitness, StatefulPolicy};

type TestBackend = Flex;
type Dev = <TestBackend as burn::tensor::backend::BackendTypes>::Device;

/// GRU hidden width for the recurrent-policy tests (small for speed).
const H: usize = 4;

fn device() -> Dev {
    Dev::default()
}

// ---------------------------------------------------------------------------
// AlternatingEnv — a tiny deterministic fixture.
//
// Reward is +1 iff the action index parity matches the step parity. A reactive
// (constant-action) policy can therefore match at most every other step, while
// a memory policy that alternates its action can match them all. Observation is
// constant, so reactivity genuinely caps the achievable score.
// ---------------------------------------------------------------------------

/// Trivial state — the env tracks step count itself; this exists only to
/// satisfy [`Environment::StateType`].
#[derive(Debug, Clone)]
struct AltState;

impl State<1> for AltState {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
struct AlternatingEnv {
    steps: usize,
    max_steps: usize,
}

impl AlternatingEnv {
    fn new(max_steps: usize) -> Self {
        Self {
            steps: 0,
            max_steps,
        }
    }
}

impl Environment<1, 1, 1> for AlternatingEnv {
    type StateType = AltState;
    type ObservationType = SantaFeAntObservation;
    type ActionType = SantaFeAntAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, SantaFeAntObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps = 0;
        Ok(SnapshotBase::running(
            SantaFeAntObservation { food_ahead: false },
            ScalarReward::new(0.0),
        ))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let matches = action.to_index() % 2 == self.steps % 2;
        let reward = ScalarReward::new(if matches { 1.0 } else { 0.0 });
        self.steps += 1;
        let obs = SantaFeAntObservation { food_ahead: false };
        let snapshot = if self.steps >= self.max_steps {
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snapshot)
    }
}

// ---------------------------------------------------------------------------
// Policies for AlternatingEnv.
// ---------------------------------------------------------------------------

/// Memory policy: emits action `Move`/`TurnLeft` by the parity of a step
/// counter threaded as `Hidden`, ignoring the (constant) observation.
#[derive(Module, Debug)]
struct AlternatingPolicy<B: Backend> {
    // Unused weights — present so the module has flattenable parameters.
    _w: Linear<B>,
}

impl<B: Backend> AlternatingPolicy<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            _w: LinearConfig::new(1, 1).init(device),
        }
    }
}

impl StatefulPolicy<TestBackend, AlternatingEnv> for AlternatingPolicy<TestBackend> {
    type Hidden = usize;

    fn reset(&self, _device: &Dev) -> usize {
        0
    }

    fn act(
        &self,
        hidden: &mut usize,
        _obs: &SantaFeAntObservation,
        _device: &Dev,
    ) -> SantaFeAntAction {
        let action = SantaFeAntAction::from_index(*hidden % 2);
        *hidden += 1;
        action
    }
}

/// Reactive baseline: always `Move` (index 0), regardless of observation.
#[derive(Module, Debug)]
struct ConstantPolicy<B: Backend> {
    _w: Linear<B>,
}

impl<B: Backend> ConstantPolicy<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            _w: LinearConfig::new(1, 1).init(device),
        }
    }
}

impl ReactivePolicy<TestBackend, AlternatingEnv> for ConstantPolicy<TestBackend> {
    fn act(&self, _obs: &SantaFeAntObservation, _device: &Dev) -> SantaFeAntAction {
        SantaFeAntAction::Move
    }
}

/// Score a single genome (all-zero weights — these policies ignore their
/// weights) through the refactored `RolloutFitness` and return its fitness.
fn score_one<M>(template: M, env_max: usize, episodes: usize) -> f32
where
    M: Module<TestBackend> + Sync + StatefulPolicy<TestBackend, AlternatingEnv> + 'static,
{
    let dev = device();
    let reshaper = ModuleReshaper::new(template);
    let num_params = reshaper.num_params();
    let mut fitness = RolloutFitness::new(
        reshaper,
        move || AlternatingEnv::new(env_max),
        episodes,
        env_max,
        dev,
    );
    let pop = Tensor::<TestBackend, 2>::zeros([1, num_params], &dev);
    let out = fitness.evaluate_batch(&pop, &dev);
    out.into_data().into_vec::<f32>().unwrap()[0]
}

#[test]
fn hidden_state_is_threaded_and_memory_beats_reactive() {
    let dev = device();
    // env_max = 6: the alternating memory policy matches every step (6); the
    // reactive constant policy matches only the even steps (3).
    let alt = score_one(AlternatingPolicy::<TestBackend>::new(&dev), 6, 1);
    let con = score_one(ConstantPolicy::<TestBackend>::new(&dev), 6, 1);
    approx::assert_relative_eq!(alt, 6.0, epsilon = 1e-6);
    approx::assert_relative_eq!(con, 3.0, epsilon = 1e-6);
    assert!(
        alt > con,
        "threaded memory policy ({alt}) must beat the reactive baseline ({con})"
    );
}

#[test]
fn reset_rezeros_hidden_per_episode() {
    // Odd episode length (5) makes cross-episode hidden leakage detectable: a
    // leaked counter would enter episode 2 at the wrong parity and lose reward.
    // A correct per-episode reset scores the full 5 in each of the 2 episodes.
    let dev = device();
    let mean = score_one(AlternatingPolicy::<TestBackend>::new(&dev), 5, 2);
    approx::assert_relative_eq!(mean, 5.0, epsilon = 1e-6);
}

#[test]
fn evaluate_batch_preserves_shape_and_averaging() {
    // Delegation to the inner ModuleEvalFn (#92): 3 rows in → 3 fitnesses out,
    // each the mean over episodes_per_eval. env_max = 4 → 4 per episode; mean
    // over 2 episodes = 4.
    let dev = device();
    let reshaper = ModuleReshaper::new(AlternatingPolicy::<TestBackend>::new(&dev));
    let num_params = reshaper.num_params();
    let mut fitness = RolloutFitness::new(reshaper, || AlternatingEnv::new(4), 2, 4, dev);
    let pop = Tensor::<TestBackend, 2>::zeros([3, num_params], &dev);
    let out = fitness.evaluate_batch(&pop, &dev);
    let values = out.into_data().into_vec::<f32>().unwrap();
    assert_eq!(values.len(), 3);
    for v in values {
        approx::assert_relative_eq!(v, 4.0, epsilon = 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Recurrent GRU policy on the real SantaFeAnt POMDP — acceptance proof that the
// seam hosts a memory policy (issue #91), the #69 AntPolicy shape lifted onto
// the production StatefulPolicy contract.
// ---------------------------------------------------------------------------

fn argmax3(logits: &[f32; 3]) -> usize {
    let mut best = 0;
    for i in 1..3 {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best
}

/// `Gru(1 -> H) -> Linear(H -> 3)`; the hidden state is a runtime
/// `Tensor<B, 2>` threaded by the rollout, never stored in the module.
#[derive(Module, Debug)]
struct GruPolicy<B: Backend> {
    gru: Gru<B>,
    head: Linear<B>,
}

impl<B: Backend> GruPolicy<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            gru: GruConfig::new(1, H, true).init(device),
            head: LinearConfig::new(H, 3).with_bias(true).init(device),
        }
    }

    /// Forward one step, mutating the hidden state, and return the 3 action
    /// logits. Shared by `act` and the recurrence test.
    fn forward_logits(
        &self,
        food_ahead: bool,
        h: &mut Tensor<B, 2>,
        device: &B::Device,
    ) -> [f32; 3] {
        let v: f32 = if food_ahead { 1.0 } else { 0.0 };
        let x = Tensor::<B, 3>::from_floats([[[v]]], device);
        let out = self.gru.forward(x, Some(h.clone())); // [1, 1, H]
        let new_h = out.reshape([1, H]); // [1, H]
        let logits = self.head.forward(new_h.clone()); // [1, 3]
        *h = new_h;
        let data = logits.into_data().into_vec::<f32>().unwrap();
        [data[0], data[1], data[2]]
    }
}

impl StatefulPolicy<TestBackend, SantaFeAnt> for GruPolicy<TestBackend> {
    type Hidden = Tensor<TestBackend, 2>;

    fn reset(&self, device: &Dev) -> Tensor<TestBackend, 2> {
        Tensor::zeros([1, H], device)
    }

    fn act(
        &self,
        hidden: &mut Tensor<TestBackend, 2>,
        obs: &SantaFeAntObservation,
        device: &Dev,
    ) -> SantaFeAntAction {
        let logits = self.forward_logits(obs.food_ahead, hidden, device);
        SantaFeAntAction::from_index(argmax3(&logits))
    }
}

#[test]
fn gru_recurrence_is_load_bearing_on_the_seam() {
    // The #69 `recurrence_carries_state` assertion, on the production seam:
    // the same percept with different prior hidden states yields different
    // logits — impossible for a reactive map.
    let dev = device();
    let policy = GruPolicy::<TestBackend>::new(&dev);
    let mut h_zero = StatefulPolicy::<TestBackend, SantaFeAnt>::reset(&policy, &dev);
    let mut h_warm = Tensor::<TestBackend, 2>::ones([1, H], &dev) * 0.5;
    let from_zero = policy.forward_logits(false, &mut h_zero, &dev);
    let from_warm = policy.forward_logits(false, &mut h_warm, &dev);
    let differs = (0..3).any(|i| (from_zero[i] - from_warm[i]).abs() > 1e-6);
    assert!(differs, "GRU hidden state did not influence the output");
}

#[test]
fn rollout_fitness_hosts_recurrent_gru_on_santa_fe() {
    // End-to-end: RolloutFitness<_, GruPolicy, SantaFeAnt> reconstructs a
    // recurrent module per row and drives it across the POMDP rollout, threading
    // the Tensor<B, 2> hidden state. Just a correctness/hosting smoke test (a
    // zero genome is not evolved), so assert a finite, in-range pellet count.
    let dev = device();
    let reshaper = ModuleReshaper::new(GruPolicy::<TestBackend>::new(&dev));
    let num_params = reshaper.num_params();
    let mut fitness = RolloutFitness::new(
        reshaper,
        || <SantaFeAnt as ConstructableEnv>::new(false),
        1,
        50,
        dev,
    );
    let pop = Tensor::<TestBackend, 2>::zeros([2, num_params], &dev);
    let out = fitness.evaluate_batch(&pop, &dev);
    let values = out.into_data().into_vec::<f32>().unwrap();
    assert_eq!(values.len(), 2);
    for pellets in values {
        assert!(
            (0.0..=89.0).contains(&pellets),
            "pellet count {pellets} out of range [0, 89]"
        );
    }
}
