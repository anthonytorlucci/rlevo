//! Test-only scaffolding for the truncation-vs-termination bootstrap-mask
//! guard shared by the six off-policy `train` loops.
//!
//! Every off-policy training loop must feed the *Bellman bootstrap mask* from
//! [`Snapshot::is_terminated`], not [`Snapshot::is_done`]. A truncation (time
//! limit) ends the episode but not the MDP, so `γ · V(next_obs)` must survive
//! in the target; zeroing it there biases every Q-value downward (Pardo et
//! al., "Time Limits in Reinforcement Learning", ICML 2018, Eq. 6).
//!
//! This module supplies the minimum needed to assert that end-to-end without a
//! physics simulator: a deterministic environment that ends its episodes with a
//! *caller-chosen* [`EpisodeStatus`], and a one-layer network per algorithm
//! model trait. Tests drive `train` for a handful of steps with
//! `learning_starts` set past the budget, so no gradient work runs and the
//! whole guard costs milliseconds.
//!
//! [`Snapshot::is_terminated`]: rlevo_core::environment::Snapshot::is_terminated
//! [`Snapshot::is_done`]: rlevo_core::environment::Snapshot::is_done

use std::marker::PhantomData;

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, activation};
use serde::{Deserialize, Serialize};

use rlevo_core::action::{BoundedAction, ContinuousAction, DiscreteAction};
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Sensor, SnapshotBase};
use rlevo_core::reward::ScalarReward;

use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use crate::algorithms::sac::sac_model::{SampleOutput, SquashedGaussianPolicy};
use crate::utils::polyak_update;

/// Observation width and (for the continuous envs) action width.
pub(crate) const OBS_DIM: usize = 2;
/// Number of discrete actions offered by [`MaskDiscreteAction`].
pub(crate) const ACTIONS: usize = 2;
/// Width of the continuous action vector.
pub(crate) const ACT_DIM: usize = 1;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Step counter; the whole MDP is a clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaskState {
    steps: usize,
}

impl State<1> for MaskState {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn numel(&self) -> usize {
        1
    }
    fn is_valid(&self) -> bool {
        true
    }
}

/// Two-feature view of the step counter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct MaskObservation {
    values: [f32; OBS_DIM],
}

impl Observation<1> for MaskObservation {
    fn shape() -> [usize; 1] {
        [OBS_DIM]
    }
}

impl<B: Backend> TensorConvertible<1, B> for MaskObservation {
    fn row_shape() -> [usize; 1] {
        [OBS_DIM]
    }
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.values);
    }
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("{e:?}"),
        })?;
        Ok(Self {
            values: [slice[0], slice[1]],
        })
    }
}

/// Binary discrete action; the env dynamics ignore it entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaskDiscreteAction(usize);

impl Action<1> for MaskDiscreteAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        self.0 < ACTIONS
    }
}

impl DiscreteAction<1> for MaskDiscreteAction {
    const ACTION_COUNT: usize = ACTIONS;

    fn to_index(&self) -> usize {
        self.0
    }
    fn from_index(index: usize) -> Self {
        assert!(index < ACTIONS, "action index out of range");
        Self(index)
    }
}

/// Single continuous action component in `[-1, 1]`; also ignored by dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MaskContinuousAction(f32);

impl Action<1> for MaskContinuousAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        self.0.is_finite()
    }
}

impl ContinuousAction<1> for MaskContinuousAction {
    const COMPONENTS: usize = 1;

    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }
    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.clamp(min, max))
    }
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1, "continuous action width");
        Self(values[0])
    }
}

impl BoundedAction<1> for MaskContinuousAction {
    fn low() -> [f32; 1] {
        [-1.0]
    }
    fn high() -> [f32; 1] {
        [1.0]
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Deterministic clock environment whose episodes end every `period` steps
/// with a *caller-chosen* [`EpisodeStatus`].
///
/// Setting `end_status` to [`EpisodeStatus::Truncated`] models a time-limited
/// task with no terminal condition (Pendulum-like); setting it to
/// [`EpisodeStatus::Terminated`] models a genuine absorbing state. Both end the
/// episode — `is_done()` is `true` either way — so the two configurations
/// differ *only* in the bootstrap mask the training loop should record. That is
/// exactly the distinction under test.
#[derive(Debug)]
pub(crate) struct MaskEnv<A> {
    state: MaskState,
    period: usize,
    end_status: EpisodeStatus,
    _action: PhantomData<A>,
}

impl<A> MaskEnv<A> {
    pub(crate) fn new(period: usize, end_status: EpisodeStatus) -> Self {
        assert!(period > 0, "period must be positive");
        Self {
            state: MaskState { steps: 0 },
            period,
            end_status,
            _action: PhantomData,
        }
    }

    fn observation(state: &MaskState) -> MaskObservation {
        #[allow(clippy::cast_precision_loss)]
        let t = state.steps as f32;
        MaskObservation {
            values: [t * 0.1, 1.0 - t * 0.1],
        }
    }
}

impl<A: Action<1> + Clone> Sensor<1, 1, 1> for MaskEnv<A> {
    type Action = A;
    type State = MaskState;
    type Observation = MaskObservation;

    fn observe(&self, _action: &A, next_state: &MaskState) -> MaskObservation {
        Self::observation(next_state)
    }
    fn observe_reset(&self, state: &MaskState) -> MaskObservation {
        Self::observation(state)
    }
}

impl<A: Action<1> + Clone> Environment<1, 1, 1> for MaskEnv<A> {
    type StateType = MaskState;
    type ObservationType = MaskObservation;
    type ActionType = A;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MaskObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = MaskState { steps: 0 };
        Ok(SnapshotBase {
            observation: Self::observation(&self.state),
            reward: ScalarReward::new(0.0),
            status: EpisodeStatus::Running,
            metadata: None,
        })
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state.steps += 1;
        let status = if self.state.steps.is_multiple_of(self.period) {
            self.end_status
        } else {
            EpisodeStatus::Running
        };
        Ok(SnapshotBase {
            observation: self.observe(&action, &self.state),
            reward: ScalarReward::new(1.0),
            status,
            metadata: None,
        })
    }
}

/// Discrete-action clock env, for DQN / C51 / QR-DQN.
pub(crate) type DiscreteMaskEnv = MaskEnv<MaskDiscreteAction>;
/// Continuous-action clock env, for DDPG / TD3 / SAC.
pub(crate) type ContinuousMaskEnv = MaskEnv<MaskContinuousAction>;

// ---------------------------------------------------------------------------
// Networks — one `Linear` each; these exist only to satisfy trait bounds.
// ---------------------------------------------------------------------------

/// `(batch, OBS_DIM)` → `(batch, ACTIONS)`. Backs the DQN model trait.
#[derive(Module, Debug)]
pub(crate) struct FlatNet<B: Backend> {
    layer: Linear<B>,
}

impl<B: Backend> FlatNet<B> {
    pub(crate) fn new(out: usize, device: &B::Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM, out).init(device),
        }
    }
    fn run(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layer.forward(obs)
    }
}

impl<B: AutodiffBackend> crate::algorithms::dqn::dqn_model::DqnModel<B, 2> for FlatNet<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        self.run(observations)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.run(observations)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, FlatNet<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

/// `(batch, OBS_DIM)` → `(batch, ACTIONS, atoms)`. Backs C51 and QR-DQN, whose
/// model traits are shape-identical.
#[derive(Module, Debug)]
pub(crate) struct AtomNet<B: Backend> {
    layer: Linear<B>,
    atoms: usize,
}

impl<B: Backend> AtomNet<B> {
    pub(crate) fn new(atoms: usize, device: &B::Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM, ACTIONS * atoms).init(device),
            atoms,
        }
    }
    fn run(&self, obs: Tensor<B, 2>) -> Tensor<B, 3> {
        let batch = obs.dims()[0];
        self.layer
            .forward(obs)
            .reshape([batch, ACTIONS, self.atoms])
    }
}

impl<B: AutodiffBackend> crate::algorithms::c51::c51_model::C51Model<B, 2> for AtomNet<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        self.run(observations)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.run(observations)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, AtomNet<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

impl<B: AutodiffBackend> crate::algorithms::qrdqn::qrdqn_model::QrDqnModel<B, 2> for AtomNet<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
        self.run(observations)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 3> {
        inner.run(observations)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, AtomNet<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

/// Deterministic actor for DDPG / TD3: `(batch, OBS_DIM)` → `(batch, OBS_DIM)`.
#[derive(Module, Debug)]
pub(crate) struct TinyActor<B: Backend> {
    layer: Linear<B>,
}

impl<B: Backend> TinyActor<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM, ACT_DIM).init(device),
        }
    }
    fn run(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::tanh(self.layer.forward(obs))
    }
}

impl<B: AutodiffBackend> DeterministicPolicy<B, 2, 2> for TinyActor<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.run(obs)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.run(obs)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, TinyActor<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

/// Continuous Q-critic for DDPG / TD3 / SAC.
#[derive(Module, Debug)]
pub(crate) struct TinyCritic<B: Backend> {
    layer: Linear<B>,
}

impl<B: Backend> TinyCritic<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM + ACT_DIM, 1).init(device),
        }
    }
    fn run(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        let joint = Tensor::cat(vec![obs, act], 1);
        self.layer.forward(joint).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> ContinuousQ<B, 2, 2> for TinyCritic<B> {
    fn forward(&self, obs: Tensor<B, 2>, act: Tensor<B, 2>) -> Tensor<B, 1> {
        self.run(obs, act)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        act: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 1> {
        inner.run(obs, act)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
        polyak_update::<B::InnerBackend, TinyCritic<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

/// Squashed-Gaussian actor for SAC. Fixed unit log-σ keeps it to one layer.
#[derive(Module, Debug)]
pub(crate) struct TinySacActor<B: Backend> {
    mean: Linear<B>,
}

impl<B: Backend> TinySacActor<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            mean: LinearConfig::new(OBS_DIM, ACT_DIM).init(device),
        }
    }
    fn sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        let mu = self.mean.forward(obs);
        let pre_tanh = mu + eps;
        let action = activation::tanh(pre_tanh);
        // Constant log-density: the fixture never asserts on SAC's entropy
        // term, only on which bootstrap mask reached the replay buffer.
        let log_prob = action
            .clone()
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .mul_scalar(0.0);
        SampleOutput { action, log_prob }
    }
}

impl<B: AutodiffBackend> SquashedGaussianPolicy<B, 2, 2> for TinySacActor<B> {
    fn action_dim(&self) -> usize {
        ACT_DIM
    }
    fn forward_sample(&self, obs: Tensor<B, 2>, eps: Tensor<B, 2>) -> SampleOutput<B, 2> {
        self.sample(obs, eps)
    }
    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
        eps: Tensor<B::InnerBackend, 2>,
    ) -> SampleOutput<B::InnerBackend, 2> {
        inner.sample(obs, eps)
    }
    fn deterministic_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::tanh(self.mean.forward(obs))
    }
}
