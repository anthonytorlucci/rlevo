//! Test-only scaffolding for the truncation-vs-termination bootstrap-mask
//! guard shared by the six off-policy `train` loops.
//!
//! Every off-policy training loop must feed the *Bellman bootstrap mask* from
//! [`EpisodeStatus::is_terminated`], not [`EpisodeStatus::is_done`]. A truncation (time
//! limit) ends the episode but not the MDP, so `$\gamma \cdot V(\text{next\_obs})$` must survive
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
//! [`EpisodeStatus::is_terminated`]: rlevo_core::environment::EpisodeStatus::is_terminated
//! [`EpisodeStatus::is_done`]: rlevo_core::environment::EpisodeStatus::is_done

use std::marker::PhantomData;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, activation};
use serde::{Deserialize, Serialize};

use rlevo_core::action::{BoundedAction, ContinuousAction, DiscreteAction};
use rlevo_core::base::{Action, HostRow, Observation};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};
use rlevo_core::reward::ScalarReward;

use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use crate::algorithms::sac::sac_model::{SampleOutput, SquashedGaussianPolicy};

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

/// Two-feature view of the step counter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct MaskObservation {
    values: [f32; OBS_DIM],
}

impl MaskObservation {
    /// Builds an observation from its two feature values.
    pub(crate) fn new(values: [f32; OBS_DIM]) -> Self {
        Self { values }
    }
}

impl Observation<1> for MaskObservation {
    fn shape() -> [usize; 1] {
        [OBS_DIM]
    }
}

impl HostRow<1> for MaskObservation {
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.values);
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
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1, "continuous action width");
        Self(values[0])
    }
}

impl BoundedAction<1> for MaskContinuousAction {
    fn low() -> &'static [f32] {
        &[-1.0]
    }
    fn high() -> &'static [f32] {
        &[1.0]
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
    /// Total [`Environment::step`] calls since construction. Deliberately
    /// **not** reset by `reset()` — `state.steps` already restarts each
    /// episode, and a NaN-injection index must address a step of the *run*, not
    /// of the current episode.
    calls: usize,
    /// Value of `calls` (1-based) whose `step` returns a `NaN` reward instead
    /// of `1.0`. `None` — the default — keeps every reward finite. See
    /// [`with_nan_reward_at`](Self::with_nan_reward_at).
    nan_reward_at: Option<usize>,
    _action: PhantomData<A>,
}

impl<A> MaskEnv<A> {
    pub(crate) fn new(period: usize, end_status: EpisodeStatus) -> Self {
        assert!(period > 0, "period must be positive");
        Self {
            state: MaskState { steps: 0 },
            period,
            end_status,
            calls: 0,
            nan_reward_at: None,
            _action: PhantomData,
        }
    }

    /// Makes the `step`th call to [`Environment::step`] return a `NaN` reward
    /// (1-based, counted from construction; the counter does not reset with the
    /// episode).
    ///
    /// Models the misbehaving environment ADR 0065 exists for — a division by
    /// zero or an exploding dynamics term that emits one non-finite reward in
    /// an otherwise healthy run. Off by default so every other fixture user
    /// keeps its all-finite reward stream.
    pub(crate) const fn with_nan_reward_at(mut self, step: usize) -> Self {
        self.nan_reward_at = Some(step);
        self
    }

    fn observation(state: MaskState) -> MaskObservation {
        #[allow(clippy::cast_precision_loss)]
        let t = state.steps as f32;
        MaskObservation {
            values: [t * 0.1, 1.0 - t * 0.1],
        }
    }
}

impl<A: Action<1> + Clone> Environment<1, 1> for MaskEnv<A> {
    type ObservationType = MaskObservation;
    type ActionType = A;

    fn reset(&mut self) -> Result<Snapshot<MaskObservation>, EnvironmentError> {
        self.state = MaskState { steps: 0 };
        Ok(Snapshot {
            observation: Self::observation(self.state),
            reward: ScalarReward::new(0.0),
            status: EpisodeStatus::Running,
            metadata: None,
        })
    }

    fn step(
        &mut self,
        _action: Self::ActionType,
    ) -> Result<Snapshot<MaskObservation>, EnvironmentError> {
        self.state.steps += 1;
        self.calls += 1;
        let status = if self.state.steps.is_multiple_of(self.period) {
            self.end_status
        } else {
            EpisodeStatus::Running
        };
        let reward = if self.nan_reward_at == Some(self.calls) {
            f32::NAN
        } else {
            1.0
        };
        Ok(Snapshot {
            observation: Self::observation(self.state),
            reward: ScalarReward::new(reward),
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
pub(crate) struct FlatNet {
    layer: Linear,
}

impl FlatNet {
    pub(crate) fn new(out: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM, out).init(device),
        }
    }
    fn run(&self, obs: Tensor<2>) -> Tensor<2> {
        self.layer.forward(obs)
    }
}

impl crate::algorithms::dqn::dqn_model::DqnModel<2> for FlatNet {
    fn forward(&self, observations: Tensor<2>) -> Tensor<2> {
        self.run(observations)
    }
}

/// `(batch, OBS_DIM)` → `(batch, ACTIONS, atoms)`. Backs C51 and QR-DQN, whose
/// model traits are shape-identical.
#[derive(Module, Debug)]
pub(crate) struct AtomNet {
    layer: Linear,
    atoms: usize,
}

impl AtomNet {
    pub(crate) fn new(atoms: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM, ACTIONS * atoms).init(device),
            atoms,
        }
    }
    fn run(&self, obs: Tensor<2>) -> Tensor<3> {
        let batch = obs.dims()[0];
        self.layer
            .forward(obs)
            .reshape([batch, ACTIONS, self.atoms])
    }
}

impl crate::algorithms::c51::c51_model::C51Model<2> for AtomNet {
    fn forward(&self, observations: Tensor<2>) -> Tensor<3> {
        self.run(observations)
    }
}

impl crate::algorithms::qrdqn::qrdqn_model::QrDqnModel<2> for AtomNet {
    fn forward(&self, observations: Tensor<2>) -> Tensor<3> {
        self.run(observations)
    }
}

/// Deterministic actor for DDPG / TD3: `(batch, OBS_DIM)` → `(batch, OBS_DIM)`.
#[derive(Module, Debug)]
pub(crate) struct TinyActor {
    layer: Linear,
}

impl TinyActor {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM, ACT_DIM).init(device),
        }
    }
    fn run(&self, obs: Tensor<2>) -> Tensor<2> {
        activation::tanh(self.layer.forward(obs))
    }
}

impl DeterministicPolicy<2, 2> for TinyActor {
    fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
        self.run(obs)
    }
}

/// Continuous Q-critic for DDPG / TD3 / SAC.
#[derive(Module, Debug)]
pub(crate) struct TinyCritic {
    layer: Linear,
}

impl TinyCritic {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(OBS_DIM + ACT_DIM, 1).init(device),
        }
    }
    fn run(&self, obs: Tensor<2>, act: Tensor<2>) -> Tensor<1> {
        let joint = Tensor::cat(vec![obs, act], 1);
        self.layer.forward(joint).squeeze_dim::<1>(1)
    }
}

impl ContinuousQ<2, 2> for TinyCritic {
    fn forward(&self, obs: Tensor<2>, act: Tensor<2>) -> Tensor<1> {
        self.run(obs, act)
    }
}

/// Squashed-Gaussian actor for SAC. Fixed unit log-`$\sigma$` keeps it to one layer.
#[derive(Module, Debug)]
pub(crate) struct TinySacActor {
    mean: Linear,
}

impl TinySacActor {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            mean: LinearConfig::new(OBS_DIM, ACT_DIM).init(device),
        }
    }
    fn sample(&self, obs: Tensor<2>, eps: Tensor<2>) -> SampleOutput<2> {
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

impl SquashedGaussianPolicy<2, 2> for TinySacActor {
    fn action_dim(&self) -> usize {
        ACT_DIM
    }
    fn forward_sample(&self, obs: Tensor<2>, eps: Tensor<2>) -> SampleOutput<2> {
        self.sample(obs, eps)
    }
    fn deterministic_action(&self, obs: Tensor<2>) -> Tensor<2> {
        activation::tanh(self.mean.forward(obs))
    }
}
