//! Quantile Regression DQN (QR-DQN) agent: actor, trainer, and replay buffer.
//!
//! The [`QrDqnAgent`] struct owns the policy network (returning raw quantile
//! values), a frozen target network, an Adam optimizer, a uniform replay
//! buffer, and the ε-greedy exploration schedule shared with DQN. Action
//! selection uses the mean-quantile Q-value `(1/N) · Σ_i θ_i(s, a)`; the
//! learning step computes target quantiles via a one-step Bellman backup on
//! the bootstrap action's quantile vector — no categorical projection — and
//! minimises the quantile Huber loss (see
//! [`crate::algorithms::qrdqn::quantile_loss`]).

use std::collections::VecDeque;
use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use crate::memory::ReplayBufferError;
use crate::metrics::{AgentStats, PerformanceRecord};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;

use crate::algorithms::dqn::exploration::EpsilonGreedy;
use crate::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfig;
use crate::algorithms::qrdqn::qrdqn_model::QrDqnModel;
use crate::algorithms::qrdqn::quantile_loss::quantile_huber_loss;
use crate::algorithms::shared::Slot;

/// Error variants returned by [`QrDqnAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum QrDqnAgentError {
    /// A tensor-to-action or action-to-tensor conversion failed.
    #[error("Tensor conversion failed: {0}")]
    TensorConversionFailed(String),
    /// The sampled or requested action is outside the valid action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// A replay buffer operation failed.
    #[error(transparent)]
    Buffer(#[from] ReplayBufferError),
    /// An I/O error occurred while saving or loading model weights.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Per-episode statistics emitted by the QR-DQN training loop.
#[derive(Debug, Clone, Copy)]
pub struct QrDqnMetrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Most recent quantile Huber loss.
    pub policy_loss: f32,
    /// Exploration rate at the end of the episode.
    pub epsilon: f32,
    /// Mean Q-value (= mean quantile) across the most recent learn step's
    /// predictions over all actions in the batch.
    pub q_mean: f32,
    /// Batch-mean standard deviation of the predicted quantile values at the
    /// taken action. A distributional analogue to C51's `entropy` diagnostic;
    /// low values flag distribution collapse (all quantiles equal).
    pub quantile_spread: f32,
}

impl PerformanceRecord for QrDqnMetrics {
    fn score(&self) -> f32 {
        self.reward
    }
    fn duration(&self) -> usize {
        self.steps
    }
}

#[derive(Debug, Clone)]
struct Transition<O: Clone> {
    obs: O,
    action_idx: usize,
    reward: f32,
    next_obs: O,
    done: bool,
}

/// Summary values returned by a single [`QrDqnAgent::learn_step`].
#[derive(Debug, Clone, Copy)]
pub struct LearnOutcome {
    /// Quantile Huber loss.
    pub loss: f32,
    /// Mean Q-value `E[Z]` (across actions and batch), for diagnostics.
    pub q_mean: f32,
    /// Batch-mean per-sample quantile spread (std of quantile values at
    /// the taken action).
    pub quantile_spread: f32,
}

/// Quantile Regression DQN agent.
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor (e.g. `1` for vector
///   observations of shape `[features]`).
/// - `DB` — rank of a batched observation tensor (= `DO + 1`).
///
/// # Field notes
///
/// - `policy_net` is held in a [`Slot`], the newtype that owns a network across
///   Burn's by-value [`Optimizer::step`](burn::optim::Optimizer::step). Every
///   read goes through `Slot::get`, and the module leaves the field only for the
///   duration of the `step` call itself inside `Slot::step_with` — the forward
///   pass, loss, and `backward` all run on a borrow, so a panic in any of them
///   leaves the agent intact. The one exception is a panic *inside* `step`,
///   which poisons the slot permanently; that window is irreducible and is
///   documented on [`Slot`].
/// - `target_net` lives on `B::InnerBackend` (the non-autodiff backend) so that
///   computing bootstrap quantiles never builds an autodiff graph.
pub struct QrDqnAgent<B, M, O, A, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    M: QrDqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    policy_net: Slot<M>,
    target_net: M::InnerModule,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    buffer: VecDeque<Transition<O>>,
    exploration: EpsilonGreedy,
    config: QrDqnTrainingConfig,
    device: B::Device,
    step: usize,
    stats: AgentStats<QrDqnMetrics>,
    _action: PhantomData<A>,
}

impl<B, M, O, A, const DO: usize, const DB: usize> std::fmt::Debug
    for QrDqnAgent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: QrDqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QrDqnAgent")
            .field("step", &self.step)
            .field("buffer_len", &self.buffer.len())
            .field("epsilon", &self.exploration.value())
            .field("config", &self.config)
            .finish()
    }
}

impl<B, M, O, A, const DO: usize, const DB: usize> QrDqnAgent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: QrDqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    /// Constructs a new agent from a pre-built policy network and config.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`QrDqnTrainingConfig::validate`](rlevo_core::config::Validate::validate).
    pub fn new(
        policy_net: M,
        config: QrDqnTrainingConfig,
        device: B::Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        let target_net = policy_net.valid();
        let adam = config.optimizer.clone();
        let optimizer = match &config.clip_grad {
            Some(clip) => adam.with_grad_clipping(Some(clip.clone())).init::<B, M>(),
            None => adam.init::<B, M>(),
        };
        // ε-greedy is reused verbatim from DQN.
        let exploration = EpsilonGreedy::new(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        );
        let stats = AgentStats::<QrDqnMetrics>::new(100);
        Ok(Self {
            policy_net: Slot::new(policy_net),
            target_net,
            optimizer,
            buffer: VecDeque::with_capacity(config.replay_buffer_capacity),
            exploration,
            config,
            device,
            step: 0,
            stats,
            _action: PhantomData,
        })
    }

    /// Current exploration rate (ε).
    pub fn epsilon(&self) -> f64 {
        self.exploration.value()
    }

    /// Current agent statistics.
    pub fn stats(&self) -> &AgentStats<QrDqnMetrics> {
        &self.stats
    }

    /// Records one completed episode into the running statistics.
    pub fn record_episode(&mut self, metrics: QrDqnMetrics) {
        self.stats.record(metrics);
    }

    /// Number of transitions currently stored.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Global step count.
    pub fn step(&self) -> usize {
        self.step
    }

    fn policy(&self) -> &M {
        self.policy_net.get()
    }

    /// ε-greedy action selection using the mean-quantile Q-value of the
    /// predicted return distribution.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> A {
        if self.exploration.should_explore(rng) {
            return A::from_index(rng.random_range(0..A::ACTION_COUNT));
        }
        self.act_greedy(obs)
    }

    /// Greedy (deterministic) action selection — the argmax over mean-quantile
    /// Q-values of the predicted return distribution.
    ///
    /// Unlike [`act`](Self::act) this never explores, so it is the policy to
    /// use for evaluation: it reflects what the network has learned without the
    /// ε-greedy exploration noise that floors at `epsilon_end`.
    pub fn act_greedy(&self, obs: &O) -> A {
        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
        let quantiles: Tensor<B, 3> = self.policy().forward(batched); // (1, A, N)
        let q: Tensor<B, 2> = quantiles.mean_dim(2).squeeze_dim::<2>(2); // (1, A)
        let idx = q.argmax(1).into_scalar();
        A::from_index(idx.elem::<i64>() as usize)
    }

    /// Snapshots the policy network onto the inner (non-autodiff) backend.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_greedy_with`](Self::act_greedy_with). Action selection never needs
    /// gradients, so running it on the inner backend avoids the per-call
    /// autodiff graph construction that [`act_greedy`](Self::act_greedy)
    /// incurs. Snapshot once after training, then reuse across many steps —
    /// the snapshot goes stale if the policy is updated again.
    pub fn inference_net(&self) -> M::InnerModule {
        self.policy().valid()
    }

    /// Greedy action selection against a pre-snapshotted inner network.
    ///
    /// Equivalent to [`act_greedy`](Self::act_greedy) but runs on the
    /// non-autodiff backend via [`inference_net`](Self::inference_net), which
    /// is dramatically cheaper for repeated single-observation inference.
    pub fn act_greedy_with(&self, net: &M::InnerModule, obs: &O) -> A {
        let obs_t: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, DB> = obs_t.unsqueeze::<DB>();
        let quantiles: Tensor<B::InnerBackend, 3> = M::forward_inner(net, batched); // (1, A, N)
        let q: Tensor<B::InnerBackend, 2> = quantiles.mean_dim(2).squeeze_dim::<2>(2); // (1, A)
        let idx = q.argmax(1).into_scalar();
        A::from_index(idx.elem::<i64>() as usize)
    }

    /// Appends a transition to the replay buffer, evicting the oldest entry
    /// when the buffer is at capacity.
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, done: bool) {
        if self.buffer.len() >= self.config.replay_buffer_capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(Transition {
            obs,
            action_idx: action.to_index(),
            reward,
            next_obs,
            done,
        });
    }

    /// Decays ε by one step.
    pub fn decay_exploration(&mut self) {
        self.exploration.decay();
    }

    /// Advances the global step counter. Called once per env step.
    pub fn on_env_step(&mut self) {
        self.step += 1;
    }

    /// Returns `true` when the agent has enough transitions to run a learn step.
    pub fn can_learn(&self) -> bool {
        self.buffer.len() >= self.config.batch_size && self.step >= self.config.learning_starts
    }

    /// Returns `true` when the agent's internal clock matches
    /// `config.train_frequency`.
    pub fn should_train(&self) -> bool {
        self.config.train_frequency > 0 && self.step.is_multiple_of(self.config.train_frequency)
    }

    /// Periodically **hard**-syncs the target network with the policy network.
    ///
    /// This method only ever performs the hard path — a full copy of the policy
    /// weights into the target. The Polyak soft update
    /// `target ← (1 − τ) · target + τ · policy` is *not* performed here; it runs
    /// once per gradient update inside [`learn_step`](Self::learn_step).
    ///
    /// The two mechanisms are mutually exclusive:
    ///
    /// - When [`QrDqnTrainingConfig::tau`] is greater than zero, soft updates
    ///   are the live mechanism and **this method is a no-op**, regardless of
    ///   `target_update_frequency`. A hard copy here would discard the
    ///   deliberate lag the soft update maintains, which is exactly what
    ///   `tau` exists to create. Because the shipped
    ///   [`Default`](QrDqnTrainingConfig::default) sets `tau = 0.005`, the hard
    ///   path is unreachable for the default configuration.
    /// - When `tau` is zero, this method copies the policy weights into the
    ///   target on every step that is a positive multiple of
    ///   `config.target_update_frequency`. A `target_update_frequency` of `0`
    ///   disables hard syncs entirely.
    ///
    /// Setting both `tau` and `target_update_frequency` to zero would freeze the
    /// target network forever and is rejected by
    /// [`QrDqnTrainingConfig::validate`](rlevo_core::config::Validate::validate),
    /// so this method can never be a silent no-op on both paths.
    pub fn sync_target(&mut self) {
        // Soft updates own the target network; the hard copy would erase the
        // Polyak lag. `tau` is validated into `[0, 1]`, so `> 0.0` is exactly
        // "soft updates enabled".
        if self.config.tau > 0.0 {
            return;
        }
        if self.config.target_update_frequency == 0 {
            return;
        }
        if self.step > 0
            && self
                .step
                .is_multiple_of(self.config.target_update_frequency)
        {
            self.target_net = self.policy().valid();
        }
    }

    /// Runs one learning step: samples a batch uniformly, computes target
    /// quantiles via a one-step Bellman backup, and minimises the quantile
    /// Huber loss against the policy's predicted quantiles.
    ///
    /// Returns `None` when [`can_learn`](Self::can_learn) is false (buffer too
    /// small or fewer than `learning_starts` steps taken). Returns
    /// [`Some(LearnOutcome)`](LearnOutcome) with the loss and diagnostic values
    /// after a successful gradient update.
    ///
    /// # Panics
    ///
    /// Panics if the agent was poisoned by a panic *inside* a previous
    /// optimizer step — the one window in which the network is out of its
    /// [`Slot`]. Such an agent cannot be recovered and must be rebuilt; see
    /// [`Slot`] for why. The forward pass, the loss, and `backward` all run
    /// against a borrow of the network, so a panic in any of them (a shape
    /// mismatch, a device error, a failed host read) leaves the agent fully
    /// usable.
    pub fn learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<LearnOutcome> {
        if !self.can_learn() {
            return None;
        }
        let batch_size = self.config.batch_size;
        let num_quantiles = self.config.num_quantiles;

        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.buffer.len()))
            .collect();

        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();

        let mut obs_flat: Vec<f32> = Vec::with_capacity(batch_size * numel_per_obs);
        let mut next_flat: Vec<f32> = Vec::with_capacity(batch_size * numel_per_obs);
        let mut action_idxs: Vec<i64> = Vec::with_capacity(batch_size);
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut dones: Vec<f32> = Vec::with_capacity(batch_size);

        for idx in &indices {
            let t = &self.buffer[*idx];
            let obs_tensor: Tensor<B::InnerBackend, DO> = t.obs.to_tensor(&self.device);
            let next_tensor: Tensor<B::InnerBackend, DO> = t.next_obs.to_tensor(&self.device);
            let obs_data = obs_tensor.into_data().convert::<f32>();
            let next_data = next_tensor.into_data().convert::<f32>();
            obs_flat.extend_from_slice(obs_data.as_slice::<f32>().expect("float data"));
            next_flat.extend_from_slice(next_data.as_slice::<f32>().expect("float data"));
            action_idxs.push(t.action_idx as i64);
            rewards.push(t.reward);
            dones.push(if t.done { 1.0 } else { 0.0 });
        }

        let mut batched_shape: Vec<usize> = Vec::with_capacity(DB);
        batched_shape.push(batch_size);
        batched_shape.extend_from_slice(&obs_shape);

        let device = self.device.clone();
        let obs_tensor: Tensor<B, DB> =
            Tensor::from_data(TensorData::new(obs_flat, batched_shape.clone()), &device);
        let next_tensor_inner: Tensor<B::InnerBackend, DB> =
            Tensor::from_data(TensorData::new(next_flat, batched_shape), &device);

        let action_tensor_1: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(action_idxs, vec![batch_size]), &device);

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let dones_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(dones, vec![batch_size]), &device);

        // --- Target quantiles (inner backend, no autodiff) ---
        let next_quantiles_all: Tensor<B::InnerBackend, 3> =
            M::forward_inner(&self.target_net, next_tensor_inner); // (B, A, N)

        // Pick bootstrap action a* = argmax_a mean_i θ_target(s', a)_i.
        let q_next: Tensor<B::InnerBackend, 2> =
            next_quantiles_all.clone().mean_dim(2).squeeze_dim::<2>(2); // (B, A)
        let a_star_2: Tensor<B::InnerBackend, 2, Int> = q_next.argmax(1); // (B, 1)
        let a_star_3: Tensor<B::InnerBackend, 3, Int> =
            a_star_2.unsqueeze_dim::<3>(2).repeat_dim(2, num_quantiles); // (B, 1, N)
        let next_quantiles_chosen: Tensor<B::InnerBackend, 2> =
            next_quantiles_all.gather(1, a_star_3).squeeze_dim::<2>(1); // (B, N)

        // Bellman backup (no projection): T θ_j = r + γ (1 − done) · θ_target_j.
        let rewards_bn: Tensor<B::InnerBackend, 2> = rewards_inner.unsqueeze_dim::<2>(1); // (B, 1)
        let dones_bn: Tensor<B::InnerBackend, 2> = dones_inner.unsqueeze_dim::<2>(1); // (B, 1)
        let keep = dones_bn.neg().add_scalar(1.0); // 1 − done  (B, 1)
        let gamma = self.config.gamma as f32;
        let target_inner: Tensor<B::InnerBackend, 2> =
            rewards_bn + keep * next_quantiles_chosen.mul_scalar(gamma);

        // --- Policy forward (autodiff) ---
        //
        // Everything from here to `step_with` runs against a borrow of the
        // network, so a panic in the forward pass, the loss, or `backward`
        // leaves `policy_net` populated and the agent usable.
        let quantiles: Tensor<B, 3> = self.policy().forward(obs_tensor); // (B, A, N)

        // Diagnostic: mean Q-value across actions and batch.
        let q_all: Tensor<B, 2> = quantiles.clone().mean_dim(2).squeeze_dim::<2>(2); // (B, A)
        let q_mean = q_all.mean().into_scalar().elem::<f32>();

        // Gather the quantile vector for the taken action → (B, N).
        let action_idx_3d: Tensor<B, 3, Int> = action_tensor_1
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, num_quantiles); // (B, 1, N)
        let pred_quantiles: Tensor<B, 2> = quantiles.gather(1, action_idx_3d).squeeze_dim::<2>(1);

        // Spread diagnostic: batch-mean std of pred_quantiles along the
        // quantile axis. Low spread ⇒ distribution collapse.
        let pred_mean: Tensor<B, 2> = pred_quantiles.clone().mean_dim(1); // (B, 1)
        let centered: Tensor<B, 2> = pred_quantiles.clone() - pred_mean;
        let variance: Tensor<B, 1> = centered.powi_scalar(2).mean_dim(1).squeeze_dim::<1>(1); // (B,)
        let spread_value = variance.sqrt().mean().into_scalar().elem::<f32>();

        // Upload the target quantile vector onto the autodiff backend.
        let target_autodiff: Tensor<B, 2> = Tensor::from_data(target_inner.into_data(), &device);

        let taus: Tensor<B, 1> = self.config.quantile_taus::<B>(&device);
        let loss_tensor =
            quantile_huber_loss(pred_quantiles, target_autodiff, taus, self.config.kappa);
        let loss_value = loss_tensor.clone().into_scalar().elem::<f32>();

        let grads = loss_tensor.backward();
        // `from_grads` takes `&M` and returns an owned, lifetime-free value, so
        // NLL ends the borrow here — the only window in which the module is out
        // of the slot is the `Optimizer::step` call inside `step_with`.
        let grads = GradientsParams::from_grads(grads, self.policy());
        self.policy_net
            .step_with(&mut self.optimizer, self.config.learning_rate, grads);

        if self.config.tau > 0.0 {
            // Clone rather than move out: the field stays intact if
            // `soft_update` panics, so a failure can't silently hard-sync
            // the target onto the policy.
            self.target_net =
                M::soft_update(self.policy(), self.target_net.clone(), self.config.tau);
        }

        Some(LearnOutcome {
            loss: loss_value,
            q_mean,
            quantile_spread: spread_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::{Autodiff, Flex};
    use burn::module::{AutodiffModule, Module, Param};
    use burn::tensor::backend::Backend;

    use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};

    use crate::algorithms::qrdqn::qrdqn_config::QrDqnTrainingConfigBuilder;
    use crate::utils::polyak_update;

    type TestBackend = Autodiff<Flex>;
    type TestInnerBackend = <TestBackend as AutodiffBackend>::InnerBackend;

    const TEST_OBS_FEATURES: usize = 4; // CartPoleObservation is [4].
    const TEST_ACTIONS: usize = 2;
    const TEST_QUANTILES: usize = 4;

    /// Minimal QR-DQN network: one weight matrix, no RNG.
    ///
    /// Every parameter is built from explicit [`TensorData`], so two nets with
    /// different `fill` values are guaranteed to differ elementwise — the
    /// target-sync tests below need a deterministic, backend-RNG-free way to
    /// put the policy and target networks provably out of sync.
    #[derive(Module, Debug)]
    struct TestQrDqnNet<B: Backend> {
        w: Param<Tensor<B, 2>>,
    }

    impl<B: Backend> TestQrDqnNet<B> {
        fn new(fill: f32, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
            let cols = TEST_ACTIONS * TEST_QUANTILES;
            let data = vec![fill; TEST_OBS_FEATURES * cols];
            let w: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(data, vec![TEST_OBS_FEATURES, cols]), device);
            Self {
                w: Param::from_tensor(w),
            }
        }

        fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
            let [batch_size, _] = observations.dims();
            observations
                .matmul(self.w.val())
                .reshape([batch_size, TEST_ACTIONS, TEST_QUANTILES])
        }
    }

    impl<B: AutodiffBackend> QrDqnModel<B, 2> for TestQrDqnNet<B> {
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
            polyak_update::<B::InnerBackend, TestQrDqnNet<B::InnerBackend>>(
                &active.valid(),
                target,
                tau as f32,
            )
        }
    }

    type TestAgent = QrDqnAgent<
        TestBackend,
        TestQrDqnNet<TestBackend>,
        CartPoleObservation,
        CartPoleAction,
        1,
        2,
    >;

    /// Reads a network's single parameter matrix back to the host.
    fn weights_of<B: Backend>(net: &TestQrDqnNet<B>) -> Vec<f32> {
        net.w
            .val()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 host read of a parameter this test just built")
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "parameter vectors must be the same length"
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Builds an agent whose policy and target networks are *deliberately* out
    /// of sync: the policy is filled with `1.0`, the target with `2.0`.
    ///
    /// This stands in for "the policy has drifted away from the target during
    /// training" without running a training loop — driving real gradient steps
    /// would couple the test to `learning_starts`, buffer fill and the ε
    /// schedule, none of which the target-sync gate depends on.
    fn diverged_agent(tau: f64, target_update_frequency: usize) -> TestAgent {
        let device: <TestBackend as burn::tensor::backend::BackendTypes>::Device =
            Default::default();
        let config = QrDqnTrainingConfigBuilder::new()
            .tau(tau)
            .target_update_frequency(target_update_frequency)
            .num_quantiles(TEST_QUANTILES)
            .build()
            .expect("valid config");
        let policy: TestQrDqnNet<TestBackend> = TestQrDqnNet::new(1.0, &device);
        let mut agent = TestAgent::new(policy, config, device).expect("valid config");
        // In-source unit tests may touch private fields (rules.md §5); this is
        // the seam that makes the gate testable without a public accessor.
        agent.target_net = TestQrDqnNet::<TestInnerBackend>::new(2.0, &device);
        agent
    }

    #[test]
    fn test_qrdqn_agent_sync_target_is_noop_when_tau_is_positive() {
        // Regression test for #182: with the default `tau = 0.005`, the
        // periodic hard copy used to fire anyway and erase the Polyak lag.
        let mut agent = diverged_agent(0.005, 100);

        let policy_before = weights_of(&agent.policy().valid());
        let target_before = weights_of(&agent.target_net);
        // Precondition — without this the assertion below would be vacuous.
        assert!(
            max_abs_diff(&policy_before, &target_before) > 1e-3,
            "precondition: policy and target must already differ before sync_target"
        );

        // Land exactly on a hard-sync boundary.
        agent.step = 100;
        agent.sync_target();

        let target_after = weights_of(&agent.target_net);
        assert!(
            max_abs_diff(&target_after, &target_before) < 1e-6,
            "target network must be untouched by sync_target when tau > 0"
        );
        assert!(
            max_abs_diff(&target_after, &policy_before) > 1e-3,
            "sync_target must not hard-copy the policy into the target while \
             Polyak soft updates (tau > 0) own the target network"
        );
    }

    #[test]
    fn test_qrdqn_agent_sync_target_hard_copies_when_tau_is_zero() {
        // Mirror of the regression test: with soft updates disabled, the hard
        // sync is the only mechanism and must still fire on schedule.
        let mut agent = diverged_agent(0.0, 100);

        let policy_before = weights_of(&agent.policy().valid());
        let target_before = weights_of(&agent.target_net);
        assert!(
            max_abs_diff(&policy_before, &target_before) > 1e-3,
            "precondition: policy and target must already differ before sync_target"
        );

        agent.step = 100;
        agent.sync_target();

        let target_after = weights_of(&agent.target_net);
        assert!(
            max_abs_diff(&target_after, &policy_before) < 1e-6,
            "with tau = 0 the target must become an exact copy of the policy at \
             a multiple of target_update_frequency"
        );
    }

    #[test]
    fn test_qrdqn_agent_sync_target_skips_non_multiple_steps_when_tau_is_zero() {
        let mut agent = diverged_agent(0.0, 100);
        let target_before = weights_of(&agent.target_net);

        agent.step = 99;
        agent.sync_target();

        assert!(
            max_abs_diff(&weights_of(&agent.target_net), &target_before) < 1e-6,
            "hard sync must only fire on multiples of target_update_frequency"
        );
    }

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = QrDqnMetrics {
            reward: 42.0,
            steps: 7,
            policy_loss: 0.5,
            epsilon: 0.1,
            q_mean: 1.0,
            quantile_spread: 2.5,
        };
        assert_eq!(m.score(), 42.0);
        assert_eq!(m.duration(), 7);
    }

    #[test]
    fn error_display_uses_thiserror_messages() {
        let err = QrDqnAgentError::InvalidAction("bad index".into());
        assert_eq!(err.to_string(), "Invalid action: bad index");
    }
}
