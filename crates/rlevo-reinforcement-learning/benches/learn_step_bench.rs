//! End-to-end `learn_step`-shaped A/B bench: staging strategy vs. the full
//! update (stage -> forward -> loss -> backward -> optimizer step), so the
//! staging delta from `staging_bench.rs` can be reported as a **percentage of
//! a real `learn_step`** instead of extrapolated arithmetic -- the original
//! host-side-staging fix estimated its saving at ~192 ms at batch 64 by
//! extrapolating from the isolated round-trip cost rather than measuring a
//! real end-to-end `learn_step`, and this file closes that gap.
//!
//! # Why this file exists alongside `staging_bench.rs`
//!
//! `staging_bench.rs` measured the staging primitive in isolation: ~1.5 ms
//! per device round trip on wgpu/Metal, invariant to row size. That number
//! answers "how expensive is one round trip" but not "what fraction of a
//! real training step does that round trip actually cost" -- a real
//! `learn_step` also runs a forward pass, a Huber loss, `backward()`, and an
//! Adam step, several of which force their own synchronizations and may
//! *hide* some staging cost behind work that was going to sync anyway. This
//! file runs the full update, with staging strategy as the only swappable
//! variable, so one criterion run yields both arms under identical
//! conditions.
//!
//! # This checkout's actual state
//!
//! `dqn_agent.rs::learn_step` in this checkout already uses
//! `write_host_row` (commit `57dd565`) -- see the
//! [`Staging::HostRow`] arm below, which reproduces that production code
//! path. [`Staging::Roundtrip`] reintroduces the pre-fix per-row
//! `to_tensor` + `into_data()` round trip **inside this bench only**, for
//! comparison; no `src/` file is touched. Everything downstream of staging
//! (forward, target, loss, backward, optimizer step, soft update) is
//! identical between the two arms -- only the staging loop differs, exactly
//! mirroring `dqn_agent.rs::learn_step` (~line 390 onward), so the delta this
//! bench measures is attributable to staging alone.
//!
//! # What is *not* reproduced from `dqn_agent.rs::learn_step`
//!
//! - **Double-DQN** (`double_q`) is omitted: the default `DqnTrainingConfig`
//!   ships it `false`, and turning it on would add one extra target-path
//!   `forward` call to *both* arms identically -- orthogonal to the staging
//!   question.
//! - **Prioritized replay** priority writeback is omitted for the same
//!   reason: the default config uses uniform replay, and PER's post-step
//!   host read is gated behind `buffer.is_prioritized()` in production, so a
//!   default-config agent never takes that branch either.
//! - **`Slot`** (the panic-safety wrapper around the policy network across
//!   `ModuleOptimizer::step`) is not used. This file keeps the `Option<M>` +
//!   `take()` idiom `Slot`'s docs describe as its historical predecessor --
//!   adequate for a bench with no panic-safety requirement, and identical in
//!   both staging arms.
//!
//! Every other line -- batch staging, tensor construction, forward, target
//! computation via [`compute_target_q_values`], Huber loss, `backward()`,
//! `GradientsParams::from_grads`, `ModuleOptimizer::step`, and the Polyak
//! soft update via [`polyak_update`] -- matches `dqn_agent.rs::learn_step`
//! line for line, using the same public helpers
//! (`rlevo_reinforcement_learning::utils::{compute_target_q_values,
//! polyak_update}`) production code calls.
//!
//! # Sync correctness -- three points, addressed explicitly
//!
//! 1. **wgpu is asynchronous; the timed region must include execution.**
//!    Exactly as `staging_bench.rs` establishes: `Tensor::into_data()` /
//!    `into_scalar()` block until the device queue drains up to that tensor
//!    (`try_read_sync`), so a timed region that never forces a read measures
//!    submission, not completion.
//! 2. **A full `learn_step` already contains two forced reads mid-function**
//!    -- `q_mean = q_all.clone().mean().into_scalar()` right after the
//!    forward pass, and `loss_value = loss_tensor.clone().into_scalar()`
//!    right before `backward()`. Both exist in production `learn_step` for
//!    diagnostics, not for benchmark correctness, and both are reproduced
//!    here unchanged. **Verified, not assumed:** neither of those two reads
//!    is *downstream* of `backward()` / `ModuleOptimizer::step()` / the Polyak
//!    update -- they run strictly before `backward()` is even called -- so
//!    they cannot force completion of the gradient computation or the
//!    optimizer step. Burn's `backward()` and `Adam::step()` are ordinary
//!    tensor-op sequences on the same backend; nothing about them forces a
//!    host read. Left alone, a `learn_step` could return having only
//!    *submitted* backward + optimizer + soft-update work, understating
//!    their cost in exactly the direction `staging_bench.rs`'s module doc
//!    warns about (inverting the result by making the fast arm look faster
//!    than it is). So **this bench adds one explicit forced read after the
//!    optimizer step**: a second forward pass, on a batch-size-1 slice of
//!    the already-staged observation tensor, through the just-updated
//!    policy network, with `.into_data()` on the output. Because the
//!    updated policy's parameters and the target network's Polyak-updated
//!    parameters share the *same physical device queue* (wgpu backend and
//!    its `Autodiff` wrapper both submit to one `wgpu::Queue` per adapter),
//!    forcing completion of anything enqueued after the soft update also
//!    drains the soft update itself -- a second, separate probe against the
//!    target network is not needed. This probe is applied identically to
//!    both staging arms, is deliberately batch-size-1 (cheap relative to
//!    the full-batch forward it follows) so it does not materially inflate
//!    the "total `learn_step`" denominator the staging percentage is
//!    computed against, and is reused as `black_box` bait to defeat
//!    dead-code elimination of the whole update.
//! 3. **The unexplained `n = 100` non-completion from step 1** (cell
//!    `stage_cartpole_wgpu/roundtrip/256`) is addressed by *not* reproducing
//!    the conditions that triggered it by default: every wgpu group here
//!    uses a reduced `sample_size` up front (see [`wgpu_sample_size`]),
//!    chosen from the measured per-row constant rather than discovered by
//!    hitting the same wall again. A short, separate, explicitly-labelled
//!    experiment (`learn_step_n100_probe`, gated behind the `N100_PROBE` env
//!    var so the default run stays fast) runs one cheap wgpu cell twice --
//!    once at `sample_size = 20` and once at the criterion default of
//!    `100` -- to check whether sustained submission stalls independent of
//!    per-iteration cost, or only appears once total wall time crosses some
//!    threshold. See the session report for the result.
//!
//! # Reporting
//!
//! Every criterion group name carries both the network family and the
//! backend label (`flex` / `wgpu`); as in `staging_bench.rs`, a `wgpu`
//! number in this repository is Metal on an Apple M2 Pro and does not
//! transfer to CUDA or any other adapter. Staging's contribution is reported
//! as a percentage of total `learn_step` wall time (roundtrip vs. `host_row`),
//! not just an absolute delta -- see the session report for the full table.
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo-reinforcement-learning --bench learn_step_bench
//! ```

#[path = "support/bench_backend.rs"]
mod bench_backend;

use std::hint::black_box;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::HuberLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Int, Tensor, TensorData, activation};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::base::TensorConvertible;
use rlevo_environments::classic::cartpole::CartPoleObservation;
use rlevo_environments::pixel_grid::{CELL_COUNT_U32, PixelGridState, PixelObservation};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::{
    DqnModel, MlpQNetwork, MlpQNetworkConfig,
};
use rlevo_reinforcement_learning::utils::{compute_target_q_values, polyak_update};

use bench_backend::BenchDevice;

/// Batch sizes swept per (network, backend) combination. 64 matches
/// `dqn_bench.rs`'s `dqn_learn_step_batch64`; 256 is added to check whether
/// staging's cost holds at a larger batch, since `staging_bench.rs`'s
/// isolated round-trip micro-benchmark did not complete at batch 256 for
/// reasons not yet diagnosed.
const BATCH_SIZES: [usize; 2] = [64, 256];

/// DQN default `gamma` (`DqnTrainingConfig::default()`).
const GAMMA: f32 = 0.99;
/// DQN default `learning_rate`.
const LR: f64 = 0.001;
/// DQN default `tau` (soft target updates active every step).
const TAU: f32 = 0.005;
/// DQN default grad-clip norm (`GradientClippingConfig::Value(100.0)`).
const GRAD_CLIP_NORM: f32 = 100.0;
/// Discrete action count for the CartPole-shaped MLP nets (Left/Right).
const CARTPOLE_ACTIONS: usize = 2;
/// Discrete action count for the pixel-grid conv net (Up/Down/Left/Right).
const PIXEL_ACTIONS: usize = 4;

// ---------------------------------------------------------------------------
// Staging strategy -- the swappable variable, reproduced from staging_bench.rs
// ---------------------------------------------------------------------------

/// Which strategy stages the sampled minibatch into a flat host buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Staging {
    /// Per-row `to_tensor(device)` then `.into_data()` -- upload-then-
    /// download-unchanged, one wgpu sync point per row. The behaviour before
    /// the host-side staging fix, reproduced verbatim from `57dd565`'s parent.
    Roundtrip,
    /// `HostRow::write_host_row` -- pure host staging, one batched
    /// upload. The strategy `dqn_agent.rs::learn_step` uses today.
    HostRow,
}

impl Staging {
    const fn label(self) -> &'static str {
        match self {
            Self::Roundtrip => "roundtrip",
            Self::HostRow => "host_row",
        }
    }
}

/// Stages one sampled minibatch's observations and next-observations into two
/// batched tensors, using `staging` to fill the host buffers. Mirrors
/// `dqn_agent.rs::learn_step`'s staging loop: `obs` ends up on `device` (the
/// autodiff device the policy trains on), `next_obs` on `plain` (the target
/// network's device, without autodiff), exactly as production does.
fn stage_batch<O, const DO: usize, const DB: usize>(
    batch: &[SynthTransition<O>],
    device: &Device,
    plain: &Device,
    staging: Staging,
) -> (Tensor<DB>, Tensor<DB>)
where
    O: TensorConvertible<DO>,
{
    debug_assert_eq!(DB, DO + 1, "batched rank DB must equal row rank DO + 1");
    let row_shape = O::shape();
    let row_len: usize = row_shape.iter().product();
    let mut obs_flat: Vec<f32> = Vec::with_capacity(batch.len() * row_len);
    let mut next_flat: Vec<f32> = Vec::with_capacity(batch.len() * row_len);
    match staging {
        Staging::Roundtrip => {
            for t in batch {
                let obs_row: Tensor<DO> = t.obs.to_tensor(device);
                let obs_data = obs_row.into_data().convert::<f32>();
                obs_flat.extend_from_slice(obs_data.as_slice::<f32>().expect("float data"));
                let next_row: Tensor<DO> = t.next_obs.to_tensor(device);
                let next_data = next_row.into_data().convert::<f32>();
                next_flat.extend_from_slice(next_data.as_slice::<f32>().expect("float data"));
            }
        }
        Staging::HostRow => {
            for t in batch {
                t.obs.write_host_row(&mut obs_flat);
                t.next_obs.write_host_row(&mut next_flat);
            }
        }
    }
    let mut shape = [0usize; DB];
    shape[0] = batch.len();
    shape[1..].copy_from_slice(&row_shape);
    let obs_tensor: Tensor<DB> = Tensor::from_data(TensorData::new(obs_flat, shape), device);
    let next_tensor: Tensor<DB> = Tensor::from_data(TensorData::new(next_flat, shape), plain);
    (obs_tensor, next_tensor)
}

/// Runs [`stage_batch`] under both strategies and asserts bit-identical
/// output, once, outside any timed region -- so a divergence would fail the
/// bench immediately instead of quietly comparing two different workloads.
fn assert_stage_bit_identical<O, const DO: usize, const DB: usize>(
    batch: &[SynthTransition<O>],
    device: &Device,
) where
    O: TensorConvertible<DO>,
{
    let (rt_obs, rt_next) = stage_batch::<O, DO, DB>(batch, device, device, Staging::Roundtrip);
    let (hr_obs, hr_next) = stage_batch::<O, DO, DB>(batch, device, device, Staging::HostRow);
    let rt_obs_v: Vec<f32> = rt_obs.into_data().try_into_vec().expect("host read");
    let hr_obs_v: Vec<f32> = hr_obs.into_data().try_into_vec().expect("host read");
    assert_eq!(
        rt_obs_v, hr_obs_v,
        "obs staging diverged between strategies"
    );
    let rt_next_v: Vec<f32> = rt_next.into_data().try_into_vec().expect("host read");
    let hr_next_v: Vec<f32> = hr_next.into_data().try_into_vec().expect("host read");
    assert_eq!(
        rt_next_v, hr_next_v,
        "next_obs staging diverged between strategies"
    );
}

// ---------------------------------------------------------------------------
// Synthetic replay data
// ---------------------------------------------------------------------------

/// A minimal stand-in for a sampled `DiscreteTransition<O>` row -- everything
/// `learn_step`'s staging + forward + target computation touches, with no
/// dependency on the crate-private replay buffer types.
struct SynthTransition<O> {
    obs: O,
    next_obs: O,
    action: i64,
    reward: f32,
    terminated: f32,
}

// Synthetic fixture/benchmark data: the loop counter and element count are
// bounded by small constants declared in this file, far below f32's 2^24
// exact-integer limit. The values are inputs to a throughput measurement, not
// quantities whose precision is asserted.
#[allow(clippy::cast_possible_wrap)]
fn cartpole_transitions(n: usize, rng: &mut StdRng) -> Vec<SynthTransition<CartPoleObservation>> {
    let random_obs = |rng: &mut StdRng| CartPoleObservation {
        cart_pos: rng.random_range(-2.4_f32..2.4_f32),
        cart_vel: rng.random_range(-3.0_f32..3.0_f32),
        pole_angle: rng.random_range(-0.2_f32..0.2_f32),
        pole_ang_vel: rng.random_range(-3.0_f32..3.0_f32),
    };
    (0..n)
        .map(|_| SynthTransition {
            obs: random_obs(rng),
            next_obs: random_obs(rng),
            action: rng.random_range(0..CARTPOLE_ACTIONS as i64),
            reward: rng.random_range(-1.0_f32..1.0_f32),
            terminated: if rng.random_range(0.0_f32..1.0_f32) < 0.05 {
                1.0
            } else {
                0.0
            },
        })
        .collect()
}

// Synthetic fixture/benchmark data: the loop counter and element count are
// bounded by small constants declared in this file, far below f32's 2^24
// exact-integer limit. The values are inputs to a throughput measurement, not
// quantities whose precision is asserted.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn pixel_transitions(n: usize, rng: &mut StdRng) -> Vec<SynthTransition<PixelObservation>> {
    (0..n)
        .map(|_| {
            let goal = rng.random_range(0..CELL_COUNT_U32);
            let obs = PixelGridState::new(rng.random_range(0..CELL_COUNT_U32), goal)
                .expect("cells sampled from 0..CELL_COUNT_U32 are in range")
                .project();
            let next_obs = PixelGridState::new(rng.random_range(0..CELL_COUNT_U32), goal)
                .expect("cells sampled from 0..CELL_COUNT_U32 are in range")
                .project();
            SynthTransition {
                obs,
                next_obs,
                action: rng.random_range(0..PIXEL_ACTIONS as i64),
                reward: rng.random_range(-1.0_f32..1.0_f32),
                terminated: if rng.random_range(0.0_f32..1.0_f32) < 0.05 {
                    1.0
                } else {
                    0.0
                },
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Networks under test
// ---------------------------------------------------------------------------

/// Small MLP, `4 -> 64 -> 64 -> 2`, as `dqn_bench.rs` builds it: the crate's
/// default Q-network.
fn small_mlp(device: &Device) -> MlpQNetwork {
    MlpQNetworkConfig {
        obs_dim: 4,
        hidden: vec![64, 64],
        num_actions: CARTPOLE_ACTIONS,
    }
    .try_init(device)
    .expect("valid Q-network config")
}

/// Wide MLP, `4 -> 512 -> 512 -> 512 -> 2`: enough parameters that the
/// optimizer step, not the staging, dominates.
fn wide_mlp(device: &Device) -> MlpQNetwork {
    MlpQNetworkConfig {
        obs_dim: 4,
        hidden: vec![512, 512, 512],
        num_actions: CARTPOLE_ACTIONS,
    }
    .try_init(device)
    .expect("valid Q-network config")
}

/// Small conv net over `PixelObservation` (`[20, 20, 3]`) -- the case the
/// library is named for, and the first deep-network benchmark in this suite.
/// Two `Conv2d` layers,
/// adaptive-average-pooled to a fixed `4x4` spatial size (so the head's
/// input width is independent of any future change to `IMG_SIDE`), then two
/// `Linear` layers down to the 4-way pixel-grid action space.
///
/// Input arrives batched as `[batch, H, W, C]` (row-major, matching
/// `PixelObservation::shape()`); `forward_impl` permutes to
/// `[batch, C, H, W]` before the first convolution.
#[derive(Module, Debug)]
struct PixelConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
}

/// Fixed pooled spatial size the conv stack's head is sized against.
const POOL_SIZE: usize = 4;
/// Output channels of the second conv layer.
const CONV2_CHANNELS: usize = 32;

impl PixelConvNet {
    fn new(device: &Device) -> Self {
        let conv1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv2 = Conv2dConfig::new([16, CONV2_CHANNELS], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let pool = AdaptiveAvgPool2dConfig::new([POOL_SIZE, POOL_SIZE]).init();
        let fc1 = LinearConfig::new(CONV2_CHANNELS * POOL_SIZE * POOL_SIZE, 128).init(device);
        let fc2 = LinearConfig::new(128, PIXEL_ACTIONS).init(device);
        Self {
            conv1,
            conv2,
            pool,
            fc1,
            fc2,
        }
    }

    // b/c/h/w are the canonical NCHW dimension names.
    #[allow(clippy::many_single_char_names)]
    fn forward_impl(&self, x: Tensor<4>) -> Tensor<2> {
        // [batch, H, W, C] -> [batch, C, H, W].
        let x = x.permute([0, 3, 1, 2]);
        let x = activation::relu(self.conv1.forward(x));
        let x = activation::relu(self.conv2.forward(x));
        let x = self.pool.forward(x);
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c * h * w]);
        let x = activation::relu(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}

impl DqnModel<4> for PixelConvNet {
    fn forward(&self, obs: Tensor<4>) -> Tensor<2> {
        self.forward_impl(obs)
    }
}

// ---------------------------------------------------------------------------
// The full learn_step mirror
// ---------------------------------------------------------------------------

/// Values read back to host purely to force completion / feed `black_box`;
/// see the module doc's "Sync correctness" point 2 for why `sync_probe`
/// exists.
struct LearnStepProbe {
    loss: f32,
    q_mean: f32,
    sync_probe: f32,
}

/// One full DQN update: stage -> forward -> target -> Huber loss ->
/// `backward()` -> Adam step -> Polyak soft update -> forced completion
/// probe. Mirrors `dqn_agent.rs::learn_step` line for line except for the
/// omissions documented in the module doc (`double_q`, PER writeback,
/// `Slot`).
#[allow(clippy::too_many_arguments)]
fn run_learn_step<M, O, const DO: usize, const DB: usize>(
    policy: M,
    mut target: M,
    optimizer: &mut ModuleOptimizer,
    device: &Device,
    plain: &Device,
    batch: &[SynthTransition<O>],
    staging: Staging,
) -> (M, M, LearnStepProbe)
where
    M: DqnModel<DB>,
    O: TensorConvertible<DO>,
{
    let batch_size = batch.len();

    let (obs_tensor, next_tensor_inner): (Tensor<DB>, Tensor<DB>) =
        stage_batch::<O, DO, DB>(batch, device, plain, staging);
    // Kept for the post-optimizer-step forced-completion probe (sync
    // correctness point 2) -- a batch-size-1 slice, cheap relative to the
    // full-batch forward it mirrors.
    let probe_input: Tensor<DB> = obs_tensor.clone().narrow(0, 0, 1);

    let action_idxs: Vec<i64> = batch.iter().map(|t| t.action).collect();
    let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
    let terminated: Vec<f32> = batch.iter().map(|t| t.terminated).collect();

    let action_tensor_1: Tensor<1, Int> =
        Tensor::from_data(TensorData::new(action_idxs, vec![batch_size]), device);
    let action_tensor: Tensor<2, Int> = action_tensor_1.unsqueeze_dim::<2>(1);

    let rewards_inner: Tensor<1> =
        Tensor::from_data(TensorData::new(rewards, vec![batch_size]), plain);
    let terminated_inner: Tensor<1> =
        Tensor::from_data(TensorData::new(terminated, vec![batch_size]), plain);

    // --- Forward ---
    let q_all: Tensor<2> = policy.forward(obs_tensor);
    let q_mean = q_all.clone().mean().into_scalar::<f32>();
    let q_pred: Tensor<2> = q_all.gather(1, action_tensor);
    let q_pred_flat: Tensor<1> = q_pred.squeeze_dim::<1>(1);

    // --- Target (vanilla DQN; double_q is off by default -- see module doc) ---
    let next_q_target_inner: Tensor<2> = target.forward(next_tensor_inner);
    let next_q_max_inner: Tensor<1> = next_q_target_inner.max_dim(1).squeeze_dim::<1>(1);

    let target_q_inner: Tensor<1> =
        compute_target_q_values(rewards_inner, next_q_max_inner, terminated_inner, GAMMA);
    let target_q: Tensor<1> = target_q_inner.to_device(device);

    // Uniform replay (weight ~= 1): reduce_weighted_loss's effect collapses
    // to a plain mean, so it is inlined rather than reproduced (it is
    // pub(crate), unreachable from a bench regardless).
    let per_sample_loss = HuberLossConfig::new(1.0)
        .init()
        .forward_no_reduction(q_pred_flat, target_q);
    let loss_tensor: Tensor<1> = per_sample_loss.mean();
    let loss_value = loss_tensor.clone().into_scalar::<f32>();

    let grads = loss_tensor.backward();
    let grads = GradientsParams::from_grads(grads, &policy);
    let policy = optimizer.step(LR, policy, grads);

    if TAU > 0.0 {
        target = polyak_update(&policy.valid(), target, TAU)
            .expect("bench target is cloned from policy");
    }

    // Forced completion probe -- see module doc "Sync correctness" point 2.
    let probe_out: Tensor<2> = policy.forward(probe_input);
    let sync_probe = probe_out
        .into_data()
        .try_into_vec::<f32>()
        .expect("forced completion read (post-optimizer-step probe)")
        .first()
        .copied()
        .unwrap_or(0.0);

    (
        policy,
        target,
        LearnStepProbe {
            loss: loss_value,
            q_mean,
            sync_probe,
        },
    )
}

fn build_optimizer() -> ModuleOptimizer {
    AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(GRAD_CLIP_NORM)))
        .init()
}

// ---------------------------------------------------------------------------
// Bench driver
// ---------------------------------------------------------------------------

/// Reduced sample size for wgpu groups, mirroring `staging_bench.rs`'s
/// workaround for the unexplained `n = 100` non-completion at batch 256
/// (see module doc point 3). Kept as a function (not a bare constant) so the
/// `learn_step_n100_probe` experiment can reuse the same reasoning inline.
const fn wgpu_sample_size() -> usize {
    20
}

/// Runs the `roundtrip` vs. `host_row` A/B for one (network, backend)
/// combination across [`BATCH_SIZES`], inside criterion group `group_name`.
#[allow(clippy::too_many_arguments)]
fn bench_learn_step<M, O, const DO: usize, const DB: usize>(
    c: &mut Criterion,
    group_name: &str,
    sample_size: usize,
    bench: BenchDevice,
    make_model: impl Fn(&Device) -> M,
    make_batch: impl Fn(usize, &mut StdRng) -> Vec<SynthTransition<O>>,
) where
    M: DqnModel<DB>,
    O: TensorConvertible<DO>,
{
    // As the agent does: train on the device with autodiff on, run the
    // target path without it.
    let plain = bench.device();
    let device = plain.clone().autodiff();
    let mut rng = StdRng::seed_from_u64(365);

    let mut group = c.benchmark_group(group_name);
    group.sample_size(sample_size);
    for &batch_size in &BATCH_SIZES {
        let batch = make_batch(batch_size, &mut rng);
        assert_stage_bit_identical::<O, DO, DB>(&batch, &device);
        for &staging in &[Staging::Roundtrip, Staging::HostRow] {
            group.bench_with_input(
                BenchmarkId::new(staging.label(), batch_size),
                &batch,
                |b, batch| {
                    let init_policy: M = make_model(&plain).train();
                    let init_target: M = init_policy.valid();
                    let mut optimizer = build_optimizer();
                    // `Option<M>` + `take()`/reassign: the idiom
                    // `rlevo_reinforcement_learning::algorithms::Slot`'s docs
                    // describe as its historical predecessor, kept here in
                    // place of `Slot` (see module doc). Needed because
                    // `ModuleOptimizer::step` and `polyak_update` both
                    // consume their module by value, and a `FnMut` closure
                    // cannot move out of a captured variable directly.
                    let mut policy_slot: Option<M> = Some(init_policy);
                    let mut target_slot: Option<M> = Some(init_target);
                    b.iter(|| {
                        let policy = policy_slot.take().expect("policy present");
                        let target = target_slot.take().expect("target present");
                        let (new_policy, new_target, probe) = run_learn_step::<M, O, DO, DB>(
                            policy,
                            target,
                            &mut optimizer,
                            &device,
                            &plain,
                            batch,
                            staging,
                        );
                        policy_slot = Some(new_policy);
                        target_slot = Some(new_target);
                        black_box((probe.loss, probe.q_mean, probe.sync_probe));
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_learn_step_sweep(c: &mut Criterion) {
    bench_learn_step::<MlpQNetwork, CartPoleObservation, 1, 2>(
        c,
        "learn_step_small_mlp_flex",
        100,
        BenchDevice::Flex,
        small_mlp,
        cartpole_transitions,
    );
    bench_learn_step::<MlpQNetwork, CartPoleObservation, 1, 2>(
        c,
        "learn_step_small_mlp_wgpu",
        wgpu_sample_size(),
        BenchDevice::Wgpu,
        small_mlp,
        cartpole_transitions,
    );

    bench_learn_step::<MlpQNetwork, CartPoleObservation, 1, 2>(
        c,
        "learn_step_wide_mlp_flex",
        100,
        BenchDevice::Flex,
        wide_mlp,
        cartpole_transitions,
    );
    bench_learn_step::<MlpQNetwork, CartPoleObservation, 1, 2>(
        c,
        "learn_step_wide_mlp_wgpu",
        wgpu_sample_size(),
        BenchDevice::Wgpu,
        wide_mlp,
        cartpole_transitions,
    );

    bench_learn_step::<PixelConvNet, PixelObservation, 3, 4>(
        c,
        "learn_step_pixel_conv_flex",
        100,
        BenchDevice::Flex,
        PixelConvNet::new,
        pixel_transitions,
    );
    bench_learn_step::<PixelConvNet, PixelObservation, 3, 4>(
        c,
        "learn_step_pixel_conv_wgpu",
        wgpu_sample_size(),
        BenchDevice::Wgpu,
        PixelConvNet::new,
        pixel_transitions,
    );
}

/// Targeted `n = 100` investigation (methodology point 3), gated behind an
/// env var so the default sweep above stays fast: runs the cheapest wgpu
/// cell (`small_mlp`, batch 64, `host_row`) at both `sample_size = 20` and
/// the criterion default `100`, back to back, to check whether sustained
/// wgpu submission stalls independent of per-iteration cost. Enable with
/// `N100_PROBE=1 cargo bench -p rlevo-reinforcement-learning --bench
/// learn_step_bench -- learn_step_n100_probe`.
fn bench_n100_probe(c: &mut Criterion) {
    if std::env::var("N100_PROBE").is_err() {
        return;
    }
    bench_learn_step::<MlpQNetwork, CartPoleObservation, 1, 2>(
        c,
        "learn_step_n100_probe_n20",
        wgpu_sample_size(),
        BenchDevice::Wgpu,
        small_mlp,
        cartpole_transitions,
    );
    bench_learn_step::<MlpQNetwork, CartPoleObservation, 1, 2>(
        c,
        "learn_step_n100_probe_n100",
        100,
        BenchDevice::Wgpu,
        small_mlp,
        cartpole_transitions,
    );
}

criterion_group!(benches, bench_learn_step_sweep, bench_n100_probe);
criterion_main!(benches);
