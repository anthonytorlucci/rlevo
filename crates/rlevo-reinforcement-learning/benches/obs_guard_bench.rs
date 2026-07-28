//! Cost measurement for the proposed **observation finiteness guard** (issue
//! #1043): should `NaN`/`Inf` observations be rejected at replay ingestion
//! (`remember`), or at the batch-staging seam inside `learn_step`?
//!
//! This file measures only. **No guard is added to `src/` by this change** --
//! every checked variant below is a bench-local prototype, so the numbers
//! describe a guard that does not exist yet.
//!
//! # What the source actually does (verified, not assumed)
//!
//! Issue #1043 speculates that "`Observation`/`HostRow<R>` may allow the check
//! to ride the existing `write_host_row` traversal at near-zero marginal cost
//! -- the row is already being walked". That is **true at one seam and false at
//! the other**, and the split is the whole reason this bench has four arms:
//!
//! - **Ingestion (`remember`) -- no traversal to fuse into.**
//!   `dqn_agent.rs::remember` (lines 386-397 in this checkout) runs the
//!   `FiniteRewardGuard` on the scalar reward and then pushes the *typed* `O`
//!   value straight into the replay buffer: `self.buffer.push(DiscreteTransition
//!   { obs, action: action.to_index(), reward, next_obs, terminated })`. No
//!   flatten, no `write_host_row`, no `Vec<f32>` anywhere on the path. A
//!   finiteness check here is therefore a **whole new `O(row_len)` pass**, and --
//!   unless a scratch buffer is threaded through the agent -- a new allocation
//!   per call as well. The [`Ingest`] arms measure exactly that.
//! - **Batch staging (`learn_step`) -- the fusion is real.**
//!   `dqn_agent.rs::learn_step` (lines 530-539) walks the sampled ids and calls
//!   `t.obs.write_host_row(&mut obs_flat)` / `t.next_obs.write_host_row(&mut
//!   next_flat)` directly into two shared flat buffers. Scanning
//!   `buf[start..]` immediately after each call is a second pass over data that
//!   is still L1-resident. The `obsguard_stage_*` arms measure that.
//!
//! # The two numbers that decide the issue
//!
//! Neither arm in isolation answers anything; only the ratios do. A guard
//! costing 300 ns is free against a 40 us device-bound `learn_step` and
//! expensive against a 2 us host-bound env step.
//!
//! 1. `(ingest_checked - ingest_baseline)` as a fraction of a full
//!    **env-step-plus-`remember` cycle** (`obsguard_env_cycle_*` groups, which
//!    drive the *real* `LunarLanderDiscrete` / `CarRacing` environments).
//! 2. `(stage_fused - stage_baseline)` as a fraction of a full **`learn_step`**
//!    (`obsguard_learn_step_*` groups, which call the *real*
//!    `DqnAgent::learn_step`, not a mirror).
//!
//! # Observation shapes
//!
//! - `LunarLanderObservation` -- 8 `f32`, `R = 1`. The small feature vector.
//! - `CarRacingObservation` -- 96x96x3 = 27 648 elements, `R = 3`. The image
//!   shape. **Backed by `Arc<[u8; 27648]>`**, so (a) a non-finite value is
//!   structurally impossible in the stored type, (b) `remember` is O(1) -- an
//!   `Arc` refcount bump, not a 110 KB copy -- and (c) `write_host_row` does a
//!   `u8 -> f32 / 255.0` conversion, so the "check" is really a *materialize
//!   then check*.
//! - [`F32Image`] -- a bench-local 96x96x3 `f32`-backed observation, same shape
//!   and same `Arc` sharing, added because `CarRacingObservation` **cannot be
//!   poisoned** (see the controls below) and because it separates the
//!   conversion cost from the scan cost. It stands in for any image env whose
//!   frames are already floats.
//!
//! Both a fixed overhead and a ns-per-element slope are recoverable from the
//! pair (8 elements vs. 27 648 elements at the same arm); a single number taken
//! at one shape would not transfer.
//!
//! # Control 1 -- poisoned rows
//!
//! Every scanning arm runs twice, over an all-finite fixture and over a
//! **poisoned** one, so no reported number is an artifact of a branch predictor
//! that has only ever seen the clean path. The poison is placed in the **last**
//! element of the row, which is the honest worst case: `Iterator::all`
//! short-circuits, so poisoning element 0 would measure a one-element scan.
//! In the **staging** arms only one row of the batch is poisoned, and the
//! poison is its last element, so the clean and poisoned cells perform
//! essentially identical work and differ only in one mispredictable branch --
//! the intended control. In the **ingestion** arms `obs` and `next_obs` share
//! one scratch buffer, so a poison at the end of `obs` lands at the halfway
//! point of that buffer and the short-circuiting spelling exits there,
//! reporting *faster* than clean; see [`check_finite_reused`]. The
//! `*_branchless` cells have no early exit and are therefore the arms whose
//! clean-vs-poisoned agreement actually demonstrates data independence.
//!
//! `CarRacingObservation` is exempt: its payload is `u8`, and no `u8` maps to a
//! non-finite `f32`. That exemption is itself a finding, reported as such --
//! only the `f32`-backed shapes ([`F32Image`], `LunarLanderObservation`) carry a
//! `poisoned` arm.
//!
//! # Control 2 -- backend and machine
//!
//! - The `obsguard_stage_*` and `obsguard_ingest_*` arms touch **no device at all**. They are
//!   pure host code (`Vec<f32>` writes and scans, a `VecDeque` push); the
//!   backend type parameter never reaches a kernel. Their group names carry the
//!   `host` label for that reason.
//! - The **denominators** do touch a device. `obsguard_learn_step_*` runs on
//!   both `flex` (CPU) and `wgpu`; `obsguard_env_cycle_*` is host-only (rapier2d
//!   physics plus, for `CarRacing`, a CPU rasterizer).
//! - On this machine `wgpu` resolves to **Metal on an Apple M2 Pro** (Darwin
//!   aarch64) -- see `support/bench_backend.rs`. A `wgpu` figure here does not
//!   transfer to CUDA, to CI (no GPU), or to any other adapter. Report every
//!   number with its backend label.
//!
//! # Allocation vs. scan
//!
//! The checked-ingestion arm ships in two variants so the two costs can be told
//! apart at the image shape, where a per-step scratch `Vec` is ~221 KB
//! (2 x 27 648 x 4 bytes):
//!
//! - `scan_reused` -- one scratch `Vec<f32>` allocated once and `clear()`ed per
//!   call (what a real guard would thread through the agent).
//! - `scan_fresh_alloc` -- `Vec::with_capacity(2 * row_len)` per call (what a
//!   naive `fn check(obs: &O) -> bool` helper would do).
//!
//! # The predicate, not the memory, is the cost
//!
//! The first run of this bench refuted an assumption built into the arm design
//! itself: the obvious `slice.iter().all(|v| v.is_finite())` is not
//! memory-bound and does not "ride along" with anything. At the image shape it
//! measured ~8x the cost of the `write_host_row` memcpy it fuses into, because
//! `Iterator::all` must short-circuit and therefore cannot be vectorized, while
//! the memcpy it follows is fully vectorized.
//!
//! Every scanning arm is therefore run in two spellings --
//! `fused_scan` / `scan_reused` (the obvious `all(is_finite)`) and
//! `fused_branchless` / `scan_reused_branchless`
//! ([`scan_finite_branchless`], a `max` reduction over the IEEE-754 exponent
//! field with no early exit). They return bit-identical verdicts, asserted
//! outside the timed region. Any conclusion drawn from the `all(is_finite)`
//! numbers alone would be a conclusion about `Iterator::all`, not about the
//! guard.
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo-reinforcement-learning --bench obs_guard_bench
//! ```

#[path = "support/bench_backend.rs"]
mod bench_backend;

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::backend::{Autodiff, Flex, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{Tensor, activation};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::{ContinuousAction, DiscreteAction};
use rlevo_core::base::{Action, HostRow, Observation, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::box2d::car_racing::{
    CarRacing, CarRacingAction, CarRacingConfig, CarRacingObservation,
};
use rlevo_environments::box2d::lunar_lander::{
    LunarLanderConfig, LunarLanderDiscrete, LunarLanderDiscreteAction, LunarLanderObservation,
};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

use bench_backend::BenchBackend;

// ---------------------------------------------------------------------------
// Shapes and sweep parameters
// ---------------------------------------------------------------------------

/// Frame side of the image-shaped observation -- matches `CarRacing`'s
/// rasterizer (`FRAME_SIZE`), which is private to `rlevo-environments`.
const FRAME: usize = 96;
/// Element count of one image row: 96 x 96 x 3 = 27 648.
const IMG_LEN: usize = FRAME * FRAME * 3;
/// Element count of one `LunarLanderObservation` row.
const LUNAR_LEN: usize = 8;

/// Minibatch sizes swept for the small feature-vector shape.
const VEC_BATCHES: [usize; 2] = [64, 256];
/// Minibatch sizes swept for the image shape. Smaller than [`VEC_BATCHES`]
/// because a 96x96x3 conv forward+backward at 256 on the CPU backend takes
/// long enough to make the sweep impractical, and the ratio of interest is
/// already resolved at 32/64.
const IMG_BATCHES: [usize; 2] = [32, 64];

/// Discrete action count used by every bench agent below.
const ACTIONS: usize = 4;
/// Replay capacity for the ingestion arms. Large enough that the buffer is at
/// capacity (steady-state push+evict, the realistic case) after prefill, small
/// enough that the image fixtures stay cheap -- every stored observation shares
/// one of [`POOL`] `Arc`s, so the resident bytes are `POOL` frames, not
/// `capacity` frames.
const INGEST_CAPACITY: usize = 4_096;
/// Number of distinct fixture observations cycled through the ingestion arms.
/// A power of two so the index wrap is a mask, and >1 so the arms are not
/// measuring a single perfectly-cached row.
const POOL: usize = 8;

// ---------------------------------------------------------------------------
// Bench-local observation: an f32-backed image of the CarRacing shape
// ---------------------------------------------------------------------------

/// A 96x96x3 `f32`-backed observation, `Arc`-shared exactly like
/// `CarRacingObservation`.
///
/// Exists for two reasons the real `CarRacingObservation` cannot serve:
/// its payload **can** be poisoned with a `NaN` (a `u8` payload cannot), and
/// its `write_host_row` is a straight `extend_from_slice` with no `u8 -> f32`
/// conversion, which separates the conversion cost from the scan cost.
#[derive(Clone)]
struct F32Image {
    px: Arc<Vec<f32>>,
}

// The payload is 27 648 floats; the derived `Debug` would be unreadable and
// would make any bench assertion failure unprintable in practice.
impl std::fmt::Debug for F32Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "F32Image({FRAME}x{FRAME}x3)")
    }
}

impl Observation<3> for F32Image {
    fn shape() -> [usize; 3] {
        [FRAME, FRAME, 3]
    }
}

impl HostRow<3> for F32Image {
    fn row_shape() -> [usize; 3] {
        [FRAME, FRAME, 3]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.px);
    }
}

impl<B: Backend> TensorConvertible<3, B> for F32Image {
    fn from_tensor(tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [FRAME, FRAME, 3] {
            return Err(TensorConversionError {
                message: format!("expected shape [{FRAME}, {FRAME}, 3], got {dims:?}"),
            });
        }
        let flat: Vec<f32> =
            tensor
                .into_data()
                .into_vec::<f32>()
                .map_err(|e| TensorConversionError {
                    message: format!("failed to read tensor data: {e:?}"),
                })?;
        Ok(Self { px: Arc::new(flat) })
    }
}

// ---------------------------------------------------------------------------
// Bench-local discrete action for the image-shaped agents
// ---------------------------------------------------------------------------

/// A 4-way discrete action, used only so a `DqnAgent` can be built over the
/// image-shaped observations. `CarRacing`'s real action space is continuous,
/// but nothing on the paths under test (`remember`, batch staging) reads the
/// action beyond `to_index()`, so the action type is not a variable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BenchAction(u8);

impl Action<1> for BenchAction {
    fn shape() -> [usize; 1] {
        [ACTIONS]
    }
    fn is_valid(&self) -> bool {
        (self.0 as usize) < ACTIONS
    }
}

impl DiscreteAction<1> for BenchAction {
    const ACTION_COUNT: usize = ACTIONS;

    /// # Panics
    ///
    /// Panics if `index >= ACTIONS`.
    fn from_index(index: usize) -> Self {
        assert!(index < ACTIONS, "BenchAction index out of range: {index}");
        // Bounded by `ACTIONS` (4) via the assertion directly above.
        #[allow(clippy::cast_possible_truncation)]
        Self(index as u8)
    }

    fn to_index(&self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Builds a `LunarLanderObservation`. When `poison`, the **last** element is
/// `NaN` (see the module doc's control 1: last, not first, so the scan is not
/// short-circuited away).
fn lunar_obs(rng: &mut StdRng, poison: bool) -> LunarLanderObservation {
    let mut values = [0.0f32; LUNAR_LEN];
    for v in &mut values {
        *v = rng.random_range(-1.5f32..1.5f32);
    }
    if poison {
        values[LUNAR_LEN - 1] = f32::NAN;
    }
    LunarLanderObservation::new(values)
}

/// Builds an [`F32Image`]. When `poison`, the last element is `NaN`.
fn f32_image(rng: &mut StdRng, poison: bool) -> F32Image {
    let mut px: Vec<f32> = Vec::with_capacity(IMG_LEN);
    for _ in 0..IMG_LEN {
        px.push(rng.random_range(0.0f32..1.0f32));
    }
    if poison {
        px[IMG_LEN - 1] = f32::NAN;
    }
    F32Image { px: Arc::new(px) }
}

/// Builds a `CarRacingObservation`. `poison` is accepted for signature parity
/// with the other fixture builders and is **ignored**: the payload is `u8`, and
/// no `u8` maps to a non-finite `f32`. See the module doc's control 1.
fn car_obs(rng: &mut StdRng, _poison: bool) -> CarRacingObservation {
    // Heap-allocated: a `Box::new([0u8; IMG_LEN])` literal would build the
    // 27 648-byte array on the stack first (clippy::large_stack_arrays).
    let mut px: Vec<u8> = vec![0u8; IMG_LEN];
    for b in &mut px {
        *b = rng.random_range(0u8..=255u8);
    }
    let boxed: Box<[u8; IMG_LEN]> = px
        .into_boxed_slice()
        .try_into()
        .expect("IMG_LEN elements by construction");
    CarRacingObservation::from_boxed(boxed)
}

// ---------------------------------------------------------------------------
// Arm 1 / arm 2 -- batch staging, with and without the fused scan
// ---------------------------------------------------------------------------

/// **Arm 1.** The staging loop exactly as `dqn_agent.rs::learn_step` runs it
/// today (lines 530-539), reduced to the observation half: for each sampled
/// row, `write_host_row` into a shared flat buffer.
fn stage_baseline<const R: usize, T: HostRow<R>>(rows: &[T], buf: &mut Vec<f32>) {
    buf.clear();
    for row in rows {
        row.write_host_row(buf);
    }
}

/// **Arm 2.** Arm 1 plus a finiteness scan of `buf[start..]` after each
/// `write_host_row` -- the fused variant, a second pass over data the previous
/// line just wrote and that is therefore still L1-resident.
///
/// The per-row verdicts are combined with a non-short-circuiting `&=` so a
/// poisoned row does not abandon the rest of the batch; the intra-row `all`
/// does short-circuit, which is why the poison sits in the last element.
fn stage_fused_scan<const R: usize, T: HostRow<R>>(rows: &[T], buf: &mut Vec<f32>) -> bool {
    buf.clear();
    let mut all_finite = true;
    for row in rows {
        let start = buf.len();
        row.write_host_row(buf);
        all_finite &= buf[start..].iter().all(|v| v.is_finite());
    }
    all_finite
}

/// IEEE-754 binary32 exponent field. A value is non-finite (`NaN` or `+-Inf`)
/// **iff** every exponent bit is set.
const EXP_MASK: u32 = 0x7F80_0000;

/// Branchless, auto-vectorizable finiteness scan of `slice`.
///
/// Added after the first measurement showed the obvious
/// `slice.iter().all(|v| v.is_finite())` running ~8x *slower* than the
/// `write_host_row` memcpy it was supposed to be riding along with -- the scan
/// is not memory-bound at all, it is bound by a short-circuiting scalar
/// predicate LLVM cannot vectorize (`Iterator::all` must stop at the first
/// failure, so it cannot be turned into a horizontal reduction).
///
/// Reformulating it as a `max` reduction over the masked exponent field removes
/// the early exit, so LLVM emits a NEON reduction instead: `acc` ends at
/// [`EXP_MASK`] exactly when some element had an all-ones exponent. The result
/// is bit-exact with `all(is_finite)`, only without the data-dependent branch.
fn scan_finite_branchless(slice: &[f32]) -> bool {
    let mut acc: u32 = 0;
    for &v in slice {
        acc = acc.max(v.to_bits() & EXP_MASK);
    }
    acc != EXP_MASK
}

/// **Arm 2b.** Arm 2 with [`scan_finite_branchless`] in place of
/// `all(is_finite)`. Same verdict, same fusion point, no early exit.
fn stage_fused_scan_branchless<const R: usize, T: HostRow<R>>(
    rows: &[T],
    buf: &mut Vec<f32>,
) -> bool {
    buf.clear();
    let mut all_finite = true;
    for row in rows {
        let start = buf.len();
        row.write_host_row(buf);
        all_finite &= scan_finite_branchless(&buf[start..]);
    }
    all_finite
}

// ---------------------------------------------------------------------------
// Arm 3 / arm 4 -- the ingestion (`remember`) path
// ---------------------------------------------------------------------------

/// Which ingestion variant a bench cell runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ingest {
    /// **Arm 3.** `remember` as it is today: the scalar `FiniteRewardGuard`,
    /// then a typed push into the replay buffer. No observation traversal.
    Baseline,
    /// **Arm 4a.** Arm 3 plus a finiteness check over `obs` and `next_obs`,
    /// staged into a scratch `Vec<f32>` **reused across calls**.
    ScanReused,
    /// **Arm 4b.** Arm 3 plus the same check, but with a fresh
    /// `Vec::with_capacity(2 * row_len)` per call -- the shape a naive
    /// standalone helper would take. At the image shape that is a ~221 KB
    /// allocation per environment step.
    ScanFreshAlloc,
    /// **Arm 4c.** [`Ingest::ScanReused`] with [`scan_finite_branchless`]
    /// instead of `all(is_finite)` -- the cheapest correct form of the check
    /// found here. See [`scan_finite_branchless`] for why the obvious spelling
    /// is the slow one.
    ScanReusedBranchless,
}

impl Ingest {
    const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ScanReused => "scan_reused",
            Self::ScanFreshAlloc => "scan_fresh_alloc",
            Self::ScanReusedBranchless => "scan_reused_branchless",
        }
    }
}

/// The prototype guard, reused-scratch form: stage both observations into
/// `scratch` and scan. This is the cheapest shape a real guard could take at
/// the ingestion seam, because there is no existing traversal to fuse into
/// (module doc, "Ingestion -- no traversal to fuse into").
///
/// # Reading the `_poisoned` cells of this arm
///
/// `obs` and `next_obs` share one scratch buffer, and the fixture's poison sits
/// in the **last element of each row**. So on a poisoned input the poison lands
/// at index `row_len - 1` of a `2 * row_len` buffer and `Iterator::all`
/// short-circuits at the halfway mark: the `_poisoned` cell measures roughly
/// *half* a scan, which is why it reports **faster** than `_clean`. That is a
/// real property of the short-circuiting spelling, not a measurement error --
/// and it is exactly the asymmetry [`check_finite_reused_branchless`] does not
/// have (its clean and poisoned cells agree to within noise, which is what makes
/// it the honest branch-prediction control).
fn check_finite_reused<const R: usize, T: HostRow<R>>(
    obs: &T,
    next_obs: &T,
    scratch: &mut Vec<f32>,
) -> bool {
    scratch.clear();
    obs.write_host_row(scratch);
    next_obs.write_host_row(scratch);
    scratch.iter().all(|v| v.is_finite())
}

/// The prototype guard, fresh-allocation form -- identical work plus one
/// `Vec` allocation and one deallocation per call.
fn check_finite_fresh<const R: usize, T: HostRow<R>>(obs: &T, next_obs: &T) -> bool {
    let row_len: usize = T::row_shape().iter().product();
    let mut scratch: Vec<f32> = Vec::with_capacity(2 * row_len);
    obs.write_host_row(&mut scratch);
    next_obs.write_host_row(&mut scratch);
    scratch.iter().all(|v| v.is_finite())
}

/// [`check_finite_reused`] with the branchless scan.
fn check_finite_reused_branchless<const R: usize, T: HostRow<R>>(
    obs: &T,
    next_obs: &T,
    scratch: &mut Vec<f32>,
) -> bool {
    scratch.clear();
    obs.write_host_row(scratch);
    next_obs.write_host_row(scratch);
    scan_finite_branchless(scratch)
}

// ---------------------------------------------------------------------------
// Networks
// ---------------------------------------------------------------------------

/// `8 -> 64 -> 64 -> 4` MLP for the `LunarLanderObservation` agents.
#[derive(Module, Debug)]
struct VecMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> VecMlp<B> {
    fn new(device: &<B as BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(LUNAR_LEN, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, ACTIONS).init(device),
        }
    }

    fn forward_impl(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(x));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for VecMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(obs)
    }
    // `tau` is an f64 hyperparameter; every tensor in this crate is f32. This is
    // the intended narrowing point.
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, VecMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

/// Fixed pooled spatial size the conv head is sized against.
const POOL_SIZE: usize = 4;
/// Output channels of the second conv layer.
const CONV2_CHANNELS: usize = 32;

/// Conv net over a `[96, 96, 3]` frame -- the same stack shape
/// `learn_step_bench.rs`'s `PixelConvNet` uses, so the two files' image
/// denominators are comparable.
#[derive(Module, Debug)]
struct ImgConvNet<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> ImgConvNet<B> {
    fn new(device: &<B as BackendTypes>::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([16, CONV2_CHANNELS], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool: AdaptiveAvgPool2dConfig::new([POOL_SIZE, POOL_SIZE]).init(),
            fc1: LinearConfig::new(CONV2_CHANNELS * POOL_SIZE * POOL_SIZE, 128).init(device),
            fc2: LinearConfig::new(128, ACTIONS).init(device),
        }
    }

    // b/c/h/w are the canonical NCHW dimension names.
    #[allow(clippy::many_single_char_names)]
    fn forward_impl(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
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

impl<B: AutodiffBackend> DqnModel<B, 4> for ImgConvNet<B> {
    fn forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        self.forward_impl(obs)
    }
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, 4>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(obs)
    }
    // `tau` is an f64 hyperparameter; every tensor in this crate is f32. This is
    // the intended narrowing point.
    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, ImgConvNet<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// Agent construction
// ---------------------------------------------------------------------------

/// Builds a `DqnAgent` whose replay buffer is prefilled to [`INGEST_CAPACITY`]
/// (so the ingestion arms measure steady-state push+evict) and whose
/// `learning_starts` is 0 (so `learn_step` runs immediately).
fn build_agent<B, M, O, const DO: usize, const DB: usize>(
    device: &B::Device,
    model: M,
    batch_size: usize,
    capacity: usize,
    fixture: &[O],
) -> DqnAgent<B, M, O, BenchAction, DO, DB>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend> + Clone,
{
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(batch_size)
        .replay_buffer_capacity(capacity)
        .learning_starts(0)
        .train_frequency(1)
        .build()
        .expect("valid config");
    let mut agent: DqnAgent<B, M, O, BenchAction, DO, DB> =
        DqnAgent::new(model, config, device.clone()).expect("valid config");
    for i in 0..capacity {
        let obs = fixture[i % fixture.len()].clone();
        let next = fixture[(i + 1) % fixture.len()].clone();
        let action = BenchAction::from_index(i % ACTIONS);
        agent.remember(obs, &action, 1.0, next, i % 97 == 0);
    }
    agent
}

// ---------------------------------------------------------------------------
// Bench driver -- arms 1 and 2 (host-only staging)
// ---------------------------------------------------------------------------

/// Runs `baseline` vs. `fused_scan` (clean and, where the payload permits,
/// poisoned) over the given batch sizes. Pure host code -- no device.
fn bench_stage<const R: usize, T>(
    c: &mut Criterion,
    group_name: &str,
    batches: &[usize],
    poisonable: bool,
    make: impl Fn(&mut StdRng, bool) -> T,
) where
    T: HostRow<R>,
{
    let mut rng = StdRng::seed_from_u64(1043);
    let row_len: usize = T::row_shape().iter().product();
    let mut group = c.benchmark_group(group_name);

    for &batch in batches {
        let clean: Vec<T> = (0..batch).map(|_| make(&mut rng, false)).collect();
        let mut buf: Vec<f32> = Vec::with_capacity(batch * row_len);

        // Correctness gate, once per cell and outside every timed region: the
        // branchless scan must return exactly what `all(is_finite)` returns, on
        // both a clean and a poisoned batch. If it does not, the timing
        // comparison below is comparing two different predicates.
        {
            let mut gate: Vec<f32> = Vec::with_capacity(batch * row_len);
            assert_eq!(
                stage_fused_scan::<R, T>(&clean, &mut gate),
                stage_fused_scan_branchless::<R, T>(&clean, &mut gate),
                "branchless scan disagreed with all(is_finite) on a clean batch"
            );
            if poisonable {
                let mut dirty: Vec<T> = (0..batch).map(|_| make(&mut rng, false)).collect();
                dirty[batch / 2] = make(&mut rng, true);
                assert!(
                    !stage_fused_scan::<R, T>(&dirty, &mut gate),
                    "all(is_finite) failed to detect the poisoned row"
                );
                assert!(
                    !stage_fused_scan_branchless::<R, T>(&dirty, &mut gate),
                    "branchless scan failed to detect the poisoned row"
                );
            }
        }

        group.bench_with_input(BenchmarkId::new("baseline", batch), &clean, |b, rows| {
            b.iter(|| {
                stage_baseline::<R, T>(black_box(rows), &mut buf);
                black_box(buf.len());
            });
        });
        group.bench_with_input(
            BenchmarkId::new("fused_scan_clean", batch),
            &clean,
            |b, rows| {
                b.iter(|| {
                    let ok = stage_fused_scan::<R, T>(black_box(rows), &mut buf);
                    black_box((ok, buf.len()));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fused_branchless_clean", batch),
            &clean,
            |b, rows| {
                b.iter(|| {
                    let ok = stage_fused_scan_branchless::<R, T>(black_box(rows), &mut buf);
                    black_box((ok, buf.len()));
                });
            },
        );

        if poisonable {
            // One poisoned row per batch, in the middle: a whole batch of
            // poison would be unrepresentative, and one is enough to force the
            // "poison found" branch through the combining `&=`.
            let mut poisoned: Vec<T> = (0..batch).map(|_| make(&mut rng, false)).collect();
            poisoned[batch / 2] = make(&mut rng, true);
            group.bench_with_input(
                BenchmarkId::new("fused_scan_poisoned", batch),
                &poisoned,
                |b, rows| {
                    b.iter(|| {
                        let ok = stage_fused_scan::<R, T>(black_box(rows), &mut buf);
                        black_box((ok, buf.len()));
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("fused_branchless_poisoned", batch),
                &poisoned,
                |b, rows| {
                    b.iter(|| {
                        let ok = stage_fused_scan_branchless::<R, T>(black_box(rows), &mut buf);
                        black_box((ok, buf.len()));
                    });
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench driver -- arms 3 and 4 (host-only ingestion)
// ---------------------------------------------------------------------------

/// Runs the four [`Ingest`] variants against a real `DqnAgent::remember`,
/// clean and (where poisonable) poisoned.
///
/// Every variant clones one fixture observation per call for `obs` and one for
/// `next_obs`, because `remember` takes them by value. That clone is present
/// **identically in all four arms**, so it cancels in the arm-4-minus-arm-3
/// delta the issue asks for; it is noted because it inflates the absolute
/// baseline slightly relative to production, where the observation is moved out
/// of the snapshot rather than cloned.
///
/// The checked arms `black_box` the verdict and push regardless, rather than
/// dropping the transition. Dropping would make the poisoned cell measure a
/// *shorter* code path (no buffer push) than the clean one, confounding the
/// branch-prediction control this arm exists to provide.
fn bench_ingest<B, M, O, const DO: usize, const DB: usize>(
    c: &mut Criterion,
    group_name: &str,
    device: &B::Device,
    make_model: impl Fn(&B::Device) -> M,
    poisonable: bool,
    make: impl Fn(&mut StdRng, bool) -> O,
) where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend> + Clone,
{
    let mut rng = StdRng::seed_from_u64(1043);
    let row_len: usize = <O as HostRow<DO>>::row_shape().iter().product();

    let clean: Vec<O> = (0..POOL).map(|_| make(&mut rng, false)).collect();
    let poisoned: Vec<O> = if poisonable {
        (0..POOL).map(|_| make(&mut rng, true)).collect()
    } else {
        Vec::new()
    };

    let mut group = c.benchmark_group(group_name);
    let mut variants: Vec<(Ingest, bool)> = vec![
        (Ingest::Baseline, false),
        (Ingest::ScanReused, false),
        (Ingest::ScanFreshAlloc, false),
        (Ingest::ScanReusedBranchless, false),
    ];
    if poisonable {
        variants.push((Ingest::ScanReused, true));
        variants.push((Ingest::ScanFreshAlloc, true));
        variants.push((Ingest::ScanReusedBranchless, true));
    }

    for (variant, poison) in variants {
        let fixture: &[O] = if poison { &poisoned } else { &clean };
        let id_name = if poison {
            format!("{}_poisoned", variant.label())
        } else {
            format!("{}_clean", variant.label())
        };
        // `learn_step` is never called here, so `batch_size` is inert; it only
        // has to validate.
        let mut agent =
            build_agent::<B, M, O, DO, DB>(device, make_model(device), 64, INGEST_CAPACITY, &clean);
        let mut scratch: Vec<f32> = Vec::with_capacity(2 * row_len);
        let mut i: usize = 0;
        group.bench_function(BenchmarkId::new(id_name, row_len), |b| {
            b.iter(|| {
                let obs = fixture[i % POOL].clone();
                let next = fixture[(i + 1) % POOL].clone();
                let action = BenchAction::from_index(i % ACTIONS);
                i = i.wrapping_add(1);
                match variant {
                    Ingest::Baseline => {}
                    Ingest::ScanReused => {
                        black_box(check_finite_reused::<DO, O>(&obs, &next, &mut scratch));
                    }
                    Ingest::ScanFreshAlloc => {
                        black_box(check_finite_fresh::<DO, O>(&obs, &next));
                    }
                    Ingest::ScanReusedBranchless => {
                        black_box(check_finite_reused_branchless::<DO, O>(
                            &obs,
                            &next,
                            &mut scratch,
                        ));
                    }
                }
                agent.remember(black_box(obs), &action, 1.0, black_box(next), false);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Denominator 1 -- a full `learn_step` (the real one)
// ---------------------------------------------------------------------------

/// Times `DqnAgent::learn_step` on a prefilled agent -- the denominator for
/// `(arm2 - arm1)`. Calls the production method, not a mirror.
fn bench_learn_step<B, M, O, const DO: usize, const DB: usize>(
    c: &mut Criterion,
    group_name: &str,
    device: &B::Device,
    sample_size: usize,
    batches: &[usize],
    make_model: impl Fn(&B::Device) -> M,
    make: impl Fn(&mut StdRng, bool) -> O,
) where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend> + Clone,
{
    let mut rng = StdRng::seed_from_u64(1043);
    let fixture: Vec<O> = (0..POOL).map(|_| make(&mut rng, false)).collect();

    let mut group = c.benchmark_group(group_name);
    group.sample_size(sample_size);
    for &batch in batches {
        // Capacity 4x the batch so sampling is not degenerate, and small enough
        // that prefill stays quick at the image shape.
        let capacity = batch * 4;
        let mut agent =
            build_agent::<B, M, O, DO, DB>(device, make_model(device), batch, capacity, &fixture);
        group.bench_function(BenchmarkId::new("learn_step", batch), |b| {
            b.iter(|| {
                let out = agent.learn_step(&mut rng).expect("learn_step");
                black_box(out.is_some());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Denominator 2 -- a full env step + `remember` cycle (the real environments)
// ---------------------------------------------------------------------------

/// Times one `env.step(action)` + `agent.remember(...)` against the **real**
/// `LunarLanderDiscrete` -- the denominator for `(arm4 - arm3)` at the small
/// feature-vector shape.
///
/// `iter_custom` is used so episode resets stay **outside** the timed region:
/// `reset()` rebuilds the whole rapier2d world, which would otherwise land as
/// a rare multi-millisecond spike inside a microsecond-scale measurement.
/// `act` is also excluded -- it is a network forward pass, and including it
/// would only enlarge the denominator and flatter the guard.
fn bench_env_cycle_lunar(c: &mut Criterion) {
    type Be = Autodiff<Flex>;
    let device = <Flex as BenchBackend>::device();
    let mut rng = StdRng::seed_from_u64(1043);
    let fixture: Vec<LunarLanderObservation> =
        (0..POOL).map(|_| lunar_obs(&mut rng, false)).collect();
    let mut agent = build_agent::<Be, VecMlp<Be>, LunarLanderObservation, 1, 2>(
        &device,
        VecMlp::new(&device),
        64,
        INGEST_CAPACITY,
        &fixture,
    );
    let mut env =
        LunarLanderDiscrete::with_config(LunarLanderConfig::default()).expect("valid config");
    let mut snap = env.reset().expect("reset");
    let mut done = false;
    let mut i: usize = 0;

    c.bench_function("obsguard_env_cycle_lunar_host/step_plus_remember", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                if done {
                    snap = env.reset().expect("reset");
                    done = false;
                }
                let obs = snap.observation().clone();
                let env_action = LunarLanderDiscreteAction::from_index(i % ACTIONS);
                let action = BenchAction::from_index(i % ACTIONS);
                i = i.wrapping_add(1);

                let t0 = Instant::now();
                let next = env.step(env_action).expect("step");
                let reward: f32 = (*next.reward()).into();
                let terminated = next.is_terminated();
                agent.remember(obs, &action, reward, next.observation().clone(), terminated);
                total += t0.elapsed();

                done = next.is_done();
                snap = next;
            }
            total
        });
    });
}

/// Same measurement against the real `CarRacing` (96x96x3 rasterized frames) --
/// the denominator for `(arm4 - arm3)` at the image shape. Resets regenerate a
/// procedural track and are excluded from the timed region for the same reason
/// as in [`bench_env_cycle_lunar`].
fn bench_env_cycle_car(c: &mut Criterion) {
    type Be = Autodiff<Flex>;
    let device = <Flex as BenchBackend>::device();
    let mut rng = StdRng::seed_from_u64(1043);
    let fixture: Vec<CarRacingObservation> = (0..POOL).map(|_| car_obs(&mut rng, false)).collect();
    let mut agent = build_agent::<Be, ImgConvNet<Be>, CarRacingObservation, 3, 4>(
        &device,
        ImgConvNet::new(&device),
        32,
        INGEST_CAPACITY,
        &fixture,
    );
    let mut env = CarRacing::with_config(CarRacingConfig::default()).expect("valid config");
    let mut snap = env.reset().expect("reset");
    let mut done = false;
    let mut i: usize = 0;

    let mut group = c.benchmark_group("obsguard_env_cycle_car_host");
    // Each iteration rasterizes a 96x96 frame; 30 samples keeps the cell inside
    // a few seconds while still yielding a real confidence interval.
    group.sample_size(30);
    group.bench_function("step_plus_remember", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                if done {
                    snap = env.reset().expect("reset");
                    done = false;
                }
                let obs = snap.observation().clone();
                let env_action = CarRacingAction::from_slice(&[0.0, 0.4, 0.0]);
                let action = BenchAction::from_index(i % ACTIONS);
                i = i.wrapping_add(1);

                let t0 = Instant::now();
                let next = env.step(env_action).expect("step");
                let reward: f32 = (*next.reward()).into();
                let terminated = next.is_terminated();
                agent.remember(obs, &action, reward, next.observation().clone(), terminated);
                total += t0.elapsed();

                done = next.is_done();
                snap = next;
            }
            total
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

fn bench_staging_arms(c: &mut Criterion) {
    bench_stage::<1, LunarLanderObservation>(
        c,
        "obsguard_stage_lunar_host",
        &VEC_BATCHES,
        true,
        lunar_obs,
    );
    bench_stage::<3, CarRacingObservation>(
        c,
        "obsguard_stage_car_host",
        &IMG_BATCHES,
        false,
        car_obs,
    );
    bench_stage::<3, F32Image>(
        c,
        "obsguard_stage_imagef32_host",
        &IMG_BATCHES,
        true,
        f32_image,
    );
}

fn bench_ingest_arms(c: &mut Criterion) {
    type Be = Autodiff<Flex>;
    let device = <Flex as BenchBackend>::device();
    bench_ingest::<Be, VecMlp<Be>, LunarLanderObservation, 1, 2>(
        c,
        "obsguard_ingest_lunar_host",
        &device,
        VecMlp::new,
        true,
        lunar_obs,
    );
    bench_ingest::<Be, ImgConvNet<Be>, CarRacingObservation, 3, 4>(
        c,
        "obsguard_ingest_car_host",
        &device,
        ImgConvNet::new,
        false,
        car_obs,
    );
    bench_ingest::<Be, ImgConvNet<Be>, F32Image, 3, 4>(
        c,
        "obsguard_ingest_imagef32_host",
        &device,
        ImgConvNet::new,
        true,
        f32_image,
    );
}

fn bench_learn_step_denominators(c: &mut Criterion) {
    let flex = <Flex as BenchBackend>::device();
    bench_learn_step::<Autodiff<Flex>, VecMlp<Autodiff<Flex>>, LunarLanderObservation, 1, 2>(
        c,
        "obsguard_learn_step_lunar_flex",
        &flex,
        100,
        &VEC_BATCHES,
        VecMlp::new,
        lunar_obs,
    );
    let wgpu = <Wgpu as BenchBackend>::device();
    bench_learn_step::<Autodiff<Wgpu>, VecMlp<Autodiff<Wgpu>>, LunarLanderObservation, 1, 2>(
        c,
        "obsguard_learn_step_lunar_wgpu",
        &wgpu,
        20,
        &VEC_BATCHES,
        VecMlp::new,
        lunar_obs,
    );
    bench_learn_step::<Autodiff<Flex>, ImgConvNet<Autodiff<Flex>>, CarRacingObservation, 3, 4>(
        c,
        "obsguard_learn_step_car_flex",
        &flex,
        20,
        &IMG_BATCHES,
        ImgConvNet::new,
        car_obs,
    );
    bench_learn_step::<Autodiff<Wgpu>, ImgConvNet<Autodiff<Wgpu>>, CarRacingObservation, 3, 4>(
        c,
        "obsguard_learn_step_car_wgpu",
        &wgpu,
        20,
        &IMG_BATCHES,
        ImgConvNet::new,
        car_obs,
    );
}

fn bench_env_cycles(c: &mut Criterion) {
    bench_env_cycle_lunar(c);
    bench_env_cycle_car(c);
}

criterion_group!(
    benches,
    bench_staging_arms,
    bench_ingest_arms,
    bench_learn_step_denominators,
    bench_env_cycles
);
criterion_main!(benches);
