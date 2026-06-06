//! Regression test for the training-loop "phantom episode" off-by-one.
//!
//! Every algorithm's training loop used to call `env.reset()` on *every*
//! terminal step before checking whether the step budget was exhausted. When
//! the final step of training is also a terminal step, that trailing reset
//! opens a fresh episode the loop never finishes. Driven through a
//! [`RecordingTap`], `reset` writes an `on_episode_start` (a new `.rec` file)
//! that never receives its `on_episode_end`, so the on-disk episode count ends
//! up one *higher* than the manifest's count — surfacing as an
//! `EpisodeCountMismatch` warning when the run is re-opened (and a junk
//! one-frame episode in the report).
//!
//! There are two distinct phantom/dangling-episode paths and this test pins
//! both, exercising the real `dqn::train` loop (the off-policy `for step`
//! shape) so a reintroduction fails here rather than only in an example run:
//!
//! 1. **Ends exactly on a terminal step** (rare). The old loop did a trailing
//!    `reset` after the final `done`, opening a one-frame phantom episode that
//!    was never stepped to completion. Fixed in the training loops by skipping
//!    the reset when the step budget is exhausted.
//! 2. **Ends mid-episode** (the common case — a step budget rarely lands on a
//!    terminal step). The in-flight episode's file is opened by
//!    `on_episode_start` but never closed/counted by `on_episode_end`. Fixed in
//!    `RecordWriter::on_run_end`, which finalises and counts the still-open
//!    episode as a truncated-by-budget episode.
//!
//! In both cases the invariant is the same: after the run,
//! `manifest.episode_count == <files on disk>` and no warning is emitted.

use std::sync::Arc;

use parking_lot::Mutex;

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

use rand::rngs::StdRng;
use rand::SeedableRng;

use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation, CartPoleState};

use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;

use rlevo_benchmarks::record::schema::FamilyPayload;
use rlevo_benchmarks::record::{EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingTap};
use rlevo_benchmarks::report::RecordedRun;

// ---------------------------------------------------------------------------
// Deterministic fixed-length environment.
//
// Every episode terminates after exactly `EP_LEN` steps, regardless of the
// action. This makes episode boundaries independent of the agent's behaviour,
// so we can pick `TOTAL_STEPS` to land the final step exactly on a terminal
// step. It reuses CartPole's observation / action / reward types purely to
// inherit their `Observation` / `Action` / `TensorConvertible` impls — no
// CartPole physics is involved.
// ---------------------------------------------------------------------------

const EP_LEN: usize = 10;

#[derive(Debug)]
struct FixedLenEnv {
    t: usize,
}

impl FixedLenEnv {
    fn new() -> Self {
        Self { t: 0 }
    }

    fn obs() -> CartPoleObservation {
        CartPoleObservation {
            cart_pos: 0.0,
            cart_vel: 0.0,
            pole_angle: 0.0,
            pole_ang_vel: 0.0,
        }
    }
}

impl Environment<1, 1, 1> for FixedLenEnv {
    type StateType = CartPoleState;
    type ObservationType = CartPoleObservation;
    type ActionType = CartPoleAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, CartPoleObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.t = 0;
        Ok(SnapshotBase::running(Self::obs(), ScalarReward::new(1.0)))
    }

    fn step(&mut self, _action: CartPoleAction) -> Result<Self::SnapshotType, EnvironmentError> {
        self.t += 1;
        let reward = ScalarReward::new(1.0);
        if self.t >= EP_LEN {
            Ok(SnapshotBase::terminated(Self::obs(), reward))
        } else {
            Ok(SnapshotBase::running(Self::obs(), reward))
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal Q-network. Learning is irrelevant here — the config keeps the agent
// in its pre-learning window for the whole run — so `soft_update` is an
// identity and never actually fires.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct DqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> DqnMlp<B> {
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 32).init(device),
            l2: LinearConfig::new(32, 2).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(observations));
        self.l2.forward(x)
    }
}

impl<B: AutodiffBackend> DqnModel<B, 2> for DqnMlp<B> {
    fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_impl(observations)
    }

    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        inner.forward_impl(observations)
    }

    fn soft_update(_active: &Self, target: Self::InnerModule, _tau: f64) -> Self::InnerModule {
        // Never reached: `target_update_frequency` exceeds `TOTAL_STEPS`.
        target
    }
}

type Be = Autodiff<Flex>;

/// Drive the real `dqn::train` loop against [`FixedLenEnv`] through a
/// [`RecordingTap`] for `total_steps`, finalise the run, and re-open it.
/// Returns the reloaded run so each test can assert on counts / warnings.
fn run_recording(total_steps: usize) -> RecordedRun {
    // Flex's process-global RNG is happier pinned to one rayon thread; this
    // test does no learning, but the convention keeps it deterministic.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    let seed: u64 = 42;

    // --- recording sink ---------------------------------------------------
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = RecordWriter::open(temp.path(), RecordingConfig::new(EnvFamily::Classic, seed))
        .expect("open writer");
    let run_dir = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template().with_algorithm("dqn");
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    // --- env + agent ------------------------------------------------------
    // Headless `Ascii` payload: the count invariant is independent of frame
    // contents, so we avoid wiring a family-specific payload source.
    let mut env: RecordingTap<FixedLenEnv, 1, 1, 1> =
        RecordingTap::new_headless(FixedLenEnv::new(), sink.clone(), |_| FamilyPayload::Ascii);

    let device = Default::default();
    <Be as Backend>::seed(&device, seed);
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(8)
        // Keep the whole run in the pre-learning window: no gradient steps,
        // no target sync — we are testing the loop's episode bookkeeping only.
        .learning_starts(total_steps + 1)
        .target_update_frequency(total_steps + 1)
        .replay_buffer_capacity(1_000)
        .build();
    let model: DqnMlp<Be> = DqnMlp::new(&device);
    let mut agent: DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2> =
        DqnAgent::new(model, config, device);

    let mut rng = StdRng::seed_from_u64(seed);
    train(&mut agent, &mut env, &mut rng, total_steps, 0).expect("training loop");

    // --- finalise ---------------------------------------------------------
    sink.lock().on_run_end(manifest);
    assert!(
        sink.lock().take_error().is_none(),
        "recording write error during run"
    );
    drop(env);
    drop(sink);

    // The tempdir must outlive the re-open, so eagerly read everything the
    // tests need before it drops at end of scope.
    let run = RecordedRun::open(&run_dir).expect("re-open recorded run");
    // `RecordedRun` holds the dir path but reads episode files lazily for
    // frame scrubbing; the count/warning surface used by these tests is
    // computed eagerly at `open`, so dropping `temp` after this is safe.
    drop(temp);
    run
}

/// Assert the count invariant: manifest count equals files on disk, equals the
/// expected episode total, and no warning surfaced.
fn assert_clean(run: &RecordedRun, expected_episodes: usize) {
    let manifest_count = run.manifest().episode_count;
    let found_count = u32::try_from(run.episodes().len()).expect("episode count fits u32");
    let expected = u32::try_from(expected_episodes).expect("episode count fits u32");

    assert_eq!(
        manifest_count, found_count,
        "manifest episode_count ({manifest_count}) must match files on disk ({found_count})"
    );
    assert_eq!(
        found_count, expected,
        "expected exactly {expected} episodes, got {found_count}"
    );
    assert!(
        run.warnings().is_empty(),
        "expected no open warnings, got: {:?}",
        run.warnings()
    );
}

/// Case 1: the budget lands exactly on a terminal step. The old trailing reset
/// opened a one-frame phantom episode; the loop fix skips it. 50 steps = 5
/// whole episodes, with step 50 being a terminal step.
#[test]
fn recording_loop_ending_on_done_has_no_phantom_episode() {
    let run = run_recording(5 * EP_LEN);
    assert_clean(&run, 5);
}

/// Case 2: the budget ends mid-episode (55 steps = 5 whole episodes + a 5-step
/// fragment). The in-flight episode's file would be left uncounted; the
/// `on_run_end` fix finalises it as a truncated-by-budget episode, so all 6
/// files are counted.
#[test]
fn recording_loop_ending_mid_episode_counts_partial() {
    let run = run_recording(5 * EP_LEN + EP_LEN / 2);
    assert_clean(&run, 6);
}
