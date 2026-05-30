//! PPO + live TUI + on-disk recording on [`CartPole`]. The full
//! env-wrapper + recording composition demonstrated end-to-end.
//!
//! Composition map:
//!
//! ```text
//!   CartPole
//!     └─ TimeLimit
//!         └─ TuiEnvTap   (frames + episode returns to live TUI)
//!             └─ RecordingTap (frame + metric stream to disk)
//!                 ↓
//!              train_discrete
//! ```
//!
//! Tracing layers: `TuiCaptureLayer` (live metric sparklines + log) and
//! `RecordingLayer` (on-disk `MetricSample` stream) installed side-by-side
//! through the same `tracing_subscriber::Registry`.
//!
//! # Run with
//!
//! ```bash
//! cargo run -p rlevo --example record_ppo_cartpole \
//!   --features viz-tui,viz-record --release
//! ```
//!
//! `--release` matters: PPO is unusable at debug speed.

use std::sync::Arc;

use parking_lot::Mutex;

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use rlevo_benchmarks::env_wrappers::TuiEnvTap;
use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingLayer, RecordingTap,
};
use rlevo_benchmarks::tui::{TuiCaptureLayer, TuiConfig, TuiRunner};

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::train_discrete;

const SEED: u64 = 42;
const TOTAL_TIMESTEPS: usize = 20_000;
const NUM_STEPS: usize = 128;
const LOG_EVERY: usize = 1_024;
const EPISODE_TIME_LIMIT: usize = 500;
const HIDDEN: usize = 64;
const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;

#[derive(Module, Debug)]
struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    fn new(
        obs_dim: usize,
        hidden: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward_impl(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = tanh(self.fc1.forward(obs));
        let h = tanh(self.fc2.forward(h));
        self.head.forward(h).squeeze_dim::<1>(1)
    }
}

impl<B: AutodiffBackend> PpoValue<B, 2> for ValueMlp<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_impl(obs)
    }
}

type Be = Autodiff<Flex>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runner = TuiRunner::start(TuiConfig::default().with_env_family(EnvFamily::Classic))?;
    let handle = runner.handle();

    let record_cfg = RecordingConfig::new(EnvFamily::Classic, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    tracing_subscriber::registry()
        .with(TuiCaptureLayer::new(handle.clone()))
        .with(RecordingLayer::new(sink.clone()))
        .try_init()?;

    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);

    let base_env = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    });
    let timed = TimeLimit::new(base_env, EPISODE_TIME_LIMIT);
    let tui_tapped: TuiEnvTap<_, 1, 1, 1> = TuiEnvTap::new(timed, handle);
    // RecordingTap wraps the TuiEnvTap so the same trajectory feeds the
    // live TUI AND the on-disk record from one wrap site.
    let mut env: RecordingTap<_, 1, 1, 1> = RecordingTap::new(tui_tapped, sink.clone());

    let policy: CategoricalPolicyHead<Be> = CategoricalPolicyHeadConfig {
        obs_dim: OBS_DIM,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .init::<Be>(&device);
    let value: ValueMlp<Be> = ValueMlp::new(OBS_DIM, HIDDEN, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(NUM_STEPS)
        .num_minibatches(4)
        .update_epochs(4)
        .learning_rate(2.5e-4)
        .clip_coef(0.2)
        .entropy_coef(0.01)
        .value_coef(0.5)
        .gamma(0.99)
        .gae_lambda(0.95)
        .build();

    let total_iterations = TOTAL_TIMESTEPS / config.batch_size().max(1);

    let mut agent: PpoAgent<
        Be,
        CategoricalPolicyHead<Be>,
        ValueMlp<Be>,
        CartPoleObservation,
        1,
        2,
    > = PpoAgent::new(policy, value, config, device, total_iterations);

    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        TOTAL_TIMESTEPS,
        LOG_EVERY,
    )?;

    // Finalise the run manifest on the shared sink before we tear the
    // TUI down. PPO has no Reporter chain to drive this, so the example
    // calls on_run_end directly.
    sink.lock().on_run_end(manifest);

    // Fail loud if recording hit a write error. `TuiRunner::drop` restores
    // the terminal on the error path.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
    }

    runner.wait_for_keypress()?;
    runner.shutdown()?;
    Ok(())
}
