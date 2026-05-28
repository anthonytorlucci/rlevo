//! Headless PPO training on [`CartPole`] → static-HTML report that
//! mounts the Leptos/WASM report client with **RL convergence plots**.
//!
//! Unlike [`report_cartpole_with_client`] (which uses a random-action
//! agent under the harness and therefore records zero `MetricSample`s),
//! this example wires the PPO training loop directly so the on-disk
//! record carries the full RL metric stream — `policy_loss` /
//! `value_loss` / `entropy` / `approx_kl` / `clip_frac` — and the
//! convergence panel surfaces them as line charts alongside the derived
//! per-episode reward / length curves.
//!
//! Composition map:
//!
//! ```text
//!   CartPole
//!     └─ TimeLimit
//!         └─ RecordingTap   (frame + metric stream to disk)
//!             ↓
//!          train_discrete
//!             ↑ tracing::info!(policy_loss = …, value_loss = …, …)
//!             └─ RecordingLayer (captures canonical RL metrics)
//! ```
//!
//! Two-step build flow:
//!
//! ```bash
//! # 1) Build the WASM client (one-time per code change).
//! cd crates/rlevo-benchmarks-report-client
//! trunk build --release
//!
//! # 2) Run this example. Trains PPO, records frames + metrics, opens
//! #    the run, and emits a single-file index.html.
//! cd ../../  # back to repo root
//! cargo run -p rlevo --example report_ppo_cartpole_with_client \
//!     --features viz-record,viz-report --release
//! ```
//!
//! `--release` matters: PPO is unusable at debug speed.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

use rlevo_benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordingConfig, RecordingLayer, RecordingTap,
};
use rlevo_benchmarks::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};

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
const TOTAL_TIMESTEPS: usize = 12_000;
const NUM_STEPS: usize = 128;
const LOG_EVERY: usize = 1_024;
const EPISODE_TIME_LIMIT: usize = 500;
const HIDDEN: usize = 64;
const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;

const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

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
    let record_cfg = RecordingConfig::new(EnvFamily::Classic, SEED);
    let writer = RecordWriter::open("runs", record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    tracing_subscriber::registry()
        .with(RecordingLayer::new(sink.clone()))
        .try_init()?;

    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);

    let base_env = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    });
    let timed = TimeLimit::new(base_env, EPISODE_TIME_LIMIT);
    let mut env: RecordingTap<_, 1, 1, 1> = RecordingTap::new(timed, sink.clone());

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

    // Finalise the run manifest before we open the directory for emit.
    if let Ok(mut s) = sink.lock() {
        s.on_run_end(manifest);
    }
    drop(env);
    drop(sink);

    let run = RecordedRun::open(&run_dir)?;
    for w in run.warnings() {
        eprintln!("warning: {w:?}");
    }

    let dist = PathBuf::from(CLIENT_DIST);
    let assets = ClientAssets::from_trunk_dist(&dist).map_err(|e| {
        format!(
            "could not load client assets from {}: {e}\n\
             Did you run `trunk build --release` in {} first?",
            dist.display(),
            "crates/rlevo-benchmarks-report-client"
        )
    })?;

    let out = run_dir.join("index.html");
    let outcome = emit_static_html(
        &run,
        &out,
        &EmitConfig {
            client_assets: Some(assets),
            ..EmitConfig::default()
        },
    )?;
    println!(
        "wrote {} ({} episodes, {} bytes{})",
        out.display(),
        outcome.episode_count,
        outcome.bytes_written,
        if outcome.size_warning {
            " — over size budget"
        } else {
            ""
        }
    );
    Ok(())
}
