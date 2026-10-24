//! Headless PPO training on [`CartPole`] → static-HTML report that
//! mounts the Leptos/WASM report client with **RL convergence plots**.
//!
//! This is the canonical "does the viz mean anything?" example: because
//! PPO actually learns, the on-disk record carries a real RL metric
//! stream — `policy_loss` / `value_loss` / `entropy` / `approx_kl` /
//! `clip_frac` — and the convergence panel surfaces them as line charts
//! alongside per-episode reward / length curves that *climb* toward the
//! optimal 500-step policy rather than flatlining at the random floor.
//!
//! Composition map:
//!
//! ```text
//!   CartPole
//!     └─ TimeLimit
//!         └─ RecordingTap   (frame + metric stream to disk)
//!             ↓
//!          train (rlevo_harness)
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
//! cargo run -p rlevo-examples --example report_ppo_cartpole_with_client \
//!     --features viz-report --release
//! ```
//!
//! `--release` matters: PPO is unusable at debug speed.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use rand::SeedableRng;
use rand::rngs::StdRng;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::Device;

use rlevo_harness::evaluator::train;
use rlevo_harness::record::{
    RecordSink, RecordWriter, RecordingConfig, RecordingLayer, RecordingTap,
};
use rlevo_harness::report::{ClientAssets, EmitConfig, RecordedRun, emit_static_html};

use rlevo_environments::classic::cartpole::{CartPole, CartPoleConfig, CartPoleObservation};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    CategoricalPolicyHead, CategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfig;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::{MlpValue, MlpValueConfig};

// Enough timesteps for the policy to climb toward the 500-step ceiling
// rather than plateauing mid-curve. CartPole PPO with `num_steps = 128`
// typically needs ~25–50k steps to solve; 50k leaves headroom while
// staying tolerable for a headless `--release` run.
const TOTAL_TIMESTEPS: usize = 50_000;

const CLIENT_DIST: &str = "crates/rlevo-benchmarks-report-client/dist";

/// Deterministic seed shared by the env, the RNG, and the recording config.
pub const SEED: u64 = 42;
/// Rollout horizon per PPO update.
pub const NUM_STEPS: usize = 128;
/// Episodes are truncated at this length via [`TimeLimit`].
pub const EPISODE_TIME_LIMIT: usize = 500;

const HIDDEN: usize = 64;
const OBS_RANK: usize = 4; // todo! this should be dim. rank = 1
const NUM_ACTIONS: usize = 2;

/// Concrete PPO agent type the examples drive — a categorical policy head
/// over the crate's default value network, `MlpValue`, both with two
/// `hidden`-wide `tanh` layers. Neither is defined here: PPO ships both.
///
/// ```text
/// pub struct PpoAgent<P, V, O, const OR: usize, const BOR: usize>
/// where
///     P: PpoPolicy<BOR>,                           --> CategoricalPolicyHead
///     V: PpoValue<BOR>,                            --> MlpValue
///     O: Observation<OR> + TensorConvertible<OR>,  --> CartPoleObservation
/// { /* private fields */ }
/// ```
///
/// todo! further discuss `CategoricalPolicyHead` and `CartPoleObservation`
///
/// links:
/// - <https://docs.rs/rlevo-reinforcement-learning/latest/rlevo_reinforcement_learning/algorithms/ppo/ppo_agent/struct.PpoAgent.html>
/// - <https://docs.rs/rlevo-environments/latest/rlevo_environments/classic/cartpole/index.html>
pub type CartPoleAgent = PpoAgent<CategoricalPolicyHead, MlpValue, CartPoleObservation, 1, 2>;

/// Builds the base [`CartPole`] wrapped in a [`TimeLimit`]; `main` adds the
/// `RecordingTap` on top of this.
///
/// `CartPole` can theoretically run forever if the agent is perfect, i.e.
/// it learns to oscillate back and forth keeping the pole upright, which
/// would stall training. So it's wrapped in `TimeLimit`, which truncates an
/// episode after `max_steps` or in this case `EPISODE_TIME_LIMIT = 500` steps.
/// This is standard RL practice: **truncation != failure**, it just means "stop
/// after maximum number of steps allowed."
///
/// The `seed = 42` makes the run reproducible.
///
/// todo! document the default values here for reference
///
/// # Panics
///
/// Panics if the hard-coded [`CartPoleConfig`] is rejected as invalid.
#[must_use]
pub fn base_env() -> TimeLimit<CartPole> {
    let base = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    })
    .expect("valid config");
    TimeLimit::new(base, EPISODE_TIME_LIMIT)
}

/// Builds a fresh PPO agent with this example's hyperparameters.
/// `total_timesteps` sizes the learning-rate / clip annealing schedule.
///
/// This wires teh policy + value + hyperparameters into a `PpoAgent`. The
/// hyperparameters are:
///
/// | Hyperparameter | Value | Meaning |
/// |---|---|---|
/// | `num_envs` | 1 | One parallel environment (no vectorization here) |
/// | `num_steps` | 128 | Collect 128 steps of experience before each update |
/// | `num_minibatches` | 4 | Split those 128 steps into 4 mini-batches of 32 |
/// | `update_epochs` | 4 | Reuse the collected data `$4\times$` per rollout (PPO's signature reuse) |
/// | `learning_rate` | 2.5e-4 | Optimizer step size |
/// | `clip_coef` | 0.2 | The PPO clip radius — the heart of the algorithm |
/// | `entropy_coef` | 0.01 | Bonus to keep the policy *exploring* (avoid collapse to one action) |
/// | `value_coef` | 0.5 | Weight of the value-loss term in the total loss |
/// | `gamma` | 0.99 | Discount factor — future rewards matter 99% as much as now |
/// | `gae_lambda` | 0.95 | Bias/variance knob for advantage estimation (GAE) |
///
/// `total_iterations = total_timesteps / batch_size` tells PPO how many updates
///  it'll do total, so it can **anneal** (gradually shrink) the learning rate
/// and clip over training. This is why `total_timesteps` is a parameter rather
/// than a constant.
///
/// todo! discuss the relationship between `total_timesteps` and
/// `TimeLimit` wrapper
///
/// # Panics
///
/// Panics if the hard-coded PPO hyperparameters are rejected as invalid.
#[must_use]
pub fn build_agent(total_timesteps: usize) -> CartPoleAgent {
    let device = Device::flex();

    let policy: CategoricalPolicyHead = CategoricalPolicyHeadConfig {
        obs_dim: OBS_RANK,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .try_init(&device)
    .expect("valid head config");
    let value = MlpValueConfig {
        obs_dim: OBS_RANK,
        hidden: vec![HIDDEN, HIDDEN],
    }
    .try_init(&device)
    .expect("valid value network config");

    let config = PpoTrainingConfig {
        num_envs: 1,
        // One parallel environment (no vectorization here)
        num_steps: NUM_STEPS,
        num_minibatches: 4,
        update_epochs: 4,
        learning_rate: 2.5e-4,
        clip_coef: 0.2,
        entropy_coef: 0.01,
        value_coef: 0.5,
        gamma: 0.99,
        gae_lambda: 0.95,
        // Gradient-norm clip at 0.5. `clip_grad` is the only clipping knob and
        // is `None` (off) by default; without it PPO can take destabilizing
        // steps. NOTE: Burn clips this **per tensor**, not across the global
        // flattened parameter vector, so it is not CleanRL's detail #10
        // global-norm clip. True global-norm clipping would need a manual
        // accumulation pass over every parameter's gradient before the
        // optimizer step, which Burn's hook doesn't provide — an open design
        // question, not a settled plan.
        clip_grad: Some(GradientClippingConfig::Norm(0.5)),
        ..Default::default()
    };

    let total_iterations = total_timesteps / config.batch_size().max(1);

    PpoAgent::new(policy, value, config, device, total_iterations).expect("valid config")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let record_cfg = RecordingConfig::for_env::<CartPole>(SEED);
    let writer = RecordWriter::open_default(record_cfg)?;
    let run_dir: PathBuf = writer.run_dir().to_path_buf();
    // v6 manifest provenance: declare the algorithm + backend so the report
    // tier picks PPO loss panels, the success threshold behind `success_rate`,
    // and build/platform reproducibility metadata (git/rustc/burn, all from
    // `rlevo-harness`' build.rs — `None` outside a checkout).
    let manifest = writer
        .manifest_template()
        .with_algorithm("ppo")
        .with_device("flex")
        .with_num_seeds(1)
        .with_success_threshold(195.0)
        .with_build_provenance();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    tracing_subscriber::registry()
        .with(RecordingLayer::new(sink.clone()))
        .try_init()?;

    let mut rng = StdRng::seed_from_u64(SEED);

    // CartPole → TimeLimit → RecordingTap. No TUI tap here — the metric
    // stream is captured headlessly and replayed by the report client.
    // Structured-only (ADR-0013): record `FamilyPayload::Classic2D` line-art.
    let mut env: RecordingTap<_, 1, 1> =
        RecordingTap::with_classic2d_payload(base_env(), sink.clone());
    let mut agent = build_agent(TOTAL_TIMESTEPS);

    train(&mut env, &mut agent, TOTAL_TIMESTEPS, usize::MAX, &mut rng)?;

    // Finalise the run manifest before we open the directory for emit.
    sink.lock().on_run_end(manifest);

    // Fail loud on a recording write error before building the report.
    if let Some(e) = sink.lock().take_error() {
        return Err(e.into());
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
