//! PPO (continuous, tanh-Gaussian head) on Pendulum with Burn's ndarray backend.
//!
//! Usage:
//!
//! ```text
//! cargo run -p rlevo-reinforcement-learning --release --example ppo_pendulum -- \
//!     --seed 42 --total-timesteps 100000 --num-steps 2048 --log-every 4096
//! ```

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppo::policies::{
    TanhGaussianPolicyHead, TanhGaussianPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;
use rlevo_reinforcement_learning::algorithms::ppo::train::train_continuous;

// ---------------------------------------------------------------------------
// Value network
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ValueMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    head: Linear<B>,
}

impl<B: Backend> ValueMlp<B> {
    fn new(obs_dim: usize, hidden: usize, device: &B::Device) -> Self {
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

// ---------------------------------------------------------------------------
// CLI + main
// ---------------------------------------------------------------------------

type Be = Autodiff<NdArray>;

struct CliArgs {
    seed: u64,
    total_timesteps: usize,
    num_steps: usize,
    log_every: usize,
}

fn parse_args() -> CliArgs {
    let mut seed = 42_u64;
    let mut total_timesteps = 100_000_usize;
    let mut num_steps = 2_048_usize;
    let mut log_every = 4_096_usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).expect("u64"),
            "--total-timesteps" => {
                total_timesteps = args.next().and_then(|v| v.parse().ok()).expect("usize");
            }
            "--num-steps" => num_steps = args.next().and_then(|v| v.parse().ok()).expect("usize"),
            "--log-every" => log_every = args.next().and_then(|v| v.parse().ok()).expect("usize"),
            other => panic!("unknown flag: {other}"),
        }
    }
    CliArgs {
        seed,
        total_timesteps,
        num_steps,
        log_every,
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();
    let args = parse_args();
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let base_env = Pendulum::with_config(PendulumConfig {
        seed: args.seed,
        ..PendulumConfig::default()
    });
    let mut env = TimeLimit::new(base_env, 200);

    // Pendulum: 3-dim obs, 1-dim action, action bounds [-2, 2].
    let policy: TanhGaussianPolicyHead<Be> = TanhGaussianPolicyHeadConfig {
        obs_dim: 3,
        hidden: 64,
        action_dim: 1,
        log_std_init: 0.0,
        action_scale: 2.0,
    }
    .init::<Be>(&device);
    let value: ValueMlp<Be> = ValueMlp::new(3, 64, &device);

    let config = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(args.num_steps)
        .num_minibatches(32)
        .update_epochs(10)
        .learning_rate(3e-4)
        .clip_coef(0.2)
        .entropy_coef(0.0) // CleanRL default for continuous
        .value_coef(0.5)
        .gamma(0.9)
        .gae_lambda(0.95)
        .action_log_std_init(0.0)
        .action_scale(2.0)
        .build();

    let total_iterations = args.total_timesteps / config.batch_size().max(1);

    let mut agent: PpoAgent<
        Be,
        TanhGaussianPolicyHead<Be>,
        ValueMlp<Be>,
        PendulumObservation,
        1,
        2,
    > = PpoAgent::new(policy, value, config, device, total_iterations);

    train_continuous::<Be, _, _, _, _, PendulumAction, _, 1, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        args.total_timesteps,
        args.log_every,
    )
    .expect("training");

    let avg = agent.stats().avg_score().unwrap_or(0.0);
    println!(
        "ppo_pendulum: final avg reward over last {} episodes: {avg:.2}",
        agent.stats().recent_history.len()
    );
}
