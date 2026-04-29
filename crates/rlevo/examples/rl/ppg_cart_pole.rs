//! Phasic Policy Gradient (PPG, discrete) on CartPole with Burn's ndarray
//! backend.
//!
//! v1 scope note: CartPole parity with PPO is the v1 target. PPG's sample
//! efficiency advantages are specific to Procgen-style CNN encoders +
//! vectorised envs; on CartPole this example should converge *similarly* to
//! PPO (not dramatically faster). Shipping it here validates that PPO's
//! `RolloutBuffer`, `PpoValue`, and `losses::*` abstractions generalise to a
//! second on-policy algorithm.
//!
//! Usage:
//!
//! ```text
//! cargo run -p evorl-rl --release --example ppg_cart_pole -- \
//!     --seed 42 --total-timesteps 50000 --num-steps 128 --log-every 4096 \
//!     --n-iteration 32 --e-aux 6 --beta-clone 1.0
//! ```

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppg::policies::{
    PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppg::ppg_agent::PpgAgent;
use rlevo_reinforcement_learning::algorithms::ppg::ppg_config::PpgConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppg::train::train_discrete;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;

// Same two-hidden-layer value MLP as the PPO CartPole example.
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

type Be = Autodiff<NdArray>;

struct CliArgs {
    seed: u64,
    total_timesteps: usize,
    num_steps: usize,
    log_every: usize,
    n_iteration: usize,
    e_aux: usize,
    beta_clone: f32,
}

fn parse_args() -> CliArgs {
    let mut seed = 42_u64;
    let mut total_timesteps = 50_000_usize;
    let mut num_steps = 128_usize;
    let mut log_every = 4_096_usize;
    let mut n_iteration = 32_usize;
    let mut e_aux = 6_usize;
    let mut beta_clone = 1.0_f32;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).expect("u64"),
            "--total-timesteps" => {
                total_timesteps = args.next().and_then(|v| v.parse().ok()).expect("usize")
            }
            "--num-steps" => num_steps = args.next().and_then(|v| v.parse().ok()).expect("usize"),
            "--log-every" => log_every = args.next().and_then(|v| v.parse().ok()).expect("usize"),
            "--n-iteration" => {
                n_iteration = args.next().and_then(|v| v.parse().ok()).expect("usize")
            }
            "--e-aux" => e_aux = args.next().and_then(|v| v.parse().ok()).expect("usize"),
            "--beta-clone" => beta_clone = args.next().and_then(|v| v.parse().ok()).expect("f32"),
            other => panic!("unknown flag: {other}"),
        }
    }
    CliArgs {
        seed,
        total_timesteps,
        num_steps,
        log_every,
        n_iteration,
        e_aux,
        beta_clone,
    }
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();
    let args = parse_args();
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let base_env = CartPole::with_config(CartPoleConfig {
        seed: args.seed,
        ..CartPoleConfig::default()
    });
    let mut env = TimeLimit::new(base_env, 500);

    let policy: PpgCategoricalPolicyHead<Be> = PpgCategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 64,
        num_actions: 2,
    }
    .init::<Be>(&device);
    let value: ValueMlp<Be> = ValueMlp::new(4, 64, &device);

    let config = PpgConfigBuilder::new()
        .with_ppo(|p| {
            use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
            PpoTrainingConfigBuilder::new()
                .num_envs(1)
                .num_steps(args.num_steps)
                .num_minibatches(4)
                .update_epochs(4)
                .learning_rate(2.5e-4)
                .clip_coef(0.2)
                .entropy_coef(0.01)
                .value_coef(0.5)
                .gamma(0.99)
                .gae_lambda(0.95)
                .anneal_lr(p.anneal_lr)
                .build()
        })
        .n_iteration(args.n_iteration)
        .e_aux(args.e_aux)
        .beta_clone(args.beta_clone)
        .build();

    let total_iterations = args.total_timesteps / config.batch_size().max(1);

    let mut agent: PpgAgent<
        Be,
        PpgCategoricalPolicyHead<Be>,
        ValueMlp<Be>,
        CartPoleObservation,
        1,
        2,
    > = PpgAgent::new(policy, value, config, device, total_iterations);

    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent,
        &mut env,
        &mut rng,
        args.total_timesteps,
        args.log_every,
    )
    .expect("training");

    let avg = agent.stats().avg_score().unwrap_or(0.0);
    println!(
        "ppg_cart_pole: final avg reward over last {} episodes: {avg:.2}",
        agent.stats().recent_history.len()
    );
}
