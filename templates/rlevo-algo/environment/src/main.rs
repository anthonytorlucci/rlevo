//! Runs your environment through rlevo's evaluation harness: a random
//! agent, and the shipped DQN trained on it, printed as a Markdown table.
//!
//! Both rows run under the one configuration in `config`, so they are
//! comparable: the same seeds, the same training budget, the same
//! evaluation. Change it in one place.

mod env;

use rlevo::burn::tensor::Device;
use rlevo::envs::wrappers::TimeLimit;
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::harness::metrics::core::{EPISODE_LENGTH_MEAN, RETURN_MEAN};
use rlevo::harness::report::BenchmarkReport;
use rlevo::harness::reporter::logging::LoggingReporter;
use rlevo::harness::suite::Suite;
use rlevo::prelude::{Agent, DiscreteAction, Observation, Rng, RngExt};
use rlevo::rl::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo::rl::algorithms::dqn::dqn_config::DqnTrainingConfig;
use rlevo::rl::algorithms::dqn::dqn_model::{MlpQNetwork, MlpQNetworkConfig};

use env::{Cell, Move, MyEnv};

/// The name the environment is registered under.
const ENV_ID: &str = "my-env";
/// Steps after which `TimeLimit` ends an episode, `Truncated`.
const TIME_LIMIT: usize = 30;

/// The configuration every row runs under.
fn config() -> EvaluatorConfig {
    EvaluatorConfig {
        // Environment steps each agent trains for before it is measured.
        train_steps: 5_000,
        // Episodes each agent is measured over, after training.
        num_episodes: 20,
        // Independent trials per agent, each from its own seed.
        num_trials_per_env: 1,
        // The harness's own cap on an episode, above the time limit.
        max_steps: 1_000,
        base_seed: 2026,
        // One thread keeps a run repeatable: Burn's CPU backend draws its
        // initial weights from one generator per process.
        num_threads: Some(1),
        ..EvaluatorConfig::default()
    }
}

/// The environment as every trial runs it: seeded by the trial, its
/// episodes cut at the time limit.
fn my_env(seed: u64) -> TimeLimit<MyEnv> {
    TimeLimit::new(MyEnv::with_seed(seed), TIME_LIMIT)
}

/// An agent that picks uniformly at random: the floor every row must clear.
struct Random;

impl<O, A: DiscreteAction<1>> Agent<O, A> for Random {
    fn act(&mut self, _obs: &O, rng: &mut dyn Rng) -> A {
        A::from_index(rng.random_range(0..A::ACTION_COUNT))
    }
}

/// The shipped DQN over its default network, seeded by the trial. It reads
/// the observation through `Cell`'s `HostRow`.
fn dqn(seed: u64) -> DqnAgent<MlpQNetwork, Cell, Move, 1, 2> {
    let device = Device::flex();
    device.seed(seed);
    let network = MlpQNetworkConfig {
        obs_dim: Cell::shape()[0],
        hidden: vec![32, 32],
        num_actions: Move::ACTION_COUNT,
    }
    .try_init(&device)
    .expect("the network's sizes are valid");
    let config = DqnTrainingConfig {
        learning_starts: 500,
        ..DqnTrainingConfig::default()
    };
    DqnAgent::new(network, config, device).expect("the DQN configuration is valid")
}

/// One row of the table: the agent's mean evaluation return and episode
/// length, averaged over its trials.
fn row(name: &str, report: &BenchmarkReport) -> Result<String, String> {
    let mut returns = 0.0;
    let mut lengths = 0.0;
    for trial in &report.trials {
        if trial.errored {
            let why = trial.error_message.as_deref().unwrap_or("no message");
            return Err(format!("{name}: a trial errored: {why}"));
        }
        returns += trial.scalars[RETURN_MEAN];
        lengths += trial.scalars[EPISODE_LENGTH_MEAN];
    }
    #[allow(clippy::cast_precision_loss)]
    let trials = report.trials.len() as f64;
    Ok(format!(
        "| {name} | {:.2} | {:.1} | {} |",
        returns / trials,
        lengths / trials,
        report.trials.len()
    ))
}

fn main() -> Result<(), String> {
    let suite = Suite::new("my-env").with_env(ENV_ID, my_env);
    let evaluator = Evaluator::new(config());
    let mut reporter = LoggingReporter::new();

    println!("| agent | return/mean | episode_length/mean | trials |");
    println!("|---|---:|---:|---:|");
    let random = evaluator.run_suite(&suite, |_seed| Random, &mut reporter);
    println!("{}", row("random", &random)?);
    let shipped = evaluator.run_suite(&suite, dqn, &mut reporter);
    println!("{}", row("DqnAgent", &shipped)?);
    Ok(())
}
