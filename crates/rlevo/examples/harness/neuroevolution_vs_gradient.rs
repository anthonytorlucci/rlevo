//! Weight-only neuroevolution against gradient-based reinforcement learning,
//! on `CartPole`, through rlevo's evaluation harness.
//!
//! Three learners train one policy shape (two hidden layers of `HIDDEN`
//! units) under one configuration: the same environment, the same seeds, the
//! same number of training steps, the same evaluation episodes. PPO (a policy
//! gradient) and DQN (a value method) learn by backpropagation. The third
//! learner has no gradient at all: CMA-ES searches the policy network's
//! weights directly, and every training episode the harness runs is one
//! fitness evaluation. A random agent is the floor. The program prints one
//! Markdown table.
//!
//! # Weight-only search as an `Agent`
//!
//! There is no weight-only strategy type: weight-only neuroevolution is any
//! flat-genome [`Strategy`] searching a network's flattened weights, with a
//! [`ModuleReshaper`] mapping a genome row to the network and back. The usual
//! driver is an `EvolutionaryLoop` scored by `rlevo::hybrid::RolloutFitness`,
//! which plays its own episodes. Here the harness plays them, so the search
//! spends exactly the training budget the gradient agents spend.
//! [`NeuroevolutionAgent`] asks CMA-ES for a population, acts with one
//! candidate per training episode, scores it by that episode's return, and
//! tells CMA-ES the scores once every candidate has played. It drives `ask`
//! and `tell` itself, so it sanitises the fitness itself, as a loop would; a
//! return is a reward, so there is nothing to negate.
//!
//! Evaluation plays the search distribution's mean, CMA-ES's estimate of the
//! best weights, not the best single candidate: one episode's return is a
//! noisy score, and the best of many noisy scores is a lucky draw.
//!
//! # Running
//!
//! ```text
//! cargo run --release -p rlevo --example neuroevolution_vs_gradient
//! ```
//!
//! Each learner runs `TRIALS` trials, each from its own seed. The table
//! gives the mean over trials and how many trials reached
//! `SOLVED_RETURN`, Gymnasium's threshold for CartPole-v1; a handful of
//! trials is still a small sample, so raise `TRIALS` before reading much
//! into a difference between rows.

use std::time::Instant;

use rlevo::burn::grad_clipping::GradientClippingConfig;
use rlevo::burn::module::Module;
use rlevo::burn::nn::{Linear, LinearConfig};
use rlevo::burn::tensor::{Device, Tensor, TensorData, activation};
use rlevo::core::fitness::Metric;
use rlevo::core::rand::SeedableRng;
use rlevo::core::rand::rngs::StdRng;
use rlevo::envs::classic::{CartPole, CartPoleAction, CartPoleConfig, CartPoleObservation};
use rlevo::envs::wrappers::TimeLimit;
use rlevo::evo::algorithms::cma_es::{CmaEs, CmaEsConfig, CmaEsState};
use rlevo::evo::fitness::sanitize_fitness_tensor;
use rlevo::evo::param_reshaper::ModuleReshaper;
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig};
use rlevo::harness::metrics::core::{EPISODE_LENGTH_MEAN, RETURN_MEAN};
use rlevo::harness::report::BenchmarkReport;
use rlevo::harness::reporter::logging::LoggingReporter;
use rlevo::harness::suite::Suite;
use rlevo::prelude::{
    Agent, DiscreteAction, EpisodeStatus, HostRow, Rng, RngExt, Strategy, Transition,
};
use rlevo::rl::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo::rl::algorithms::dqn::dqn_config::DqnTrainingConfig;
use rlevo::rl::algorithms::dqn::dqn_model::{MlpQNetwork, MlpQNetworkConfig};
use rlevo::rl::algorithms::ppo::policies::{CategoricalPolicyHead, CategoricalPolicyHeadConfig};
use rlevo::rl::algorithms::ppo::ppo_agent::PpoAgent;
use rlevo::rl::algorithms::ppo::ppo_config::PpoTrainingConfig;
use rlevo::rl::algorithms::ppo::ppo_value::{MlpValue, MlpValueConfig};

/// Steps after which `TimeLimit` ends a `CartPole` episode, `Truncated`.
const TIME_LIMIT: usize = 500;
/// Environment steps every learner trains for before it is measured.
const TRAIN_STEPS: usize = 50_000;
/// Width of both hidden layers of every learner's policy network.
const HIDDEN: usize = 16;
/// `CartPole`'s observation width and action count.
const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 2;
/// Independent trials per learner, each from its own seed.
const TRIALS: usize = 5;
/// The mean evaluation return at which a trial counts as solved:
/// Gymnasium's reward threshold for CartPole-v1.
const SOLVED_RETURN: f64 = 475.0;
/// The counter [`NeuroevolutionAgent`] reports its completed generations
/// under.
const GENERATIONS: &str = "neuroevolution/generations";

/// The configuration every row runs under.
fn config() -> EvaluatorConfig {
    EvaluatorConfig {
        train_steps: TRAIN_STEPS,
        // Episodes each learner is measured over, after training.
        num_episodes: 10,
        num_trials_per_env: TRIALS,
        max_steps: TIME_LIMIT,
        base_seed: 2026,
        // One thread keeps a run repeatable: Burn's CPU backend draws its
        // initial weights from one generator per process.
        num_threads: Some(1),
        ..EvaluatorConfig::default()
    }
}

/// `CartPole`, seeded by the trial, with its episodes cut at the time limit.
fn cartpole(seed: u64) -> TimeLimit<CartPole> {
    let config = CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    };
    TimeLimit::new(
        CartPole::with_config(config).expect("the default CartPole configuration is valid"),
        TIME_LIMIT,
    )
}

/// The policy whose weights evolve: `OBS_DIM -> HIDDEN -> HIDDEN ->
/// NUM_ACTIONS`, tanh, acting on the largest output. PPO's policy head and
/// DQN's Q-network below have the same shape.
#[derive(Module, Debug)]
struct PolicyNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

impl PolicyNet {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(OBS_DIM, HIDDEN).init(device),
            l2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            l3: LinearConfig::new(HIDDEN, NUM_ACTIONS).init(device),
        }
    }

    /// The action whose output is largest.
    fn act(&self, obs: &CartPoleObservation) -> CartPoleAction {
        let mut row = Vec::with_capacity(OBS_DIM);
        obs.write_host_row(&mut row);
        let x =
            Tensor::<2>::from_data(TensorData::new(row, [1, OBS_DIM]), &self.l1.weight.device());
        let h = activation::tanh(self.l1.forward(x));
        let h = activation::tanh(self.l2.forward(h));
        let index = self
            .l3
            .forward(h)
            .argmax(1)
            .into_data()
            .try_into_vec::<i32>()
            .expect("an argmax reads back as integers")[0];
        CartPoleAction::from_index(usize::try_from(index).expect("an argmax is not negative"))
    }
}

/// CMA-ES over [`PolicyNet`]'s flattened weights, as an agent the harness
/// trains: one candidate per training episode, one `tell` per population.
struct NeuroevolutionAgent {
    strategy: CmaEs,
    config: CmaEsConfig,
    reshaper: ModuleReshaper<PolicyNet>,
    /// The state the population being scored was asked from.
    state: CmaEsState,
    /// The state `ask` returned with the population; `tell` takes it back.
    asked: CmaEsState,
    population: Tensor<2>,
    /// Returns of the candidates that have played so far, in row order.
    returns: Vec<f32>,
    episode_return: f32,
    /// The network acting in training: the candidate now being scored.
    candidate: PolicyNet,
    /// The network acting in evaluation: the distribution's mean.
    mean: PolicyNet,
    /// Every draw `ask` and `tell` make comes from here, seeded by the trial.
    rng: StdRng,
    device: Device,
    training: bool,
    generations: u64,
}

impl NeuroevolutionAgent {
    fn new(seed: u64) -> Self {
        let device = Device::flex();
        let reshaper = ModuleReshaper::new(PolicyNet::new(&device));
        let mut config = CmaEsConfig::default_for(reshaper.num_params());
        // CMA-ES samples its first mean uniformly in `bounds`; small initial
        // weights keep the first policies away from saturating tanh.
        config.bounds = rlevo::core::bounds::Bounds::new(-0.5, 0.5);
        config.initial_sigma = 0.5;
        let strategy = CmaEs::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let state = strategy.init(&config, &mut rng, &device);
        let (population, asked) = strategy.ask(&config, &state, &mut rng, &device);
        let candidate = row(&reshaper, &population, 0);
        let mean = mean(&reshaper, &state, &device);
        Self {
            strategy,
            config,
            reshaper,
            state,
            asked,
            population,
            returns: Vec::new(),
            episode_return: 0.0,
            candidate,
            mean,
            rng,
            device,
            training: false,
            generations: 0,
        }
    }

    /// Scores the population, tells CMA-ES, and asks it for the next one.
    fn next_generation(&mut self) {
        let n = self.returns.len();
        let scores = Tensor::<1>::from_data(
            TensorData::new(std::mem::take(&mut self.returns), [n]),
            &self.device,
        );
        let (state, _metrics) = self.strategy.tell(
            &self.config,
            self.population.clone(),
            sanitize_fitness_tensor(scores),
            self.asked.clone(),
            &mut self.rng,
        );
        let (population, asked) =
            self.strategy
                .ask(&self.config, &state, &mut self.rng, &self.device);
        self.mean = mean(&self.reshaper, &state, &self.device);
        self.state = state;
        self.asked = asked;
        self.population = population;
        self.generations += 1;
    }
}

/// Row `index` of `population`, as a network.
fn row(reshaper: &ModuleReshaper<PolicyNet>, population: &Tensor<2>, index: usize) -> PolicyNet {
    let n = reshaper.num_params();
    reshaper.unflatten(
        population
            .clone()
            .slice([index..index + 1, 0..n])
            .reshape([n]),
    )
}

/// The mean of the search distribution, as a network.
fn mean(reshaper: &ModuleReshaper<PolicyNet>, state: &CmaEsState, device: &Device) -> PolicyNet {
    let weights = state.mean().to_vec();
    let n = weights.len();
    reshaper.unflatten(Tensor::<1>::from_data(
        TensorData::new(weights, [n]),
        device,
    ))
}

impl Agent<CartPoleObservation, CartPoleAction> for NeuroevolutionAgent {
    fn act(&mut self, obs: &CartPoleObservation, _rng: &mut dyn Rng) -> CartPoleAction {
        if self.training {
            self.candidate.act(obs)
        } else {
            self.mean.act(obs)
        }
    }

    fn on_transition(
        &mut self,
        transition: Transition<&CartPoleObservation, &CartPoleAction>,
        _rng: &mut dyn Rng,
    ) {
        if self.training {
            self.episode_return += transition.reward;
        }
    }

    /// The episode's return is the candidate's fitness. The next candidate
    /// plays the next episode; after the last one, the next generation.
    fn on_episode_end(&mut self, _status: EpisodeStatus) {
        if !self.training {
            return;
        }
        self.returns.push(std::mem::take(&mut self.episode_return));
        if self.returns.len() == self.config.pop_size {
            self.next_generation();
        }
        self.candidate = row(&self.reshaper, &self.population, self.returns.len());
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.episode_return = 0.0;
    }

    fn emit_metrics(&self) -> Vec<Metric> {
        vec![Metric::Counter {
            name: GENERATIONS.to_string(),
            count: self.generations,
        }]
    }
}

/// An agent that picks uniformly at random: the floor every row must clear.
struct Random;

impl<O, A: DiscreteAction<1>> Agent<O, A> for Random {
    fn act(&mut self, _obs: &O, rng: &mut dyn Rng) -> A {
        A::from_index(rng.random_range(0..A::ACTION_COUNT))
    }
}

/// PPO with a categorical policy head of [`PolicyNet`]'s shape.
fn ppo(seed: u64) -> PpoAgent<CategoricalPolicyHead, MlpValue, CartPoleObservation, 1, 2> {
    let device = Device::flex();
    device.seed(seed);
    let policy = CategoricalPolicyHeadConfig {
        obs_dim: OBS_DIM,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .try_init(&device)
    .expect("the policy head's sizes are valid");
    let value = MlpValueConfig {
        obs_dim: OBS_DIM,
        hidden: vec![HIDDEN, HIDDEN],
    }
    .try_init(&device)
    .expect("the value network's sizes are valid");
    let config = PpoTrainingConfig {
        num_envs: 1,
        num_steps: 128,
        num_minibatches: 4,
        update_epochs: 4,
        learning_rate: 2.5e-4,
        clip_grad: Some(GradientClippingConfig::Norm(0.5)),
        ..PpoTrainingConfig::default()
    };
    let iterations = TRAIN_STEPS / config.batch_size();
    PpoAgent::new(policy, value, config, device, iterations)
        .expect("the PPO configuration is valid")
}

/// DQN at its default configuration, with a Q-network of [`PolicyNet`]'s
/// shape. Of the three learners it is the most sensitive to its settings at
/// this budget: the rate and exploration schedule the agent template uses,
/// chosen for a 64-unit network, do worse here than these defaults.
fn dqn(seed: u64) -> DqnAgent<MlpQNetwork, CartPoleObservation, CartPoleAction, 1, 2> {
    let device = Device::flex();
    device.seed(seed);
    let network = MlpQNetworkConfig {
        obs_dim: OBS_DIM,
        hidden: vec![HIDDEN, HIDDEN],
        num_actions: NUM_ACTIONS,
    }
    .try_init(&device)
    .expect("the network's sizes are valid");
    DqnAgent::new(network, DqnTrainingConfig::default(), device)
        .expect("the DQN configuration is valid")
}

/// Weights a learner trains, counted once from a freshly built learner.
fn weights() -> [usize; 3] {
    let device = Device::flex();
    let evolved = PolicyNet::new(&device).num_params();
    let ppo = CategoricalPolicyHeadConfig {
        obs_dim: OBS_DIM,
        hidden: HIDDEN,
        num_actions: NUM_ACTIONS,
    }
    .try_init(&device)
    .expect("the policy head's sizes are valid")
    .num_params()
        + MlpValueConfig {
            obs_dim: OBS_DIM,
            hidden: vec![HIDDEN, HIDDEN],
        }
        .try_init(&device)
        .expect("the value network's sizes are valid")
        .num_params();
    let dqn = MlpQNetworkConfig {
        obs_dim: OBS_DIM,
        hidden: vec![HIDDEN, HIDDEN],
        num_actions: NUM_ACTIONS,
    }
    .try_init(&device)
    .expect("the network's sizes are valid")
    .num_params();
    [evolved, ppo, dqn]
}

/// One row of the table: the learner's mean evaluation return and episode
/// length over its trials, how many trials it solved, how it updated its
/// weights, and how long its trials took.
fn table_row(
    name: &str,
    weights: &str,
    updates: &str,
    report: &BenchmarkReport,
    seconds: f64,
) -> Result<String, String> {
    let mut returns = 0.0;
    let mut lengths = 0.0;
    let mut solved = 0;
    for trial in &report.trials {
        if trial.errored {
            let why = trial.error_message.as_deref().unwrap_or("no message");
            return Err(format!("{name}: a trial errored: {why}"));
        }
        returns += trial.scalars[RETURN_MEAN];
        lengths += trial.scalars[EPISODE_LENGTH_MEAN];
        if trial.scalars[RETURN_MEAN] >= SOLVED_RETURN {
            solved += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let trials = report.trials.len() as f64;
    Ok(format!(
        "| {name} | {weights} | {updates} | {:.1} | {:.1} | {solved}/{} | {seconds:.1} |",
        returns / trials,
        lengths / trials,
        report.trials.len(),
    ))
}

/// A counter every trial of `report` reported, averaged over the trials.
fn counter(report: &BenchmarkReport, name: &str) -> u64 {
    let total: u64 = report
        .trials
        .iter()
        .map(|t| t.counters.get(name).copied().unwrap_or(0))
        .sum();
    total / report.trials.len().max(1) as u64
}

fn main() -> Result<(), String> {
    let suite = Suite::new("cartpole").with_env("CartPole", cartpole);
    let evaluator = Evaluator::new(config());
    let mut reporter = LoggingReporter::new();
    let [evolved, ppo_weights, dqn_weights] = weights();

    println!(
        "CartPole, {TRAIN_STEPS} training steps per trial, then {} evaluation episodes; \
         {TRIALS} trials per learner, means over trials.",
        config().num_episodes
    );
    println!();
    println!(
        "| learner | weights | updates | return/mean | episode_length/mean | solved | seconds |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");

    let start = Instant::now();
    let random = evaluator.run_suite(&suite, |_seed| Random, &mut reporter);
    let seconds = start.elapsed().as_secs_f64();
    println!("{}", table_row("random", "0", "none", &random, seconds)?);

    let start = Instant::now();
    let evolution = evaluator.run_suite(&suite, NeuroevolutionAgent::new, &mut reporter);
    let seconds = start.elapsed().as_secs_f64();
    let generations = format!("{} generations", counter(&evolution, GENERATIONS));
    println!(
        "{}",
        table_row(
            "CMA-ES (no gradient)",
            &evolved.to_string(),
            &generations,
            &evolution,
            seconds
        )?
    );

    let start = Instant::now();
    let policy_gradient = evaluator.run_suite(&suite, ppo, &mut reporter);
    let seconds = start.elapsed().as_secs_f64();
    let iterations = format!(
        "{} PPO iterations",
        counter(&policy_gradient, "rl/iterations")
    );
    println!(
        "{}",
        table_row(
            "PPO",
            &ppo_weights.to_string(),
            &iterations,
            &policy_gradient,
            seconds
        )?
    );

    let start = Instant::now();
    let value = evaluator.run_suite(&suite, dqn, &mut reporter);
    let seconds = start.elapsed().as_secs_f64();
    let steps = format!("{} gradient steps", counter(&value, "rl/gradient_updates"));
    println!(
        "{}",
        table_row("DQN", &dqn_weights.to_string(), &steps, &value, seconds)?
    );
    Ok(())
}
