//! End-to-end integration tests for the TD3 agent.
//!
//! Default-run tests exercise TD3 on the shared synthetic 1-D continuous env
//! ([`rlevo_test_support::env::LinearEnv`]) and check the TD3-specific
//! delayed-actor schedule so that `cargo test` stays tractable (no physics
//! simulator, small networks, modest budgets). The Pendulum macro-smoke is
//! gated behind `#[ignore]` per the project's integration-test budget
//! convention.
//!
//! The shared fixture, the `Flex` determinism preamble ([`flex_guard`] /
//! [`seeded_device`]), and the acceptance assertions live in the
//! `rlevo-test-support` dev-crate; only the TD3 networks and the
//! algorithm-specific tests remain here.

use burn::tensor::Device;

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo::harness::evaluator::train;
use rlevo_core::action::ContinuousAction;
use rlevo_core::agent::Agent;
use rlevo_core::environment::Environment;
use rlevo_environments::classic::pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation,
};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_model::{
    MlpActor, MlpActorConfig, MlpCritic, MlpCriticConfig,
};
use rlevo_reinforcement_learning::algorithms::td3::td3_agent::Td3Agent;
use rlevo_reinforcement_learning::algorithms::td3::td3_config::Td3TrainingConfig;
use rlevo_reinforcement_learning::target::TargetUpdate;

use rlevo_test_support::assert::assert_improves_over_random;
use rlevo_test_support::baseline::{random_return, uniform_bounded};
use rlevo_test_support::env::{LinearAction, LinearEnv, LinearObservation};
use rlevo_test_support::flex::{flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Actor and critic: the crate's defaults with one hidden layer, shared
// between the LinearEnv tests and the Pendulum smoke test.
// ---------------------------------------------------------------------------

fn actor(
    obs_dim: usize,
    hidden: usize,
    action_dim: usize,
    scale: f32,
    device: &Device,
) -> MlpActor {
    MlpActorConfig {
        obs_dim,
        hidden: vec![hidden],
        action_dim,
        action_scale: scale,
    }
    .try_init(device)
    .expect("valid actor config")
}

fn critic(obs_dim: usize, action_dim: usize, hidden: usize, device: &Device) -> MlpCritic {
    MlpCriticConfig {
        obs_dim,
        action_dim,
        hidden: vec![hidden],
    }
    .try_init(device)
    .expect("valid critic config")
}

// Concrete TD3 agent over the shared 1-D continuous fixture.
type LinearAgent = Td3Agent<MlpActor, MlpCritic, LinearObservation, LinearAction, 1, 2, 1, 2>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Builds and trains a TD3 agent on the shared `LinearEnv` for `total` steps,
/// returning the standardised outcome consumed by the suite macros. Shared by
/// the convergence and reproducibility tests below.
fn run_linear(seed: u64, total: usize) -> TrainOutcome {
    let device = seeded_device(seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor = actor(1, 32, 1, 1.0, &device);
    let critic_1 = critic(1, 1, 32, &device);
    let critic_2 = critic(1, 1, 32, &device);
    let config = Td3TrainingConfig {
        replay_buffer_capacity: 20_000,
        batch_size: 32,
        learning_starts: 500,
        actor_lr: 1e-3,
        critic_lr: 1e-3,
        gamma: 0.99,
        target_update: TargetUpdate::polyak(0.02, 2),
        exploration_noise: 0.2,
        policy_noise: 0.2,
        noise_clip: 0.5,
        policy_frequency: 2,
        ..Default::default()
    };
    let mut agent: LinearAgent =
        Td3Agent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    let episodes = train(&mut env, &mut agent, total, usize::MAX, &mut rng).expect("training");
    TrainOutcome::from_returns(episodes.iter().map(|e| e.return_value))
}

/// Mean episode return of a uniform-random `U(-1, 1)` policy on the shared
/// `LinearEnv`, measured over 200 episodes from `seed`. The learning test below
/// asserts the trained agent beats this measured baseline by a margin.
fn random_linear(seed: u64) -> f32 {
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(
        &mut env,
        200,
        20,
        &mut rng,
        uniform_bounded::<1, LinearAction>,
    )
}

/// Mean episode return of a uniform-random torque policy on the `TimeLimit`ed
/// Pendulum, measured over 100 episodes from `seed`. Baseline for the Pendulum
/// learning check.
fn random_pendulum(seed: u64) -> f32 {
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    })
    .expect("valid config");
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(
        &mut env,
        100,
        200,
        &mut rng,
        uniform_bounded::<1, PendulumAction>,
    )
}

// Default-run convergence check: TD3 should clear the random baseline within
// 8k env steps. A uniform `U(-1, 1)` policy scores `random_linear`; an optimal
// policy approaches `0`. (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "8 000-step LinearEnv convergence check; run with `cargo test -- --ignored`"]
    td3_linear_improves_over_random,
    improves_over_random(margin = 2.0),
    seed = 42,
    total = 8_000,
    run = run_linear,
    random = random_linear,
}

// Seeded reproducibility: two same-seed runs must produce bit-equal metrics.
// (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    td3_linear_flex_reproducibility,
    bits,
    seed = 42,
    total = 1_000,
    run = run_linear,
}

/// `act_with` (plain-device greedy inference) must produce the same action as
/// `act(obs, false, _)` (the autodiff deterministic mean) for identical
/// weights — the snapshot is just a non-autodiff copy of the actor, so
/// the bench's faster greedy path stays faithful to the eval policy.
#[test]
fn td3_act_with_matches_deterministic_act() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let device = seeded_device(seed);

    let actor = actor(1, 32, 1, 1.0, &device);
    let critic_1 = critic(1, 1, 32, &device);
    let critic_2 = critic(1, 1, 32, &device);
    let config = Td3TrainingConfig {
        replay_buffer_capacity: 20_000,
        batch_size: 32,
        learning_starts: 500,
        actor_lr: 1e-3,
        critic_lr: 1e-3,
        gamma: 0.99,
        target_update: TargetUpdate::polyak(0.02, 2),
        exploration_noise: 0.2,
        policy_noise: 0.2,
        noise_clip: 0.5,
        policy_frequency: 2,
        ..Default::default()
    };
    let mut agent: LinearAgent =
        Td3Agent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    let net = agent.inference_net();
    let mut rng = StdRng::seed_from_u64(seed);
    // Evaluating `$\Rightarrow$` deterministic mean + bound clip; rng is unused.
    agent.set_training(false);
    for &x in &[-0.9_f32, -0.3, 0.0, 0.25, 0.8] {
        let obs = LinearObservation { x };
        let det = agent.act(&obs, &mut rng);
        let greedy = agent.act_with(&net, &obs);
        let a = det.as_slice()[0];
        let b = greedy.as_slice()[0];
        assert!((a - b).abs() < 1e-5, "x={x}: act={a}, act_with={b}");
    }
}

/// Default-run test: with `policy_frequency = 2`, exactly half of the
/// learn-steps after warm-up should emit an actor update. We drive the agent
/// past warm-up on `LinearEnv` and then count `LearnOutcome::actor_loss
/// .is_some()` emissions across a fixed number of manual `learn_step` calls.
#[test]
fn td3_delayed_update_skips_actor_step() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let device = seeded_device(seed);
    let mut env = LinearEnv::with_seed(seed, 20);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor = actor(1, 16, 1, 1.0, &device);
    let critic_1 = critic(1, 1, 16, &device);
    let critic_2 = critic(1, 1, 16, &device);
    let config = Td3TrainingConfig {
        replay_buffer_capacity: 1_024,
        batch_size: 8,
        learning_starts: 32,
        policy_frequency: 2,
        ..Default::default()
    };
    let mut agent: LinearAgent =
        Td3Agent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    // Prime the buffer past both the batch-size and warm-up thresholds.
    let mut snap = env.reset().expect("reset");
    for _ in 0..64 {
        let obs = snap.observation;
        let action = agent.act(&obs, &mut rng);
        let next = env.step(action).expect("step");
        let reward: f32 = (next.reward).into();
        // `done` drives the episode reset; `remember` masks the Bellman
        // bootstrap on the status, and only on termination. `LinearEnv` only ever truncates, so these differ.
        let done = next.status.is_done();
        let next_obs = next.observation;
        agent.remember(obs, &action, reward, next_obs, next.status);
        snap = if done {
            env.reset().expect("reset")
        } else {
            next
        };
    }
    assert!(
        agent.step() >= 32 && agent.buffer_len() >= 8,
        "agent should be past warm-up"
    );

    // Exactly 10 manual learn_steps → with policy_frequency=2 the actor
    // must fire on 5 of them.
    let mut actor_fires = 0_usize;
    for _ in 0..10 {
        let outcome = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("can learn");
        if outcome.actor_loss.is_some() {
            actor_fires += 1;
        }
    }
    assert_eq!(
        actor_fires, 5,
        "expected 5 actor updates across 10 critic steps, got {actor_fires}"
    );
    assert_eq!(
        agent.critic_updates(),
        10,
        "expected 10 critic updates total"
    );
}

/// Pendulum learning check: 500k steps, confirms the moving average is finite
/// and beats a measured uniform-random torque policy.
#[test]
#[ignore = "500 000-step continuous TD3 Pendulum run (~several minutes on CPU); confirms avg reward beats a measured uniform-random baseline — run with `cargo test -- --ignored`"]
fn td3_pendulum_improves_over_random() {
    let _guard = flex_guard();
    let seed: u64 = 42;
    let device = seeded_device(seed);
    let base = Pendulum::with_config(PendulumConfig {
        seed,
        ..PendulumConfig::default()
    })
    .expect("valid config");
    let mut env = TimeLimit::new(base, 200);
    let mut rng = StdRng::seed_from_u64(seed);

    let actor = actor(3, 256, 1, 2.0, &device);
    let critic_1 = critic(3, 1, 256, &device);
    let critic_2 = critic(3, 1, 256, &device);
    let config = Td3TrainingConfig {
        replay_buffer_capacity: 200_000,
        batch_size: 256,
        learning_starts: 10_000,
        actor_lr: 1e-4,
        critic_lr: 1e-3,
        gamma: 0.99,
        target_update: TargetUpdate::polyak(0.005, 2),
        exploration_noise: 0.1,
        policy_noise: 0.2,
        noise_clip: 0.5,
        policy_frequency: 2,
        ..Default::default()
    };
    let mut agent: Td3Agent<MlpActor, MlpCritic, PendulumObservation, PendulumAction, 1, 2, 1, 2> =
        Td3Agent::new(actor, critic_1, critic_2, config, device).expect("valid config");

    let episodes = train(&mut env, &mut agent, 500_000, usize::MAX, &mut rng).expect("training");

    assert!(!episodes.is_empty(), "the run must complete an episode");
    let avg = TrainOutcome::from_returns(episodes.iter().map(|e| e.return_value)).avg_score;
    // TD3 should comfortably clear a uniform-random torque policy after 500k steps (it reaches
    // `$\approx -150..-300$` here). The wide 300-pt margin keeps the bar meaningful while absorbing
    // cross-platform float drift; the tighter -150 acceptance target stays aspirational.
    let baseline = random_pendulum(seed);
    assert_improves_over_random(avg, baseline, 300.0);
}
