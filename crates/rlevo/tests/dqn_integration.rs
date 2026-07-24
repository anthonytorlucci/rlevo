//! End-to-end integration tests for the DQN agent.
//!
//! These tests wire [`DqnAgent`] against the shared seeded `CartPole` fixture
//! ([`rlevo_test_support::env::cartpole_seeded`]) and assert learning
//! behaviour. They intentionally use modest step budgets so `cargo test`
//! remains tractable; the longer macro-bench targets live in
//! `benches/dqn_bench.rs`.
//!
//! The `Flex` determinism preamble ([`flex_guard`] / [`seeded_device`]) and the
//! acceptance assertions live in the `rlevo-test-support` dev-crate; only the
//! DQN network and the algorithm-specific tests remain here.

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, activation};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_model::DqnModel;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;
use rlevo_reinforcement_learning::target::TargetUpdate;
use rlevo_reinforcement_learning::utils::{PolyakError, polyak_update};

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Shared MLP (mirror of the dqn_cart_pole example). Implements DQN's model
// trait and so stays test-local.
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct DqnMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> DqnMlp<B> {
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 64).init(device),
            l2: LinearConfig::new(64, 64).init(device),
            l3: LinearConfig::new(64, 2).init(device),
        }
    }

    fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.l1.forward(observations));
        let x = activation::relu(self.l2.forward(x));
        self.l3.forward(x)
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

    #[allow(clippy::cast_possible_truncation)]
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError> {
        polyak_update::<B::InnerBackend, DqnMlp<B::InnerBackend>>(
            &active.valid(),
            target,
            tau as f32,
        )
    }
}

type Agent = DqnAgent<Be, DqnMlp<Be>, CartPoleObservation, CartPoleAction, 1, 2>;

/// Constructs a deterministic [`DqnAgent`] for `CartPole`.
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce bit-for-bit identical initial weights. Callers must hold the
/// [`flex_guard`] lock for the duration of the test (see that function for the
/// process-global Flex RNG / rayon-pinning rationale).
///
/// The hyperparameters (ε-greedy schedule, target update, γ, learning rate,
/// buffer capacity) are tuned for the 4-observation / 2-action `CartPole` task
/// and intentionally conservative so training converges well within 30 000
/// steps.
///
/// The target rule is `polyak(0.005, 1)` — a soft update on every gradient
/// step. This is the behaviour this test has always exercised: it used to be
/// spelled `.tau(0.005).target_update_frequency(500)`, in which the `500` was
/// **inert** (the hard path self-gated to a no-op whenever τ > 0). Transcribing
/// the `500` as a cadence would be a 500× slowdown of the soft update, not a
/// faithful translation — the convergence budget below is calibrated against
/// the per-gradient-step schedule that actually ran.
fn fresh_agent(seed: u64) -> Agent {
    let device = seeded_device::<Be>(seed);
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.9995)
        .learning_starts(1_000)
        .train_frequency(4)
        .replay_buffer_capacity(50_000)
        .double_q(false)
        .build()
        .expect("valid config");
    let model: DqnMlp<Be> = DqnMlp::new(&device);
    DqnAgent::new(model, config, device).expect("valid config")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Builds and trains a DQN agent on the shared seeded `CartPole` for `total`
/// steps, returning the standardised outcome consumed by the suite macros.
/// Shared by the convergence and reproducibility tests below.
fn run_cartpole(seed: u64, total: usize) -> TrainOutcome {
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, total, 0).expect("training");
    let stats = agent.stats();
    TrainOutcome {
        avg_score: stats.avg_score().unwrap_or(0.0),
        rewards: stats.recent_history.iter().map(|m| m.reward).collect(),
    }
}

// `CartPole` target: after 30k steps with a 2x64 MLP, the 100-episode moving
// average should comfortably exceed 100. The smoke run during development
// reached ~183 on seed=42; we assert a conservative floor of 100. (Generated
// by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "30 000-step CartPole training run (several minutes on CPU); confirms avg reward ≥ 100 — run with `cargo test -- --ignored`"]
    dqn_cartpole_converges,
    reaches(100.0),
    seed = 42,
    total = 30_000,
    run = run_cartpole,
}

// Reproducibility smoke test: two back-to-back runs from the same seed produce
// identical reward sequences. (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    #[ignore = "6 000-step reproducibility check (two sequential CartPole runs); run with `cargo test -- --ignored`"]
    dqn_cartpole_flex_reproducibility,
    seq,
    seed = 123,
    total = 3_000,
    run = run_cartpole,
}

/// Smoke test: a short run completes with finite rewards and a populated
/// buffer. Protects against silent regressions that would NaN-out.
/// The `flex_guard` lock serializes execution within this binary so
/// Burn's process-global Flex RNG stays isolated.
#[test]
#[ignore = "2 500-step Flex backend smoke run (~30 s); checks for NaN rewards and non-empty replay buffer — run with `cargo test -- --ignored`"]
fn dqn_cartpole_produces_finite_rewards() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = fresh_agent(seed);
    train(&mut agent, &mut env, &mut rng, 2_500, 0).expect("training");
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    let rewards: Vec<f32> = agent
        .stats()
        .recent_history
        .iter()
        .map(|m| m.reward)
        .collect();
    assert_all_finite("reward", &rewards);
}

// ---------------------------------------------------------------------------
// Hard-copy cadence, end to end (ADR 0058 / 0059)
//
// Every other exercise of `TargetUpdate::hard` in the workspace drives
// `learn_step` directly from a unit test. This one runs it through the real
// `train()` loop, which is the only place the two counters `hard(n)` sits
// between — `train_frequency` (environment steps) and the gradient-update
// cadence — actually interact.
// ---------------------------------------------------------------------------

/// Hard-copy cadence, in gradient updates.
const HARD_EVERY: usize = 5;
/// Environment steps between *attempted* gradient updates.
const HARD_TRAIN_FREQUENCY: usize = 4;
/// Environment steps before the first attempted gradient update.
///
/// `68`, not a round `64`, and the offset is load-bearing: the first attempted
/// update lands at env step 68 = 17·4, so gradient update `u` happens at env
/// step `4·(u + 16)`. Because `16 % HARD_EVERY != 0`, the gradient-update gate
/// `u % 5 == 0` and an env-step gate `4·(u+16) % 5 == 0` fire on **disjoint**
/// updates. That makes this test a discriminator for ADR 0059's unit, not just
/// for `hard(n)` firing at all — a gate wrongly reading the env-step counter
/// copies at env steps 80/100/120/140, none of which is a leg boundary below.
/// A `learning_starts` of `64` would have made the two gates coincide exactly
/// and the unit error invisible here.
const HARD_LEARNING_STARTS: usize = 68;

/// Cumulative env-step marks for the run's three legs.
///
/// Chosen so the gradient-update count lands on a firing of `hard(5)` at leg 1
/// (20 updates), strictly between firings at leg 2 (24), and on the next firing
/// at leg 3 (25) — see [`hard_expected_updates`].
const HARD_LEG1_END: usize = 144;
const HARD_LEG2_END: usize = 160;
const HARD_LEG3_END: usize = 164;

/// The gradient-update count after `total_steps` environment steps, derived
/// from the loop's own two gates rather than hard-coded: `train()` attempts a
/// learn step when `step % train_frequency == 0`, and `learn_step` reaches a
/// loss (and so advances the counter) once `step >= learning_starts`. The
/// buffer gate is slack here — `learning_starts` exceeds `batch_size`, and one
/// transition is pushed per step.
fn hard_expected_updates(total_steps: usize) -> usize {
    (1..=total_steps)
        .filter(|s| s.is_multiple_of(HARD_TRAIN_FREQUENCY) && *s >= HARD_LEARNING_STARTS)
        .count()
}

/// Flattens one network's parameters to the host, element-wise.
///
/// Deliberately not a `sum()` proxy: per-element optimizer steps very nearly
/// cancel in a sum, which would turn "did the target change" into a coin flip.
fn hard_net_weights<B: Backend>(net: &DqnMlp<B>) -> Vec<f32> {
    fn host<B: Backend, const D: usize>(t: Tensor<B, D>, out: &mut Vec<f32>) {
        out.extend(
            t.into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("host read"),
        );
    }
    let mut out = Vec::new();
    for linear in [&net.l1, &net.l2, &net.l3] {
        host(linear.weight.val(), &mut out);
        if let Some(bias) = linear.bias.as_ref() {
            host(bias.val(), &mut out);
        }
    }
    out
}

fn hard_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "weight snapshots must have equal length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// A `hard(n)` config driven through the real `train()` loop.
///
/// Asserts both halves of the cadence contract end-to-end, because either half
/// alone is satisfiable by a broken gate: a gate that *never* fires passes the
/// "frozen between firings" leg, and a gate that *always* fires passes the
/// "exact copy at a firing" leg. Only the pair pins `hard(5)`. The leg
/// boundaries additionally separate the gradient-update unit from the env-step
/// unit — see [`HARD_LEARNING_STARTS`].
///
/// Deliberately small — a 4-64-64-2 MLP, batch 16, 164 environment steps and 25
/// gradient updates — so it stays un-`#[ignore]`d and runs in the PR gate,
/// unlike the convergence and reproducibility runs above. It asserts a
/// *mechanism*, not learning progress, so it needs no convergence budget.
// Bit-exactness *is* the property under test: at τ = 1.0 `polyak_update`
// assigns the active parameter verbatim, so a fired hard copy leaves a
// max |Δw| of exactly zero and an unfired one leaves the previous snapshot
// byte-identical. A tolerance would let a τ < 1 soft update — the mechanism
// this test exists to distinguish `hard(n)` from — pass as a "copy".
#[allow(clippy::float_cmp)]
// Three legs of one run, kept in one function because each leg consumes the
// agent, env and RNG state the previous leg left behind; splitting them would
// mean threading that state through helpers for no gain in readability.
#[allow(clippy::too_many_lines)]
#[test]
fn dqn_cartpole_hard_target_copies_inside_train() {
    let _guard = flex_guard();
    let seed: u64 = 4242;
    let device = seeded_device::<Be>(seed);
    let config = DqnTrainingConfigBuilder::new()
        .batch_size(16)
        .gamma(0.99)
        .target_update(TargetUpdate::hard(HARD_EVERY))
        .learning_rate(1e-3)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay(0.99)
        .learning_starts(HARD_LEARNING_STARTS)
        .train_frequency(HARD_TRAIN_FREQUENCY)
        .replay_buffer_capacity(2_000)
        .double_q(false)
        .build()
        .expect("valid config");
    let mut agent: Agent =
        DqnAgent::new(DqnMlp::<Be>::new(&device), config, device).expect("valid config");
    let mut env = cartpole_seeded(seed);
    let mut rng = StdRng::seed_from_u64(seed);

    let initial_target = hard_net_weights(agent.target_net());

    // -- Leg 1: land exactly on a firing -----------------------------------
    train(&mut agent, &mut env, &mut rng, HARD_LEG1_END, 0).expect("training");

    let leg1_updates = hard_expected_updates(HARD_LEG1_END);
    assert_eq!(
        agent.gradient_updates(),
        leg1_updates,
        "the cadence counter must count gradient updates, not the {HARD_LEG1_END} \
         environment steps the loop just ran (ADR 0059)"
    );
    assert!(
        leg1_updates >= 2 * HARD_EVERY && leg1_updates.is_multiple_of(HARD_EVERY),
        "budget error: leg 1 must cross hard({HARD_EVERY}) at least twice and end on \
         a firing, got {leg1_updates} updates"
    );

    // The run itself completed healthily.
    assert!(agent.buffer_len() > 0, "buffer should have transitions");
    let rewards: Vec<f32> = agent
        .stats()
        .recent_history
        .iter()
        .map(|m| m.reward)
        .collect();
    assert!(
        !rewards.is_empty(),
        "the run must have completed at least one episode, or the finiteness check \
         below is vacuous"
    );
    assert_all_finite("reward", &rewards);

    // The target really hard-copied: it left its initialisation, and it now
    // equals the policy bit-for-bit. τ = 0.005 would leave it lagged instead.
    let copied = hard_net_weights(agent.target_net());
    let moved = hard_max_abs_diff(&copied, &initial_target);
    assert!(
        moved > 0.0,
        "after {leg1_updates} gradient updates the target must have been copied at \
         least once (max |Δw| from init = {moved:e})"
    );
    let policy = hard_net_weights(&agent.inference_net());
    assert_eq!(
        hard_max_abs_diff(&copied, &policy),
        0.0,
        "update {leg1_updates} is a multiple of {HARD_EVERY}, so τ = 1.0 must copy the \
         post-step policy exactly"
    );

    // -- Leg 2: stop strictly between firings ------------------------------
    train(
        &mut agent,
        &mut env,
        &mut rng,
        HARD_LEG2_END - HARD_LEG1_END,
        0,
    )
    .expect("training");

    let leg2_updates = hard_expected_updates(HARD_LEG2_END);
    assert_eq!(agent.gradient_updates(), leg2_updates);
    assert!(
        !leg2_updates.is_multiple_of(HARD_EVERY),
        "budget error: leg 2 must end off-cadence, got {leg2_updates} updates"
    );
    assert_eq!(
        hard_max_abs_diff(&hard_net_weights(agent.target_net()), &copied),
        0.0,
        "no update in ({leg1_updates}, {leg2_updates}] is a multiple of {HARD_EVERY}, so \
         the target must be untouched"
    );
    let drifted = hard_max_abs_diff(&hard_net_weights(&agent.inference_net()), &copied);
    assert!(
        drifted > 0.0,
        "precondition failed: the policy must have moved away from the copy (max |Δw| \
         = {drifted:e}), or the frozen-target assertion above is vacuous"
    );

    // -- Leg 3: the next firing copies again -------------------------------
    train(
        &mut agent,
        &mut env,
        &mut rng,
        HARD_LEG3_END - HARD_LEG2_END,
        0,
    )
    .expect("training");

    let leg3_updates = hard_expected_updates(HARD_LEG3_END);
    assert_eq!(agent.gradient_updates(), leg3_updates);
    assert!(leg3_updates.is_multiple_of(HARD_EVERY));
    let refreshed = hard_net_weights(agent.target_net());
    assert_eq!(
        hard_max_abs_diff(&refreshed, &hard_net_weights(&agent.inference_net())),
        0.0,
        "update {leg3_updates} is the next multiple of {HARD_EVERY}: the target must be \
         an exact copy of the policy again"
    );
    assert!(
        hard_max_abs_diff(&refreshed, &copied) > 0.0,
        "the refresh must pick up the policy's drift, not re-copy the stale weights"
    );
}
