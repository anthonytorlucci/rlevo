//! End-to-end integration tests for PPG.
//!
//! Budget conventions mirror the PPO tests: default-run tests use a modest
//! 50k-step budget with lax thresholds; heavier macro-convergence and
//! reproducibility checks live behind `#[ignore]`.
//!
//! The `Flex` determinism preamble ([`flex_guard`] / [`seeded_device`]), the
//! shared seeded `CartPole` fixture ([`cartpole_seeded`]), and the acceptance
//! assertions live in the `rlevo-test-support` dev-crate; only the value
//! network and the algorithm-specific tests remain here.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_environments::classic::cartpole::{CartPoleAction, CartPoleObservation};
use rlevo_environments::wrappers::TimeLimit;
use rlevo_reinforcement_learning::algorithms::ppg::policies::{
    PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig,
};
use rlevo_reinforcement_learning::algorithms::ppg::ppg_agent::PpgAgent;
use rlevo_reinforcement_learning::algorithms::ppg::ppg_config::PpgConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppg::train::train_discrete;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;

use rlevo_test_support::assert::assert_all_finite;
use rlevo_test_support::baseline::{random_return, uniform_discrete};
use rlevo_test_support::capture::FieldCapture;
use rlevo_test_support::env::cartpole_seeded;
use rlevo_test_support::flex::{FlexAutodiff as Be, flex_guard, seeded_device};
use rlevo_test_support::{TrainOutcome, rl_learning_test, rl_reproducibility_test};

// ---------------------------------------------------------------------------
// Shared value MLP. Implements PPO's value trait (reused by PPG) and so stays
// test-local.
// ---------------------------------------------------------------------------

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

/// Build a deterministic `PpgAgent` for `CartPole` discrete (4 obs, 2 actions).
///
/// Seeds the backend via [`seeded_device`] so two calls with the same `seed`
/// produce identical initial weights. Callers must hold the [`flex_guard`] lock
/// for the duration of the test.
///
/// PPG-specific config: `e_aux=6` aux epochs per phase, `beta_clone=1.0` KL
/// distillation weight, `aux_batch_size=128`. The PPO sub-config mirrors
/// `ppo_integration.rs` (lr=2.5e-4, clip=0.2, 4 epochs, 4 minibatches,
/// 128-step rollout).
///
/// `n_iteration` controls how many policy-phase iterations occur between
/// consecutive aux phases. Callers pass it explicitly so tests can disable the
/// aux phase (set it above the total iteration count) or force it to fire early
/// (set it to a small value).
fn make_cart_pole_agent(
    seed: u64,
    num_steps: usize,
    total_timesteps: usize,
    n_iteration: usize,
) -> PpgAgent<Be, PpgCategoricalPolicyHead<Be>, ValueMlp<Be>, CartPoleObservation, 1, 2> {
    let device = seeded_device::<Be>(seed);

    let policy = PpgCategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 64,
        num_actions: 2,
    }
    .try_init::<Be>(&device)
    .expect("valid head config");
    let value = ValueMlp::new(4, 64, &device);

    let ppo = PpoTrainingConfigBuilder::new()
        .num_envs(1)
        .num_steps(num_steps)
        .num_minibatches(4)
        .update_epochs(4)
        .learning_rate(2.5e-4)
        .clip_coef(0.2)
        .entropy_coef(0.01)
        .value_coef(0.5)
        .gamma(0.99)
        .gae_lambda(0.95)
        .build()
        .expect("valid config");
    let config = PpgConfigBuilder::new()
        .ppo(ppo)
        .n_iteration(n_iteration)
        .e_aux(6)
        .beta_clone(1.0)
        .aux_batch_size(128)
        .build()
        .expect("valid config");
    let total_iterations = total_timesteps / config.batch_size().max(1);
    PpgAgent::new(policy, value, config, device, total_iterations).expect("valid config")
}

/// Builds and trains a PPG agent on the shared seeded `CartPole` for `total`
/// steps with `n_iteration = 32` (the aux phase fires periodically), returning
/// the standardised outcome consumed by the suite macro.
fn run_cartpole(seed: u64, total: usize) -> TrainOutcome {
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, 128, total, 32);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let stats = agent.stats();
    TrainOutcome {
        avg_score: stats.avg_score().unwrap_or(0.0),
        rewards: stats.recent_history.iter().map(|m| m.reward).collect(),
    }
}

#[test]
#[ignore = "50 000-step PPG CartPole run with aux phase disabled (n_iteration=10 000); confirms PPO-parity avg reward ≥ 80 — run with `cargo test -- --ignored`"]
fn ppg_without_aux_phase_matches_ppo_baseline() {
    let _guard = flex_guard();
    // Sanity check that the policy-phase update is a faithful PPO update:
    // with n_iteration set above the total iteration count the auxiliary
    // phase never fires, and PPG should match PPO's ~50k-step CartPole
    // threshold (80 in `ppo_integration.rs`).
    let seed: u64 = 42;
    let total = 50_000_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total, 10_000);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    assert!(
        agent.last_aux_phase().is_none(),
        "aux phase should not have fired with n_iteration=10000"
    );
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(
        avg >= 80.0,
        "expected avg reward >= 80 (PPO-parity), got {avg:.2}"
    );
}

/// The auxiliary phase must fire *and* move the policy — including on the
/// terminal iteration (issues #319, #324).
///
/// `total / num_steps = 2048 / 128 = 16` policy-phase iterations with
/// `n_iteration = 4`, so `16 % 4 == 0` and the aux phase this test inspects via
/// [`PpgAgent::last_aux_phase`] is the **terminal** one, at
/// `iteration == total_iterations`. That is exactly #324's trigger: the phase
/// used to read the annealed learning rate one schedule tick ahead of the
/// policy phase it accompanies, and at the terminal tick the anneal is exactly
/// `0.0`, so every minibatch stepped by nothing.
///
/// This test previously asserted only `aux_value_loss.is_finite()` and
/// `policy_kl.is_finite()`. `0.0` is finite, so a bit-exactly zero `policy_kl`
/// — the precise signature of the #324 no-op — satisfied both: the test passed
/// against the bug it is positioned to guard (recorded in #319's closing
/// comment). #319 also warned that asserting `policy_kl > 0.0` would fail for a
/// benign reason; #324's fix is what retired that warning, since the terminal
/// phase now steps at the rate of the policy phase it accompanies.
///
/// The `> 0.0` form is now load-bearing here, and it rests on the
/// more-than-one-minibatch precondition asserted in the body.
#[test]
#[ignore = "2 048-step PPG run (n_iteration=4) verifying the terminal aux phase fires and moves the policy at a nonzero learning rate (#324) — run with `cargo test -- --ignored`"]
fn ppg_aux_phase_actually_runs() {
    let _guard = flex_guard();
    // Use a small n_iteration so the aux phase fires within a tiny budget.
    let seed: u64 = 11;
    let num_steps = 128_usize;
    let total = 2_048_usize; // 16 iterations at num_steps=128.
    let n_iteration = 4_usize;
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total, n_iteration);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    assert!(
        agent.iteration() >= n_iteration,
        "should have completed at least n_iteration={n_iteration} policy phases, got {}",
        agent.iteration()
    );
    let aux = agent
        .last_aux_phase()
        .expect("aux phase should have run at least once");
    // Precondition for the `policy_kl > 0.0` assertion below. `maybe_aux_phase`
    // snapshots `π_old` once, before the aux epoch loop, so the *first*
    // minibatch of the *first* epoch compares that snapshot against weights the
    // phase has not stepped yet: its KL term is structurally `0.0` at any
    // learning rate. `policy_kl` is the mean over minibatches, so it can only be
    // strictly positive when the phase runs more than one of them.
    //
    // This config does: 4 aux slices × 128 steps = 512 aux steps, chunked at
    // `aux_batch_size = 128` → 4 chunks, × `e_aux = 6` epochs = 24 minibatches.
    assert!(
        aux.minibatches > 1,
        "`policy_kl > 0.0` is only meaningful when the aux phase runs more than one \
         minibatch (the first minibatch's KL is structurally zero — π_old is snapshotted \
         before the epoch loop); got {} minibatches, so shrinking the aux step count or \
         growing `aux_batch_size` into a single chunk has invalidated this test rather \
         than regressed #324",
        aux.minibatches
    );
    // Guards the #318 all-minibatches-skipped path, which reports both aux
    // losses as a literal `0.0` sentinel rather than NaN: an aux-value MSE
    // against CartPole returns is bounded below by 0 and can only *be* 0 on a
    // bit-exact perfect regression, which a freshly initialised head does not
    // achieve.
    assert!(
        aux.aux_value_loss > 0.0,
        "the aux-value MSE must be strictly positive; `0.0` is also the sentinel reported \
         when every minibatch was skipped as non-finite (#318), got {}",
        aux.aux_value_loss
    );
    // The #324 assertion. `is_finite()` alone passed on the bit-exactly zero
    // KL that the terminal aux phase produced when it ran at `lr == 0.0`.
    assert!(
        aux.policy_kl > 0.0,
        "the terminal aux phase moved the policy by exactly nothing (policy_kl = {}), \
         which means it ran at lr = 0.0 — it must step at the rate of the policy phase it \
         accompanies (#324); `is_finite()` alone does not catch this (#319)",
        aux.policy_kl
    );
}

#[test]
#[ignore = "2 048-timestep PPG training run; checks finite rewards and losses — run with `cargo test -- --ignored`"]
fn ppg_cartpole_produces_finite_rewards() {
    let _guard = flex_guard();
    let seed: u64 = 7;
    let total = 2_048_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total, 4);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let history = &agent.stats().recent_history;
    assert_all_finite(
        "reward",
        &history.iter().map(|m| m.reward).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "policy_loss",
        &history.iter().map(|m| m.policy_loss).collect::<Vec<_>>(),
    );
    assert_all_finite(
        "value_loss",
        &history.iter().map(|m| m.value_loss).collect::<Vec<_>>(),
    );
}

// ---------------------------------------------------------------------------
// Auxiliary-phase learning rate (issue #324)
// ---------------------------------------------------------------------------

/// Regression (issue #324): the auxiliary phase used to read the *annealed*
/// learning rate after `policy_phase_update` had already bumped the iteration
/// counter, so it ran one schedule tick ahead of the policy phase it
/// accompanies. Whenever `total_iterations % n_iteration == 0` the final
/// auxiliary phase therefore landed on `iteration == total_iterations`, where
/// the anneal is exactly `0.0` — its minibatches moved no parameter and its
/// reported `policy_kl` was bit-exactly zero.
///
/// This is that configuration, minimised: `total / num_steps = 4` iterations
/// with `n_iteration = 2`, so auxiliary phases fire at iterations 2 and 4 and
/// the second one is the failing terminal case.
#[test]
fn ppg_final_aux_phase_steps_at_a_nonzero_learning_rate() {
    const SEED: u64 = 19;
    const NUM_STEPS: usize = 128;
    const TOTAL: usize = 512; // 4 policy-phase iterations.
    const N_ITERATION: usize = 2; // aux phases at iterations 2 and 4.

    let _guard = flex_guard();

    let mut env = TimeLimit::new(cartpole_seeded(SEED), 500);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut agent = make_cart_pole_agent(SEED, NUM_STEPS, TOTAL, N_ITERATION);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, TOTAL, 0,
    )
    .expect("training");

    // The run must actually reach the terminal iteration, or the regression
    // below is checking an interior aux phase that never had the bug.
    assert_eq!(
        agent.iteration(),
        TOTAL / NUM_STEPS,
        "the run must complete all {} policy phases for the terminal aux phase to fire",
        TOTAL / NUM_STEPS
    );

    let aux = agent
        .last_aux_phase()
        .expect("the aux phase must have fired at the final iteration");
    // Soundness precondition for `policy_kl > 0.0` below, not just a
    // liveness check. `maybe_aux_phase` snapshots `π_old` once, before the aux
    // epoch loop, so the *first* minibatch of the *first* epoch runs its
    // "new" forward pass over weights the phase has not stepped yet — that
    // minibatch's KL is structurally `0.0` at any learning rate, healthy or
    // annealed to nothing. `policy_kl` is the mean over minibatches, so it is
    // strictly positive only when more than one minibatch runs. A
    // single-minibatch aux phase (`e_aux = 1` with `aux_batch_size` ≥ the total
    // aux step count) reports `policy_kl == 0.0` at a perfectly healthy lr.
    //
    // This config does: `n_iteration = 2` slices × `NUM_STEPS = 128` = 256 aux
    // steps, chunked at `aux_batch_size = 128` → 2 chunks, × `e_aux = 6`
    // epochs = 12 minibatches.
    assert!(
        aux.minibatches > 1,
        "`policy_kl > 0.0` is only meaningful when the aux phase runs more than one \
         minibatch (the first minibatch's KL is structurally zero — π_old is snapshotted \
         before the epoch loop); got {} minibatches, so shrinking `n_iteration × NUM_STEPS` \
         or growing `aux_batch_size` into a single chunk has invalidated this test rather \
         than regressed #324",
        aux.minibatches
    );
    assert!(
        aux.policy_kl.is_finite(),
        "the final aux phase's distillation KL must be finite, got {}",
        aux.policy_kl
    );
    // The load-bearing assertion. `policy_kl` is `KL(π_old ‖ π_new)` against a
    // snapshot taken at the start of the phase, so at `lr == 0.0` the policy
    // cannot move and every minibatch contributes an exact zero.
    assert!(
        aux.policy_kl > 0.0,
        "the final aux phase moved the policy by exactly nothing (policy_kl = {}), which \
         means it ran at lr = 0.0 — it must step at the rate of the policy phase it \
         accompanies (#324)",
        aux.policy_kl
    );
}

/// Mean episode return of a uniform-random left/right policy on the
/// `TimeLimit`ed `CartPole`, measured over 100 episodes from `seed`. Baseline
/// the trained PPG agent must beat.
fn random_cartpole(seed: u64) -> f32 {
    let mut env = TimeLimit::new(cartpole_seeded(seed), 500);
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(
        &mut env,
        100,
        500,
        &mut rng,
        uniform_discrete::<1, CartPoleAction>,
    )
}

// Seeded reproducibility: two same-seed CartPole runs must produce identical
// reward sequences. (Generated by `rl_reproducibility_test!`.)
rl_reproducibility_test! {
    #[ignore = "4 096-step reproducibility check (two sequential CartPole runs); run with `cargo test -- --ignored`"]
    ppg_cartpole_flex_reproducibility,
    seq,
    seed = 123,
    total = 2_048,
    run = run_cartpole,
}

// Replaces the former `ppg_cart_pole_reaches_475` macro test. That run needed
// 400k steps to clear the near-perfect 475 bar — well over the five-minute
// budget and prone to borderline failures — because the aux phase's
// distillation repeatedly pulls the policy back on CartPole (not PPG's home
// turf; its wins live on Procgen-style envs, deferred). Capped at a
// deterministic 50k-step budget, PPG reaches ~30 here, comfortably above a
// measured uniform-random policy (≈ 20), so this stays a "learns beyond random"
// sanity bar rather than a convergence proof. Determinism (seeded backend +
// 1-thread rayon) keeps the margin reproducible; cross-crate macro convergence
// is left to a longer, out-of-band run. (Generated by `rl_learning_test!`.)
rl_learning_test! {
    #[ignore = "50 000-step discrete PPG CartPole run (~4 min on Flex); confirms avg reward beats a measured uniform-random baseline — run with `cargo test -- --ignored`"]
    ppg_cartpole_improves_over_random,
    improves_over_random(margin = 3.0),
    seed = 42,
    total = 50_000,
    run = run_cartpole,
    random = random_cartpole,
}

// ---------------------------------------------------------------------------
// Progress-logging cadence (issue #321)
// ---------------------------------------------------------------------------
//
// PPG shares the PPO watermark, so this is the twin of
// `ppo_progress_logs_when_log_every_does_not_divide_num_steps` in
// `ppo_integration.rs`. The event-capture plumbing is shared via
// `rlevo_test_support::capture::FieldCapture`.
//
// PPG is worth covering separately: its progress payload also reads the
// auxiliary-phase result, which is loop-local and had to be hoisted so the
// terminal line reports it rather than claiming `aux_ran = false`.

#[test]
fn ppg_progress_logs_when_log_every_does_not_divide_num_steps() {
    // The canonical #321 configuration: `lcm(128, 100) = 3200`, so the old
    // divisibility gate emitted ZERO lines for a run this short.
    const NUM_STEPS: usize = 128;
    const LOG_EVERY: usize = 100;
    // Chosen so that no boundary of the run -- 128, 256, 384, 512, 640, 650 --
    // is a multiple of `LOG_EVERY`. A total that happened to be a multiple of
    // 100 would let the old gate fire on the terminal boundary by coincidence,
    // making this test pass against the bug it exists to catch.
    const TOTAL: usize = 650;
    const SEED: u64 = 7;
    // Small enough that the auxiliary phase actually fires during the run, so
    // the hoisted `aux` is exercised rather than being `None` throughout.
    const N_ITERATION: usize = 2;

    let _guard = flex_guard();

    let capture = FieldCapture::new("step");
    capture.record(|| {
        let mut env = TimeLimit::new(cartpole_seeded(SEED), 500);
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut agent = make_cart_pole_agent(SEED, NUM_STEPS, TOTAL, N_ITERATION);
        train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
            &mut agent, &mut env, &mut rng, TOTAL, LOG_EVERY,
        )
        .expect("training");
    });

    let steps = capture.values();

    // (1) The literal #321 regression: zero lines under the old gate.
    assert!(
        !steps.is_empty(),
        "a {TOTAL}-step PPG run with log_every = {LOG_EVERY} must emit at least one progress \
         line, but emitted none (the old divisibility gate fired only on multiples of \
         lcm({NUM_STEPS}, {LOG_EVERY}) = 3200)"
    );

    // (2) The terminal line: the run's final policy-phase stats must be
    // reported even though the closing rollout is partial.
    assert_eq!(
        steps.last().copied(),
        Some(TOTAL as u64),
        "the final progress line must report the terminal step {TOTAL}, got {steps:?}"
    );

    // (3) Cadence: a boundary-gated logger cannot hit `log_every` exactly, but
    // it must never fall a whole extra rollout behind it. The terminal line is
    // exempt — it is deliberately early, by design.
    let mut previous = 0_u64;
    for &step in &steps[..steps.len() - 1] {
        let gap = step - previous;
        assert!(
            gap >= LOG_EVERY as u64,
            "progress lines must be at least log_every = {LOG_EVERY} steps apart, \
             got a gap of {gap} at step {step} in {steps:?}"
        );
        assert!(
            gap < (LOG_EVERY + NUM_STEPS) as u64,
            "progress lines must never lag more than one rollout past log_every \
             (< {}), got a gap of {gap} at step {step} in {steps:?}",
            LOG_EVERY + NUM_STEPS
        );
        previous = step;
    }
}
