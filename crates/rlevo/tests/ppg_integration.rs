//! End-to-end integration tests for PPG.
//!
//! Budget conventions mirror the PPO tests: default-run tests use a modest
//! 50k-step budget with lax thresholds; heavier macro-convergence and
//! reproducibility checks live behind `#[ignore]`.

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
use rlevo_reinforcement_learning::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::ppo::ppo_value::PpoValue;

#[derive(Module, Debug)]
struct ValueMlp<B: Backend> {
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

fn make_cart_pole_agent(
    seed: u64,
    num_steps: usize,
    total_timesteps: usize,
    n_iteration: usize,
) -> PpgAgent<Be, PpgCategoricalPolicyHead<Be>, ValueMlp<Be>, CartPoleObservation, 1, 2> {
    let device = Default::default();
    <Be as Backend>::seed(&device, seed);

    let policy = PpgCategoricalPolicyHeadConfig {
        obs_dim: 4,
        hidden: 64,
        num_actions: 2,
    }
    .init::<Be>(&device);
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
        .build();
    let config = PpgConfigBuilder::new()
        .ppo(ppo)
        .n_iteration(n_iteration)
        .e_aux(6)
        .beta_clone(1.0)
        .aux_batch_size(128)
        .build();
    let total_iterations = total_timesteps / config.batch_size().max(1);
    PpgAgent::new(policy, value, config, device, total_iterations)
}

#[test]
fn ppg_cart_pole_reaches_modest_threshold() {
    // CartPole is not PPG's home turf: the auxiliary phase's distillation
    // periodically pulls the policy back, slowing convergence relative to
    // PPO. PPG's sample-efficiency wins live on Procgen-style envs with CNN
    // encoders + vectorised rollout, both deferred. The threshold here is
    // a sanity bar that PPG learns *something* beyond random (random ≈ 20).
    let seed: u64 = 42;
    let total = 50_000_usize;
    let num_steps = 128_usize;

    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total, 32);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(avg >= 30.0, "expected avg reward >= 30, got {avg:.2}");
}

#[test]
fn ppg_without_aux_phase_matches_ppo_baseline() {
    // Sanity check that the policy-phase update is a faithful PPO update:
    // with n_iteration set above the total iteration count the auxiliary
    // phase never fires, and PPG should match PPO's ~50k-step CartPole
    // threshold (80 in `ppo_integration.rs`).
    let seed: u64 = 42;
    let total = 50_000_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
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

#[test]
fn ppg_aux_phase_actually_runs() {
    // Use a small n_iteration so the aux phase fires within a tiny budget.
    let seed: u64 = 11;
    let num_steps = 128_usize;
    let total = 2_048_usize; // 16 iterations at num_steps=128.
    let n_iteration = 4_usize;
    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
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
    assert!(aux.minibatches > 0);
    assert!(aux.aux_value_loss.is_finite());
    assert!(aux.policy_kl.is_finite());
}

#[test]
#[ignore = "perturbs Burn's global ndarray RNG; run with --test-threads=1"]
fn ppg_short_run_produces_finite_rewards() {
    let seed: u64 = 7;
    let total = 2_048_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total, 4);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    for (i, m) in agent.stats().recent_history.iter().enumerate() {
        assert!(m.reward.is_finite(), "non-finite reward at episode {i}");
        assert!(
            m.policy_loss.is_finite(),
            "non-finite policy_loss at ep {i}"
        );
        assert!(m.value_loss.is_finite(), "non-finite value_loss at ep {i}");
    }
}

#[test]
#[ignore = "macro convergence; ~2-5 min on ndarray"]
fn ppg_cart_pole_reaches_475() {
    let seed: u64 = 42;
    let total = 400_000_usize;
    let num_steps = 128_usize;
    let mut env = TimeLimit::new(
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }),
        500,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut agent = make_cart_pole_agent(seed, num_steps, total, 32);
    train_discrete::<Be, _, _, _, _, CartPoleAction, _, 1, 1, 2>(
        &mut agent, &mut env, &mut rng, total, 0,
    )
    .expect("training");
    let avg = agent.stats().avg_score().unwrap_or(0.0);
    assert!(avg >= 475.0, "expected avg reward >= 475, got {avg:.2}");
}
