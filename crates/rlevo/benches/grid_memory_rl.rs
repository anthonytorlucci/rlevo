//! Random baseline vs. feedforward DQN on [`MemoryEnv`] — quality + throughput.
//!
//! ## What this bench is actually showing
//!
//! `MemoryEnv` is a POMDP recall task: a cue object (a Key or a Ball, both
//! green) is shown once in the start room, the agent walks a corridor out of
//! sight of it, and at the far end must issue `Done` facing the fork object
//! *whose type equals the cue*. The cue type and the fork order are sampled
//! afresh every episode, and the layout (`size >= 11`, **Invariant M** in the
//! [`memory`](rlevo_environments::grids::memory) module docs) guarantees the cue
//! is outside the egocentric view from every cell the agent can answer from.
//!
//! This bench runs [`MemoryConfig::default`] (currently `size = 13`,
//! `max_steps = 845`) rather than pinning its own literals, so it always
//! benchmarks the configuration rlevo actually ships.
//!
//! **That was not true until ADR 0043 landed (issue #109).** Before the fix the
//! cue was a compile-time constant and the reward was keyed to a fixed
//! coordinate, so a feedforward DQN could solve the environment outright — this
//! bench's previous module doc claimed the opposite ("a memoryless network
//! cannot carry the cue across the corridor") and was simply wrong about the
//! environment it was benchmarking. The claim is true *now*.
//!
//! ## The number to read: success rate, against the 50% ceiling
//!
//! At the fork the observation is **cue-invariant** — two episodes differing only
//! in cue type produce byte-identical observations while demanding opposite
//! turns. A reactive/memoryless policy therefore cannot do better than guess
//! between the two arms, which caps it at **50%** success:
//!
//! | Policy | Success-rate ceiling |
//! |---|---|
//! | Any memoryless policy (random, feedforward DQN, …) | **50%** — chance on the binary fork |
//! | Memory-augmented (recurrent) policy | can approach **100%** |
//!
//! The ceiling is 50%, **not** 0%: a reactive policy that reaches the fork and
//! commits still collects the reward on the half of episodes where its guess
//! happens to be right. That distinction is why this bench prints **success
//! rate** as the headline column and pins a literal `reactive ceiling` reference
//! row under it — mean reward alone smears the ceiling into an uninterpretable
//! scalar, because the reward also decays with episode length.
//!
//! ## What the numbers actually come out as (and why)
//!
//! A ceiling is not an achievement. Both policies benchmarked here currently
//! score **≈ 0%**, which is *below* the 50% ceiling, for a reason that has
//! nothing to do with memory:
//!
//! - **Random** fires `Done` with probability `1/7` at every step, so it ends the
//!   episode after ~7 actions — long before it could have walked the corridor
//!   (`size - 3 == 10` forward steps at the default size). It essentially never
//!   reaches the fork to guess at all.
//! - **The feedforward DQN**, at the `TRAIN_TIMESTEPS` budget below, does not
//!   learn to traverse the corridor either: the reward is sparse (paid only on a
//!   correct `Done` at the fork) and the horizon is long. It is capped at 50% *in
//!   principle* and sits near 0% *in practice*.
//!
//! So read the table as two separate failures, not one: a score **under** 50%
//! means the policy never got to the fork; a score **at** ~50% would mean it got
//! there and guessed; a score **consistently over** ~55% on a large evaluation
//! would be a red flag that the cue has leaked back into the observation and
//! issue #109 has regressed.
//!
//! rlevo ships no recurrent policy today, so the memory-augmented row above has
//! no column here. The feedforward DQN is a *control*, not a success story.
//!
//! 1. **Quality comparison** — trains a DQN, then prints success rate (headline)
//!    and mean terminal reward for the random policy vs. the greedy DQN policy,
//!    against the reactive ceiling.
//! 2. **Throughput** — a Criterion group timing per-step rollout cost of the
//!    random policy vs. DQN-greedy inference.
//!
//! The flatten + MLP model over the `7×7×3` observation is shared with the
//! other `*_dqn` benches via [`support`].
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo --bench grid_memory_rl
//! ```

#[path = "support/value_nets.rs"]
mod support;

use std::hint::black_box;

use burn::backend::{Autodiff, Flex};

use criterion::{BenchmarkId, Criterion, Throughput};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::grids::core::{GridObservation, OBS_CHANNELS, VIEW_SIZE};
use rlevo_environments::grids::{GridAction, MemoryConfig, MemoryEnv};

use rlevo_reinforcement_learning::algorithms::dqn::dqn_agent::DqnAgent;
use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
use rlevo_reinforcement_learning::algorithms::dqn::train::train;
use rlevo_reinforcement_learning::target::TargetUpdate;

use support::GridMlpDqn;

const SEED: u64 = 2026;
/// Flattened observation width: `7 * 7 * 3 = 147`.
const OBS_FEATURES: usize = VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS;
const ACTIONS: usize = GridAction::ACTION_COUNT;
const TRAIN_TIMESTEPS: usize = 40_000;
const EVAL_EPISODES: usize = 200;
/// Success-rate ceiling for any policy without memory: the fork is binary and
/// the observation at the decision cell is cue-invariant, so the arm is a coin
/// flip. A reactive policy is capped here — it is **not** pinned to zero.
const REACTIVE_CEILING: f32 = 0.5;

type Backend_ = Autodiff<Flex>;
type MemoryAgent = DqnAgent<Backend_, GridMlpDqn<Backend_>, GridObservation, GridAction, 3, 4>;

/// The shipped default (`size`, `max_steps`) under this bench's fixed seed.
///
/// Deliberately *not* a set of local `SIZE` / `MAX_STEPS` literals: those drifted
/// out of step with `MemoryConfig::default()` once, and a bench that silently
/// measures a configuration nobody runs is worse than no bench. Overriding only
/// `seed` keeps the geometry pinned to whatever rlevo actually ships.
fn memory_config() -> MemoryConfig {
    MemoryConfig {
        seed: SEED,
        ..MemoryConfig::default()
    }
}

/// Trains a feedforward DQN on a fresh, seeded `MemoryEnv`. See the module
/// docs: this is a *control*, not an attempt to solve the task — a memoryless
/// policy is capped at the ~50% chance rate on the binary fork.
fn train_dqn() -> MemoryAgent {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut env = MemoryEnv::with_config(memory_config(), false).expect("valid config");

    let config = DqnTrainingConfigBuilder::new()
        .batch_size(64)
        .gamma(0.99)
        .target_update(TargetUpdate::polyak(0.005, 1))
        .learning_rate(5e-4)
        .epsilon_start(1.0)
        .epsilon_end(0.01)
        .epsilon_decay(0.9997)
        .learning_starts(500)
        .train_frequency(4)
        .replay_buffer_capacity(20_000)
        .double_q(true)
        .build()
        .expect("valid config");

    let model: GridMlpDqn<Backend_> = GridMlpDqn::new(OBS_FEATURES, 128, ACTIONS, &device);
    let mut agent: MemoryAgent = DqnAgent::new(model, config, device).expect("valid config");

    train(&mut agent, &mut env, &mut rng, TRAIN_TIMESTEPS, 0).expect("dqn training loop");
    agent
}

/// Runs one episode under `next_action`. Returns the terminal reward and
/// whether the correct fork was chosen (reward > 0).
fn roll_out(
    env: &mut MemoryEnv,
    mut next_action: impl FnMut(&GridObservation) -> GridAction,
) -> (f32, bool) {
    let mut snapshot = env.reset().expect("reset");
    loop {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        if snapshot.is_done() {
            let reward = f32::from(*snapshot.reward());
            return (reward, reward > 0.0);
        }
    }
}

/// Estimates `(mean_reward, success_rate)` over `EVAL_EPISODES` episodes.
#[allow(clippy::cast_precision_loss)]
fn evaluate(mut next_action: impl FnMut(&GridObservation) -> GridAction) -> (f32, f32) {
    let mut env = MemoryEnv::with_config(memory_config(), false).expect("valid config");
    let mut total_reward = 0.0_f32;
    let mut successes = 0_usize;
    for _ in 0..EVAL_EPISODES {
        let (reward, solved) = roll_out(&mut env, &mut next_action);
        total_reward += reward;
        successes += usize::from(solved);
    }
    let n = EVAL_EPISODES as f32;
    (total_reward / n, successes as f32 / n)
}

fn print_quality_comparison(agent: &MemoryAgent) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let (rand_reward, rand_success) =
        evaluate(|_obs| GridAction::from_index(rng.random_range(0..ACTIONS)));

    let mut eval_rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let (dqn_reward, dqn_success) = evaluate(|obs| agent.act(obs, &mut eval_rng));

    let cfg = memory_config();
    println!();
    println!(
        "MemoryEnv policy quality | size={} max_steps={} \
         episodes={EVAL_EPISODES} train_steps={TRAIN_TIMESTEPS}",
        cfg.size, cfg.max_steps
    );
    println!("  Success rate is the headline. Both policies below are memoryless, so both are");
    println!("  capped at chance on the binary fork; only a recurrent policy (not yet in rlevo)");
    println!("  can exceed that. A score *below* the ceiling means the policy never reached");
    println!("  the fork at all; a score consistently *above* it means the cue has leaked.");
    println!("  policy             success_rate   mean_reward");
    println!("  random             {rand_success:>12.3}  {rand_reward:>12.4}");
    println!("  dqn (greedy)       {dqn_success:>12.3}  {dqn_reward:>12.4}");
    println!("  reactive ceiling   {REACTIVE_CEILING:>12.3}  {:>12}", "-");
    println!();
}

/// Drives `steps` of a `next_action`-chosen rollout, resetting on episode end.
fn rollout_steps(steps: usize, mut next_action: impl FnMut(&GridObservation) -> GridAction) {
    let mut env = MemoryEnv::with_config(memory_config(), false).expect("valid config");
    let mut snapshot = env.reset().expect("reset");
    for _ in 0..steps {
        let action = next_action(snapshot.observation());
        snapshot = env.step(action).expect("step");
        if snapshot.is_done() {
            snapshot = env.reset().expect("reset");
        }
    }
}

fn bench_policies(c: &mut Criterion, agent: &MemoryAgent) {
    let mut group = c.benchmark_group("grid_memory_policy_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(Throughput::Elements(steps as u64));

        group.bench_with_input(BenchmarkId::new("random", steps), &steps, |b, &steps| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(SEED);
                rollout_steps(black_box(steps), |_obs| {
                    GridAction::from_index(rng.random_range(0..ACTIONS))
                });
            });
        });

        group.bench_with_input(
            BenchmarkId::new("dqn_greedy", steps),
            &steps,
            |b, &steps| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(SEED);
                    rollout_steps(black_box(steps), |obs| agent.act(obs, &mut rng));
                });
            },
        );
    }
    group.finish();
}

fn main() {
    let agent = train_dqn();
    print_quality_comparison(&agent);

    let mut criterion = Criterion::default().configure_from_args();
    bench_policies(&mut criterion, &agent);
    criterion.final_summary();
}
