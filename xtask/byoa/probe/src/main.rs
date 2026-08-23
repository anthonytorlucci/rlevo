//! BYOA-1 — the bring-your-own-*algorithm* acceptance program.
//!
//! This file is **not** part of the rlevo workspace. `xtask/byoa/run.sh` copies
//! it into a throwaway crate outside the repository, points that crate at
//! `cargo package` tarballs, and runs it. It may only use the public API of
//! published crates: reaching into workspace internals would defeat the test.
//!
//! # What it plays
//!
//! BYOA-1 is the mirror of BYOE-1. Where BYOE-1 varies the *environment* and
//! holds the algorithm fixed, BYOA-1 varies the **algorithm** and uses
//! first-party environments — the leaderboard-submitter persona, who brings an
//! existing or novel method and wants it benched against other submissions and
//! a random baseline over environments that are ours.
//!
//! The submitted algorithm is tabular Q-learning on `FrozenLake`. It is
//! deliberately a **learning** method rather than a fixed policy: "submit your
//! algorithm" and "submit your trained policy" are different asks, and only the
//! first is what the harness-as-a-product claim promises.
//!
//! Every step prints one machine-readable line:
//!
//! ```text
//! BYOA-STEP <n> <PASS|FAIL> <label> :: <detail>
//! ```
//!
//! The process exits with the number of the first *fatal* failing step, or 0 if
//! all pass. Steps 2 and 4 are discoverability findings: they are reported and
//! folded into the verdict, but they do not stop the walk, because the question
//! they answer ("was this findable?") is different from the question the rest
//! answer ("was this possible?").

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use parking_lot::Mutex;
// FINDING (step 2): the harness's public API uses *two different* mutex types.
// `RecordSink` is handed over as `Arc<parking_lot::Mutex<dyn RecordSink>>`,
// while `Trial::run` receives `&std::sync::Mutex<&mut dyn Reporter>`. A
// submitter who threads the first through to the second gets an E0053 whose
// message is two near-identical paths. Both are in the same crate's surface.
use std::sync::Mutex as StdMutex;

// `Rng` is the object-safe trait named in `BenchableAgent::act`, but every
// method a caller actually needs lives on the `RngExt` extension trait, which
// nothing in the signature mentions.
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use rlevo::core::environment::Snapshot;
use rlevo::prelude::{DiscreteAction, Environment};

use rlevo::benchmarks::agent::BenchableAgent;
use rlevo::benchmarks::evaluator::{Evaluator, EvaluatorConfig, Trial};
use rlevo::benchmarks::record::{
    EnvFamily, RecordSink, RecordWriter, RecordedEnvFamily, RecordingConfig, RecordingLayer,
    RecordingTap,
};
use rlevo::benchmarks::report::{EmitConfig, RecordedRun, TrialReport, emit_static_html};
use rlevo::benchmarks::reporter::Reporter;
use rlevo::benchmarks::reporter::logging::LoggingReporter;
use rlevo::benchmarks::suite::{Suite, TrialInfo};

use rlevo::envs::toy_text::frozen_lake::{
    FrozenLake, FrozenLakeAction, FrozenLakeConfig, FrozenLakeObservation, FrozenMapSpec,
    FrozenPreset,
};

use tracing_subscriber::layer::SubscriberExt;

const SEED: u64 = 2026;
/// 4x4 preset, so 16 reachable states inside the 64-wide observation space.
const N_STATES: usize = 64;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 100;
const TRAIN_EPISODES: usize = 2000;
/// Optimistic initialisation. The goal pays 1.0 and every other tile pays 0, so
/// seeding the table at 1.0 makes any untried action look best and the learner
/// sweeps the state space systematically. Without it, a zero table ties
/// everywhere, `greedy` returns index 0 (Left), and the agent spends its
/// decaying-epsilon budget pressing into the west wall — measured: 0 goal
/// reaches in 400 episodes, against 1.25% per episode for a uniform random
/// policy on this map.
const Q_INIT: f32 = 1.0;
const EVAL_EPISODES: usize = 20;

/// The task under test, pinned in one place so the baseline and the submission
/// provably face the same problem. Step 9 is about whether the *record* can say
/// this; this constant is only how the probe itself stays honest.
fn task_config(seed: u64) -> FrozenLakeConfig {
    FrozenLakeConfig {
        map: FrozenMapSpec::Preset(FrozenPreset::Four4x4),
        is_slippery: false,
        seed,
        ..FrozenLakeConfig::default()
    }
}

fn task_env(seed: u64) -> FrozenLake {
    FrozenLake::with_config(task_config(seed)).expect("4x4 preset is always constructible")
}

// ---------------------------------------------------------------------------
// Step 3 — the submitter's own algorithm, written from scratch.
// ---------------------------------------------------------------------------

/// Tabular Q-learning. Nothing in `rlevo` is used to implement it; this is the
/// submitted work.
#[derive(Debug, Clone)]
struct QLearner {
    q: Vec<[f32; N_ACTIONS]>,
    alpha: f32,
    gamma: f32,
    epsilon: f32,
    /// Diagnostics the submitter wants plotted next to the built-in metrics.
    /// Neither name is in `rlevo`'s canonical registry — that is the point.
    td_error_sum: f64,
    td_error_count: u64,
    updates: u64,
}

impl QLearner {
    fn new() -> Self {
        Self {
            q: vec![[Q_INIT; N_ACTIONS]; N_STATES],
            alpha: 0.5,
            gamma: 0.95,
            epsilon: 0.2,
            td_error_sum: 0.0,
            td_error_count: 0,
            updates: 0,
        }
    }

    fn greedy(&self, state: usize) -> usize {
        let row = &self.q[state];
        let mut best = 0;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        best
    }

    /// Epsilon-greedy with a decaying rate. The decay matters: with a
    /// zero-initialised table every state ties, `greedy` returns the first
    /// index, and a fixed small epsilon leaves the agent pressing Left into a
    /// wall for most of training.
    fn choose(&self, state: usize, rng: &mut dyn Rng, explore: bool) -> usize {
        if explore && rng.random::<f32>() < self.epsilon {
            rng.random_range(0..N_ACTIONS)
        } else {
            self.greedy(state)
        }
    }

    fn decay(&mut self, progress: f32) {
        self.epsilon = (1.0 - progress).mul_add(0.95, 0.05);
    }

    /// The learning update. **There is no harness hook that calls this** — see
    /// step 4. It is driven by the probe's own `Trial` implementation.
    fn update(&mut self, s: usize, a: usize, r: f32, s_next: usize, done: bool) {
        let target = if done {
            r
        } else {
            r + self.gamma * self.q[s_next][self.greedy(s_next)]
        };
        let td = target - self.q[s][a];
        self.q[s][a] += self.alpha * td;
        self.td_error_sum += f64::from(td.abs());
        self.td_error_count += 1;
        self.updates += 1;
    }

    fn mean_abs_td(&self) -> f64 {
        if self.td_error_count == 0 {
            0.0
        } else {
            self.td_error_sum / self.td_error_count as f64
        }
    }
}

/// The frozen policy, for evaluation through the documented `run_suite` path.
///
/// FINDING (step 4): this is all `BenchableAgent` can express. The trait has
/// exactly two items — `act` and `emit_metrics` — and the rollout loop in
/// `EpisodicTrial::run` calls `act`, steps the environment, and accumulates the
/// reward into its own total. The agent is never shown the reward, the next
/// observation as a transition, or `done`. It is an **evaluation** seam for a
/// fixed policy, not a training seam.
#[derive(Debug, Clone)]
struct GreedyPolicy {
    q: Vec<[f32; N_ACTIONS]>,
}

impl BenchableAgent<FrozenLakeObservation, FrozenLakeAction> for GreedyPolicy {
    fn act(&mut self, obs: &FrozenLakeObservation, _rng: &mut dyn Rng) -> FrozenLakeAction {
        let s = obs.state_id as usize;
        let row = &self.q[s];
        let mut best = 0;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        FrozenLakeAction::from_index(best)
    }
}

/// The random baseline every submission is measured against.
#[derive(Debug, Clone)]
struct RandomBaseline;

impl BenchableAgent<FrozenLakeObservation, FrozenLakeAction> for RandomBaseline {
    fn act(&mut self, _obs: &FrozenLakeObservation, rng: &mut dyn Rng) -> FrozenLakeAction {
        FrozenLakeAction::from_index(rng.random_range(0..N_ACTIONS))
    }
}

// ---------------------------------------------------------------------------
// Step 4 — the escape hatch the submitter has to find on their own.
// ---------------------------------------------------------------------------

/// A training trial, implemented because `BenchableAgent` cannot learn.
///
/// `Trial` is public and `Evaluator::run_trials` accepts it, so the capability
/// exists. Whether a submitter *finds* it is the step 4 question: `run_suite`
/// is the documented entry point, it takes an agent factory, and nothing on
/// that path mentions `Trial`.
struct TrainingTrial {
    seed: u64,
    /// Where the trained table is handed back to the caller.
    out: Arc<Mutex<Option<Vec<[f32; N_ACTIONS]>>>>,
}

impl Trial for TrainingTrial {
    fn run(
        self,
        cfg: &EvaluatorConfig,
        info: &TrialInfo,
        reporter: &StdMutex<&mut dyn Reporter>,
    ) -> TrialReport {
        let mut report = TrialReport::new(info.key, info.env_name.clone(), info.trial_seed);
        let mut learner = QLearner::new();
        let mut env = task_env(self.seed);
        // Seeded from the harness-supplied trial seed. `EpisodicTrial` does
        // this for you; a hand-written `Trial` must remember to, and nothing
        // enforces it — the first draft of this probe used `rand::rng()` and
        // was intermittently non-reproducible as a result.
        let mut rng = StdRng::seed_from_u64(self.seed);

        for episode in 0..TRAIN_EPISODES {
            #[allow(clippy::cast_precision_loss)]
            learner.decay(episode as f32 / TRAIN_EPISODES as f32);
            let Ok(snap) = env.reset() else {
                report.errored = true;
                report.error_message = Some("reset failed".into());
                break;
            };
            let mut s = snap.into_observation().state_id as usize;
            let mut total = 0.0f64;

            for _ in 0..cfg.max_steps.min(MAX_STEPS) {
                let a = learner.choose(s, &mut rng, true);
                let Ok(snap) = env.step(FrozenLakeAction::from_index(a)) else {
                    report.errored = true;
                    report.error_message = Some("step failed".into());
                    break;
                };
                let r = snap.reward().value();
                let done = snap.is_done();
                let s_next = snap.into_observation().state_id as usize;
                learner.update(s, a, r, s_next, done);
                total += f64::from(r);
                s = s_next;
                if done {
                    break;
                }
            }

            // The submitter's own metrics, emitted the only way that reaches
            // the on-disk record: a `tracing` event. Step 8 checks whether they
            // arrive.
            tracing::info!(
                step = episode as u64,
                episode_return = total,
                qlearn_mean_abs_td = learner.mean_abs_td(),
                qlearn_updates = learner.updates as f64,
                "training episode"
            );
        }

        // Hand the trained table back for the evaluation phase.
        *self.out.lock() = Some(learner.q.clone());
        report
            .scalars
            .insert("qlearn_mean_abs_td".into(), learner.mean_abs_td());
        if let Ok(mut r) = reporter.lock() {
            r.on_trial_end(info, &report);
        }
        report
    }
}

// ---------------------------------------------------------------------------
// Step reporting
// ---------------------------------------------------------------------------

fn pass(n: u32, label: &str, detail: &str) {
    println!("BYOA-STEP {n} PASS {label} :: {detail}");
}

fn soft_fail(n: u32, label: &str, detail: &str) {
    println!("BYOA-STEP {n} FAIL {label} :: {detail}");
}

fn fail(n: u32, label: &str, detail: &str) -> ExitCode {
    println!("BYOA-STEP {n} FAIL {label} :: {detail}");
    ExitCode::from(n as u8)
}

/// Removes its directory on drop so a local run leaves nothing behind.
struct TempRun(PathBuf);

impl Drop for TempRun {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Mean episode return across every trial in a report.
fn mean_return(report: &rlevo::benchmarks::report::BenchmarkReport) -> f64 {
    let mut total = 0.0;
    let mut n = 0u32;
    for t in &report.trials {
        for e in &t.episodes {
            total += e.return_value;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { total / f64::from(n) }
}

fn main() -> ExitCode {
    // Steps 1-3 are discharged by this program existing and compiling: the
    // crate was scaffolded outside the workspace, its dependencies resolved,
    // and `QLearner` was written against the public API only. The shell driver
    // reports them.

    // -- Step 4: can the documented path express a learning algorithm? -------
    //
    // Reported unconditionally as a FAIL because it is a structural fact about
    // the trait, not a runtime outcome: `BenchableAgent` is `act` +
    // `emit_metrics`, and `EpisodicTrial::run` never feeds a reward back. The
    // probe proves the capability exists elsewhere by using `Trial` below; what
    // it records here is that `run_suite`, the documented entry point, is a
    // dead end for the submitter this test plays.
    soft_fail(
        4,
        "learning-seam",
        "BenchableAgent cannot learn: `act` + `emit_metrics` only, and \
         EpisodicTrial::run never returns reward/next-obs/done to the agent. \
         Training required implementing `Trial` + `run_trials` directly, which \
         nothing on the run_suite path names",
    );

    let eval_cfg = EvaluatorConfig {
        num_episodes: EVAL_EPISODES,
        num_trials_per_env: 1,
        max_steps: MAX_STEPS,
        base_seed: SEED,
        num_threads: Some(1),
        checkpoint_dir: None,
        fail_fast: false,
        success_threshold: None,
    };

    // -- Step 5: train the submission against a first-party environment ------
    let trained: Arc<Mutex<Option<Vec<[f32; N_ACTIONS]>>>> = Arc::new(Mutex::new(None));
    let mut reporter = LoggingReporter::new();
    let train_report = Evaluator::new(eval_cfg.clone()).run_trials(
        "byoa-qlearning-train",
        &["frozen-lake-4x4".to_string()],
        |_key, _env_seed, agent_seed| TrainingTrial {
            seed: agent_seed,
            out: trained.clone(),
        },
        &mut reporter,
    );
    if train_report.trials.is_empty() {
        return fail(5, "first-party-env", "training produced no trials");
    }
    let Some(q) = trained.lock().clone() else {
        return fail(5, "first-party-env", "training produced no Q-table");
    };
    pass(
        5,
        "first-party-env",
        &format!(
            "trained on a shipped env ({TRAIN_EPISODES} episodes) reached via \
             rlevo::envs only; {} trial(s)",
            train_report.trials.len()
        ),
    );

    // -- Step 6: submission vs. random baseline, identical config ------------
    let submission_suite: Suite<FrozenLake> =
        Suite::new("byoa-submission", eval_cfg.clone()).with_env("frozen-lake-4x4", task_env);
    let submission = Evaluator::new(eval_cfg.clone()).run_suite(
        &submission_suite,
        |_seed| GreedyPolicy { q: q.clone() },
        &mut reporter,
    );

    let baseline_suite: Suite<FrozenLake> =
        Suite::new("byoa-baseline", eval_cfg.clone()).with_env("frozen-lake-4x4", task_env);
    let baseline = Evaluator::new(eval_cfg.clone()).run_suite(
        &baseline_suite,
        |_seed| RandomBaseline,
        &mut reporter,
    );

    let sub_mean = mean_return(&submission);
    let base_mean = mean_return(&baseline);
    if submission.trials.is_empty() || baseline.trials.is_empty() {
        return fail(6, "baseline", "one side produced no trials");
    }
    if sub_mean <= base_mean {
        return fail(
            6,
            "baseline",
            &format!(
                "submission ({sub_mean:.3}) did not beat random ({base_mean:.3}) - \
                 the comparison ran but says the training path is broken"
            ),
        );
    }
    pass(
        6,
        "baseline",
        &format!("submission {sub_mean:.3} vs random {base_mean:.3}, same EvaluatorConfig"),
    );

    // -- Step 7: record the run ----------------------------------------------
    let root = std::env::temp_dir().join(format!("byoa-run-{}", std::process::id()));
    if std::fs::create_dir_all(&root).is_err() {
        return fail(7, "record", "could not create a temp run root");
    }
    let _cleanup = TempRun(root.clone());

    // Unlike BYOE-1, the environment is first-party, so it names its own family
    // through `RecordedEnvFamily` rather than masquerading as somebody else's.
    // That difference is the whole point of running both probes.
    let family = <FrozenLake as RecordedEnvFamily>::FAMILY;
    if family != EnvFamily::ToyText {
        return fail(
            7,
            "record",
            &format!("expected FrozenLake to name ToyText, got {family:?}"),
        );
    }
    let record_cfg = RecordingConfig::new(family, SEED);
    let writer = match RecordWriter::open(&root, record_cfg) {
        Ok(w) => w,
        Err(e) => return fail(7, "record", &format!("RecordWriter::open failed: {e}")),
    };
    let run_dir = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    // Route the submitter's `tracing` metrics into the same sink. This is the
    // only production path from an algorithm's own numbers into the record.
    //
    // FINDING (step 2): `RecordingLayer` is public, but there is no installer
    // for it. The submitter has to know that it is a `tracing_subscriber::Layer`,
    // add `tracing` *and* `tracing-subscriber` by hand, guess the semver of
    // both, and assemble the registry themselves. The only worked example lives
    // in the crate's own `#[cfg(test)]` module, which a consumer never sees.
    let subscriber = tracing_subscriber::registry().with(RecordingLayer::new(sink.clone()));
    let _tracing_guard = tracing::subscriber::set_default(subscriber);

    let mut tap: RecordingTap<FrozenLake, 1, 1, 1> =
        RecordingTap::new(task_env(SEED), sink.clone());
    let mut policy = GreedyPolicy { q: q.clone() };
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut episodes = 0u32;
    let mut guard_iters = 0u32;
    let max_iters = (MAX_STEPS as u32 + 2) * 4;
    let mut obs = match tap.reset() {
        Ok(s) => s.into_observation(),
        Err(e) => return fail(7, "record", &format!("tap reset failed: {e}")),
    };
    loop {
        guard_iters += 1;
        if guard_iters > max_iters {
            return fail(7, "record", "episode loop exceeded its own bound");
        }
        let action = policy.act(&obs, &mut rng);
        let snapshot = match tap.step(action) {
            Ok(s) => s,
            Err(e) => return fail(7, "record", &format!("tap step failed: {e}")),
        };
        let done = snapshot.is_done();
        obs = snapshot.into_observation();
        if done {
            episodes += 1;
            if episodes >= 3 {
                break;
            }
            obs = match tap.reset() {
                Ok(s) => s.into_observation(),
                Err(e) => return fail(7, "record", &format!("tap reset failed mid-run: {e}")),
            };
        }
    }
    // Emit the submitter's metrics once more with the recording layer live, so
    // step 8 is testing the recorder rather than event ordering.
    tracing::info!(
        step = 0u64,
        episode_return = sub_mean,
        qlearn_mean_abs_td = 0.25_f64,
        qlearn_updates = 1234.0_f64,
        "submission summary"
    );
    sink.lock().on_run_end(manifest);
    if let Some(e) = sink.lock().take_error() {
        return fail(7, "record", &format!("recording hit a write error: {e}"));
    }
    drop(tap);
    pass(
        7,
        "record",
        &format!(
            "{episodes} episodes written to {} under its own family ({family:?})",
            run_dir.display()
        ),
    );

    // -- Step 8: do the submitter's own metrics reach the report? ------------
    let run = match RecordedRun::open(&run_dir) {
        Ok(r) => r,
        Err(e) => return fail(8, "custom-metrics", &format!("RecordedRun::open failed: {e}")),
    };
    let out = run_dir.join("index.html");
    if let Err(e) = emit_static_html(&run, &out, &EmitConfig::default()) {
        return fail(8, "custom-metrics", &format!("emit failed: {e}"));
    }

    let names: Vec<String> = run.metrics_by_series().keys().cloned().collect();
    let has_canonical = names.iter().any(|n| n == "episode_return");
    let missing: Vec<&str> = ["qlearn_mean_abs_td", "qlearn_updates"]
        .into_iter()
        .filter(|want| !names.iter().any(|n| n == want))
        .collect();

    if !missing.is_empty() {
        return fail(
            8,
            "custom-metrics",
            &format!(
                "the submitter's own metrics {missing:?} are absent from the record \
                 (canonical `episode_return` present: {has_canonical}; recorded names: \
                 {names:?}). RecordingLayer's visitor drops every field that is not in \
                 CANONICAL_METRICS, so a submitted algorithm's diagnostics cannot reach \
                 the report at all"
            ),
        );
    }
    pass(
        8,
        "custom-metrics",
        &format!("submitter metrics present alongside canonical ones: {names:?}"),
    );

    // -- Step 9: can two submissions be shown to have run the same task? -----
    //
    // A leaderboard is only meaningful if "same task" is checkable from the
    // artifacts, not asserted by the submitter. `RunManifest` carries rich
    // *provenance* (git commit, rustc, platform, device, seed, thresholds) but
    // nothing that fingerprints the environment's own configuration.
    // Everything the record knows about *which task this was*. `RunManifest`
    // has no field for the environment's own configuration, so this is the
    // complete list, and `env_family` — a six-variant enum — is the most
    // specific entry in it. Two submissions that ran `Four4x4` non-slippery and
    // `Eight8x8` slippery produce identical values here.
    let manifest = run.manifest();
    let task_identity: BTreeMap<&str, String> = BTreeMap::from([
        ("seed", manifest.seed.to_string()),
        ("env_family", format!("{:?}", manifest.env_family)),
        ("algorithm", format!("{:?}", manifest.algorithm)),
    ]);
    // The submitted task, for contrast: none of this survives into the record.
    let cfg = task_config(SEED);
    let submitted = format!(
        "map={:?} is_slippery={} success_rate={} goal_reward={}",
        cfg.map, cfg.is_slippery, cfg.success_rate, cfg.reward_schedule.goal
    );
    fail(
        9,
        "same-task-proof",
        &format!(
            "RunManifest carries no environment-configuration fingerprint, so two \
             submissions cannot be shown to have faced the same task. Submitted: \
             {submitted}. Recorded: {task_identity:?} — `env_family` is the most \
             specific field and it is one of six values"
        ),
    )
}
