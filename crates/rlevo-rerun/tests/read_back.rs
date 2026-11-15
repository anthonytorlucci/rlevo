//! What the sink writes, read back from the files.
//!
//! Each test drives a [`RerunSink`] (by hand through the `Reporter` calls, or
//! through a real `Evaluator` run), then opens the `.rrd` files with rerun's
//! own reader and checks what is in them. Nothing here trusts what the sink
//! says it wrote.

mod common;

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use rerun::external::re_log_types::{EntityPath, EntityPathPart};

use common::{path, read, read_as_written, trial_file};

use rlevo_core::environment::EpisodeStatus;
use rlevo_harness::metrics::Metric;
use rlevo_harness::report::{BenchmarkReport, EpisodeSummary, Phase, Provenance, TrialReport};
use rlevo_harness::reporter::{Boundary, Reporter};
use rlevo_harness::suite::{SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink};

// ── Driving the sink by hand ────────────────────────────────────────────────

const RUN_ID: &str = "00000000000000a7";

fn suite(env_ids: &[&str]) -> SuiteInfo {
    suite_of_run(RUN_ID, env_ids)
}

fn suite_of_run(run_id: &str, env_ids: &[&str]) -> SuiteInfo {
    WRITING.with_borrow_mut(|w| w.2 = run_id.to_string());
    SuiteInfo::new(
        "suite".to_string(),
        env_ids.iter().map(ToString::to_string).collect(),
        1,
        None,
        run_id.to_string(),
    )
}

fn suite_end(sink: &mut RerunSink) {
    sink.on_suite_end(&BenchmarkReport::new("suite".to_string(), 0));
}

fn errored(trial: &TrialInfo, message: &str) -> TrialReport {
    let mut report = report(trial);
    report.errored = true;
    report.error_message = Some(message.to_string());
    report
}

/// The status entry of `rec`'s trial, if it has one.
fn status(rec: &common::Recorded, entity: &str) -> Option<String> {
    let text = rerun::TextDocument::descriptor_text().component;
    rec.store
        .iter_physical_chunks()
        .filter(|chunk| *chunk.entity_path() == path(entity))
        .flat_map(|chunk| {
            chunk
                .iter_slices::<String>(text)
                .flatten()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
        })
        .next()
}

fn trial(env_idx: usize, trial_idx: usize, env_id: &str) -> TrialInfo {
    TrialInfo::new(
        TrialKey { env_idx, trial_idx },
        env_id.to_string(),
        1000 + trial_idx as u64,
    )
}

fn report(trial: &TrialInfo) -> TrialReport {
    let mut report = TrialReport::new(trial.key, trial.env_id.clone(), trial.trial_seed);
    report.provenance = Some(Provenance::this_build());
    report
}

thread_local! {
    /// The directory and label of the sink this test built, and the run it
    /// last started: what a trial's file is named from. Each test runs on a
    /// thread of its own.
    static WRITING: RefCell<(PathBuf, String, String)> = RefCell::default();
}

fn sink(dir: &Path, label: &str) -> RerunSink {
    WRITING.with_borrow_mut(|w| {
        w.0 = dir.to_path_buf();
        w.1 = label.replace(' ', "_");
    });
    RerunSink::new(RerunConfig::new(dir, label)).expect("the directory can be created")
}

/// The file the sink writes `trial` to, by the documented rule. The names
/// in these tests need no cleaning beyond a space.
fn file_of(trial: &TrialInfo) -> PathBuf {
    WRITING.with_borrow(|(dir, label, run_id)| {
        trial_file(
            dir,
            run_id,
            label,
            trial.key.env_idx,
            &trial.env_id.replace([' ', '/'], "_"),
            trial.key.trial_idx,
        )
    })
}

fn scalar(name: &str, value: f64) -> Metric {
    Metric::Scalar {
        name: name.to_string(),
        value,
    }
}

fn counter(name: &str, count: u64) -> Metric {
    Metric::Counter {
        name: name.to_string(),
        count,
    }
}

/// Ends `trial` and returns the path its file is, or will be, at.
fn end(sink: &mut RerunSink, trial: &TrialInfo, report: &TrialReport) -> PathBuf {
    sink.on_trial_end(trial, report);
    file_of(trial)
}

/// Three training episodes then two evaluation ones, each followed by a
/// metrics read, as an episodic trial delivers them. Lengths and returns
/// differ, so no two rows can be swapped unseen.
fn play_episodic(sink: &mut RerunSink, trial: &TrialInfo) {
    let mut step = 0_u64;
    for (index, (length, ret)) in [(5_usize, 1.5), (7, -2.0), (3, 4.25)]
        .into_iter()
        .enumerate()
    {
        step += length as u64;
        sink.on_episode(
            trial,
            &EpisodeSummary::new(
                Phase::Training,
                index,
                ret,
                length,
                EpisodeStatus::Terminated,
            ),
        );
        sink.on_metrics(
            trial,
            Boundary::TrainingEpisode { index, step },
            &[counter("rl/env_steps", step), scalar("rl/loss", ret * 10.0)],
        );
    }
    for (index, (length, ret)) in [(4_usize, 9.0), (6, 8.0)].into_iter().enumerate() {
        step += length as u64;
        sink.on_episode(
            trial,
            &EpisodeSummary::new(
                Phase::Evaluation,
                index,
                ret,
                length,
                EpisodeStatus::Truncated,
            ),
        );
        sink.on_metrics(
            trial,
            Boundary::EvaluationEpisode { index, step },
            &[counter("rl/env_steps", step)],
        );
    }
}

fn names(list: &[&str]) -> BTreeSet<String> {
    list.iter().map(ToString::to_string).collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Rows 6, 7, 8 and 10: every episode's return and length, training and
/// evaluation apart, each on its episode timeline and on `step`, and on no
/// other; the agent's metrics at each read. No wall-clock timeline.
#[test]
fn an_episodic_trial_reads_back_episode_by_episode() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    let trial = trial(0, 0, "cartpole");
    sink.on_suite_start(&suite(&["cartpole"]));
    sink.on_trial_start(&trial);
    play_episodic(&mut sink, &trial);
    let file = end(&mut sink, &trial, &report(&trial));

    let rec = read(&file);
    assert_eq!(rec.application_id, "rlevo-episodic");
    assert_eq!(rec.recording_id, RUN_ID);

    let train = path("/metrics/episode_return/train/ppo/cartpole/0");
    assert_eq!(
        rec.series(&train, "episode"),
        [(0, 1.5), (1, -2.0), (2, 4.25)]
    );
    assert_eq!(
        rec.series(&train, "step"),
        [(5, 1.5), (12, -2.0), (15, 4.25)]
    );
    assert_eq!(rec.timelines(&train), names(&["episode", "step"]));

    let eval = path("/metrics/episode_return/eval/ppo/cartpole/0");
    assert_eq!(rec.series(&eval, "eval_episode"), [(0, 9.0), (1, 8.0)]);
    assert_eq!(rec.series(&eval, "step"), [(19, 9.0), (25, 8.0)]);
    assert_eq!(rec.timelines(&eval), names(&["eval_episode", "step"]));

    let length = path("/metrics/episode_length/train/ppo/cartpole/0");
    assert_eq!(
        rec.series(&length, "episode"),
        [(0, 5.0), (1, 7.0), (2, 3.0)]
    );
    let length = path("/metrics/episode_length/eval/ppo/cartpole/0");
    assert_eq!(rec.series(&length, "step"), [(19, 4.0), (25, 6.0)]);

    // The counter is read in both phases; the loss only while training.
    let steps = path("/metrics/rl/env_steps/ppo/cartpole/0");
    assert_eq!(
        rec.series(&steps, "episode"),
        [(0, 5.0), (1, 12.0), (2, 15.0)]
    );
    assert_eq!(rec.series(&steps, "eval_episode"), [(0, 19.0), (1, 25.0)]);
    assert_eq!(
        rec.series(&steps, "step"),
        [(5, 5.0), (12, 12.0), (15, 15.0), (19, 19.0), (25, 25.0)]
    );
    let loss = path("/metrics/rl/loss/ppo/cartpole/0");
    assert_eq!(
        rec.series(&loss, "step"),
        [(5, 15.0), (12, -20.0), (15, 42.5)]
    );
    assert_eq!(rec.timelines(&loss), names(&["episode", "step"]));

    assert!(
        rec.entities().iter().all(|e| !e.starts_with("/status")),
        "a trial that did not error has no status entry: {:?}",
        rec.entities()
    );
}

/// Rows 1 and 2: the trial's identity, task and build, as properties under
/// the trial's own name, and the rerun version that wrote the file.
#[test]
fn a_trials_properties_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    let trial = trial(1, 2, "cartpole");
    sink.on_suite_start(&suite(&["other", "cartpole"]));
    sink.on_trial_start(&trial);
    play_episodic(&mut sink, &trial);
    let mut report = report(&trial);
    report.task_hash = Some("fnv1a64-canon-v1:0123".to_string());
    report.episodes = vec![EpisodeSummary::new(
        Phase::Evaluation,
        0,
        1.0,
        1,
        EpisodeStatus::Terminated,
    )];
    let file = end(&mut sink, &trial, &report);

    let rec = read(&file);
    let props = path("/__properties/trial/ppo/cartpole/2");
    let get = |name: &'static str| rec.text(&props, name);
    assert_eq!(get("env_id").as_deref(), Some("cartpole"));
    assert_eq!(get("env_idx").as_deref(), Some("1"));
    assert_eq!(get("trial_idx").as_deref(), Some("2"));
    assert_eq!(get("seed").as_deref(), Some("1002"));
    assert_eq!(get("run_label").as_deref(), Some("ppo"));
    assert_eq!(get("run_id").as_deref(), Some(RUN_ID));
    assert_eq!(get("task_hash").as_deref(), Some("fnv1a64-canon-v1:0123"));
    assert_eq!(get("errored").as_deref(), Some("false"));
    assert_eq!(get("eval_episodes").as_deref(), Some("1"));
    assert_eq!(
        get("rlevo_version"),
        Some(Provenance::this_build().rlevo_version)
    );
    assert_eq!(get("platform"), Some(Provenance::this_build().platform));
    let sdk = get("rerun_version").expect("the file names the rerun that wrote it");
    assert!(
        sdk.starts_with("0.") && sdk.split('.').count() == 3,
        "the SDK version is a release number: {sdk}"
    );
    assert_eq!(
        file.file_name().and_then(|n| n.to_str()),
        Some("ppo.1-cartpole.2.rrd")
    );
}

/// Rows 11 and 12: a generation trial's metrics, one row per generation, on
/// `generation` alone, under the generation application ID.
#[test]
fn a_generation_trial_reads_back_generation_by_generation() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "cmaes");
    let trial = trial(0, 0, "sphere");
    sink.on_suite_start(&suite(&["sphere"]));
    sink.on_trial_start(&trial);
    for (generation, best) in [(1_usize, 10.0), (2, 5.0), (3, 2.5), (4, 1.25)] {
        sink.on_metrics(
            &trial,
            Boundary::Generation { generation },
            &[
                scalar("ea/best_fitness", best),
                scalar("coea/best_fitness_a", -best),
            ],
        );
    }
    let file = end(&mut sink, &trial, &report(&trial));

    let rec = read(&file);
    assert_eq!(rec.application_id, "rlevo-generation");
    assert_eq!(rec.recording_id, RUN_ID);
    let best = path("/metrics/ea/best_fitness/cmaes/sphere/0");
    assert_eq!(
        rec.series(&best, "generation"),
        [(1, 10.0), (2, 5.0), (3, 2.5), (4, 1.25)]
    );
    assert_eq!(rec.timelines(&best), names(&["generation"]));
    let coea = path("/metrics/coea/best_fitness_a/cmaes/sphere/0");
    assert_eq!(rec.series(&coea, "generation").len(), 4);
}

/// Row 4: a trial that errored before its first episode still has a file,
/// with a status entry carrying the error and its properties.
#[test]
fn an_errored_trial_has_a_status_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    let trial = trial(0, 0, "cartpole");
    sink.on_suite_start(&suite(&["cartpole"]));
    sink.on_trial_start(&trial);
    let report = errored(&trial, "reset failed: no such device");
    let file = end(&mut sink, &trial, &report);
    suite_end(&mut sink);

    let rec = read(&file);
    assert_eq!(
        status(&rec, "/status/ppo/cartpole/0").as_deref(),
        Some("errored: reset failed: no such device")
    );
    let props = path("/__properties/trial/ppo/cartpole/0");
    assert_eq!(rec.text(&props, "errored").as_deref(), Some("true"));
}

/// A trial that errors before its first generation says nothing of its
/// kind, and may end before any sibling has said. Its file waits for the
/// suite's end and is filed with the run: under the application ID its
/// siblings have, which with the recording ID is what makes one recording.
/// Opened at once as an episodic trial, it would sit outside the run.
#[test]
fn a_trial_that_errors_before_any_boundary_is_filed_with_its_run() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ga");
    let (early, late) = (trial(0, 0, "sphere"), trial(0, 1, "sphere"));
    sink.on_suite_start(&suite(&["sphere"]));
    sink.on_trial_start(&early);
    sink.on_trial_start(&late);

    // The first trial errors and ends before the second reaches a generation.
    let early_file = end(&mut sink, &early, &errored(&early, "bad genome"));
    assert!(
        !early_file.exists(),
        "nothing says yet what kind of run this is"
    );
    assert_eq!(sink.files(), Vec::<PathBuf>::new());

    for generation in 1..=2_usize {
        sink.on_metrics(
            &late,
            Boundary::Generation { generation },
            &[scalar("ea/best_fitness", 1.0)],
        );
    }
    let late_file = end(&mut sink, &late, &report(&late));
    suite_end(&mut sink);
    assert_eq!(sink.files(), [late_file.clone(), early_file.clone()]);

    let (early_rec, late_rec) = (read(&early_file), read(&late_file));
    assert_eq!(late_rec.application_id, "rlevo-generation");
    assert_eq!(early_rec.application_id, late_rec.application_id);
    assert_eq!(early_rec.recording_id, late_rec.recording_id);
    assert_eq!(
        status(&early_rec, "/status/ga/sphere/0").as_deref(),
        Some("errored: bad genome")
    );
}

/// A sink handed to a second run starts afresh: the second run's files are
/// its own, and the first run's kind does not decide how a trial of the
/// second is filed.
#[test]
fn what_a_sink_knows_of_one_run_does_not_reach_the_next() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "x");

    // A generation run.
    let first = trial(0, 0, "sphere");
    sink.on_suite_start(&suite_of_run("00000000000000a1", &["sphere"]));
    sink.on_trial_start(&first);
    sink.on_metrics(
        &first,
        Boundary::Generation { generation: 1 },
        &[scalar("ea/best_fitness", 1.0)],
    );
    let first_file = end(&mut sink, &first, &report(&first));
    suite_end(&mut sink);
    assert_eq!(sink.files(), std::slice::from_ref(&first_file));

    // Then a run whose only trial errors at once.
    let second = trial(0, 0, "cartpole");
    sink.on_suite_start(&suite_of_run("00000000000000a2", &["cartpole"]));
    assert_eq!(sink.files(), Vec::<PathBuf>::new());
    sink.on_trial_start(&second);
    let file = end(&mut sink, &second, &errored(&second, "reset failed"));
    suite_end(&mut sink);

    assert_eq!(sink.files(), std::slice::from_ref(&file));
    // Each run has a directory of its own, so the second, of the same label,
    // environment index and trial, did not write over the first.
    assert_ne!(file.parent(), first_file.parent());
    assert_eq!(read(&first_file).recording_id, "00000000000000a1");
    let rec = read(&file);
    assert_eq!(rec.recording_id, "00000000000000a2");
    assert_eq!(
        rec.application_id, "rlevo-episodic",
        "no trial of this run said its kind, so the earlier run's does not stand in"
    );
}

/// An environment or a run named with free text: the name is one path part,
/// whatever it contains, and the file name is safe.
#[test]
fn free_text_names_are_escaped_in_paths_and_cleaned_in_file_names() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "my run");
    let trial = trial(0, 3, "Cart Pole/v1");
    sink.on_suite_start(&suite(&["Cart Pole/v1"]));
    sink.on_trial_start(&trial);
    sink.on_episode(
        &trial,
        &EpisodeSummary::new(Phase::Evaluation, 0, 2.0, 1, EpisodeStatus::Terminated),
    );
    sink.on_metrics(
        &trial,
        Boundary::EvaluationEpisode { index: 0, step: 1 },
        &[],
    );
    let file = end(&mut sink, &trial, &report(&trial));

    assert_eq!(
        file.file_name().and_then(|n| n.to_str()),
        Some("my_run.0-Cart_Pole_v1.3.rrd")
    );
    let rec = read(&file);
    let entity = EntityPath::new(
        [
            "metrics",
            "episode_return",
            "eval",
            "my run",
            "Cart Pole/v1",
            "3",
        ]
        .into_iter()
        .map(EntityPathPart::new)
        .collect(),
    );
    assert_eq!(rec.series(&entity, "eval_episode"), [(0, 2.0)]);
}

/// A histogram holds every sample since the trial began, so it is not
/// logged at each read; the metrics beside it are.
#[test]
fn a_histogram_is_left_out_and_its_neighbours_are_kept() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    let trial = trial(0, 0, "cartpole");
    sink.on_suite_start(&suite(&["cartpole"]));
    sink.on_trial_start(&trial);
    sink.on_metrics(
        &trial,
        Boundary::TrainingEpisode { index: 0, step: 3 },
        &[
            Metric::Histogram {
                name: "rl/td_errors".to_string(),
                values: vec![1.0, 2.0],
            },
            scalar("rl/loss", 0.5),
        ],
    );
    let file = end(&mut sink, &trial, &report(&trial));

    let rec = read(&file);
    assert_eq!(
        rec.series(&path("/metrics/rl/loss/ppo/cartpole/0"), "step"),
        [(3, 0.5)]
    );
    assert!(
        rec.entities().iter().all(|e| !e.contains("td_errors")),
        "{:?}",
        rec.entities()
    );
}

/// Two trials open at once, their calls interleaved as parallel trials'
/// are: each file holds its own trial's rows only, and both share the run's
/// recording ID.
#[test]
fn interleaved_trials_keep_their_rows_apart() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    let (a, b) = (trial(0, 0, "cartpole"), trial(0, 1, "cartpole"));
    sink.on_suite_start(&suite(&["cartpole"]));
    sink.on_trial_start(&a);
    sink.on_trial_start(&b);
    for (index, ret) in [1.0, 2.0, 3.0].into_iter().enumerate() {
        for (trial, sign) in [(&a, 1.0), (&b, -1.0)] {
            sink.on_episode(
                trial,
                &EpisodeSummary::new(
                    Phase::Evaluation,
                    index,
                    sign * ret,
                    2,
                    EpisodeStatus::Terminated,
                ),
            );
        }
    }
    // No metrics read followed these episodes (as with a sink that wants
    // none), so each is logged without a step when the next arrives, and
    // the last when its trial ends.
    let file_a = end(&mut sink, &a, &report(&a));
    let file_b = end(&mut sink, &b, &report(&b));
    assert_ne!(file_a, file_b);
    assert_eq!(sink.files(), [file_a.clone(), file_b.clone()]);

    let (rec_a, rec_b) = (read(&file_a), read(&file_b));
    assert_eq!(rec_a.recording_id, rec_b.recording_id);
    let of = |idx: usize| path(&format!("/metrics/episode_return/eval/ppo/cartpole/{idx}"));
    assert_eq!(
        rec_a.series(&of(0), "eval_episode"),
        [(0, 1.0), (1, 2.0), (2, 3.0)]
    );
    assert_eq!(
        rec_b.series(&of(1), "eval_episode"),
        [(0, -1.0), (1, -2.0), (2, -3.0)]
    );
    assert_eq!(rec_a.series(&of(1), "eval_episode"), []);
    assert_eq!(rec_b.series(&of(0), "eval_episode"), []);
    assert_eq!(rec_a.timelines(&of(0)), names(&["eval_episode"]));
}

/// Rows logged a while apart are still written together. rerun's default
/// writes every 200 ms, which made nearly every row a chunk of its own and a
/// trial's file many times the size of its data.
#[test]
fn rows_logged_a_while_apart_share_a_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    let trial = trial(0, 0, "cartpole");
    sink.on_suite_start(&suite(&["cartpole"]));
    sink.on_trial_start(&trial);
    for (index, step) in [(0_usize, 3_u64), (1, 6), (2, 9)] {
        sink.on_metrics(
            &trial,
            Boundary::TrainingEpisode { index, step },
            &[scalar("rl/loss", 1.0)],
        );
        // Longer than rerun's default interval between writes.
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    let file = end(&mut sink, &trial, &report(&trial));

    let rec = read_as_written(&file);
    let loss = path("/metrics/rl/loss/ppo/cartpole/0");
    assert_eq!(rec.series(&loss, "step").len(), 3);
    assert_eq!(rec.chunks(&loss), 1);
}

/// The instructions name the directory, the version that wrote the files,
/// and a viewer bound to this machine; there are none before a trial ends.
#[test]
fn viewer_instructions_name_the_version_and_bind_locally() {
    let dir = tempfile::tempdir().unwrap();
    let mut sink = sink(dir.path(), "ppo");
    assert_eq!(sink.viewer_instructions(), None);
    let trial = trial(0, 0, "cartpole");
    sink.on_suite_start(&suite(&["cartpole"]));
    sink.on_trial_start(&trial);
    let file = end(&mut sink, &trial, &report(&trial));
    suite_end(&mut sink);
    let sdk = read(&file)
        .text(&path("/__properties/trial/ppo/cartpole/0"), "rerun_version")
        .expect("the file names the rerun that wrote it");

    let how = sink.viewer_instructions().expect("a trial has ended");
    assert!(
        how.contains(&format!("cargo binstall rerun-cli@{sdk}")),
        "{how}"
    );
    assert!(
        how.contains(&format!(
            "cargo install rerun-cli --version ={sdk} --locked"
        )),
        "{how}"
    );
    assert!(how.contains("rerun --bind 127.0.0.1"), "{how}");
    assert!(
        how.contains(&dir.path().join(RUN_ID).display().to_string()),
        "the instructions name the run's own directory: {how}"
    );
}
