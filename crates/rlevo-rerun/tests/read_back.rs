//! What the sink writes, read back from the files.
//!
//! Each test drives a [`RerunSink`] (by hand through the `Reporter` calls, or
//! through a real `Evaluator` run), then opens the `.rrd` files with rerun's
//! own reader and checks what is in them. Nothing here trusts what the sink
//! says it wrote.

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use rerun::external::re_log_types::{EntityPath, EntityPathPart};

use common::{path, read, read_as_written};

use rlevo_core::environment::EpisodeStatus;
use rlevo_harness::metrics::Metric;
use rlevo_harness::report::{BenchmarkReport, EpisodeSummary, Phase, Provenance, TrialReport};
use rlevo_harness::reporter::{Boundary, Reporter};
use rlevo_harness::suite::{SuiteInfo, TrialInfo, TrialKey};
use rlevo_rerun::{RerunConfig, RerunSink};

// ── Driving the sink by hand ────────────────────────────────────────────────

const RUN_ID: &str = "00000000000000a7";

fn suite(env_ids: &[&str]) -> SuiteInfo {
    SuiteInfo::new(
        "suite".to_string(),
        env_ids.iter().map(ToString::to_string).collect(),
        1,
        None,
        RUN_ID.to_string(),
    )
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

fn sink(dir: &Path, label: &str) -> RerunSink {
    RerunSink::new(RerunConfig::new(dir, label)).expect("the directory can be created")
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

/// Ends `trial` the way the harness does: asks for the recording, copies it
/// into the report, then calls `on_trial_end`. Returns the file's path.
fn end(sink: &mut RerunSink, trial: &TrialInfo, mut report: TrialReport) -> PathBuf {
    let recording = sink.recording(trial).expect("the sink records this trial");
    report.recording_path = Some(recording.path.clone());
    report.recording_sdk_version = Some(recording.sdk_version);
    sink.on_trial_end(trial, &report);
    recording.path
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
    let file = end(&mut sink, &trial, report(&trial));

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
    let sdk = sink.recording(&trial).unwrap().sdk_version;
    let file = end(&mut sink, &trial, report);

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
    assert_eq!(get("rerun_version"), Some(sdk.clone()));
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
    let file = end(&mut sink, &trial, report(&trial));

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
    let mut report = report(&trial);
    report.errored = true;
    report.error_message = Some("reset failed: no such device".to_string());
    let file = end(&mut sink, &trial, report);

    let rec = read(&file);
    let text = rerun::TextDocument::descriptor_text().component;
    let status = rec
        .store
        .iter_physical_chunks()
        .filter(|chunk| *chunk.entity_path() == path("/status/ppo/cartpole/0"))
        .flat_map(|chunk| {
            chunk
                .iter_slices::<String>(text)
                .flatten()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
        })
        .next();
    assert_eq!(
        status.as_deref(),
        Some("errored: reset failed: no such device")
    );
    let props = path("/__properties/trial/ppo/cartpole/0");
    assert_eq!(rec.text(&props, "errored").as_deref(), Some("true"));
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
    let file = end(&mut sink, &trial, report(&trial));

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
    let file = end(&mut sink, &trial, report(&trial));

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
    let file_a = end(&mut sink, &a, report(&a));
    let file_b = end(&mut sink, &b, report(&b));
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
    let file = end(&mut sink, &trial, report(&trial));

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
    let sdk = sink.recording(&trial).unwrap().sdk_version;
    end(&mut sink, &trial, report(&trial));
    sink.on_suite_end(&BenchmarkReport::new("suite".to_string(), 0));

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
    assert!(how.contains(&dir.path().display().to_string()), "{how}");
}
