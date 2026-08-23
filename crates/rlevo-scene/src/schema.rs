//! The record schema: everything an `episode_*.rec` file contains.
//!
//! Encoding is bincode 2.x with the explicit [`bincode::config::standard()`]
//! configuration locked at write time and stamped into
//! [`EpisodeRecordHeader::format_version`], so a loader can refuse mismatched
//! files cleanly. The version stamp sits at the very start of every file, so a
//! reader can peek it with a 16-byte read before committing to a decode pass.
//!
//! # One definition, not three
//!
//! These types used to exist three times: as producer-facing `*Snapshot` types
//! in `rlevo-core::render`, as `*Payload` "bincode-stable mirrors" here, and
//! again in the report client's `wire.rs`, with a 495-line drift-guard test
//! holding the copies together. [`FamilyPayload`] now carries the snapshot types
//! directly and the mirrors are gone, so drift is not a category any more.
//!
//! The mirrors existed to insulate the wire format from producer-type churn,
//! which was a real concern. The replacement is that a change to a snapshot type
//! **is** a wire change, and this crate is where "does this break the format?"
//! is the first review question. [`FORMAT_VERSION`] is the enforcement.
//!
//! # What is not here
//!
//! `RunManifest` and the `RecordedEnvFamily` trait stay in `rlevo-benchmarks`.
//! The manifest reaches the report as **JSON**, not bincode, and a
//! field-name-keyed format degrades gracefully where a positional one corrupts
//! silently — so mirroring it is safe and sharing it would drag
//! `ObjectiveSense` out of `rlevo-core`. See the crate docs.
//!
//! Producing a [`RunId`] also stays there: it needs a clock and an RNG, and this
//! crate has neither.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::payload::{
    Box2dSnapshot, Classic2DSnapshot, GridSnapshot, Landscape2DSnapshot, Locomotion2DSnapshot,
    TabularSnapshot,
};
use crate::scene::{SceneDescriptor, ScenePose};
use crate::styled::StyledFrame;

/// Current on-disk schema version. The writer stamps this into every
/// [`EpisodeRecordHeader`]; the loader rejects any file that does not
/// carry exactly this value.
pub const FORMAT_VERSION: u16 = 9;

/// Oldest on-disk version this loader accepts. Equal to
/// [`FORMAT_VERSION`] — no backward compatibility is maintained
/// before the first production release.
pub const MIN_SUPPORTED_VERSION: u16 = 9;

/// Locked bincode configuration shared by writer and loader. Kept as a
/// helper rather than a constant because `bincode::config::Configuration`
/// is non-const.
#[must_use]
pub fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Identifies a recording run. Format: `YYYYMMDD-HHMMSS-<6 hex>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunId(pub String);

/// Environment family classification used both to choose a
/// default frame stride and to pick the report-tier panel mode.
///
/// A variant names the **report adapter that decodes an environment's frames**
/// — this is a *render* family, not the `rlevo-environments` module an env
/// lives in and not a taxonomy of task type. Nothing else keys off it: metrics
/// and charts render family-independently, so the family only selects the frame
/// decoder and the default stride. `rlevo-benchmarks` carries the pieces that
/// consume it — `default_frame_stride` and the `RecordedEnvFamily` trait,
/// which records the invariant this places on each environment. Both are
/// named in prose: they live in a crate that depends on this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EnvFamily {
    /// Classic control tasks (`CartPole`, `MountainCar`, `Acrobot`, …).
    Classic,
    /// Discrete grid-world environments.
    Grids,
    /// Tabular toy-text environments (`FrozenLake`, `Taxi`, …).
    ToyText,
    /// 2D rigid-body physics environments.
    Box2d,
    /// Continuous locomotion environments (no ASCII rendering path).
    Locomotion,
    /// Optimisation-landscape search environments.
    Landscapes,
}

/// Per-family rich payload.
///
/// **Variant ordering is wire-format-stable** — new variants append at
/// the end so existing bincode tags keep decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FamilyPayload {
    /// Plain-text / styled-text rendering — the default for ASCII-able
    /// envs.
    Ascii,
    /// Landscape (search-space) projection for `landscapes` envs.
    Landscape2D(Landscape2DSnapshot),
    /// Rigid-body world for `box2d` envs.
    Box2dBodies(Box2dSnapshot),
    /// Sagittal-plane stick figure for `locomotion` envs — their
    /// canonical view, since locomotion has no ASCII path.
    Locomotion2D(Locomotion2DSnapshot),
    /// Structured tile grid for `grids` (Minigrid-style) envs. Added in
    /// `FORMAT_VERSION = 5`; the report renders SVG from this instead of
    /// the legacy ASCII path.
    Grid(GridSnapshot),
    /// Structured layout for `toy_text` envs — a tile grid (`FrozenLake` /
    /// `CliffWalking` / `Taxi`) or a card table (`Blackjack`). Added in
    /// `FORMAT_VERSION = 5`.
    TabularText(TabularSnapshot),
    /// Structured 2-D line-art for `classic` physics envs (`CartPole` /
    /// `Pendulum` / `MountainCar` / `Acrobot`). Added in `FORMAT_VERSION = 5`.
    Classic2D(Classic2DSnapshot),
    /// Per-frame placements for a scene whose geometry was recorded once, at
    /// episode start, as a [`SceneDescriptor`] in
    /// [`RecordChunk::Scene`](crate::codec::RecordChunk::Scene). Added in
    /// `FORMAT_VERSION = 9`.
    ///
    /// This is the variant every continuous-geometry family migrates onto:
    /// `Landscape2D`, `Box2dBodies`, `Locomotion2D`, and `Classic2D` all
    /// describe the same thing — posed geometry in a viewport — through four
    /// shapes, four client adapters, and four tap constructors. They are
    /// retained while their producers move across one family at a time.
    Scene(ScenePose),
}

/// Identifies the harness trial that produced an episode.
///
/// Mirrors `suite::TrialKey` but is wire-owned (`u32` fields, serde) so the
/// record schema does not depend on the harness types. `None` on an
/// [`EpisodeRecordHeader`] means the episode came from a non-harness
/// producer (a training loop or landscape driver that drives the sink
/// directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialRef {
    /// Zero-based index of the environment within the suite.
    pub env_index: u32,
    /// Zero-based seed-repetition index for that environment.
    pub trial_index: u32,
}

/// Whether an episode was recorded during training (exploration on) or
/// evaluation (deterministic / greedy). Added in `FORMAT_VERSION = 6`.
///
/// Eval rollouts are recorded as separate episode files tagged
/// [`EpisodeKind::Evaluation`]; the metric samples captured into that file
/// inherit the tag by file membership, so no per-sample discriminator is
/// needed. `Training` is the default for existing producers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EpisodeKind {
    /// Exploration rollout produced while the policy is being updated.
    #[default]
    Training,
    /// Deterministic rollout produced to measure the current policy.
    Evaluation,
}

/// Why a [`CheckpointRef`] was written. Added in `FORMAT_VERSION = 6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CheckpointKind {
    /// Saved on a periodic step interval during training.
    Periodic,
    /// Saved because it is the best-performing checkpoint so far.
    Best,
    /// Saved at the end of the run.
    Final,
}

/// Serialisation format of a saved learner checkpoint, mirroring the Burn
/// file recorder that wrote it. Added in `FORMAT_VERSION = 6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CheckpointFormat {
    /// `NamedMpkFileRecorder` (`.mpk`).
    NamedMpk,
    /// `NamedMpkGzFileRecorder` (`.mpk.gz`).
    NamedMpkGz,
    /// `BinFileRecorder` (`.bin`).
    Bin,
    /// `PrettyJsonFileRecorder` (`.json`).
    Json,
    /// Any other recorder.
    Other,
}

/// A reference to one Burn-saved learner checkpoint.
///
/// Deep-RL produces a trained artifact — the policy / value networks — that
/// the record cannot otherwise point to. Burn owns model serialisation via
/// its `Recorder` trait, so the record **references** the saved file(s) and
/// never embeds weights. This is the deep-RL analog of
/// [`PopulationSample::best_genome_digest`] (a pointer, not the genome).
/// Carried only in `rlevo-benchmarks`' `RunManifest::checkpoints` — named in
/// prose because the manifest travels as JSON and stays in that crate; pure
/// evolutionary-optimisation runs leave that list empty. Added in
/// `FORMAT_VERSION = 6`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRef {
    /// Global training step at which the checkpoint was written.
    pub step: u32,
    /// Why this checkpoint was written.
    pub kind: CheckpointKind,
    /// Burn recorder format used for the artifact.
    pub format: CheckpointFormat,
    /// Path to the artifact, relative to the run directory.
    pub path: String,
    /// Metric (e.g. eval return) at save time, if known.
    pub metric: Option<f64>,
    /// 128-bit content digest, like [`PopulationSample::best_genome_digest`].
    pub digest: Option<[u8; 16]>,
}

/// Header written at the start of every `episode_*.rec` file. Carries
/// run-level identification + the format-version stamp the loader uses
/// to refuse mismatched files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRecordHeader {
    /// Wire-format version stamp; loader rejects any value ≠ [`FORMAT_VERSION`].
    pub format_version: u16,
    /// Unique identifier of the parent recording run.
    pub run_id: RunId,
    /// RNG seed used for this episode.
    pub seed: u64,
    /// Environment family — determines the payload decoder on the report tier.
    pub env_family: EnvFamily,
    /// Unix timestamp (seconds) when this episode file was opened.
    pub created_at: i64,
    /// Trial that produced this episode, or `None` for non-harness
    /// producers. Added in `FORMAT_VERSION = 4`.
    pub trial: Option<TrialRef>,
    /// Whether this episode was a training or evaluation rollout. Added in
    /// `FORMAT_VERSION = 6`; defaults to [`EpisodeKind::Training`].
    pub kind: EpisodeKind,
}

/// One captured frame, written length-prefixed to the per-episode
/// file. `action` is bincode-encoded so heterogeneous action types
/// (Discrete / Continuous / multi-dim) share one carrier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRecord {
    /// Zero-based step index within the episode.
    pub step: u32,
    /// Bincode-encoded action taken to reach this frame; empty on the reset frame.
    pub action: Vec<u8>,
    /// Scalar reward received at this step.
    pub reward: f32,
    /// Plain-text env rendering, or `None` for headless envs.
    pub ascii: Option<String>,
    /// Styled-text env rendering, or `None` for headless envs.
    pub styled: Option<StyledFrame>,
    /// Family-specific rich payload for report-tier rendering.
    pub family_payload: FamilyPayload,
}

/// One scalar metric sample with the global training step (PPO) or
/// generation index (EA) it was emitted at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSample {
    /// Global training step (RL) or generation index (EA) at emission.
    pub step: u32,
    /// Metric name, e.g. `"policy_loss"` or `"episode_return"`.
    pub name: String,
    /// Scalar value of the metric.
    pub value: f64,
}

/// One per-generation snapshot of an evolutionary-algorithm population.
///
/// Carried by `RecordChunk::Population` in the on-disk stream
/// (`FORMAT_VERSION = 3` and later). RL-only runs never emit this
/// chunk; EA runs emit one per call to `RecordSink::on_population_sample`,
/// typically once per generation.
///
/// **Wire fields not yet fully populated**:
///
/// - `parents_of_best` is emitted empty until a Strategy-side lineage
///   trait extension lands. The field is present so the schema does not
///   need a further bump when the lineage DAG renders.
/// - `inner_rl_returns` is `None` from pure-EA producers; populated by
///   a future hybrid driver to feed an `(inner_rl_return, fitness)`
///   scatter panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationSample {
    /// Zero-based generation index.
    pub generation: u32,
    /// Per-genome fitness scores in population order.
    pub fitnesses: Vec<f32>,
    /// Optional behavioural or genotypic diversity estimate.
    pub diversity: Option<f32>,
    /// Index into `fitnesses` of the highest-scoring genome.
    ///
    /// Loaders must bounds-check this against `fitnesses.len()` before
    /// indexing; a malformed producer could write an out-of-range value.
    pub best_index: u32,
    /// 128-bit hash of the best genome's serialised representation.
    pub best_genome_digest: Option<[u8; 16]>,
    /// Digests of the best genome's parent genomes (empty until lineage DAG lands).
    pub parents_of_best: Vec<[u8; 16]>,
    /// Per-genome inner RL returns for hybrid drivers; `None` from pure-EA producers.
    pub inner_rl_returns: Option<Vec<f32>>,
}

/// In-memory aggregate of one on-disk episode file. Used by tests and
/// the slim record-file decoder helper; the streaming writer never materialises
/// this whole struct in memory during normal recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Run-level identification and format-version stamp.
    pub header: EpisodeRecordHeader,
    /// All captured frames in episode order.
    pub frames: Vec<FrameRecord>,
    /// Scalar metrics emitted during the episode.
    pub metrics: Vec<MetricSample>,
    /// EA population snapshots; empty for RL-only runs.
    #[serde(default)]
    pub population_samples: Vec<PopulationSample>,
    /// Static scene geometry, recorded once at episode start. `None` for every
    /// producer that has not migrated to [`FamilyPayload::Scene`].
    ///
    /// Singular, not a list: an episode describes one scene, and a second
    /// descriptor in the same file means a producer re-declared its geometry
    /// mid-episode. The decoder keeps the first and ignores the rest rather
    /// than failing, since a pose keyed against the first descriptor is still
    /// renderable. Added in `FORMAT_VERSION = 9`.
    #[serde(default)]
    pub scene: Option<SceneDescriptor>,
}

/// Free-form hyperparameter map captured in the run manifest. Per-algorithm
/// structured fields land in a follow-on milestone once an actual report-tier
/// consumer needs them.
///
/// [`BTreeMap`] is used rather than `HashMap` so bincode serialisation is
/// deterministic and key-sorted, which simplifies diff-based comparison of
/// run manifests.
pub type Hyperparameters = BTreeMap<String, String>;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::payload::Point2;
    use crate::styled::{StyledLine, StyledSpan};

    /// The policy invariant: no backward compatibility is maintained, so the
    /// oldest accepted version is always the current one. This is what must
    /// never drift; it holds at every version and needs no edit when one is
    /// bumped.
    #[test]
    fn min_supported_version_equals_format_version() {
        assert_eq!(
            MIN_SUPPORTED_VERSION, FORMAT_VERSION,
            "no backward compatibility is maintained across a format bump, so \
             the loader's floor must move with it",
        );
    }

    /// Tripwire against an *accidental* bump. The version is stamped into every
    /// recording and into the built WASM bundle, and
    /// [`MIN_SUPPORTED_VERSION`] moves with it, so a stray edit here silently
    /// invalidates every recording a reader might still hold.
    ///
    /// Deliberately named without the number in it. The previous name spelled
    /// the version out (`..._is_seven_and_min_supported_is_seven`), so every
    /// bump renamed the test as well as editing it — the same name-drift this
    /// suite avoids everywhere else.
    ///
    /// Bumping is legitimate; it is a *coordination* point, not a prohibition.
    /// When you do, update this literal and the changelog table in the crate
    /// docs. The client no longer keeps a copy of the constant to update —
    /// there is one definition now.
    #[test]
    fn format_version_is_pinned() {
        assert_eq!(FORMAT_VERSION, 9);
    }

    #[test]
    fn family_payload_round_trips_each_rich_variant() {
        let cases = [
            FamilyPayload::Ascii,
            FamilyPayload::Landscape2D(Landscape2DSnapshot {
                bounds_x: (-1.0, 1.0),
                bounds_y: (-1.0, 1.0),
                current: Point2::new(0.1, 0.2),
                best: None,
                trail: vec![],
                label: "sphere".into(),
            }),
            FamilyPayload::Box2dBodies(Box2dSnapshot {
                world_bounds: (Point2::new(-1.0, -1.0), Point2::new(1.0, 1.0)),
                bodies: vec![],
                contacts: vec![],
            }),
            FamilyPayload::Locomotion2D(Locomotion2DSnapshot {
                joints: vec![],
                bones: vec![],
                ground_y: Some(0.0),
                com: None,
                contacts: vec![],
            }),
        ];
        for c in &cases {
            let bytes = bincode::serde::encode_to_vec(c, bincode_config()).unwrap();
            let (decoded, _): (FamilyPayload, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
            assert_eq!(c, &decoded);
        }
    }

    fn sample_header() -> EpisodeRecordHeader {
        EpisodeRecordHeader {
            format_version: FORMAT_VERSION,
            run_id: RunId("20260527-120000-abc123".into()),
            seed: 42,
            env_family: EnvFamily::Classic,
            created_at: 1_700_000_000,
            trial: Some(TrialRef {
                env_index: 1,
                trial_index: 2,
            }),
            kind: EpisodeKind::Training,
        }
    }

    fn sample_frame() -> FrameRecord {
        FrameRecord {
            step: 7,
            action: vec![1, 2, 3],
            reward: -0.5,
            ascii: Some("pole\n|".into()),
            styled: Some(StyledFrame {
                lines: vec![StyledLine {
                    spans: vec![StyledSpan::raw("pole")],
                }],
            }),
            family_payload: FamilyPayload::Ascii,
        }
    }

    fn sample_metric() -> MetricSample {
        MetricSample {
            step: 1024,
            name: "policy_loss".into(),
            value: 0.0123,
        }
    }

    #[test]
    fn header_round_trip() {
        let h = sample_header();
        let bytes = bincode::serde::encode_to_vec(&h, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecordHeader, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(h, decoded);
        assert_eq!(
            decoded.trial,
            Some(TrialRef {
                env_index: 1,
                trial_index: 2
            })
        );
    }

    #[test]
    fn header_round_trip_without_trial() {
        let h = EpisodeRecordHeader {
            trial: None,
            ..sample_header()
        };
        let bytes = bincode::serde::encode_to_vec(&h, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecordHeader, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(h, decoded);
        assert!(decoded.trial.is_none());
    }

    #[test]
    fn header_round_trips_each_episode_kind() {
        for kind in [EpisodeKind::Training, EpisodeKind::Evaluation] {
            let h = EpisodeRecordHeader {
                kind,
                ..sample_header()
            };
            let bytes = bincode::serde::encode_to_vec(&h, bincode_config()).unwrap();
            let (decoded, _): (EpisodeRecordHeader, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
            assert_eq!(h, decoded);
            assert_eq!(decoded.kind, kind);
        }
    }

    #[test]
    fn checkpoint_ref_round_trips() {
        let c = CheckpointRef {
            step: 50_000,
            kind: CheckpointKind::Best,
            format: CheckpointFormat::NamedMpk,
            path: "checkpoints/best.mpk".into(),
            metric: Some(241.7),
            digest: Some([9u8; 16]),
        };
        let bytes = bincode::serde::encode_to_vec(&c, bincode_config()).unwrap();
        let (decoded, _): (CheckpointRef, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(c, decoded);
    }

    #[test]
    fn frame_round_trip_with_styled() {
        let f = sample_frame();
        let bytes = bincode::serde::encode_to_vec(&f, bincode_config()).unwrap();
        let (decoded, _): (FrameRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(f, decoded);
    }

    #[test]
    fn frame_round_trip_without_styled() {
        let f = FrameRecord {
            styled: None,
            ascii: None,
            ..sample_frame()
        };
        let bytes = bincode::serde::encode_to_vec(&f, bincode_config()).unwrap();
        let (decoded, _): (FrameRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(f, decoded);
    }

    #[test]
    fn metric_sample_round_trip() {
        let m = sample_metric();
        let bytes = bincode::serde::encode_to_vec(&m, bincode_config()).unwrap();
        let (decoded, _): (MetricSample, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(m, decoded);
    }

    #[test]
    fn full_record_round_trip() {
        let rec = EpisodeRecord {
            header: sample_header(),
            frames: vec![sample_frame(), sample_frame()],
            metrics: vec![sample_metric()],
            population_samples: vec![sample_population()],
            scene: None,
        };
        let bytes = bincode::serde::encode_to_vec(&rec, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecord, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(rec, decoded);
    }

    fn sample_population() -> PopulationSample {
        PopulationSample {
            generation: 4,
            fitnesses: vec![0.5, 0.4, 0.3, 0.6, 0.8],
            diversity: Some(0.12),
            best_index: 2,
            best_genome_digest: Some([7u8; 16]),
            parents_of_best: vec![[3u8; 16], [11u8; 16]],
            inner_rl_returns: None,
        }
    }

    #[test]
    fn population_sample_round_trip() {
        let p = sample_population();
        let bytes = bincode::serde::encode_to_vec(&p, bincode_config()).unwrap();
        let (decoded, _): (PopulationSample, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(p, decoded);
    }
}
