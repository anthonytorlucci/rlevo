//! The scene recording path, end to end: a `ScenePayloadSource` env through
//! `RecordingTap` to a file on disk and back.
//!
//! ADR 0082 decision 4 splits what every other family fuses: geometry is
//! recorded **once per episode**, poses **per frame**. That split is the reason
//! a 3-D record is affordable at all, and it is also the only thing here a unit
//! test cannot see. `SceneDescriptor` and `ScenePose` round-trip through
//! bincode in `rlevo-scene`'s own tests; what those cannot check is whether the
//! two halves are emitted at the right rates, in the right order, by the tap.
//!
//! Three properties, each of which fails silently if broken:
//!
//! 1. **One descriptor per episode, N poses.** A descriptor per frame is the
//!    naive implementation and produces a correct-looking report at many times
//!    the size — nothing in the picture reveals it.
//! 2. **Descriptor before the first pose.** A reader that meets a pose first
//!    has no node ids to resolve it against. The reset frame carries a pose, so
//!    the ordering has exactly one frame of slack.
//! 3. **Node ids survive the round trip and still address the poses.** This is
//!    the keyed-not-positional invariant, and it is invisible to a
//!    value-equality round-trip test that happens to keep its nodes in order.

#![cfg(feature = "record")]

use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use rlevo_benchmarks::record::{
    EnvFamily, FamilyPayload, InMemoryRecordSink, RecordSink, RecordWriter, RecordingConfig,
    RecordingTap, read_episode_record,
};
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::environment::{Environment, EnvironmentError, Snapshot, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_scene::{
    Bone, Extent, Geometry, NodeId, Point3, SceneDescriptor, SceneNode, ScenePayloadSource,
    ScenePose, Transform3,
};

// ---------------------------------------------------------------------------
// A minimal two-link arm that describes itself as a scene.
// ---------------------------------------------------------------------------

const ARM: NodeId = NodeId(0);
const GROUND: NodeId = NodeId(1);

#[derive(Debug, Clone, Copy)]
struct ArmObs {
    angle: f32,
}

impl Observation<1> for ArmObs {
    fn shape() -> [usize; 1] {
        [1]
    }
}

#[derive(Debug, Clone, Copy)]
struct ArmState;

impl State<1> for ArmState {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
    fn numel(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ArmAction(u8);

impl Action<1> for ArmAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

struct Arm {
    angle: f32,
    step: u32,
    ends_at: u32,
}

impl Arm {
    const fn new(ends_at: u32) -> Self {
        Self {
            angle: 0.0,
            step: 0,
            ends_at,
        }
    }
}

impl ScenePayloadSource for Arm {
    fn scene_descriptor(&self) -> SceneDescriptor {
        SceneDescriptor {
            nodes: vec![
                SceneNode {
                    id: ARM,
                    geometry: Geometry::Skeleton {
                        vertices: vec![
                            Point3::ORIGIN,
                            Point3::new(0.0, 1.0, 0.0),
                            Point3::new(0.0, 2.0, 0.0),
                        ],
                        bones: vec![Bone::new(0, 1), Bone::new(1, 2)],
                    },
                    role: "arm".into(),
                    static_transform: None,
                },
                SceneNode {
                    id: GROUND,
                    geometry: Geometry::Box {
                        half_extents: Point3::new(5.0, 0.05, 5.0),
                    },
                    role: "ground".into(),
                    // Never moves, so no pose need ever mention it.
                    static_transform: Some(Transform3::IDENTITY),
                },
            ],
            bounds: (
                Extent::new(-5.0, 5.0).expect("valid"),
                Extent::new(0.0, 3.0).expect("valid"),
                Extent::new(-5.0, 5.0).expect("valid"),
            ),
            background: Some("ground_plane".into()),
        }
    }

    fn scene_pose(&self) -> ScenePose {
        // Rotation about world-z, scalar-first: (cos(t/2), 0, 0, sin(t/2)).
        let half = self.angle * 0.5;
        ScenePose {
            transforms: vec![(
                ARM,
                Transform3 {
                    position: Point3::ORIGIN,
                    orientation: [half.cos(), 0.0, 0.0, half.sin()],
                },
            )],
            markers: vec![("tip".into(), Point3::new(self.angle, 2.0, 0.0))],
        }
    }
}

impl Environment<1, 1, 1> for Arm {
    type StateType = ArmState;
    type ObservationType = ArmObs;
    type ActionType = ArmAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, ArmObs, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.angle = 0.0;
        self.step = 0;
        Ok(SnapshotBase::running(
            ArmObs { angle: 0.0 },
            ScalarReward(0.0),
        ))
    }

    fn step(&mut self, _action: ArmAction) -> Result<Self::SnapshotType, EnvironmentError> {
        self.angle += 0.1;
        self.step += 1;
        let obs = ArmObs { angle: self.angle };
        let reward = ScalarReward(1.0);
        Ok(if self.step >= self.ends_at {
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        })
    }
}

/// Drives `tap` through `episodes` complete episodes of `steps` steps each.
///
/// Bounded by construction. An episode loop that ran until the env said it was
/// done would hang rather than fail if the tap stopped forwarding termination,
/// and a hang in CI reads as flake rather than as this test.
fn drive(tap: &mut RecordingTap<Arm, 1, 1, 1>, episodes: u32, steps: u32) {
    for _ in 0..episodes {
        let snap = tap.reset().expect("reset");
        let mut last = snap.observation().angle;
        for _ in 0..steps {
            let snap = tap.step(ArmAction(0)).expect("step");
            // The tap must forward the inner env's own snapshot, not a
            // reconstruction. A wrapper that returned a default-constructed
            // observation would record perfectly good frames and hand the
            // caller a frozen env.
            let angle = snap.observation().angle;
            assert!(angle > last, "tap did not forward the inner observation");
            last = angle;
            if snap.is_done() {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// In-memory: rate, ordering, and the keyed invariant.
// ---------------------------------------------------------------------------

/// Each episode gets its own descriptor, and every frame in it carries a pose.
///
/// This test does **not** pin the *once* half of "once per episode", despite
/// what the property sounds like: `InMemoryRecordSink::on_scene` keeps the
/// first descriptor and drops the rest, so a tap re-declaring geometry on every
/// frame is invisible here. Verified by mutation — that mutant survives this
/// test and dies in `descriptor_chunk_precedes_the_first_frame_chunk`, which
/// counts chunks in the raw stream. Do not add the count assertion here; it
/// would read as coverage while asserting the sink's guard, not the tap.
#[test]
fn each_episode_gets_a_descriptor_and_every_frame_carries_a_pose() {
    let sink = Arc::new(Mutex::new(InMemoryRecordSink::new()));
    let handle: Arc<Mutex<dyn RecordSink>> = sink.clone();
    let mut tap = RecordingTap::<Arm, 1, 1, 1>::with_scene_payload(Arm::new(4), handle);

    drive(&mut tap, 2, 4);
    drop(tap);

    let sink = sink.lock();
    assert_eq!(sink.episodes.len(), 2, "two episodes recorded");

    for (idx, ep) in &sink.episodes {
        let descriptor = ep
            .scene
            .as_ref()
            .unwrap_or_else(|| panic!("episode {idx} recorded no scene descriptor"));
        assert_eq!(
            descriptor.nodes.len(),
            2,
            "episode {idx}: descriptor survived intact",
        );
        // The reset frame plus four steps. Every one of them carries a pose,
        // and none of them carries geometry.
        assert_eq!(ep.frames.len(), 5, "episode {idx}: reset frame + 4 steps");
        for frame in &ep.frames {
            match &frame.family_payload {
                FamilyPayload::Scene(pose) => {
                    assert_eq!(
                        pose.transforms.len(),
                        1,
                        "episode {idx}: only the moving node is posed; the \
                         ground carries a static_transform and must not be \
                         re-sent every frame",
                    );
                }
                other => panic!("episode {idx}: expected a Scene payload, got {other:?}"),
            }
        }
    }
}

#[test]
fn every_posed_id_addresses_a_declared_node() {
    // The keyed-not-positional invariant, stated as the property a renderer
    // actually needs. A positional design would pass a round-trip test and fail
    // this one the moment the descriptor's node order changed.
    let sink = Arc::new(Mutex::new(InMemoryRecordSink::new()));
    let handle: Arc<Mutex<dyn RecordSink>> = sink.clone();
    let mut tap = RecordingTap::<Arm, 1, 1, 1>::with_scene_payload(Arm::new(3), handle);
    drive(&mut tap, 1, 3);
    drop(tap);

    let sink = sink.lock();
    let ep = sink.episodes.values().next().expect("one episode");
    let declared: Vec<NodeId> = ep
        .scene
        .as_ref()
        .expect("descriptor")
        .nodes
        .iter()
        .map(|n| n.id)
        .collect();

    for frame in &ep.frames {
        let FamilyPayload::Scene(pose) = &frame.family_payload else {
            panic!("expected a Scene payload");
        };
        assert!(pose.is_consistent(), "pose fails its own invariants");
        for (id, _) in &pose.transforms {
            assert!(
                declared.contains(id),
                "pose names {id:?}, which the descriptor does not declare",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// On disk: the chunk stream, which is what the report client actually reads.
// ---------------------------------------------------------------------------

#[test]
fn scene_survives_the_file_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = RecordingConfig {
        // Stride 1: this test is about what reaches the file, and a default
        // stride would drop frames and make the pose count a function of the
        // family table rather than of the drive loop.
        frame_stride: Some(1),
        ..RecordingConfig::new(EnvFamily::Locomotion, 7)
    };
    let writer = RecordWriter::open(dir.path(), cfg).expect("open writer");
    let run_dir = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));

    let mut tap = RecordingTap::<Arm, 1, 1, 1>::with_scene_payload(Arm::new(3), sink.clone());
    drive(&mut tap, 1, 3);
    drop(tap);
    sink.lock().on_run_end(manifest);
    assert!(
        sink.lock().take_error().is_none(),
        "recording reported a write error",
    );
    drop(sink);

    let path = run_dir.join("episode_000000.rec");
    let record = read_episode_record(&path).expect("decode episode file");

    let descriptor = record.scene.expect("descriptor reached the file");
    assert!(
        descriptor.is_consistent(),
        "descriptor decoded but fails its structural invariants",
    );
    assert_eq!(descriptor.background.as_deref(), Some("ground_plane"));
    assert!(
        (descriptor.bounds.1.hi() - 3.0).abs() < f32::EPSILON,
        "Extent decoded through its try_from and kept its value",
    );

    let ground = descriptor
        .nodes
        .iter()
        .find(|n| n.id == GROUND)
        .expect("ground node");
    assert_eq!(
        ground.static_transform,
        Some(Transform3::IDENTITY),
        "the static transform is what spares the ground a per-frame pose",
    );

    assert_eq!(record.frames.len(), 4, "reset frame + 3 steps");
    let FamilyPayload::Scene(first) = &record.frames[0].family_payload else {
        panic!("first frame is not a Scene payload");
    };
    assert_eq!(first.transforms[0].0, ARM);
    assert_eq!(first.markers[0].0, "tip");
}

/// The ordering property, asserted against the raw chunk stream rather than the
/// decoded aggregate — `EpisodeRecord` sorts the chunks into fields, so it
/// cannot show that the descriptor was written *before* the reset frame's pose.
#[test]
fn descriptor_chunk_precedes_the_first_frame_chunk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = RecordingConfig {
        frame_stride: Some(1),
        ..RecordingConfig::new(EnvFamily::Locomotion, 7)
    };
    let writer = RecordWriter::open(dir.path(), cfg).expect("open writer");
    let run_dir = writer.run_dir().to_path_buf();
    let manifest = writer.manifest_template();
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(writer));
    let mut tap = RecordingTap::<Arm, 1, 1, 1>::with_scene_payload(Arm::new(2), sink.clone());
    drive(&mut tap, 1, 2);
    drop(tap);
    sink.lock().on_run_end(manifest);
    drop(sink);

    let bytes = std::fs::read(run_dir.join("episode_000000.rec")).expect("read file");
    let order = chunk_kinds(&bytes);
    let scene_at = order
        .iter()
        .position(|k| *k == "Scene")
        .expect("a Scene chunk was written");
    let first_frame_at = order
        .iter()
        .position(|k| *k == "Frame")
        .expect("a Frame chunk was written");
    assert!(
        scene_at < first_frame_at,
        "geometry must precede the first pose that keys against it; got {order:?}",
    );
    assert_eq!(
        order.iter().filter(|k| **k == "Scene").count(),
        1,
        "exactly one descriptor per episode; got {order:?}",
    );
}

/// The two decoders must agree on the first-wins rule.
///
/// `rlevo-scene`'s `decode_episode_record` is what the report client reads
/// with; `rlevo-benchmarks`' `read_episode_record` is what these tests read
/// with. They are separate implementations of the same framing, so a
/// divergence would make this suite assert something the client never does —
/// exactly the fork ADR 0081 removed from the chunk type itself, still present
/// in the two loops over it.
#[test]
fn both_decoders_keep_the_first_descriptor() {
    use rlevo_scene::codec::{RecordChunk, decode_episode_record};
    use rlevo_scene::{EpisodeRecordHeader, FORMAT_VERSION, FrameRecord, RunId, bincode_config};

    fn framed<T: Serialize>(value: &T) -> Vec<u8> {
        let bytes = bincode::serde::encode_to_vec(value, bincode_config()).expect("encode");
        let len = u32::try_from(bytes.len()).expect("fits");
        let mut out = Vec::from(len.to_le_bytes());
        out.extend(bytes);
        out
    }

    let env = Arm::new(1);
    let first = env.scene_descriptor();
    let mut second = env.scene_descriptor();
    second.background = Some("replaced".into());

    let mut bytes = Vec::from(FORMAT_VERSION.to_le_bytes());
    bytes.resize(16, 0);
    bytes.extend(framed(&EpisodeRecordHeader {
        format_version: FORMAT_VERSION,
        run_id: RunId("test".into()),
        seed: 0,
        env_family: EnvFamily::Locomotion,
        created_at: 0,
        trial: None,
        kind: rlevo_scene::EpisodeKind::Training,
    }));
    bytes.extend(framed(&RecordChunk::Scene(first)));
    bytes.extend(framed(&RecordChunk::Frame(FrameRecord {
        step: 0,
        action: Vec::new(),
        reward: 0.0,
        ascii: None,
        styled: None,
        family_payload: FamilyPayload::Scene(env.scene_pose()),
    })));
    bytes.extend(framed(&RecordChunk::Scene(second)));

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("episode_000000.rec");
    std::fs::write(&path, &bytes).expect("write");

    let via_file = read_episode_record(&path).expect("benchmarks decoder");
    let via_slice = decode_episode_record(&bytes).expect("scene decoder");

    assert_eq!(
        via_file.scene, via_slice.scene,
        "the two decoders disagree about which descriptor survives",
    );
    assert_eq!(
        via_file.scene.expect("descriptor").background.as_deref(),
        Some("ground_plane"),
        "both must keep the first, not the replacement",
    );
}

/// Reads the file's chunks in stream order and names each one.
fn chunk_kinds(bytes: &[u8]) -> Vec<&'static str> {
    use rlevo_scene::codec::RecordChunk;

    let cfg = bincode::config::standard();
    let mut cursor = 16; // preamble
    let mut kinds = Vec::new();
    // The header is a bare `EpisodeRecordHeader`, not a `RecordChunk`; skip it
    // by its own length prefix before the chunk loop begins.
    let skip = |cursor: &mut usize| -> Option<Vec<u8>> {
        if bytes.len().saturating_sub(*cursor) < 4 {
            return None;
        }
        let len = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().ok()?) as usize;
        *cursor += 4;
        let end = cursor.checked_add(len)?;
        if end > bytes.len() {
            return None;
        }
        let payload = bytes[*cursor..end].to_vec();
        *cursor = end;
        Some(payload)
    };
    skip(&mut cursor).expect("header");
    while let Some(payload) = skip(&mut cursor) {
        let (chunk, _): (RecordChunk, usize) =
            bincode::serde::decode_from_slice(&payload, cfg).expect("decode chunk");
        kinds.push(match chunk {
            RecordChunk::Frame(_) => "Frame",
            RecordChunk::Metrics(_) => "Metrics",
            RecordChunk::Population(_) => "Population",
            RecordChunk::Scene(_) => "Scene",
        });
    }
    kinds
}
