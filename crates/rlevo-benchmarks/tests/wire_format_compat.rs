//! Cross-crate guard for the M5.1 wire-format mirror.
//!
//! `rlevo-benchmarks-report-client/src/wire.rs` redeclares the bincode
//! record types so the client can decode `.rec` payloads in the
//! browser without depending on `rlevo-benchmarks` (which has native-only
//! transitive deps). This test exercises both sides on the same
//! populated `EpisodeRecord` and asserts they round-trip identically.
//!
//! If this test fails, the schema and the mirror have drifted. Update
//! the mirror to match (and bump `FORMAT_VERSION` if the wire format
//! genuinely changed).

#![cfg(feature = "record")]

use rlevo_benchmarks::record::{
    Box2dPayload as NativeBox2dPayload, EnvFamily as NativeFamily,
    EpisodeRecord as NativeRecord, EpisodeRecordHeader as NativeHeader,
    FORMAT_VERSION as NATIVE_VERSION, FamilyPayload as NativePayload,
    FrameRecord as NativeFrame, Landscape2DPayload as NativeLandscapePayload,
    Locomotion2DPayload as NativeLocomotionPayload, MetricSample as NativeMetric,
    RunId as NativeRunId, bincode_config,
};
use rlevo_benchmarks_report_client::wire as client;
use rlevo_core::render::{
    BodyKind, Color, Modifier, Point2, RigidBody2D, SpanStyle, StyledFrame, StyledLine,
    StyledSpan,
};

fn populated_native_record() -> NativeRecord {
    NativeRecord {
        header: NativeHeader {
            format_version: NATIVE_VERSION,
            run_id: NativeRunId("sync-test".into()),
            seed: 11,
            env_family: NativeFamily::Classic,
            created_at: 1_700_000_000,
        },
        frames: vec![
            NativeFrame {
                step: 0,
                action: vec![1, 2, 3],
                reward: 1.5,
                ascii: Some("step 0".into()),
                styled: Some(StyledFrame {
                    lines: vec![StyledLine {
                        spans: vec![StyledSpan::new(
                            "x",
                            SpanStyle::default().fg(Color::Red).bold(),
                        )],
                    }],
                }),
                family_payload: NativePayload::Ascii,
            },
            NativeFrame {
                step: 1,
                action: vec![],
                reward: -0.25,
                ascii: None,
                styled: None,
                family_payload: NativePayload::Ascii,
            },
            NativeFrame {
                step: 2,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::Landscape2D(NativeLandscapePayload {
                    bounds_x: (-5.0, 5.0),
                    bounds_y: (-5.0, 5.0),
                    current: Point2::new(0.5, -0.25),
                    best: Some(Point2::new(0.0, 0.0)),
                    trail: vec![Point2::new(0.1, 0.2)],
                    label: "sphere".into(),
                }),
            },
            NativeFrame {
                step: 3,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::Box2dBodies(NativeBox2dPayload {
                    world_bounds: (Point2::new(-10.0, -1.0), Point2::new(10.0, 8.0)),
                    bodies: vec![RigidBody2D {
                        vertices: vec![
                            Point2::new(-0.5, -0.5),
                            Point2::new(0.5, -0.5),
                            Point2::new(0.5, 0.5),
                            Point2::new(-0.5, 0.5),
                        ],
                        position: Point2::new(1.0, 2.0),
                        rotation_rad: 0.25,
                        kind: BodyKind::Hull,
                    }],
                    contacts: vec![Point2::new(0.0, 0.0)],
                }),
            },
            NativeFrame {
                step: 4,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::Locomotion2D(NativeLocomotionPayload {
                    joints: vec![Point2::new(0.0, 1.0), Point2::new(0.5, 1.5)],
                    bones: vec![(0, 1)],
                    ground_y: 0.0,
                    com: Some(Point2::new(0.25, 1.25)),
                    contacts: vec![Point2::new(0.0, 0.0)],
                }),
            },
        ],
        metrics: vec![
            NativeMetric {
                step: 100,
                name: "policy_loss".into(),
                value: 0.01,
            },
            NativeMetric {
                step: 200,
                name: "entropy".into(),
                value: 0.6789,
            },
        ],
    }
}

#[test]
fn format_version_constants_agree() {
    assert_eq!(NATIVE_VERSION, client::FORMAT_VERSION);
}

#[test]
fn native_encode_decodes_via_client_wire_types() {
    let native = populated_native_record();
    let bytes = bincode::serde::encode_to_vec(&native, bincode_config()).unwrap();
    let (mirrored, _): (client::EpisodeRecord, usize) =
        bincode::serde::decode_from_slice(&bytes, client::bincode_config()).unwrap();

    assert_eq!(mirrored.header.format_version, native.header.format_version);
    assert_eq!(mirrored.header.run_id.0, native.header.run_id.0);
    assert_eq!(mirrored.header.seed, native.header.seed);
    assert_eq!(mirrored.header.created_at, native.header.created_at);
    assert_eq!(mirrored.frames.len(), native.frames.len());
    for (m, n) in mirrored.frames.iter().zip(native.frames.iter()) {
        assert_eq!(m.step, n.step);
        assert_eq!(m.action, n.action);
        assert!((m.reward - n.reward).abs() < 1e-6);
        assert_eq!(m.ascii, n.ascii);
        assert_eq!(m.styled.is_some(), n.styled.is_some());
        // FamilyPayload variant tags must round-trip — order of
        // variants in the enum is part of the wire contract.
        match (&m.family_payload, &n.family_payload) {
            (client::FamilyPayload::Ascii, NativePayload::Ascii) => {}
            (client::FamilyPayload::Landscape2D(mc), NativePayload::Landscape2D(nc)) => {
                assert_eq!(mc.label, nc.label);
                assert_eq!(mc.trail.len(), nc.trail.len());
            }
            (client::FamilyPayload::Box2dBodies(mc), NativePayload::Box2dBodies(nc)) => {
                assert_eq!(mc.bodies.len(), nc.bodies.len());
                assert_eq!(mc.contacts.len(), nc.contacts.len());
            }
            (client::FamilyPayload::Locomotion2D(mc), NativePayload::Locomotion2D(nc)) => {
                assert_eq!(mc.joints.len(), nc.joints.len());
                assert_eq!(mc.bones.len(), nc.bones.len());
                assert_eq!(mc.com.is_some(), nc.com.is_some());
            }
            (other_m, other_n) => panic!(
                "family_payload variant mismatch: client={other_m:?} native={other_n:?}"
            ),
        }
    }
    assert_eq!(mirrored.metrics.len(), native.metrics.len());
    for (m, n) in mirrored.metrics.iter().zip(native.metrics.iter()) {
        assert_eq!(m.step, n.step);
        assert_eq!(m.name, n.name);
        assert!((m.value - n.value).abs() < 1e-9);
    }
}

#[test]
fn client_decode_episode_record_walks_full_wire_stream() {
    // Build the same on-disk byte layout the RecordWriter emits:
    // [16-byte preamble][length-prefixed header][length-prefixed RecordChunk]*.
    let native = populated_native_record();
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(&NATIVE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&[0u8; 14]);

    let header_bytes =
        bincode::serde::encode_to_vec(&native.header, bincode_config()).unwrap();
    bytes.extend_from_slice(&u32::try_from(header_bytes.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&header_bytes);

    // Native RecordChunk type is private inside writer.rs; emulate it
    // by hand using the same enum tag layout. The client's RecordChunk
    // mirror is what we are testing.
    for frame in &native.frames {
        // Bincode-encode the native frame directly — the wire layout
        // is byte-identical across the host/client mirror by
        // construction, so we don't manually rebuild the variant.
        let native_chunk_bytes =
            bincode::serde::encode_to_vec(frame, bincode_config()).unwrap();
        // Wrap into a client-side RecordChunk::Frame by prefixing the
        // enum tag (0 = Frame).
        let mut chunk_bytes: Vec<u8> = Vec::with_capacity(native_chunk_bytes.len() + 1);
        chunk_bytes.push(0u8); // RecordChunk::Frame
        chunk_bytes.extend_from_slice(&native_chunk_bytes);
        bytes.extend_from_slice(&u32::try_from(chunk_bytes.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&chunk_bytes);
    }
    let metrics_chunk = client::RecordChunk::Metrics(
        native
            .metrics
            .iter()
            .map(|m| client::MetricSample {
                step: m.step,
                name: m.name.clone(),
                value: m.value,
            })
            .collect(),
    );
    let metrics_payload =
        bincode::serde::encode_to_vec(&metrics_chunk, client::bincode_config()).unwrap();
    bytes.extend_from_slice(&u32::try_from(metrics_payload.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&metrics_payload);

    let decoded = client::decode_episode_record(&bytes).expect("client decoder accepts native bytes");
    assert_eq!(decoded.frames.len(), native.frames.len());
    assert_eq!(decoded.metrics.len(), 2);
    assert_eq!(decoded.header.seed, 11);
    // Spot-check that the rich-payload variants survived the wire.
    assert!(matches!(
        decoded.frames[2].family_payload,
        client::FamilyPayload::Landscape2D(_)
    ));
    assert!(matches!(
        decoded.frames[3].family_payload,
        client::FamilyPayload::Box2dBodies(_)
    ));
    assert!(matches!(
        decoded.frames[4].family_payload,
        client::FamilyPayload::Locomotion2D(_)
    ));

    // Suppress the otherwise-unused imports for the styled types: we
    // exercise them via SpanStyle::default() above but want a hard ref
    // to surface "field renamed" failures here too.
    let _ = (Color::Red, Modifier::default(), SpanStyle::default(),
             StyledSpan::raw("x"), StyledLine::default(), StyledFrame::default());
}
