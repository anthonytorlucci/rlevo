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
    EnvFamily as NativeFamily, EpisodeRecord as NativeRecord,
    EpisodeRecordHeader as NativeHeader, FORMAT_VERSION as NATIVE_VERSION,
    FamilyPayload as NativePayload, FrameRecord as NativeFrame, MetricSample as NativeMetric,
    RunId as NativeRunId, bincode_config,
};
use rlevo_benchmarks_report_client::wire as client;
use rlevo_core::render::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};

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
        let chunk = client::RecordChunk::Frame(client::FrameRecord {
            step: frame.step,
            action: frame.action.clone(),
            reward: frame.reward,
            ascii: frame.ascii.clone(),
            styled: frame.styled.as_ref().map(|sf| client::StyledFrame {
                lines: sf
                    .lines
                    .iter()
                    .map(|line| client::StyledLine {
                        spans: line
                            .spans
                            .iter()
                            .map(|sp| client::StyledSpan {
                                text: sp.text.clone(),
                                style: client::SpanStyle::default(),
                            })
                            .collect(),
                    })
                    .collect(),
            }),
            family_payload: client::FamilyPayload::Ascii,
        });
        let payload = bincode::serde::encode_to_vec(&chunk, client::bincode_config()).unwrap();
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&payload);
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
    assert_eq!(decoded.frames.len(), 2);
    assert_eq!(decoded.metrics.len(), 2);
    assert_eq!(decoded.header.seed, 11);

    // Suppress the otherwise-unused imports for the styled types: we
    // exercise them via SpanStyle::default() above but want a hard ref
    // to surface "field renamed" failures here too.
    let _ = (Color::Red, Modifier::default(), SpanStyle::default(),
             StyledSpan::raw("x"), StyledLine::default(), StyledFrame::default());
}
