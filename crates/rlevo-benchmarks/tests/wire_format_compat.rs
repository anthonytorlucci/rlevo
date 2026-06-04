//! Cross-crate guard for the report-client wire-format mirror.
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
    Box2dPayload as NativeBox2dPayload, CheckpointFormat as NativeCheckpointFormat,
    CheckpointKind as NativeCheckpointKind, CheckpointRef as NativeCheckpointRef,
    EnvFamily as NativeFamily, EpisodeKind as NativeEpisodeKind, EpisodeRecord as NativeRecord,
    EpisodeRecordHeader as NativeHeader, Classic2DPayload as NativeClassic2DPayload,
    FORMAT_VERSION as NATIVE_VERSION, FamilyPayload as NativePayload, FrameRecord as NativeFrame,
    GridPayload as NativeGridPayload, Landscape2DPayload as NativeLandscapePayload,
    Locomotion2DPayload as NativeLocomotionPayload, MetricSample as NativeMetric,
    PopulationSample as NativePopulationSample, RunId as NativeRunId, RunManifest as NativeManifest,
    TabularPayload as NativeTabularPayload, TrialRef as NativeTrialRef, bincode_config,
};
use rlevo_benchmarks_report_client::wire as client;
use rlevo_core::render::{
    BodyKind, CardTable, Classic2DBody, Classic2DRole, Classic2DSnapshot, Color, GridAgentMarker,
    GridDir, GridTile, Modifier, Point2, RigidBody2D, SpanStyle, StyledFrame, StyledLine,
    StyledSpan, TabularCell, TabularGrid, TabularLayout, TabularMarker, TabularMarkerKind,
    TabularSnapshot,
};

#[allow(clippy::too_many_lines)]
fn populated_native_record() -> NativeRecord {
    NativeRecord {
        header: NativeHeader {
            format_version: NATIVE_VERSION,
            run_id: NativeRunId("sync-test".into()),
            seed: 11,
            env_family: NativeFamily::Classic,
            created_at: 1_700_000_000,
            trial: Some(NativeTrialRef {
                env_index: 3,
                trial_index: 4,
            }),
            kind: NativeEpisodeKind::Evaluation,
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
            NativeFrame {
                step: 5,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::Grid(NativeGridPayload {
                    width: 2,
                    height: 2,
                    tiles: vec![
                        GridTile::Wall,
                        GridTile::Floor,
                        GridTile::Goal,
                        GridTile::Lava,
                    ],
                    agent: GridAgentMarker {
                        x: 1,
                        y: 0,
                        dir: GridDir::East,
                        carrying: None,
                    },
                }),
            },
            NativeFrame {
                step: 6,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::TabularText(NativeTabularPayload::from(
                    TabularSnapshot {
                        layout: TabularLayout::Grid(TabularGrid {
                            width: 2,
                            height: 1,
                            cells: vec![TabularCell::Start, TabularCell::Goal],
                            markers: vec![TabularMarker {
                                x: 0,
                                y: 0,
                                kind: TabularMarkerKind::Agent,
                            }],
                        }),
                    },
                )),
            },
            NativeFrame {
                step: 7,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::TabularText(NativeTabularPayload::from(
                    TabularSnapshot {
                        layout: TabularLayout::Cards(CardTable {
                            player_cards: vec![1, 10],
                            player_total: 21,
                            usable_ace: true,
                            dealer_cards: vec![7],
                            dealer_showing: 7,
                        }),
                    },
                )),
            },
            NativeFrame {
                step: 8,
                action: vec![],
                reward: 0.0,
                ascii: None,
                styled: None,
                family_payload: NativePayload::Classic2D(NativeClassic2DPayload::from(
                    Classic2DSnapshot {
                        bodies: vec![
                            Classic2DBody {
                                points: vec![Point2::new(-2.4, 0.0), Point2::new(2.4, 0.0)],
                                role: Classic2DRole::Track,
                                closed: false,
                            },
                            Classic2DBody {
                                points: vec![Point2::new(0.0, 0.1), Point2::new(0.0, 1.1)],
                                role: Classic2DRole::Pole,
                                closed: false,
                            },
                        ],
                        bounds: (Point2::new(-2.6, -0.4), Point2::new(2.6, 1.6)),
                    },
                )),
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
        population_samples: vec![
            NativePopulationSample {
                generation: 0,
                fitnesses: vec![0.5, 0.4, 0.3, 0.2, 0.1],
                diversity: Some(0.42),
                best_index: 4,
                best_genome_digest: Some([9u8; 16]),
                parents_of_best: vec![[1u8; 16], [2u8; 16]],
                inner_rl_returns: None,
            },
            NativePopulationSample {
                generation: 1,
                fitnesses: vec![0.4, 0.35, 0.25, 0.2, 0.05],
                diversity: Some(0.38),
                best_index: 4,
                best_genome_digest: None,
                parents_of_best: Vec::new(),
                inner_rl_returns: Some(vec![10.0, 11.5, 12.25, 9.75, 13.0]),
            },
        ],
    }
}

// Compile-time guard: if either constant is bumped without updating the
// other, this fires before any test is run.
const _: () = assert!(
    NATIVE_VERSION == client::FORMAT_VERSION,
    "FORMAT_VERSION mismatch: bump wire.rs to match schema.rs (or vice-versa)",
);
const _: () = assert!(
    rlevo_benchmarks::record::MIN_SUPPORTED_VERSION
        == client::MIN_SUPPORTED_VERSION,
    "MIN_SUPPORTED_VERSION mismatch: bump wire.rs to match schema.rs (or vice-versa)",
);

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
    // Trial provenance survives the host→client bincode round-trip.
    let trial = mirrored.header.trial.expect("trial provenance decoded");
    assert_eq!(trial.env_index, 3);
    assert_eq!(trial.trial_index, 4);
    // v6 episode kind survives the host→client round-trip.
    assert_eq!(mirrored.header.kind, client::EpisodeKind::Evaluation);
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
            (client::FamilyPayload::Grid(mc), NativePayload::Grid(nc)) => {
                assert_eq!(mc.width, nc.width);
                assert_eq!(mc.height, nc.height);
                assert_eq!(mc.tiles.len(), nc.tiles.len());
                assert_eq!(mc.agent.x, nc.agent.x);
                assert_eq!(mc.agent.y, nc.agent.y);
            }
            (client::FamilyPayload::TabularText(mc), NativePayload::TabularText(nc)) => {
                match (&mc.layout, &nc.layout) {
                    (client::TabularLayout::Grid(mg), TabularLayout::Grid(ng)) => {
                        assert_eq!(mg.width, ng.width);
                        assert_eq!(mg.cells.len(), ng.cells.len());
                        assert_eq!(mg.markers.len(), ng.markers.len());
                    }
                    (client::TabularLayout::Cards(mcard), TabularLayout::Cards(ncard)) => {
                        assert_eq!(mcard.player_total, ncard.player_total);
                        assert_eq!(mcard.usable_ace, ncard.usable_ace);
                        assert_eq!(mcard.dealer_showing, ncard.dealer_showing);
                    }
                    (om, on) => panic!("tabular layout mismatch: client={om:?} native={on:?}"),
                }
            }
            (client::FamilyPayload::Classic2D(mc), NativePayload::Classic2D(nc)) => {
                assert_eq!(mc.bodies.len(), nc.bodies.len());
                for (mb, nb) in mc.bodies.iter().zip(nc.bodies.iter()) {
                    assert_eq!(mb.points.len(), nb.points.len());
                    assert_eq!(mb.closed, nb.closed);
                }
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
    assert_eq!(mirrored.population_samples.len(), native.population_samples.len());
    for (m, n) in mirrored
        .population_samples
        .iter()
        .zip(native.population_samples.iter())
    {
        assert_eq!(m.generation, n.generation);
        assert_eq!(m.fitnesses.len(), n.fitnesses.len());
        for (a, b) in m.fitnesses.iter().zip(n.fitnesses.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(m.diversity.is_some(), n.diversity.is_some());
        assert_eq!(m.best_index, n.best_index);
        assert_eq!(m.best_genome_digest, n.best_genome_digest);
        assert_eq!(m.parents_of_best, n.parents_of_best);
        assert_eq!(m.inner_rl_returns.is_some(), n.inner_rl_returns.is_some());
    }
}

#[test]
#[cfg(feature = "json")]
fn manifest_provenance_and_checkpoints_mirror_via_json() {
    // The report client reads the manifest as JSON; guard that the v6
    // provenance fields + checkpoint list survive the host→client decode.
    let mut native = NativeManifest::new(NativeRunId("m6".into()), 7, NativeFamily::Classic, 1);
    native.algorithm = Some("ppo".into());
    native.num_seeds = Some(10);
    native.success_threshold = Some(195.0);
    native.checkpoints = vec![NativeCheckpointRef {
        step: 5000,
        kind: NativeCheckpointKind::Best,
        format: NativeCheckpointFormat::NamedMpk,
        path: "checkpoints/best.mpk".into(),
        metric: Some(241.7),
        digest: Some([5u8; 16]),
    }];

    let json = serde_json::to_string(&native).unwrap();
    let mirrored: client::RunManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(mirrored.algorithm.as_deref(), Some("ppo"));
    assert_eq!(mirrored.num_seeds, Some(10));
    assert_eq!(mirrored.success_threshold, Some(195.0));
    assert_eq!(mirrored.checkpoints.len(), 1);
    assert_eq!(mirrored.checkpoints[0].path, "checkpoints/best.mpk");
    assert_eq!(mirrored.checkpoints[0].kind, client::CheckpointKind::Best);
    assert_eq!(mirrored.checkpoints[0].step, 5000);
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

    for sample in &native.population_samples {
        let chunk = client::RecordChunk::Population(client::PopulationSample {
            generation: sample.generation,
            fitnesses: sample.fitnesses.clone(),
            diversity: sample.diversity,
            best_index: sample.best_index,
            best_genome_digest: sample.best_genome_digest,
            parents_of_best: sample.parents_of_best.clone(),
            inner_rl_returns: sample.inner_rl_returns.clone(),
        });
        let payload =
            bincode::serde::encode_to_vec(&chunk, client::bincode_config()).unwrap();
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&payload);
    }

    let decoded = client::decode_episode_record(&bytes).expect("client decoder accepts native bytes");
    assert_eq!(decoded.frames.len(), native.frames.len());
    assert_eq!(decoded.metrics.len(), 2);
    assert_eq!(decoded.population_samples.len(), native.population_samples.len());
    assert_eq!(decoded.population_samples[0].generation, 0);
    assert_eq!(decoded.population_samples[1].generation, 1);
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
