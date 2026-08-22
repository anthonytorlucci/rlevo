//! Guards that env **wrappers** forward every payload source, not a subset.
//!
//! A wrapper that changes *when* an episode ends, or that tees frames to a
//! panel, does not change what the environment looks like. It should therefore
//! record exactly as richly as the env it wraps. Both shipped wrappers
//! forwarded only `Classic2DPayloadSource`, and the gap failed quietly:
//!
//! - Grid and `toy_text` envs implement `AsciiRenderable`, so
//!   `RecordingTap::with_grid_payload(TimeLimit::new(env, ..))` did not
//!   compile, but `RecordingTap::new(TimeLimit::new(env, ..))` did — recording
//!   `FamilyPayload::Ascii` frames correctly tagged `family = Grids`, which the
//!   report rendered through its ASCII fallback. Right family, degraded view,
//!   nothing failed. That is the shape worth pinning: **the wrong-but-working
//!   call is the one a driver reaches for once the right one is rejected.**
//! - Locomotion envs have no `AsciiRenderable` at all, so a wrapped locomotion
//!   env was simply unrecordable.
//!
//! Three layers here, each catching something the others cannot:
//!
//! 1. [`timelimit_forwards_every_payload_source`] is compile-time: it fails to
//!    build if a forwarding impl is missing.
//! 2. [`wrapped_grid_env_records_structured_payload`] is behavioural — it
//!    checks the recorded frames actually carry `FamilyPayload::Grid`. A
//!    forwarding impl that compiled but returned a stub would pass layer 1.
//! 3. [`every_payload_source_is_forwarded_by_every_wrapper`] reads the trait
//!    list out of `rlevo-core` at test time and checks both wrapper files
//!    against it, so a *seventh* payload source cannot land unforwarded. This
//!    is the only layer that fails for a trait nobody has written a test for
//!    yet — which is exactly how this defect got in.
//!
//! Layers 1 and 2 cover `TimeLimit` only. `TuiEnvTap` lives behind
//! `rlevo-benchmarks`'s `tui` feature, which the pull-request gate does not
//! enable for that crate — its forwarding impls are compiled by
//! `viz-examples.yml` (full feature set) and checked for completeness here by
//! layer 3, which needs no feature because it reads source text.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use rlevo_core::environment::Environment;
use rlevo_core::render::payload::{
    Box2dPayloadSource, Classic2DPayloadSource, GridPayloadSource, Landscape2DPayloadSource,
    Locomotion2DPayloadSource, TabularPayloadSource,
};

use rlevo_benchmarks::record::schema::FamilyPayload;
use rlevo_benchmarks::record::{InMemoryRecordSink, RecordSink, RecordingTap};

use rlevo_environments::grids::core::action::GridAction;
use rlevo_environments::grids::empty::{EmptyConfig, EmptyEnv};
use rlevo_environments::wrappers::TimeLimit;

// ---------------------------------------------------------------------------
// Layer 1 — compile-time forwarding
// ---------------------------------------------------------------------------

/// Each body asserts `TimeLimit<E>: Trait` given only `E: Trait`.
///
/// The *generic* body is the proof, not the instantiation: rustc discharges
/// `TimeLimit<E>: GridPayloadSource` from `E: GridPayloadSource` when it
/// type-checks `grid`, so a missing forwarder fails at the definition. Deleting
/// `TimeLimit`'s `GridPayloadSource` impl reports exactly
/// ``the trait bound `TimeLimit<E>: GridPayloadSource` is not satisfied`` here,
/// naming `E` rather than any concrete env. The calls below exist only to keep
/// these from reading as dead code; the type they pass is uninhabited and never
/// built.
#[test]
fn timelimit_forwards_every_payload_source() {
    fn classic2d<E: Classic2DPayloadSource>() {
        fn requires<T: Classic2DPayloadSource>() {}
        requires::<TimeLimit<E>>();
    }
    fn grid<E: GridPayloadSource>() {
        fn requires<T: GridPayloadSource>() {}
        requires::<TimeLimit<E>>();
    }
    fn tabular<E: TabularPayloadSource>() {
        fn requires<T: TabularPayloadSource>() {}
        requires::<TimeLimit<E>>();
    }
    fn box2d<E: Box2dPayloadSource>() {
        fn requires<T: Box2dPayloadSource>() {}
        requires::<TimeLimit<E>>();
    }
    fn locomotion2d<E: Locomotion2DPayloadSource>() {
        fn requires<T: Locomotion2DPayloadSource>() {}
        requires::<TimeLimit<E>>();
    }
    fn landscape2d<E: Landscape2DPayloadSource>() {
        fn requires<T: Landscape2DPayloadSource>() {}
        requires::<TimeLimit<E>>();
    }

    classic2d::<AnySource>();
    grid::<AnySource>();
    tabular::<AnySource>();
    box2d::<AnySource>();
    locomotion2d::<AnySource>();
    landscape2d::<AnySource>();
}

/// Uninhabited stand-in implementing all six payload sources.
///
/// Uninhabited on purpose: `match *self {}` discharges every method body
/// without constructing a single snapshot type, so this stays correct as
/// snapshot shapes change and needs no cargo feature to name.
enum AnySource {}

impl Classic2DPayloadSource for AnySource {
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        match *self {}
    }
}
impl GridPayloadSource for AnySource {
    fn grid_snapshot(&self) -> rlevo_core::render::payload::GridSnapshot {
        match *self {}
    }
}
impl TabularPayloadSource for AnySource {
    fn tabular_snapshot(&self) -> rlevo_core::render::payload::TabularSnapshot {
        match *self {}
    }
}
impl Box2dPayloadSource for AnySource {
    fn box2d_snapshot(&self) -> rlevo_core::render::payload::Box2dSnapshot {
        match *self {}
    }
}
impl Locomotion2DPayloadSource for AnySource {
    fn locomotion2d_snapshot(&self) -> rlevo_core::render::payload::Locomotion2DSnapshot {
        match *self {}
    }
}
impl Landscape2DPayloadSource for AnySource {
    fn landscape2d_snapshot(&self) -> rlevo_core::render::payload::Landscape2DSnapshot {
        match *self {}
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — behavioural
// ---------------------------------------------------------------------------

/// A `TimeLimit`-wrapped gridworld records **structured** frames.
///
/// Pins the defect rather than its inverse: before the forwarders existed this
/// test would not compile at all, and the workaround a driver reached for
/// (`RecordingTap::new`) produced `FamilyPayload::Ascii` — which is what the
/// second half asserts is *not* what comes out now.
#[test]
fn wrapped_grid_env_records_structured_payload() {
    let env = EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false).expect("valid config");
    let sink = Arc::new(Mutex::new(InMemoryRecordSink::new()));
    let sink_dyn: Arc<Mutex<dyn RecordSink>> = sink.clone();

    let mut tap = RecordingTap::<_, 3, 3, 1>::with_grid_payload(TimeLimit::new(env, 8), sink_dyn);

    tap.reset().expect("reset");
    // Bounded by construction: a fixed count, not "until done" — a forwarding
    // regression must fail this test, never hang it.
    for _ in 0..4 {
        if tap.step(GridAction::Forward).is_err() {
            break;
        }
    }

    let recorded = sink.lock();
    let episode = recorded.episodes.get(&0).expect("episode 0 recorded");
    assert!(
        !episode.frames.is_empty(),
        "the tap recorded no frames at all, so the payload assertion below \
         would pass vacuously",
    );
    for (i, frame) in episode.frames.iter().enumerate() {
        assert!(
            matches!(frame.family_payload, FamilyPayload::Grid(_)),
            "frame {i} carries FamilyPayload::{}, not ::Grid — the wrapper is \
             swallowing the inner env's GridPayloadSource and the report will \
             silently fall back to ASCII",
            payload_variant(&frame.family_payload),
        );
    }
}

/// Variant name only. The payloads themselves carry whole tile grids and body
/// lists, so `{:?}` on a failure would bury the one word that matters.
fn payload_variant(p: &FamilyPayload) -> &'static str {
    match p {
        FamilyPayload::Ascii => "Ascii",
        FamilyPayload::Grid(_) => "Grid",
        FamilyPayload::TabularText(_) => "TabularText",
        FamilyPayload::Classic2D(_) => "Classic2D",
        FamilyPayload::Box2dBodies(_) => "Box2dBodies",
        FamilyPayload::Locomotion2D(_) => "Locomotion2D",
        FamilyPayload::Landscape2D(_) => "Landscape2D",
        _ => "<unnamed variant — add it here>",
    }
}

// ---------------------------------------------------------------------------
// Layer 3 — completeness against the trait list
// ---------------------------------------------------------------------------

/// Wrapper source files that must forward every payload source, relative to
/// the workspace root.
static WRAPPERS: &[(&str, &str)] = &[
    (
        "TimeLimit",
        "crates/rlevo-environments/src/wrappers/time_limit.rs",
    ),
    (
        "TuiEnvTap",
        "crates/rlevo-benchmarks/src/env_wrappers/tui_env_tap.rs",
    ),
];

/// `CARGO_MANIFEST_DIR` is `crates/rlevo` for an integration test here.
fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/rlevo always has two ancestors")
}

/// Every `pub trait *PayloadSource` declared in `rlevo-core`.
fn declared_payload_sources() -> Vec<String> {
    let path = workspace_root().join("crates/rlevo-core/src/render/payload.rs");
    let src =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut found: Vec<String> = src
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub trait "))
        .filter_map(|rest| rest.split(|c: char| !c.is_alphanumeric()).next())
        .filter(|name| name.ends_with("PayloadSource"))
        .map(str::to_owned)
        .collect();
    found.sort_unstable();
    found.dedup();
    assert!(
        found.len() >= 6,
        "found only {found:?} in payload.rs — the scan stopped matching the \
         source, so this whole test has silently stopped checking anything",
    );
    found
}

/// A seventh payload source added to `rlevo-core` must be forwarded by every
/// wrapper. Nothing else notices: an unforwarded source is not a compile
/// error, it is a wrapper that quietly records less than the env it wraps.
#[test]
fn every_payload_source_is_forwarded_by_every_wrapper() {
    let traits = declared_payload_sources();
    let mut missing = Vec::new();

    for (wrapper, rel) in WRAPPERS {
        let path = workspace_root().join(rel);
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        for t in &traits {
            // The forwarding impl names the trait in `impl .. <Trait> for ..`;
            // requiring `for` on the same statement keeps a `use` or a doc
            // mention from counting as a forwarder.
            let forwarded = src
                .split("impl")
                .any(|chunk| chunk.contains(t.as_str()) && chunk.contains(" for "));
            if !forwarded {
                missing.push(format!("{wrapper} does not forward {t}"));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "wrapper(s) missing payload forwarding:\n  {}\n\nA wrapper must \
         forward every payload source: it changes when an episode ends or \
         where frames are teed, never what the env looks like. An \
         unforwarded source does not break the build — the caller falls back \
         to `RecordingTap::new` and silently records ASCII frames under the \
         correct family tag.",
        missing.join("\n  "),
    );
}
