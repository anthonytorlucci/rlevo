//! Guards the [`Box2dPayloadSource`] impls for `BipedalWalker` and
//! `CarRacing`.
//!
//! Both envs implement `AsciiRenderable` but not this trait, so
//! `RecordingTap::with_box2d_payload` did not compile for either and a driver
//! fell back to `RecordingTap::new` — recording `FamilyPayload::Ascii` frames
//! under a correct `family = Box2d` tag, which the box2d adapter then rendered
//! through its generic view instead of drawing rigid bodies.
//!
//! # The NaN case is the interesting one
//!
//! `CarRacing`'s wheel fixed-joints are stiff enough that the car body and all
//! four wheels reach `(NaN, NaN)` within roughly ten to twenty steps under
//! essentially any action, and `CarRacingState::is_valid` has no finiteness
//! predicate to catch it. That divergence is a separate open defect, but it
//! meets this payload head-on: NaN written to the wire propagates through the
//! report's scale arithmetic into unparseable SVG coordinates, so the viewer
//! gets a blank panel and no message.
//!
//! [`no_nan_reaches_the_wire_from_a_diverged_car`] pins the invariant that
//! survives either outcome — **no non-finite value ever leaves the payload** —
//! rather than asserting that divergence happens. If the joints are fixed
//! upstream the test still passes for the right reason; if they are not, the
//! guard keeps the corruption out of the recording.
#![cfg(feature = "box2d")]

use rlevo_core::environment::Environment;
use rlevo_core::render::{Box2dPayloadSource, Box2dSnapshot};

use rlevo_environments::box2d::bipedal_walker::action::BipedalWalkerAction;
use rlevo_environments::box2d::bipedal_walker::config::BipedalWalkerConfig;
use rlevo_environments::box2d::bipedal_walker::env::BipedalWalker;
use rlevo_environments::box2d::car_racing::action::CarRacingAction;
use rlevo_environments::box2d::car_racing::config::CarRacingConfig;
use rlevo_environments::box2d::car_racing::env::CarRacing;

/// Every scalar that can reach the wire format must be finite.
fn assert_all_finite(name: &str, step: usize, snap: &Box2dSnapshot) {
    let (min, max) = snap.world_bounds;
    for (label, v) in [
        ("world_bounds.min.x", min.x),
        ("world_bounds.min.y", min.y),
        ("world_bounds.max.x", max.x),
        ("world_bounds.max.y", max.y),
    ] {
        assert!(
            v.is_finite(),
            "{name} step {step}: {label} is {v} — the report's scale \
             arithmetic turns this into unparseable SVG coordinates and \
             renders a blank panel with no warning",
        );
    }
    for (i, b) in snap.bodies.iter().enumerate() {
        assert!(
            b.position.x.is_finite() && b.position.y.is_finite(),
            "{name} step {step}: body {i} position is ({}, {})",
            b.position.x,
            b.position.y,
        );
        assert!(
            b.rotation_rad.is_finite(),
            "{name} step {step}: body {i} rotation is {}",
            b.rotation_rad,
        );
        for (j, v) in b.vertices.iter().enumerate() {
            assert!(
                v.x.is_finite() && v.y.is_finite(),
                "{name} step {step}: body {i} vertex {j} is ({}, {})",
                v.x,
                v.y,
            );
        }
    }
}

/// A frame the payload refuses to describe must be refused *legibly*.
///
/// Degenerate `world_bounds` is the one input the box2d adapter rejects with a
/// visible message; an empty body list under valid bounds would just render an
/// empty scene, which is the silent outcome this batch exists to remove.
fn assert_empty_means_refused(name: &str, step: usize, snap: &Box2dSnapshot) {
    if snap.bodies.is_empty() {
        let (min, max) = snap.world_bounds;
        assert!(
            (max.x - min.x).abs() < f32::EPSILON && (max.y - min.y).abs() < f32::EPSILON,
            "{name} step {step}: emitted no bodies but non-degenerate bounds \
             ({min:?}, {max:?}) — the report will draw an empty scene rather \
             than say it cannot render",
        );
    }
}

#[test]
fn bipedal_walker_payload_stays_finite_under_full_throttle() {
    let mut env = BipedalWalker::with_config(BipedalWalkerConfig::default()).expect("valid");
    env.reset().expect("reset");

    let snap = env.box2d_snapshot();
    assert_all_finite("BipedalWalker", 0, &snap);
    assert!(
        !snap.bodies.is_empty(),
        "a freshly reset walker must emit bodies; an empty payload renders as \
         nothing and is indistinguishable from the ASCII fallback",
    );

    // Fixed iteration count, not "until done" — a regression must fail this
    // test, never hang it.
    for step in 1..=60 {
        if env
            .step(BipedalWalkerAction([1.0, -1.0, 1.0, -1.0]))
            .is_err()
        {
            break;
        }
        let snap = env.box2d_snapshot();
        assert_all_finite("BipedalWalker", step, &snap);
        assert_empty_means_refused("BipedalWalker", step, &snap);
    }
}

/// The walker's hull, four leg segments, and terrain all reach the payload —
/// the ASCII tier draws one flat line for terrain regardless of relief, so the
/// segments are what this payload adds over it.
#[test]
fn bipedal_walker_payload_carries_hull_legs_and_terrain() {
    use rlevo_core::render::BodyKind;

    let mut env = BipedalWalker::with_config(BipedalWalkerConfig::default()).expect("valid");
    env.reset().expect("reset");
    let snap = env.box2d_snapshot();

    let count = |k: BodyKind| snap.bodies.iter().filter(|b| b.kind == k).count();
    assert_eq!(count(BodyKind::Hull), 1, "one hull");
    assert_eq!(
        count(BodyKind::Leg),
        4,
        "two upper and two lower leg segments"
    );
    assert!(
        count(BodyKind::Ground) > 0,
        "at least one terrain segment must be in view; the 200-unit base slab \
         is skipped deliberately, so zero here means the walker floats over \
         nothing",
    );
}

/// No non-finite value leaves the payload, whether or not the wheel joints
/// have been fixed upstream.
///
/// Deliberately not asserting *that* the solver diverges: pinning the defect
/// would make this test fail the day it is fixed, for the wrong reason.
#[test]
fn no_nan_reaches_the_wire_from_a_diverged_car() {
    let mut env = CarRacing::with_config(CarRacingConfig::default()).expect("valid");
    env.reset().expect("reset");

    let snap = env.box2d_snapshot();
    assert_all_finite("CarRacing", 0, &snap);
    assert!(
        !snap.bodies.is_empty(),
        "a freshly reset car must emit bodies — the car, its wheels, and the \
         track tiles in view",
    );

    // 60 steps is well past the ten-to-twenty where the joints are known to
    // diverge, and bounded so a regression fails rather than hangs.
    for step in 1..=60 {
        if env.step(CarRacingAction::new(1.0, 1.0, 0.0)).is_err() {
            break;
        }
        let snap = env.box2d_snapshot();
        assert_all_finite("CarRacing", step, &snap);
        assert_empty_means_refused("CarRacing", step, &snap);
    }
}

/// The track is the thing the ASCII tier cannot show at all — `viewport()`'s
/// own comment says the report tier owns track rendering via this payload.
#[test]
fn car_racing_payload_carries_the_track() {
    use rlevo_core::render::BodyKind;

    let mut env = CarRacing::with_config(CarRacingConfig::default()).expect("valid");
    env.reset().expect("reset");
    let snap = env.box2d_snapshot();

    let tiles = snap
        .bodies
        .iter()
        .filter(|b| matches!(b.kind, BodyKind::Ground | BodyKind::Goal))
        .count();
    assert!(
        tiles > 0,
        "no track tiles in view at reset — the car spawns on the centreline, \
         so an empty track means the viewport cull is rejecting everything",
    );
    assert_eq!(
        snap.bodies
            .iter()
            .filter(|b| b.kind == BodyKind::Hull)
            .count(),
        1,
        "the car body",
    );
    assert_eq!(
        snap.bodies
            .iter()
            .filter(|b| b.kind == BodyKind::Wheel)
            .count(),
        4,
        "four wheels",
    );
}
