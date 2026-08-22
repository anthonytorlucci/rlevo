//! Guards the [`Locomotion2DPayloadSource`] impls for all four locomotion
//! environments.
//!
//! Locomotion envs deliberately have **no** `AsciiRenderable` impl, so this
//! payload is their only rendering pathway in the entire stack. Three of the
//! four (`Swimmer`, `Reacher`, `InvertedDoublePendulum`) shipped without it:
//! `RecordingTap::new` did not compile for want of `AsciiRenderable`, and
//! `with_locomotion_payload` did not compile for want of this trait, so the
//! only constructor that accepted them was
//! `new_headless(env, sink, |_| FamilyPayload::Ascii)` — which records frames
//! that render as "(no rendered frame)".
//!
//! # What these tests check, and why geometrically
//!
//! A payload impl that compiles proves nothing: every one of these is a
//! projection, and a transposed axis or a mis-reflected joint produces a
//! perfectly well-typed stick figure of the wrong thing. The bone lengths are
//! the invariant that catches it — each is a rigid body, so the distance
//! between the joints it connects must equal its configured length at every
//! frame, whatever the pose. [`assert_shared_invariants`] covers what is true
//! of all four; the per-env tests add the geometry only that env knows.
#![cfg(feature = "locomotion")]

use rlevo_core::environment::Environment;
use rlevo_core::render::{Locomotion2DPayloadSource, Locomotion2DSnapshot, Point2};

use rlevo_environments::locomotion::inverted_double_pendulum::{
    config::InvertedDoublePendulumConfig, env::InvertedDoublePendulum,
};
use rlevo_environments::locomotion::inverted_pendulum::{
    config::InvertedPendulumConfig, env::InvertedPendulum,
};
use rlevo_environments::locomotion::reacher::{config::ReacherConfig, env::Reacher};
use rlevo_environments::locomotion::swimmer::{
    action::SwimmerAction, config::SwimmerConfig, env::Swimmer,
};

fn dist(a: Point2, b: Point2) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// Invariants every locomotion payload must satisfy, whatever the env.
fn assert_shared_invariants(name: &str, snap: &Locomotion2DSnapshot) {
    assert!(
        !snap.joints.is_empty(),
        "{name}: payload has no joints, so the report renders an empty frame \
         — indistinguishable from the ASCII fallback this trait exists to \
         replace",
    );
    assert!(
        !snap.bones.is_empty(),
        "{name}: payload has no bones, so the joints render as loose dots",
    );
    for (i, j) in snap.joints.iter().enumerate() {
        assert!(
            j.x.is_finite() && j.y.is_finite(),
            "{name}: joint {i} is non-finite ({}, {}) — the report's bounds \
             computation silently drops non-finite points, so this degrades \
             the frame rather than failing it",
            j.x,
            j.y,
        );
    }
    for &(a, b) in &snap.bones {
        let n = snap.joints.len();
        assert!(
            (a as usize) < n && (b as usize) < n,
            "{name}: bone ({a}, {b}) indexes outside {n} joints; the report \
             silently skips out-of-range bones, so the limb just vanishes",
        );
    }
    if let Some(gy) = snap.ground_y {
        assert!(gy.is_finite(), "{name}: ground_y is non-finite");
    }
}

/// Envs with a real floor report it; envs without one must report `None`.
///
/// This is the distinction the payload could not express before: `ground_y`
/// was a bare `f32`, so a top-down zero-gravity env had to name a floor that
/// does not exist and the adapter drew it across the frame.
#[test]
fn only_envs_with_a_floor_report_a_ground_line() {
    let ip = InvertedPendulum::with_config(InvertedPendulumConfig::default()).expect("valid");
    assert_eq!(
        ip.locomotion2d_snapshot().ground_y,
        Some(0.0),
        "InvertedPendulum's cart rests on the world-z floor plane",
    );

    let idp = InvertedDoublePendulum::with_config(InvertedDoublePendulumConfig::default())
        .expect("valid");
    assert_eq!(
        idp.locomotion2d_snapshot().ground_y,
        Some(0.0),
        "InvertedDoublePendulum's cart rests on the world-z floor plane",
    );

    let swimmer = Swimmer::with_config(SwimmerConfig::default()).expect("valid");
    assert_eq!(
        swimmer.locomotion2d_snapshot().ground_y,
        None,
        "Swimmer floats in zero gravity viewed from above; a ground line \
         would draw a seabed through the middle of the frame",
    );

    let reacher = Reacher::with_config(ReacherConfig::default()).expect("valid");
    assert_eq!(
        reacher.locomotion2d_snapshot().ground_y,
        None,
        "Reacher is a top-down zero-gravity manipulator with no floor",
    );
}

#[test]
fn inverted_pendulum_payload_holds_its_invariants() {
    let env = InvertedPendulum::with_config(InvertedPendulumConfig::default()).expect("valid");
    assert_shared_invariants("InvertedPendulum", &env.locomotion2d_snapshot());
}

/// The two poles are rigid capsules, so each bone must measure exactly its
/// configured length — the check that a mis-reflected joint fails.
#[test]
fn inverted_double_pendulum_bones_match_pole_length() {
    let config = InvertedDoublePendulumConfig::default();
    let pole_length = config.pole_length;
    let env = InvertedDoublePendulum::with_config(config).expect("valid");
    let snap = env.locomotion2d_snapshot();

    assert_shared_invariants("InvertedDoublePendulum", &snap);
    assert_eq!(snap.joints.len(), 3, "cart, mid-hinge, tip");

    let lower = dist(snap.joints[0], snap.joints[1]);
    let upper = dist(snap.joints[1], snap.joints[2]);
    // The lower bone spans cart *centre* to the mid-hinge, so it carries the
    // cart's half-height on top of the pole; the upper bone is pole-only.
    assert!(
        upper > pole_length * 0.9 && upper < pole_length * 1.1,
        "upper bone should measure the pole length {pole_length}, got {upper} \
         — a reflection through the wrong anchor lands here",
    );
    assert!(
        lower > pole_length * 0.9,
        "lower bone should be at least the pole length {pole_length}, got \
         {lower}",
    );
    assert!(
        snap.joints[2].y > snap.joints[0].y,
        "at reset both poles are near-upright, so the tip must sit above the \
         cart; if it does not, the projection has taken the wrong axis pair",
    );
}

/// Both links are rigid, so the two bones must measure their configured
/// lengths — and the shoulder is pinned at the world origin.
#[test]
fn reacher_bones_match_link_lengths_and_target_is_carried() {
    let config = ReacherConfig::default();
    let (l1, l2) = (config.link1_length, config.link2_length);
    let env = Reacher::with_config(config).expect("valid");
    let snap = env.locomotion2d_snapshot();

    assert_shared_invariants("Reacher", &snap);
    assert_eq!(snap.joints.len(), 3, "shoulder, elbow, fingertip");

    assert!(
        dist(snap.joints[0], Point2::new(0.0, 0.0)) < 1e-6,
        "the shoulder is a fixed body at the world origin, got {:?}",
        snap.joints[0],
    );

    let upper = dist(snap.joints[0], snap.joints[1]);
    let fore = dist(snap.joints[1], snap.joints[2]);
    assert!(
        (upper - l1).abs() < l1 * 0.05,
        "upper arm should measure link1_length {l1}, got {upper}",
    );
    assert!(
        (fore - l2).abs() < l2 * 0.05,
        "forearm should measure link2_length {l2}, got {fore}",
    );

    assert_eq!(
        snap.contacts.len(),
        1,
        "the target disc must reach the report — a Reacher frame without it \
         shows an arm waving at nothing",
    );
}

/// Three equal rigid capsules chained end to end: every bone must measure one
/// segment length — **while the body is bent**, not only at reset.
///
/// The bend is the whole test. At reset every segment sits at θ ≈ 0, so its
/// quaternion is ≈ `[1, 0, 0, 0]` and *any* component read as the z-term
/// yields `atan2(0, 1) = 0`; the endpoints come out right by accident and the
/// reconstruction is never exercised. Verified: with only a reset-state
/// assertion, swapping `orientation[3]` for `orientation[1]` in the impl left
/// this test green. Driving the joints to real angles first is what gives it
/// teeth.
#[test]
fn swimmer_bones_match_segment_length_while_bent() {
    let config = SwimmerConfig::default();
    let seg_len = config.segment_length;
    let mut env = Swimmer::with_config(config).expect("valid");

    env.reset().expect("reset");
    // Opposing full-scale torques fold the body into a distinctly non-straight
    // pose. Fixed iteration count, not "until bent" — a broken reconstruction
    // must fail this test, never hang it.
    for _ in 0..40 {
        env.step(SwimmerAction([1.0, -1.0])).expect("step");
    }

    let snap = env.locomotion2d_snapshot();
    assert_shared_invariants("Swimmer", &snap);
    assert_eq!(snap.joints.len(), 4, "nose, two inner joints, tail");
    assert_eq!(snap.bones.len(), 3, "three segments");

    // Confirm the premise of this test before relying on it: if the swimmer is
    // still straight, the bone-length assertions below are as vacuous as they
    // were at reset.
    let straight = dist(snap.joints[0], snap.joints[3]);
    assert!(
        straight < seg_len * 2.9,
        "the swimmer never bent (nose-to-tail {straight} ≈ 3 × {seg_len}), so \
         the bone-length checks below prove nothing about the rotation \
         reconstruction",
    );

    for &(a, b) in &snap.bones {
        let d = dist(snap.joints[a as usize], snap.joints[b as usize]);
        assert!(
            (d - seg_len).abs() < seg_len * 0.05,
            "bone ({a}, {b}) measures {d}, expected segment_length {seg_len} \
             — the endpoint reconstruction reads each capsule's rotation about \
             world-z, so a wrong quaternion component shows up as a wrong \
             bone length here",
        );
    }
}
