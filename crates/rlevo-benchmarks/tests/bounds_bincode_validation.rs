//! `Bounds` validation holds under bincode, not just under self-describing
//! formats.
//!
//! ADR 0081 deferred one verification to landing: `Bounds` validates on
//! deserialization via `#[serde(try_from = "(f32, f32)", into = "(f32, f32)")]`,
//! which is well understood for JSON, and the record codec is **bincode**. If
//! validation did not run there, the plan was to validate at construction
//! instead. It does run, so the newtype carries its invariant onto the wire and
//! ADR 0082 can adopt `Bounds` for `SceneDescriptor` unchanged.
//!
//! This file is the executed answer, kept as a guard rather than deleted: the
//! property belongs to a serde attribute that a future edit could drop without
//! any other test noticing. It travels with `Bounds` into `rlevo-scene`.
//!
//! The control test is load-bearing. Two of these assert that decoding *fails*,
//! which malformed bytes would also produce — so one test proves the very same
//! bytes decode cleanly as a plain `(f32, f32)`. Without it, the suite would
//! stay green if `Bounds` stopped deserializing at all.

#![cfg(feature = "record")]

use rlevo_core::bounds::Bounds;

fn cfg() -> bincode::config::Configuration {
    bincode::config::standard()
}

#[test]
fn valid_bounds_round_trips_through_bincode() {
    let b = Bounds::new(-2.0, 6.0);
    let bytes = bincode::serde::encode_to_vec(b, cfg()).expect("encode");
    let (back, _): (Bounds, usize) =
        bincode::serde::decode_from_slice(&bytes, cfg()).expect("decode");
    assert_eq!(back.lo(), -2.0);
    assert_eq!(back.hi(), 6.0);
}

#[test]
fn bounds_serializes_as_a_bare_tuple() {
    // If `into = "(f32, f32)"` holds, the wire shape is identical to a tuple,
    // which is what makes the ADR 0082 migration wire-compatible.
    let b = Bounds::new(-2.0, 6.0);
    let as_bounds = bincode::serde::encode_to_vec(b, cfg()).expect("encode bounds");
    let as_tuple = bincode::serde::encode_to_vec((-2.0f32, 6.0f32), cfg()).expect("encode tuple");
    assert_eq!(
        as_bounds, as_tuple,
        "Bounds must occupy the same bytes as a bare (f32, f32)"
    );
}

#[test]
fn inverted_range_is_rejected_when_decoded_as_bounds() {
    // Encode a raw inverted tuple, then try to read it back as `Bounds`.
    // This is exactly the hostile-file case ADR 0027 exists to stop.
    let bytes = bincode::serde::encode_to_vec((6.0f32, -2.0f32), cfg()).expect("encode");
    let result: Result<(Bounds, usize), _> = bincode::serde::decode_from_slice(&bytes, cfg());
    assert!(
        result.is_err(),
        "an inverted (6.0, -2.0) decoded into a Bounds -- try_from validation \
         does NOT run under bincode, so ADR 0081 must validate at construction"
    );
}

#[test]
fn control_the_same_bytes_decode_fine_as_a_plain_tuple() {
    // Without this, the two rejection tests above could be passing because the
    // bytes are malformed rather than because validation ran.
    let bytes = bincode::serde::encode_to_vec((6.0f32, -2.0f32), cfg()).expect("encode");
    let (tuple, _): ((f32, f32), usize) =
        bincode::serde::decode_from_slice(&bytes, cfg()).expect("the bytes are a valid tuple");
    assert_eq!(tuple, (6.0, -2.0));
}

#[test]
fn nan_endpoint_is_rejected_when_decoded_as_bounds() {
    let bytes = bincode::serde::encode_to_vec((0.0f32, f32::NAN), cfg()).expect("encode");
    let result: Result<(Bounds, usize), _> = bincode::serde::decode_from_slice(&bytes, cfg());
    assert!(
        result.is_err(),
        "a NaN endpoint decoded into a Bounds -- try_from validation does NOT \
         run under bincode"
    );
}
