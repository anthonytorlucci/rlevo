//! Positive half of the [`RecordableAction`] guard: every `Serialize` type
//! satisfies the bound with nothing to write by hand, which is the property
//! that makes a one-line derive the whole fix for a missing one.
//!
//! The negative half — that a non-`Serialize` action type is *rejected*, both
//! at a direct bound and through an associated type on a wrapper — lives as
//! `compile_fail` doctests on the trait itself in `src/record/action.rs`.
//! It has to: rustdoc collects doctests from library targets only, so a
//! `compile_fail` block written in this file would never be compiled and
//! would pass by never running.
#![cfg(feature = "record")]

use rlevo_benchmarks::record::RecordableAction;

fn assert_recordable<T: RecordableAction>() {}

#[test]
fn a_derived_action_type_satisfies_the_bound() {
    #[derive(serde::Serialize)]
    #[allow(dead_code)]
    enum Boiler {
        Off,
        On,
    }

    assert_recordable::<Boiler>();
}

/// The blanket impl covers the shapes real action types take — plain enums
/// above, and the primitive / vector payloads continuous actions carry.
#[test]
fn primitive_and_container_action_payloads_satisfy_the_bound() {
    assert_recordable::<u8>();
    assert_recordable::<i64>();
    assert_recordable::<f32>();
    assert_recordable::<Vec<f32>>();
    assert_recordable::<[f64; 3]>();
}
