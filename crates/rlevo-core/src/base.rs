//! Core traits for reinforcement learning abstractions.
//!
//! This module defines the foundational vocabulary used throughout `rlevo-core`:
//! observations, actions, and the host row a tensor is built from. The conversion to a Burn tensor itself,
//! `TensorConvertible`, exists only under the `burn` feature. All other modules
//! depend on these primitives.

#[cfg(feature = "burn")]
use burn::tensor::{Device, Tensor, TensorData};
use std::fmt::Debug;

/// The `Observation` trait defines how an agent perceives the world. It
/// represents something that can be observed from the environment.
///
/// # Serde is not part of this contract
///
/// The contract is exactly `Debug + Clone + Send + Sync`: an observation must be
/// inspectable, copyable into a buffer, and movable across threads. Serialization
/// is *not* required. Nothing in the agent/environment loop — replay storage
/// included — persists an observation by way of this bound, so demanding
/// `Serialize + Deserialize` of every implementor taxed types that never cross a
/// wire and pushed hand-written impls onto types serde cannot derive for.
///
/// A consumer that genuinely needs to persist a domain type declares the bound at
/// **its own** seam, where the requirement actually arises. `RecordingTap`'s
/// `where E::ActionType: Serialize` is the reference shape (see
/// `rlevo_harness::record::env_tap`: the bound sits on `RecordingTap`'s
/// `Environment` impl and on the inherent impl that holds its private
/// `capture_frame`). Do not reinstate a universal serde bound here to save a
/// downstream `where` clause.
///
/// Concrete observation types are still expected to derive `Serialize` /
/// `Deserialize` (`docs/rules.md` §8) — this removes a universal *obligation*,
/// not the capability. `Action` is likewise deliberately serde-free.
/// (ADR 0064.)
pub trait Observation<const R: usize>: Debug + Clone + Send + Sync {
    /// Returns the size of each axis in this observation space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// size of that axis — the number of elements along it. All values must be
    /// greater than zero.
    ///
    /// For a type that also implements [`HostRow`], this is the shape of its
    /// row: [`HostRow::write_host_row`] pushes exactly
    /// `shape().iter().product()` values, and a tensor built from the row has
    /// this shape.
    fn shape() -> [usize; R];
}

/// Base trait for all action types in reinforcement learning environments.
///
/// This trait defines the minimal interface that all actions must implement, regardless
/// of their underlying representation (discrete, continuous, or hybrid). It ensures actions
/// are debuggable, clonable, and can validate themselves.
///
/// # Design Rationale
///
/// The `Action` trait is intentionally minimal and framework-agnostic:
/// - `Debug`: Required for logging and debugging agents
/// - `Clone`: Actions may be stored in replay buffers or used multiple times
/// - `Sized`: Enables efficient stack allocation and compile-time optimization
/// - `is_valid()`: Allows runtime validation of action constraints
///
/// # Implementing Action
///
/// When implementing this trait, ensure `is_valid()` checks all constraints:
/// - Range bounds for numeric values
/// - Finiteness for floating-point values
/// - Structural invariants (e.g., array dimensions)
/// - Environment-specific rules (e.g., available moves in a game state)
pub trait Action<const R: usize>: Debug + Clone + Sized {
    /// Returns the size of each axis in this action space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// cardinality of that axis — the number of possible values along it. All
    /// values must be greater than zero.
    ///
    /// # Not a component count
    ///
    /// `shape()` describes **axis cardinality only**. It is *never* a source of
    /// the flattened scalar component count of a continuous action: neither
    /// `shape().iter().product()` nor `R` is universally correct, because the
    /// workspace permits both a rank-1 action with `shape() == [C]` and a
    /// rank-`C` action with `shape() == [1; C]`. Use
    /// [`ContinuousAction::COMPONENTS`](crate::action::ContinuousAction::COMPONENTS),
    /// which is the single authority for that quantity. See ADR 0038 and
    /// ADR 0053 §8.
    fn shape() -> [usize; R];

    /// Validates whether this action satisfies all constraints.
    ///
    /// This method checks if the action is legal according to its type's invariants.
    /// It does **not** check environment-specific legality (e.g., whether a move
    /// is valid in the current game state)—that's the environment's responsibility.
    ///
    /// # Returns
    ///
    /// Returns `true` if the action satisfies all structural constraints, `false` otherwise.
    fn is_valid(&self) -> bool;
}

/// Host-side, backend-independent row serialization of an observation.
///
/// This is the layout half of tensor conversion: how an observation flattens
/// into a row-major `f32` row. The row's shape is the observation's own
/// [`Observation::shape`], so it is stated once. Neither depends on which Burn
/// backend the row is eventually uploaded to, so this trait carries **no** `B`
/// parameter and names nothing from Burn.
///
/// Only an observation writes a row: no learner converts an action or a state
/// to a tensor. Actions reach a network through
/// [`DiscreteAction::to_index`](crate::action::DiscreteAction::to_index) and
/// [`ContinuousAction::as_slice`](crate::action::ContinuousAction::as_slice).
///
/// Implementing it is all a type does for tensor conversion. With this crate's
/// `burn` feature on, `TensorConvertible` is implemented for every `HostRow`
/// type and uploads the row; without it, nothing here depends on Burn. A batch
/// is staged by calling [`write_host_row`] once per item into one buffer,
/// because staging never touches a device.
///
/// # Type Parameters
///
/// - `R`: Rank of a single row.
///
/// # Invariants
///
/// - A type implements `HostRow` at exactly **one** rank `R`, the rank of its
///   `Observation`. Implementing it at two ranks makes every unqualified
///   `write_host_row` or `shape` call on a concrete value of that type
///   ambiguous (E0283).
/// - `write_host_row` pushes exactly `shape().iter().product()` values.
///
/// [`write_host_row`]: HostRow::write_host_row
pub trait HostRow<const R: usize>: Observation<R> {
    /// Appends the row-major `f32` payload of `self` to `buf`.
    ///
    /// This is the primitive from which both single-item conversion
    /// (`TensorConvertible::to_tensor`) and whole-batch staging (one call per
    /// item into one buffer) are built, so the two can never disagree on
    /// element order.
    ///
    /// # Contract
    ///
    /// - Push **exactly** `shape().iter().product()` values, in **row-major**
    ///   order matching [`shape`].
    /// - Push **plain `f32`** — do *not* pre-convert to a backend element type.
    ///   Burn's `TensorData::new` performs the element-type conversion at upload
    ///   time.
    ///   This trait has no backend parameter, so there is nothing to convert to.
    /// - **Append**; never clear or truncate `buf`. Batch staging relies on
    ///   successive rows being concatenated into one contiguous buffer.
    ///
    /// [`shape`]: Observation::shape
    fn write_host_row(&self, buf: &mut Vec<f32>);

    /// Returns `true` if every value this type writes into a row is finite —
    /// i.e. the row contains no `NaN` and no `$\pm\infty$`.
    ///
    /// This is a statement about the **row**, not about the domain value: it
    /// asks whether the `f32` payload [`write_host_row`] produces is safe to
    /// upload. A caller uses it as an admission predicate at a data-ingestion
    /// boundary (ADR 0067): a replay buffer drops a transition whose
    /// observation row is non-finite, because a poisoned row cannot be
    /// detected downstream — on the CPU backend `relu` maps `NaN` to `0.0`, so
    /// the network yields a finite, in-domain, `is_valid()` action from an
    /// erased observation and no loss, Q-value, or action check ever fires.
    ///
    /// # Arguments
    ///
    /// - `scratch`: caller-owned staging buffer. The default body **clears** it,
    ///   writes one row into it, and leaves the row there. Reusing one buffer
    ///   across calls is the whole point of the parameter: without it the
    ///   default body would allocate on every env step (~110 KB for an
    ///   f32-backed image observation). Its contents afterwards are an
    ///   implementation detail — an override may not touch it at all.
    ///
    /// # Overriding
    ///
    /// Override this **only** for a type whose row is structurally incapable of
    /// being non-finite (an integer-backed payload: `u8 -> f32` is total). Such
    /// an override returns `true` without materializing the row, which is both
    /// an assertion and a load-bearing performance decision — the default body
    /// would convert 27 648 bytes to `f32` every env step for a `$96 \times 96 \times 3$` frame.
    ///
    /// An override **must** carry a compile-time witness for its structural
    /// claim — a concrete type ascription against the payload field, e.g.
    /// `let _: &[u8] = &self.pixels;` — so that changing the payload type
    /// breaks the build rather than the guarantee. The witness must not be
    /// spelled via `f32::from` or an `Into<f32>` bound: `impl<T> From<T> for T`
    /// means `f32: Into<f32>`, so those keep compiling after an `f32` refactor
    /// and guarantee nothing.
    ///
    /// [`write_host_row`]: HostRow::write_host_row
    #[must_use]
    fn row_is_finite(&self, scratch: &mut Vec<f32>) -> bool {
        scratch.clear();
        self.write_host_row(scratch);
        row_slice_is_finite(scratch)
    }
}

/// Branchless finiteness test over a staged `f32` row: `true` if no element is
/// `NaN` or `$\pm\infty$`.
///
/// # Why this spelling, and why not `iter().all(f32::is_finite)`
///
/// **Do not "simplify" this to `buf.iter().all(|v| v.is_finite())`.** That
/// rewrite is the single most likely future regression here, and it is a real
/// one: measured at ~9 GB/s against a ~62 GB/s memcpy on an M2 Pro — `$8\times$` the
/// cost of the write it rides — because [`Iterator::all`] must short-circuit
/// and therefore cannot lower to a horizontal SIMD reduction. It is also
/// *data-dependent*: how long it runs depends on where the first non-finite
/// element sits, which confounds any benchmark control. (ADR 0067 §Decision 1.)
///
/// # Derivation
///
/// 1. An IEEE-754 `binary32` value is non-finite (`NaN` or `$\pm\infty$`) **iff** its
///    8-bit exponent field is all ones.
/// 2. `v.to_bits() & 0x7F80_0000` masks each element down to exactly that
///    field, clearing the sign bit and the mantissa. Clearing the **sign** is
///    load-bearing: `-NaN` and `-Inf` must reduce identically to their positive
///    counterparts.
/// 3. `0x7F80_0000` is the **maximum** value the masked field can take, so a
///    `u32::max` reduction over every element equals `0x7F80_0000` **iff** at
///    least one element is non-finite.
/// 4. `max` is associative and commutative and never short-circuits, so LLVM
///    lowers the fold to a horizontal vector reduction, restoring ~`$1\times$` fusion
///    with the write that produced the buffer.
///
/// An empty row has no non-finite element, and the fold's `0` identity yields
/// `true` — which is the correct answer.
/// Deliberately **not** `pub`: `rlevo-core` is a contract crate (rules §1), and
/// this is the body of a provided trait method, not a new public surface.
#[must_use]
fn row_slice_is_finite(buf: &[f32]) -> bool {
    /// IEEE-754 `binary32` exponent-field mask: sign and mantissa cleared.
    const EXPONENT_MASK: u32 = 0x7F80_0000;

    buf.iter()
        .fold(0u32, |acc, &v| acc.max(v.to_bits() & EXPONENT_MASK))
        != EXPONENT_MASK
}

/// Conversion from a domain type to a Burn tensor, behind the `burn` feature.
///
/// The backend-independent half of the conversion — the row layout — lives on
/// the [`HostRow`] supertrait; this trait adds only the device-facing half.
/// It is implemented for every [`HostRow`] type by a blanket impl, so no type
/// writes it, or overrides [`to_tensor`](Self::to_tensor), by hand.
///
/// Conversion runs one way: nothing in the workspace turns a tensor back into
/// a domain value. The decoders in use read a host value directly
/// (`DiscreteAction::from_index`, `ContinuousAction::from_slice`).
///
/// # Type Parameters
///
/// - `R`: Rank of the tensor produced.
#[cfg(feature = "burn")]
pub trait TensorConvertible<const R: usize>: HostRow<R> {
    /// Converts `self` into a tensor on `device`.
    ///
    /// Stages one row into a host `Vec<f32>` with [`write_host_row`] and uploads
    /// it with a single [`Tensor::from_data`], in the shape [`shape`] reports.
    /// A batch staged with the same row writer therefore has the same element
    /// order.
    ///
    /// [`shape`]: Observation::shape
    /// [`write_host_row`]: HostRow::write_host_row
    fn to_tensor(&self, device: &Device) -> Tensor<R> {
        let row: [usize; R] = Self::shape();
        let mut buf: Vec<f32> = Vec::with_capacity(row.iter().product());
        self.write_host_row(&mut buf);
        debug_assert_eq!(buf.len(), row.iter().product::<usize>());
        Tensor::from_data(TensorData::new(buf, row), device)
    }
}

#[cfg(feature = "burn")]
impl<const R: usize, T: HostRow<R>> TensorConvertible<R> for T {}

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    // The `as f32` casts below are on small grid coordinates, well inside the
    // range f32 represents exactly.
    #![allow(clippy::float_cmp, clippy::cast_precision_loss)]
    use super::*;

    /// A rank-2 observation: a 2x2 grid of bits.
    ///
    /// **The absence of `Serialize` / `Deserialize` here is load-bearing.** This
    /// mock is the compile-time guard that [`Observation`] carries no serde
    /// supertrait (ADR 0064): its supertrait list is exactly
    /// `Debug + Clone + Send + Sync`, and if anyone re-adds
    /// `Serialize + for<'de> Deserialize<'de>` to it, `rlevo-core`'s own test
    /// build stops compiling on the `impl Observation<2>` below. Do not
    /// "helpfully" derive serde for this type — a consumer that needs to persist
    /// an observation declares the bound at its own seam.
    #[derive(Debug, Clone, PartialEq)]
    struct SerdeFreeObservation {
        pixels: [[u8; 2]; 2],
    }

    impl Observation<2> for SerdeFreeObservation {
        fn shape() -> [usize; 2] {
            [2, 2]
        }
    }

    #[test]
    fn test_observation_needs_no_serde() {
        let obs = SerdeFreeObservation {
            pixels: [[1, 0], [0, 1]],
        };
        assert_eq!(SerdeFreeObservation::shape(), [2, 2]);
        assert_eq!(obs.clone(), obs);
    }

    // ========================================================================
    // TensorConvertible: the derived `to_tensor`, through the blanket impl
    // ========================================================================

    #[cfg(feature = "burn")]
    mod to_tensor {
        use super::*;

        /// Rank-1 test row: three scalar features, shape `[3]`.
        #[derive(Clone, Debug)]
        struct Vec3(f32, f32, f32);

        impl Observation<1> for Vec3 {
            fn shape() -> [usize; 1] {
                [3]
            }
        }

        impl HostRow<1> for Vec3 {
            fn write_host_row(&self, buf: &mut Vec<f32>) {
                buf.extend_from_slice(&[self.0, self.1, self.2]);
            }
        }

        /// Rank-3 test row: a `[2, 2, 1]` image-like payload.
        #[derive(Clone, Debug)]
        struct Img([f32; 4]);

        impl Observation<3> for Img {
            fn shape() -> [usize; 3] {
                [2, 2, 1]
            }
        }

        impl HostRow<3> for Img {
            fn write_host_row(&self, buf: &mut Vec<f32>) {
                buf.extend_from_slice(&self.0);
            }
        }

        /// The derived `to_tensor` produces the same data/shape as the old manual
        /// `Tensor::from_floats` path for a rank-1 row.
        #[test]
        fn test_derived_to_tensor_rank1_matches_manual() {
            let device: Device = Device::flex();
            let item: Vec3 = Vec3(1.5, -2.5, 3.5);

            let derived: Tensor<1> = <Vec3 as TensorConvertible<1>>::to_tensor(&item, &device);
            let manual: Tensor<1> = Tensor::from_floats([1.5_f32, -2.5, 3.5], &device);

            assert_eq!(derived.dims(), manual.dims());
            let derived_v: Vec<f32> = derived
                .into_data()
                .try_into_vec::<f32>()
                .expect("f32 host read of a tensor this test just built");
            let manual_v: Vec<f32> = manual
                .into_data()
                .try_into_vec::<f32>()
                .expect("f32 host read of a tensor this test just built");
            assert_eq!(derived_v, manual_v);
        }

        /// The derived `to_tensor` produces the same data/shape as the old manual
        /// `TensorData::new` path for a rank-3 row.
        #[test]
        fn test_derived_to_tensor_rank3_matches_manual() {
            let device: Device = Device::flex();
            let item: Img = Img([0.1, 0.2, 0.3, 0.4]);

            let derived: Tensor<3> = <Img as TensorConvertible<3>>::to_tensor(&item, &device);
            let manual: Tensor<3> = Tensor::from_data(
                TensorData::new(vec![0.1_f32, 0.2, 0.3, 0.4], [2, 2, 1]),
                &device,
            );

            assert_eq!(derived.dims(), manual.dims());
            let derived_v: Vec<f32> = derived
                .into_data()
                .try_into_vec::<f32>()
                .expect("f32 host read of a tensor this test just built");
            let manual_v: Vec<f32> = manual
                .into_data()
                .try_into_vec::<f32>()
                .expect("f32 host read of a tensor this test just built");
            assert_eq!(derived_v, manual_v);
        }
    }

    // ===== `row_is_finite` / `row_slice_is_finite` =====

    /// The reference implementation the branchless reduction must agree with.
    ///
    /// This is deliberately the spelling ADR 0067 §Decision 1 forbids in
    /// production (short-circuiting, ~`$8\times$` the cost, data-dependent): here it is
    /// the *oracle*, and having it in the test file is what makes the
    /// production reduction checkable rather than merely asserted.
    fn oracle_is_finite(buf: &[f32]) -> bool {
        buf.iter().all(|v| v.is_finite())
    }

    /// Every f32 shape that has ever been a footgun in this predicate.
    ///
    /// `-NAN` and `f32::from_bits(0xFFC0_0000)` are the load-bearing entries:
    /// a mask that failed to clear the sign bit would answer them wrong.
    fn finiteness_corpus() -> Vec<f32> {
        vec![
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0xFFC0_0000), // an explicitly negative quiet NaN
            f32::from_bits(0x7FC0_0001), // a positive NaN with a set mantissa bit
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::from_bits(1), // smallest positive subnormal
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1.0,
            -1.0,
            1e-30,
            1e30,
        ]
    }

    /// The branchless exponent-field reduction must agree with
    /// `iter().all(is_finite)` on every element of the corpus, at every
    /// position of a row, and on the empty row.
    #[test]
    fn test_row_slice_is_finite_matches_all_is_finite_oracle() {
        assert_eq!(
            row_slice_is_finite(&[]),
            oracle_is_finite(&[]),
            "an empty row has no non-finite element, so it must read as finite"
        );

        let clean = [1.0_f32, -2.5, 0.0, 3.75, -0.125];
        assert!(
            row_slice_is_finite(&clean),
            "a clean row must be accepted: {clean:?}"
        );

        for poison in finiteness_corpus() {
            // Single-element row: the degenerate case where the poisoned
            // element is simultaneously first, last and only.
            let single = [poison];
            assert_eq!(
                row_slice_is_finite(&single),
                oracle_is_finite(&single),
                "single-element row disagreed with the oracle for {poison} \
                 (bits {:#010x})",
                poison.to_bits()
            );

            // First, middle and last position of a row that is otherwise
            // clean. `all` short-circuits at the first bad element and the
            // reduction does not, so all three positions must be covered.
            for position in [0, clean.len() / 2, clean.len() - 1] {
                let mut row = clean;
                row[position] = poison;
                assert_eq!(
                    row_slice_is_finite(&row),
                    oracle_is_finite(&row),
                    "row poisoned at index {position} with {poison} (bits \
                     {:#010x}) disagreed with the oracle: {row:?}",
                    poison.to_bits()
                );
            }
        }
    }

    /// A negative NaN must be rejected. Called out on its own because it is the
    /// exact failure a mask that forgets to clear the sign bit produces, and
    /// that failure is invisible in a corpus of positive NaNs.
    #[test]
    fn test_row_slice_is_finite_rejects_negative_nan() {
        for bad in [
            -f32::NAN,
            f32::from_bits(0xFFC0_0000),
            f32::NEG_INFINITY,
            f32::from_bits(0xFF80_0000), // -Inf by bit pattern
        ] {
            assert!(
                bad.is_sign_negative(),
                "this test only means anything for sign-negative values; \
                 {bad} is not one"
            );
            assert!(
                !row_slice_is_finite(&[1.0, bad, 2.0]),
                "a sign-negative non-finite value ({bad}, bits {:#010x}) must \
                 be rejected — the exponent mask has to clear the sign bit",
                bad.to_bits()
            );
        }
    }

    /// A row of every corpus value at once is non-finite, and the same row with
    /// the non-finite entries removed is finite.
    #[test]
    fn test_row_slice_is_finite_on_mixed_rows() {
        let all = finiteness_corpus();
        assert_eq!(
            row_slice_is_finite(&all),
            oracle_is_finite(&all),
            "the full corpus as one row disagreed with the oracle"
        );
        assert!(
            !row_slice_is_finite(&all),
            "the corpus contains NaN and ±Inf, so it must not read as finite"
        );

        let finite_only: Vec<f32> = all.into_iter().filter(|v| v.is_finite()).collect();
        assert!(
            row_slice_is_finite(&finite_only),
            "the corpus with every non-finite entry removed must read as finite"
        );
    }

    /// Fixed-width row used to exercise the *provided* `row_is_finite` body
    /// (staging via `write_host_row`), as opposed to the raw slice predicate.
    #[derive(Debug, Clone)]
    struct TestFiniteRow([f32; 4]);

    impl Observation<1> for TestFiniteRow {
        fn shape() -> [usize; 1] {
            [4]
        }
    }

    impl HostRow<1> for TestFiniteRow {
        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&self.0);
        }
    }

    /// The default body must agree with the oracle applied to the row that
    /// `write_host_row` actually produces.
    #[test]
    fn test_host_row_row_is_finite_default_body() {
        let mut scratch = Vec::new();

        let clean = TestFiniteRow([1.0, -2.0, 0.0, 4.5]);
        assert!(
            clean.row_is_finite(&mut scratch),
            "a clean row must be admitted by the default body"
        );

        for poison in finiteness_corpus() {
            for position in 0..4 {
                let mut values = [1.0_f32, -2.0, 0.0, 4.5];
                values[position] = poison;
                let row = TestFiniteRow(values);
                assert_eq!(
                    row.row_is_finite(&mut scratch),
                    oracle_is_finite(&values),
                    "the default body disagreed with the oracle for {poison} \
                     at index {position}"
                );
            }
        }
    }

    /// The default body must `clear` the scratch buffer first: a leftover
    /// `NaN` from a previous, unrelated staging call must not condemn a clean
    /// row (nor a leftover clean row rescue a poisoned one).
    #[test]
    fn test_host_row_row_is_finite_clears_scratch() {
        let mut scratch = vec![f32::NAN, f32::INFINITY, -1.0];
        let clean = TestFiniteRow([1.0, -2.0, 0.0, 4.5]);
        assert!(
            clean.row_is_finite(&mut scratch),
            "stale non-finite values left in `scratch` must be cleared, not \
             counted against the row under test"
        );
        assert_eq!(
            scratch.len(),
            4,
            "after the call `scratch` holds exactly the staged row"
        );
        assert_eq!(
            scratch,
            vec![1.0_f32, -2.0, 0.0, 4.5],
            "the staged row must be the one `write_host_row` produces"
        );

        let poisoned = TestFiniteRow([1.0, f32::NAN, 0.0, 4.5]);
        assert!(
            !poisoned.row_is_finite(&mut scratch),
            "a clean buffer left over from the previous call must not rescue a \
             poisoned row"
        );
    }
}
