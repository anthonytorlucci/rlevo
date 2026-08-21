//! Core traits for reinforcement learning abstractions.
//!
//! This module defines the foundational vocabulary used throughout `rlevo-core`:
//! rewards, observations, states, actions, transition dynamics, and tensor
//! conversion. All other modules depend on these primitives.

use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use std::fmt::Debug;

/// Generic update function: how something evolves over time.
///
/// Parameterized over the input stimulus and the output type it transforms.
pub trait UpdateFunction<Input, Output> {
    /// Computes the next value given the current value and an input.
    fn update(&self, current: &Output, input: &Input) -> Output;
}

/// A scalar reward signal emitted by an environment each step.
///
/// # Finiteness
///
/// Implementors **should** yield finite values. `Reward: Into<f32>` erases every
/// reward to an `f32` at the framework's ingestion points, so a `NaN` or an
/// infinity arriving there is an environment bug — not user input that some
/// layer is expected to sanitise.
///
/// `rlevo-core` enforces none of this. There is no `is_finite` check on this
/// trait, no bound that could express one, and no validation in
/// [`ScalarReward::new`](crate::reward::ScalarReward::new) (ADR 0065 closed that
/// question). What follows is what the framework *actually does* when the
/// obligation is broken, and it is three different things at three places:
///
/// - **Dropped at replay ingestion.** The reinforcement-learning crate's reward
///   guard (`FiniteRewardGuard`) refuses to store a transition whose reward is
///   non-finite, counts every such drop, and logs a warning on an escalating
///   schedule. The transition never reaches the buffer, so learning is not
///   poisoned. (ADR 0065.)
/// - **Transited, deliberately, by the reported score.** Episode return
///   accumulates the raw value, so an affected episode reports a non-finite
///   return and `AgentStats::avg_score` transits it rather than sanitising it.
///   This is intentional: a non-finite return is a true statement about a run
///   whose environment emitted a non-finite reward, and hiding it would make the
///   bug harder to find, not rarer. (ADR 0065 and ADR 0070.)
/// - **Counted and reported by the benchmarks harness.** The evaluator
///   accumulates raw (matching the above), logs a warning, and records a
///   `return/non_finite_steps` counter on the trial report. (ADR 0078.)
///
/// ## Both signs corrupt a report, and neither announces itself
///
/// This is not a `NaN`-only hazard. For trial returns `[1, 5, X, 20, 100]` with a
/// success threshold of `10`, the harness's own aggregates come out as:
///
/// | `X`     | mean | median | std   | min | max   | success rate |
/// | ------- | ---- | ------ | ----- | --- | ----- | ------------ |
/// | finite  | 27.0 | 9.0    | 41.42 | 1   | 100   | 0.40         |
/// | `NaN`   | NaN  | 20.0   | NaN   | 1   | 100   | 0.40         |
/// | `+inf`  | inf  | 20.0   | NaN   | 1   | inf   | 0.60         |
///
/// `NaN` is scored as a *failure* the agent may not have earned; `+inf` is scored
/// as a *success* it certainly did not, and poisons `max` besides. In both cases
/// the median silently shifts (9.0 to 20.0) while min stays finite-looking, so a
/// report can present a poisoned aggregate next to plausible order statistics
/// with no single tell. Treat a non-finite reward as a defect to fix upstream in
/// the environment, not as a condition to detect downstream.
///
/// The enforcement that does exist lives at the points where the reward has
/// already been erased to `f32` — by then the type that produced it is gone, and
/// with it any chance of a useful diagnostic. This section is the obligation;
/// meeting it is the implementor's job.
pub trait Reward: Clone + std::ops::Add<Output = Self> + Into<f32> + Debug {
    /// Returns the additive identity for this reward type (typically `0.0`).
    fn zero() -> Self;
}

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
/// `crates/rlevo-benchmarks/src/record/env_tap.rs`, lines 300 and 322). Do not
/// reinstate a universal serde bound here to save a downstream `where` clause.
///
/// Concrete observation types are still expected to derive `Serialize` /
/// `Deserialize` (`docs/rules.md` §8) — this removes a universal *obligation*,
/// not the capability. `Action` and `Reward` are likewise deliberately
/// serde-free. (ADR 0064.)
pub trait Observation<const R: usize>: Debug + Clone + Send + Sync {
    /// The rank of this observation space — i.e. the number of axes (tensor
    /// order), *not* the size of any axis.
    ///
    /// "Rank" here means the count of indices needed to address an element
    /// (`NumPy` `ndim`, Burn's `Tensor<B, R>`), not matrix rank or CP-decomposition
    /// rank. This is automatically set to match the const generic parameter `R`.
    const RANK: usize = R;

    /// Returns the size of each axis in this observation space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// cardinality of that axis — the number of possible values along it. All
    /// values must be greater than zero.
    fn shape() -> [usize; R];
}

/// The complete state of an environment (Markov property).
///
/// # Observation production has moved off `State`
///
/// A `State` no longer produces its own observation. In the POMDP tuple
/// ⟨S, A, T, R, Ω, O⟩ the emission model `O` is a property of the *environment*,
/// not of a state value, so it lives on the env-side
/// [`Sensor`](crate::environment::Sensor) trait with the canonical signature
/// `O(a, s')`. `State` retains only what genuinely belongs to a point in state
/// space: its rank, [`shape`](State::shape), [`numel`](State::numel), and
/// [`is_valid`](State::is_valid). (ADR 0047, superseding ADR 0019.)
pub trait State<const R: usize>: Debug + Clone + Send + Sync {
    /// The rank of this state space — i.e. the number of axes (tensor order),
    /// *not* the size of any axis.
    ///
    /// "Rank" here means the count of indices needed to address an element
    /// (`NumPy` `ndim`, Burn's `Tensor<B, R>`), not matrix rank or CP-decomposition
    /// rank. This is automatically set to match the const generic parameter `R`.
    const RANK: usize = R;

    /// Returns the size of each axis in this state space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// cardinality of that axis — the number of possible values along it. All
    /// values must be greater than zero.
    fn shape() -> [usize; R];

    /// Validates whether this state satisfies all constraints.
    ///
    /// This method checks if the state is legal according to its type's invariants.
    /// It does **not** check environment-specific legality - that's the environment's responsibility.
    ///
    /// # Returns
    ///
    /// Returns `true` if the state satisfies all structural constraints, `false` otherwise.
    fn is_valid(&self) -> bool;

    /// Returns the total number of scalar elements in this state's representation.
    ///
    /// This value is critical for:
    /// - Allocating buffers for state serialization
    /// - Determining neural network input layer dimensions
    /// - Validating state transformations (e.g., flattening/unflattening)
    ///
    /// # Relationship to Shape
    ///
    /// For consistency, `numel()` must equal the product of all dimensions returned by
    /// [`shape()`](State::shape). The default implementation enforces this by computing
    /// the product directly. Override only if the state uses a non-product layout.
    ///
    /// # Returns
    ///
    /// The total number of scalar elements needed to represent this state.
    fn numel(&self) -> usize {
        Self::shape().iter().product()
    }
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
    /// The rank of this action space — i.e. the number of axes (tensor order),
    /// *not* the size of any axis.
    ///
    /// "Rank" here means the count of indices needed to address an element
    /// (`NumPy` `ndim`, Burn's `Tensor<B, R>`), not matrix rank or CP-decomposition
    /// rank. This is automatically set to match the const generic parameter `R`.
    const RANK: usize = R;

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

/// Deterministic environment transition dynamics.
///
/// ```math
/// s_{t+1} = f(s_t, a_t)
/// ```
///
/// This trait covers only **deterministic** transitions. Stochastic dynamics
/// (where the successor state is drawn from a distribution) are not modeled
/// here; environments with stochastic transitions implement that logic internally
/// inside [`crate::environment::Environment::step`].
pub trait TransitionDynamics<const SR: usize, const AR: usize, S: State<SR>, A: Action<AR>> {
    /// Returns the successor state after applying `action` to `state`.
    fn transition(&self, state: &S, action: &A) -> S;
}

/// Error returned when a tensor cannot be converted to or from a domain type.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Invalid tensor conversion: {message}")]
pub struct TensorConversionError {
    /// Human-readable description of why the conversion failed.
    pub message: String,
}

/// Host-side, backend-independent row serialization.
///
/// This is the layout half of tensor conversion: how a value flattens into a
/// row-major `f32` row, and what shape that row has. Neither answer depends on
/// which Burn backend the row is eventually uploaded to, so this trait carries
/// **no** `B` parameter — a fact the pre-split [`TensorConvertible`] could only
/// state in prose.
///
/// Implement this alongside [`TensorConvertible`], which requires it as a
/// supertrait and derives [`TensorConvertible::to_tensor`] from it. The
/// whole-batch staging helper [`stack_to_tensor`] needs *only* this trait,
/// because staging never touches a device.
///
/// # Type Parameters
///
/// - `R`: Rank of a single row.
///
/// # Invariants
///
/// - A type implements `HostRow` at exactly **one** rank `R`. Implementing it at
///   two ranks makes every unqualified `write_host_row` / `row_shape` call on a
///   concrete value of that type ambiguous (E0283).
/// - `write_host_row` pushes exactly `row_shape().iter().product()` values.
///
/// [`stack_to_tensor`]: crate::base::stack_to_tensor
pub trait HostRow<const R: usize> {
    /// Returns the per-item ("row") shape of the tensor this type serializes to.
    ///
    /// This is the shape of a **single** value — rank `R`, with each axis size
    /// fixed by the domain type (e.g. `[8]` for an 8-feature observation, or
    /// `[H, W, C]` for an image). It is the layout that [`write_host_row`] must
    /// fill, and the shape [`to_tensor`] wraps around the written buffer.
    ///
    /// The product of the returned axes is the number of `f32` scalars a single
    /// row occupies, which is exactly how many values [`write_host_row`] must
    /// push.
    ///
    /// [`write_host_row`]: HostRow::write_host_row
    /// [`to_tensor`]: TensorConvertible::to_tensor
    fn row_shape() -> [usize; R];

    /// Appends the row-major `f32` payload of `self` to `buf`.
    ///
    /// This is the primitive from which both single-item conversion
    /// ([`to_tensor`]) and whole-batch staging ([`stack_to_tensor`]) are
    /// derived, guaranteeing the two can never disagree on element order.
    ///
    /// # Contract
    ///
    /// - Push **exactly** `row_shape().iter().product()` values, in **row-major**
    ///   order matching [`row_shape`].
    /// - Push **plain `f32`** — do *not* pre-convert to a backend element type.
    ///   [`TensorData::new`] performs the element-type conversion at upload time.
    ///   This trait has no backend parameter, so there is nothing to convert to.
    /// - **Append**; never clear or truncate `buf`. Batch staging relies on
    ///   successive rows being concatenated into one contiguous buffer.
    ///
    /// [`row_shape`]: HostRow::row_shape
    /// [`to_tensor`]: TensorConvertible::to_tensor
    /// [`stack_to_tensor`]: crate::base::stack_to_tensor
    fn write_host_row(&self, buf: &mut Vec<f32>);

    /// Returns `true` if every value this type writes into a row is finite —
    /// i.e. the row contains no `NaN` and no `±Inf`.
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
    /// would convert 27 648 bytes to `f32` every env step for a 96×96×3 frame.
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
/// `NaN` or `±Inf`.
///
/// # Why this spelling, and why not `iter().all(f32::is_finite)`
///
/// **Do not "simplify" this to `buf.iter().all(|v| v.is_finite())`.** That
/// rewrite is the single most likely future regression here, and it is a real
/// one: measured at ~9 GB/s against a ~62 GB/s memcpy on an M2 Pro — 8× the
/// cost of the write it rides — because [`Iterator::all`] must short-circuit
/// and therefore cannot lower to a horizontal SIMD reduction. It is also
/// *data-dependent*: how long it runs depends on where the first non-finite
/// element sits, which confounds any benchmark control. (ADR 0067 §Decision 1.)
///
/// # Derivation
///
/// 1. An IEEE-754 `binary32` value is non-finite (`NaN` or `±Inf`) **iff** its
///    8-bit exponent field is all ones.
/// 2. `v.to_bits() & 0x7F80_0000` masks each element down to exactly that
///    field, clearing the sign bit and the mantissa. Clearing the **sign** is
///    load-bearing: `-NaN` and `-Inf` must reduce identically to their positive
///    counterparts.
/// 3. `0x7F80_0000` is the **maximum** value the masked field can take, so a
///    `u32::max` reduction over every element equals `0x7F80_0000` **iff** at
///    least one element is non-finite.
/// 4. `max` is associative and commutative and never short-circuits, so LLVM
///    lowers the fold to a horizontal vector reduction, restoring ~1× fusion
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

/// Bidirectional conversion between a domain type and a Burn tensor.
///
/// The backend-independent half of the conversion — the row layout — lives on
/// the [`HostRow`] supertrait; this trait adds only the device-facing half.
///
/// # Type Parameters
///
/// - `R`: Rank of the tensor produced.
/// - `B`: Burn backend.
///
/// # Invariants
///
/// Two clauses, both required of every implementor, for any valid `x` and any
/// device `d`:
///
/// 1. **Tensor-image fidelity.** `from_tensor(x.to_tensor(d))?.to_tensor(d)`
///    equals `x.to_tensor(d)` — decoding a tensor and re-encoding the result is
///    a no-op *on the tensor*. Everything the tensor carries survives a decode
///    unchanged, so no consumer that only ever sees tensors can observe a
///    difference.
/// 2. **No fabrication.** Any field [`write_host_row`] does not write must
///    decode to a value that *represents absence* — `Option::None`, or a
///    dedicated type whose decoded variant says "unknown". It must **never**
///    decode to a plausible in-domain value. A default that looks like real
///    data is the failure this clause exists to forbid: it is
///    indistinguishable from a measurement downstream.
///
/// A type whose `write_host_row` covers **every** field satisfies clause 1 in
/// the stronger form `from_tensor(x.to_tensor(d)) == Ok(x)` — a total round
/// trip — and satisfies clause 2 vacuously. That is the expected case and what
/// nearly every implementor in this workspace does; clause 1 is a **floor**
/// that keeps a partial encoding honest, not permission to be partial. Omit a
/// field from the row only when the domain says the tensor should not carry it,
/// and document the omission on both the field and `from_tensor`.
///
/// [`write_host_row`]: HostRow::write_host_row
///
/// # Errors
///
/// `from_tensor` returns [`TensorConversionError`] when the tensor's shape,
/// dtype, or contents violate the domain type's invariants (see
/// [`State::is_valid`] / [`Action::is_valid`]).
pub trait TensorConvertible<const R: usize, B: Backend>: HostRow<R> + Sized {
    /// Converts `self` into a tensor on `device`.
    ///
    /// # Do not override
    ///
    /// This method has a default body derived from [`row_shape`] and
    /// [`write_host_row`]: it stages one row into a host `Vec<f32>` and uploads
    /// it with a single [`Tensor::from_data`]. Implementors **must not** provide
    /// their own `to_tensor` — doing so would let the single-item layout drift
    /// from the batched layout produced by [`stack_to_tensor`], defeating the
    /// whole point of the shared row-writer primitive.
    ///
    /// [`row_shape`]: HostRow::row_shape
    /// [`write_host_row`]: HostRow::write_host_row
    /// [`stack_to_tensor`]: crate::base::stack_to_tensor
    fn to_tensor(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, R> {
        let row: [usize; R] = Self::row_shape();
        let mut buf: Vec<f32> = Vec::with_capacity(row.iter().product());
        self.write_host_row(&mut buf);
        debug_assert_eq!(buf.len(), row.iter().product::<usize>());
        Tensor::from_data(TensorData::new(buf, row), device)
    }

    /// Reconstructs a value from a tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor's shape or contents
    /// do not describe a valid instance of `Self`.
    fn from_tensor(tensor: Tensor<B, R>) -> Result<Self, TensorConversionError>;
}

/// Stages a whole batch of rows into one host buffer and uploads it as a single
/// tensor.
///
/// Each item's [`write_host_row`] payload is concatenated into one contiguous
/// `Vec<f32>`, which is then uploaded with a **single** [`Tensor::from_data`]
/// call. This is materially cheaper than converting each item to its own tensor
/// and calling [`Tensor::stack`], which incurs one host→device upload *per item*
/// plus a concatenation kernel. Because both this function and the derived
/// [`TensorConvertible::to_tensor`] draw from the same
/// [`write_host_row`]/[`row_shape`] primitives, the batched layout is guaranteed
/// to match `stack`-ing the individual rows.
///
/// The produced tensor has rank `BR = R + 1` and shape `[items.len(), ..row]`,
/// i.e. a leading batch axis followed by the per-item [`row_shape`].
///
/// # Type Parameters
///
/// - `R`: rank of a single row.
/// - `BR`: rank of the batched tensor; must equal `R + 1`.
/// - `T`: the row type, [`HostRow<R>`]. Note the bound is [`HostRow`], not
///   [`TensorConvertible`]: staging is host-only, so the row type need not name
///   a backend at all.
/// - `B`: Burn backend the assembled buffer is uploaded to.
///
/// # The `BR = R + 1` contract
///
/// Stable Rust cannot express `R + 1` in a const-generic position, so `BR` is a
/// separate parameter checked at runtime. This function is the **single
/// chokepoint** for that invariant: the leading `assert_eq!` runs before the
/// shape array is assembled, which is what makes the subsequent
/// `shape[1..].copy_from_slice(&row)` sound (it would panic on a length
/// mismatch otherwise).
///
/// # Note
///
/// That this helper has no in-tree call sites is deliberate, not neglect. The
/// six off-policy agents (`dqn`, `c51`, `qrdqn`, `ddpg`, `td3`, `sac`) fill five
/// buffers — observations, next-observations, actions, rewards, terminated
/// flags — in a *single* pass over the sampled ids; routing observations through
/// this function would first require materializing an intermediate `Vec` of
/// those rows, splitting that pass in two. They call [`write_host_row`] directly
/// instead — the same primitive used here — and keep the one batched upload.
/// This remains the recommended primitive for the common case: batching a
/// homogeneous slice of [`HostRow`] rows into a single upload.
///
/// # Panics
///
/// Panics if `BR != R + 1`.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::Tensor;
/// use rlevo_core::base::{stack_to_tensor, HostRow, TensorConversionError, TensorConvertible};
///
/// #[derive(Clone)]
/// struct Point {
///     x: f32,
///     y: f32,
/// }
///
/// impl HostRow<1> for Point {
///     fn row_shape() -> [usize; 1] {
///         [2]
///     }
///     fn write_host_row(&self, buf: &mut Vec<f32>) {
///         buf.push(self.x);
///         buf.push(self.y);
///     }
/// }
///
/// impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for Point {
///     fn from_tensor(_tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
///         unimplemented!()
///     }
/// }
///
/// type B = Flex;
/// let device = Default::default();
/// let items: Vec<Point> = vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }];
/// let batched: Tensor<B, 2> = stack_to_tensor::<1, 2, Point, B>(&items, &device);
/// assert_eq!(batched.dims(), [2, 2]);
/// ```
///
/// [`write_host_row`]: HostRow::write_host_row
/// [`row_shape`]: HostRow::row_shape
pub fn stack_to_tensor<const R: usize, const BR: usize, T, B>(
    items: &[T],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, BR>
where
    T: HostRow<R>,
    B: Backend,
{
    assert_eq!(BR, R + 1, "batched rank BR must equal row rank R + 1");
    let row: [usize; R] = T::row_shape();
    let row_len: usize = row.iter().product();
    let mut buf: Vec<f32> = Vec::with_capacity(items.len() * row_len);
    for item in items {
        item.write_host_row(&mut buf);
    }
    debug_assert_eq!(buf.len(), items.len() * row_len);
    let mut shape: [usize; BR] = [0usize; BR];
    shape[0] = items.len();
    shape[1..].copy_from_slice(&row); // sound only because the BR == R + 1 assert above ran first — keep the ordering
    Tensor::from_data(TensorData::new(buf, shape), device)
}

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    // The `as f32` casts below are on small grid coordinates, well inside the
    // range f32 represents exactly.
    #![allow(clippy::float_cmp, clippy::cast_precision_loss)]
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Simple scalar reward implementation for testing
    #[derive(Clone, Debug, PartialEq)]
    struct TestReward(f32);

    impl Reward for TestReward {
        fn zero() -> Self {
            TestReward(0.0)
        }
    }

    impl std::ops::Add for TestReward {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            TestReward(self.0 + other.0)
        }
    }

    impl From<TestReward> for f32 {
        fn from(reward: TestReward) -> f32 {
            reward.0
        }
    }

    // ===== Basic Reward Trait Tests =====

    /// Test that `zero()` creates a neutral element for addition
    #[test]
    fn test_reward_zero_is_additive_identity() {
        let zero = TestReward::zero();
        let reward = TestReward(42.5);

        // zero + reward should equal reward
        let result = zero.clone() + reward.clone();
        assert_eq!(result, reward);

        // reward + zero should equal reward
        let result = reward.clone() + zero.clone();
        assert_eq!(result, reward);
    }

    /// Test that rewards can be added together
    #[test]
    fn test_reward_addition() {
        let reward1 = TestReward(10.0);
        let reward2 = TestReward(25.5);
        let result = reward1 + reward2;

        assert_eq!(result, TestReward(35.5));
    }

    /// Test that negative rewards can be added
    #[test]
    fn test_reward_negative_addition() {
        let positive = TestReward(100.0);
        let negative = TestReward(-30.0);
        let result = positive + negative;

        assert_eq!(result, TestReward(70.0));
    }

    /// Test that rewards can be converted to f32
    #[test]
    fn test_reward_into_f32() {
        let reward = TestReward(42.5);
        let as_f32: f32 = reward.into();

        assert_eq!(as_f32, 42.5);
    }

    /// Test that zero reward converts to 0.0
    #[test]
    fn test_reward_zero_into_f32() {
        let zero = TestReward::zero();
        let as_f32: f32 = zero.into();

        assert_eq!(as_f32, 0.0);
    }

    /// Test that rewards are cloneable
    #[test]
    fn test_reward_clone() {
        let original = TestReward(123.456);
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    /// Test that rewards implement Debug
    #[test]
    fn test_reward_debug() {
        let reward = TestReward(42.0);
        let debug_str = format!("{reward:?}");

        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("TestReward"));
    }

    // ===== Arithmetic Properties Tests =====

    /// Test accumulated reward through chained additions
    #[test]
    fn test_reward_accumulation() {
        let mut accumulated = TestReward::zero();
        let rewards = vec![TestReward(10.0), TestReward(20.0), TestReward(15.0)];

        for reward in rewards {
            accumulated = accumulated + reward;
        }

        assert_eq!(accumulated, TestReward(45.0));
    }

    /// Test reward trait with floating point precision
    #[test]
    fn test_reward_floating_point_precision() {
        let r1 = TestReward(0.1);
        let r2 = TestReward(0.2);
        let result = r1 + r2;

        // Account for floating point imprecision
        let expected = 0.3;
        let as_f32: f32 = result.into();
        assert!((as_f32 - expected).abs() < 1e-6);
    }

    /// Test addition associativity: (a + b) + c == a + (b + c)
    #[test]
    fn test_reward_addition_associativity() {
        let r1 = TestReward(5.0);
        let r2 = TestReward(10.0);
        let r3 = TestReward(15.0);

        let left = (r1.clone() + r2.clone()) + r3.clone();
        let right = r1 + (r2 + r3);

        assert_eq!(left, right);
    }

    /// Test addition commutativity: a + b == b + a
    #[test]
    fn test_reward_addition_commutativity() {
        let r1 = TestReward(7.5);
        let r2 = TestReward(12.5);

        let left = r1.clone() + r2.clone();
        let right = r2 + r1;

        assert_eq!(left, right);
    }

    // ===== Special Values Tests =====

    /// Test reward arithmetic with large values
    #[test]
    fn test_reward_large_values() {
        let large1 = TestReward(1e6);
        let large2 = TestReward(1e6);

        let result = large1 + large2;
        let result_f32: f32 = result.into();

        assert_eq!(result_f32, 2e6);
    }

    /// Test reward arithmetic with small values
    #[test]
    fn test_reward_small_values() {
        let small1 = TestReward(1e-6);
        let small2 = TestReward(1e-6);

        let result = small1 + small2;
        let result_f32: f32 = result.into();

        assert!((result_f32 - 2e-6).abs() < 1e-7);
    }

    /// Test mixed positive and negative rewards
    #[test]
    fn test_reward_mixed_signs() {
        let positive = TestReward(10.0);
        let negative = TestReward(-5.0);

        let pos_then_neg = positive.clone() + negative.clone();
        let pos_then_neg_f32: f32 = pos_then_neg.into();

        let neg_then_pos = negative.clone() + positive.clone();
        let neg_then_pos_f32: f32 = neg_then_pos.into();

        assert_eq!(pos_then_neg_f32, 5.0);
        assert_eq!(neg_then_pos_f32, 5.0);
    }

    /// ========================================================================
    /// `GameState` example to test the State trait implementation
    /// ========================================================================
    #[derive(Debug, Clone, PartialEq)]
    enum GameState {
        Menu,
        Playing { level: u8 },
        GameOver { score: u32 },
    }

    impl State<1> for GameState {
        fn shape() -> [usize; 1] {
            [3] // 3 features: state_id, level, score
        }

        fn is_valid(&self) -> bool {
            match self {
                GameState::Playing { level } => *level > 0 && *level <= 10,
                _ => true,
            }
        }

        fn numel(&self) -> usize {
            // Encode as 3 features: [state_id, level, score]
            3
        }
    }

    /// Test state validation for each state variant
    #[test]
    fn test_game_state_validation() {
        // Menu state should always be valid
        let menu_state = GameState::Menu;
        assert!(menu_state.is_valid(), "Menu state should always be valid");

        // GameOver state should always be valid
        let game_over_state = GameState::GameOver { score: 1000 };
        assert!(
            game_over_state.is_valid(),
            "GameOver state should always be valid"
        );

        // Playing state with valid levels should be valid
        for level in 1..=10 {
            let playing_state = GameState::Playing { level };
            assert!(
                playing_state.is_valid(),
                "Playing state with level {level} should be valid"
            );
        }

        // Playing state with invalid levels should be invalid
        let invalid_levels = [0, 11, 255];
        for level in invalid_levels {
            let invalid_state = GameState::Playing { level };
            assert!(
                !invalid_state.is_valid(),
                "Playing state with level {level} should be invalid"
            );
        }
    }

    /// Test that numel returns 3 for all state variants
    #[test]
    fn test_game_state_numel() {
        let test_states = [
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::GameOver { score: 1000 },
        ];

        for state in test_states {
            assert_eq!(
                state.numel(),
                3,
                "Number of elements should be 3 for all states"
            );
        }
    }

    /// Test that shape returns [3] for all state variants
    #[test]
    fn test_game_state_shape() {
        let test_states = [
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::GameOver { score: 1000 },
        ];

        for _state in test_states {
            assert_eq!(
                GameState::shape(),
                [3],
                "Shape should be [3] for all states"
            );
        }
    }

    /// Test the invariant: `numel()` should equal product of `shape()`
    #[test]
    fn test_game_state_consistency() {
        let test_states = [
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::GameOver { score: 1000 },
        ];

        for state in test_states {
            let numel = state.numel();
            let shape_product: usize = GameState::shape().iter().product();
            assert_eq!(
                numel, shape_product,
                "numel({numel}) should equal shape product({shape_product})"
            );
        }
    }

    /// Test that filtering states by validity works correctly
    #[test]
    fn test_game_state_filtering() {
        let states = vec![
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::Playing { level: 0 }, // Invalid
            GameState::GameOver { score: 1000 },
        ];

        let valid_states: Vec<_> = states.into_iter().filter(super::State::is_valid).collect();

        assert_eq!(
            valid_states.len(),
            3,
            "Should have 3 valid states out of 4 total"
        );
        assert!(
            valid_states.iter().all(super::State::is_valid),
            "All filtered states should be valid"
        );

        // Verify the invalid state was filtered out
        assert!(
            !valid_states.contains(&GameState::Playing { level: 0 }),
            "Invalid playing state should be filtered out"
        );
    }

    /// Test edge cases for Playing state level bounds
    #[test]
    fn test_playing_state_edge_cases() {
        // Test boundary values
        let min_valid_level = GameState::Playing { level: 1 };
        assert!(
            min_valid_level.is_valid(),
            "Level 1 should be valid (minimum valid)"
        );

        let max_valid_level = GameState::Playing { level: 10 };
        assert!(
            max_valid_level.is_valid(),
            "Level 10 should be valid (maximum valid)"
        );

        let below_min = GameState::Playing { level: 0 };
        assert!(
            !below_min.is_valid(),
            "Level 0 should be invalid (below minimum)"
        );

        let above_max = GameState::Playing { level: 11 };
        assert!(
            !above_max.is_valid(),
            "Level 11 should be invalid (above maximum)"
        );
    }

    /// ========================================================================
    /// `GridPosition` example to test the State trait implementation
    /// ========================================================================
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct GridPosition {
        x: i32,
        y: i32,
        max_x: i32,
        max_y: i32,
    }

    impl State<1> for GridPosition {
        fn shape() -> [usize; 1] {
            [2] // 2 coordinates: x, y
        }

        fn is_valid(&self) -> bool {
            self.x >= 0 && self.y >= 0 && self.x < self.max_x && self.y < self.max_y
        }

        fn numel(&self) -> usize {
            2 // x and y coordinates
        }
    }

    impl GridPosition {
        /// Flatten the grid position to a vector of f32 values
        fn flatten(&self) -> Vec<f32> {
            vec![
                self.x as f32,
                self.y as f32,
                self.max_x as f32,
                self.max_y as f32,
            ]
        }
    }

    /// Test `GridPosition` validation
    #[test]
    fn test_grid_position_validation() {
        let valid = GridPosition {
            x: 5,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        assert!(valid.is_valid(), "x, y should be valid.");
        //
        let invalid = GridPosition {
            x: 15,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        assert!(
            !invalid.is_valid(),
            "x is larger than max_x and therefore invalid."
        );
    }

    /// Test `GridPosition` flatten
    #[test]
    fn test_grid_position_flattening() {
        let pos1 = GridPosition {
            x: 3,
            y: 7,
            max_x: 10,
            max_y: 10,
        };
        let pos2 = GridPosition {
            x: 0,
            y: 0,
            max_x: 10,
            max_y: 10,
        };
        let pos3 = GridPosition {
            x: 9,
            y: 9,
            max_x: 10,
            max_y: 10,
        };
        let flat1 = pos1.flatten();
        let flat2 = pos2.flatten();
        let flat3 = pos3.flatten();

        assert_eq!(flat1, vec![3.0, 7.0, 10.0, 10.0]);
        assert_eq!(flat2, vec![0.0, 0.0, 10.0, 10.0]);
        assert_eq!(flat3, vec![9.0, 9.0, 10.0, 10.0]);
    }

    /// Test that `GridPosition` numel matches shape product
    #[test]
    fn test_grid_position_consistency() {
        let pos = GridPosition {
            x: 5,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        let numel = pos.numel();
        let shape_product: usize = GridPosition::shape().iter().product();
        assert_eq!(
            numel, shape_product,
            "numel should equal shape product for GridPosition"
        );
        assert_eq!(numel, 2, "GridPosition should have numel of 2");
    }

    /// Test State trait const RANK value
    #[test]
    fn test_state_rank_constant() {
        assert_eq!(
            <GameState as State<1>>::RANK,
            1,
            "GameState should have RANK = 1"
        );
        assert_eq!(
            <GridPosition as State<1>>::RANK,
            1,
            "GridPosition should have RANK = 1"
        );
    }

    // ========================================================================
    // TensorConvertible: derived `to_tensor` + `stack_to_tensor`
    // ========================================================================

    use burn::backend::Flex;

    /// Backend used by the tensor-conversion tests.
    type TcB = Flex;

    /// Rank-1 test row: three scalar features, shape `[3]`.
    #[derive(Clone, Debug)]
    struct Vec3(f32, f32, f32);

    impl HostRow<1> for Vec3 {
        fn row_shape() -> [usize; 1] {
            [3]
        }
        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&[self.0, self.1, self.2]);
        }
    }

    impl<B: Backend> TensorConvertible<1, B> for Vec3 {
        fn from_tensor(_tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by these tests")
        }
    }

    /// Rank-3 test row: a `[2, 2, 1]` image-like payload.
    #[derive(Clone, Debug)]
    struct Img([f32; 4]);

    impl HostRow<3> for Img {
        fn row_shape() -> [usize; 3] {
            [2, 2, 1]
        }
        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&self.0);
        }
    }

    impl<B: Backend> TensorConvertible<3, B> for Img {
        fn from_tensor(_tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by these tests")
        }
    }

    /// `stack_to_tensor` must produce exactly what `Tensor::stack` of the
    /// individually-converted rows produces — bit-identical data and shape.
    #[test]
    fn test_stack_to_tensor_matches_manual_stack() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let items: Vec<Vec3> = vec![
            Vec3(1.0, 2.0, 3.0),
            Vec3(4.0, 5.0, 6.0),
            Vec3(7.0, 8.0, 9.0),
        ];

        let batched: Tensor<TcB, 2> = stack_to_tensor::<1, 2, Vec3, TcB>(&items, &device);

        let per_item: Vec<Tensor<TcB, 1>> = items
            .iter()
            .map(|i| <Vec3 as TensorConvertible<1, TcB>>::to_tensor(i, &device))
            .collect();
        let manual: Tensor<TcB, 2> = Tensor::stack(per_item, 0);

        assert_eq!(batched.dims(), manual.dims());
        let batched_v: Vec<f32> = batched
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let manual_v: Vec<f32> = manual
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        assert_eq!(batched_v, manual_v);
    }

    /// `stack_to_tensor` panics when the batched rank does not equal `R + 1`.
    #[test]
    #[should_panic(expected = "batched rank BR must equal row rank R + 1")]
    fn test_stack_to_tensor_wrong_rank_panics() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let items: Vec<Vec3> = vec![Vec3(1.0, 2.0, 3.0)];
        // BR = 3, but R + 1 = 2 → must panic.
        let _bad: Tensor<TcB, 3> = stack_to_tensor::<1, 3, Vec3, TcB>(&items, &device);
    }

    /// The derived `to_tensor` produces the same data/shape as the old manual
    /// `Tensor::from_floats` path for a rank-1 row.
    #[test]
    fn test_derived_to_tensor_rank1_matches_manual() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let item: Vec3 = Vec3(1.5, -2.5, 3.5);

        let derived: Tensor<TcB, 1> =
            <Vec3 as TensorConvertible<1, TcB>>::to_tensor(&item, &device);
        let manual: Tensor<TcB, 1> = Tensor::from_floats([1.5_f32, -2.5, 3.5], &device);

        assert_eq!(derived.dims(), manual.dims());
        let derived_v: Vec<f32> = derived
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let manual_v: Vec<f32> = manual
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        assert_eq!(derived_v, manual_v);
    }

    /// The derived `to_tensor` produces the same data/shape as the old manual
    /// `TensorData::new` path for a rank-3 row.
    #[test]
    fn test_derived_to_tensor_rank3_matches_manual() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let item: Img = Img([0.1, 0.2, 0.3, 0.4]);

        let derived: Tensor<TcB, 3> = <Img as TensorConvertible<3, TcB>>::to_tensor(&item, &device);
        let manual: Tensor<TcB, 3> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, 0.2, 0.3, 0.4], [2, 2, 1]),
            &device,
        );

        assert_eq!(derived.dims(), manual.dims());
        let derived_v: Vec<f32> = derived
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let manual_v: Vec<f32> = manual
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        assert_eq!(derived_v, manual_v);
    }

    // ===== `row_is_finite` / `row_slice_is_finite` =====

    /// The reference implementation the branchless reduction must agree with.
    ///
    /// This is deliberately the spelling ADR 0067 §Decision 1 forbids in
    /// production (short-circuiting, ~8× the cost, data-dependent): here it is
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

    impl HostRow<1> for TestFiniteRow {
        fn row_shape() -> [usize; 1] {
            [4]
        }

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
