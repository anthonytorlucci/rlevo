//! [`RecordableAction`] — the recording tier's requirement on action types.
//!
//! Recording an episode means writing each action to disk, so
//! [`RecordingTap`](crate::record::RecordingTap) needs the environment's
//! `ActionType` to be [`serde::Serialize`]. The
//! [`Action`](rlevo_core::base::Action) trait itself is only
//! `Debug + Clone + Sized` and deliberately stays that way: a serde
//! supertrait on a domain trait would tax every environment for a capability
//! only this crate uses, so the requirement is declared here, at the
//! consuming seam (`docs/rules.md`, ADR 0064).
//!
//! That design is right, and it used to be undiscoverable. Bounding the tap
//! on `Serialize` directly meant a researcher whose action type lacked the
//! derive got told their *wrapper* did not implement `Environment`, followed
//! by serde's list of the couple of hundred types that do implement
//! `Serialize` — with no sentence anywhere naming the thing to do. This
//! trait exists to replace that with the one-line fix.

use serde::Serialize;

/// Blanket-implemented marker for action types the recording tier can write.
///
/// Every [`Serialize`] type implements this; there is nothing to write by
/// hand, and nothing to import. It exists only so the compiler can say what
/// went wrong when an action type is *not* `Serialize`:
///
/// ```text
/// error[E0277]: `Boiler` cannot be recorded: an environment's action type
///               must implement `serde::Serialize`
///     |
/// 351 |   RecordingTap::new(Thermostat::with_seed(SEED), sink.clone());
///     |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ this action type is
///     |                                                 not `Serialize`
///     |
/// help: the trait `RecordableAction` is not implemented for `Boiler`
///     |
///  93 | enum Boiler {
///     | ^^^^^^^^^^^
///     = note: add `#[derive(serde::Serialize)]` to `Boiler`
/// note: required by a bound in `RecordingTap::<E, D, SD, AD>::new`
/// ```
///
/// Transcribed from a `cargo xtask byoe` run with the probe's derive
/// removed, not composed by hand.
///
/// # Why the constructors carry the bound too
///
/// The requirement is only *used* by the `Environment` impl, so bounding
/// that impl alone would be enough to make the code correct. It is not
/// enough to make it explain itself: a bound checked during method
/// resolution (`tap.reset()`) fails as `E0599`, and
/// `#[diagnostic::on_unimplemented]` does not apply to `E0599` — the
/// researcher gets "the method `reset` exists ... but its trait bounds were
/// not satisfied" with the message dropped. Repeating the bound on every
/// constructor moves the failure to the caller's own
/// `RecordingTap::new(...)` line, where it is an `E0277` and the message
/// applies. That is the transcript above.
///
/// # Rejection is pinned; the message is not
///
/// A non-`Serialize` action type is rejected at a direct bound:
///
/// ```compile_fail,E0277
/// use rlevo_benchmarks::record::RecordableAction;
///
/// #[derive(Debug, Clone)]
/// enum Boiler { Off, On }
///
/// fn needs_recordable<T: RecordableAction>(_: T) {}
/// needs_recordable(Boiler::Off);
/// ```
///
/// and through an associated type on a wrapper — the shape
/// [`RecordingTap`](crate::record::RecordingTap) actually has, and the one
/// where `do_not_recommend` is load-bearing:
///
/// ```compile_fail,E0277
/// use rlevo_benchmarks::record::RecordableAction;
///
/// trait Env { type ActionType; }
/// struct Tap<E>(E);
/// impl<E: Env> Env for Tap<E> where E::ActionType: RecordableAction {
///     type ActionType = E::ActionType;
/// }
///
/// #[derive(Debug, Clone)]
/// enum Boiler { Off, On }
/// struct Thermostat;
/// impl Env for Thermostat { type ActionType = Boiler; }
///
/// fn record<E: Env>(_: E) {}
/// record(Tap(Thermostat));
/// ```
///
/// **`compile_fail` asserts an error code and nothing more.** Deleting either
/// diagnostic attribute leaves both blocks above green while restoring exactly
/// the unhelpful output this trait exists to prevent — the message is the
/// point, and the message is what is not covered. Pinning stderr needs a
/// `trybuild`-style dev-dependency; until one is added, the transcript above
/// is the reference, and `cargo xtask byoe` exercises the live path.
///
/// Two attributes make that happen and both are load-bearing:
/// [`on_unimplemented`] supplies the text, and
/// [`do_not_recommend`] on the blanket impl below stops rustc unwinding
/// through it to report the underlying `Boiler: Serialize` obligation
/// instead — which is what it does by default, and which is precisely the
/// unhelpful message this trait replaces. Removing either one silently
/// restores the old diagnostic.
///
/// [`on_unimplemented`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-diagnosticon_unimplemented-attribute
/// [`do_not_recommend`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-diagnosticdo_not_recommend-attribute
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be recorded: an environment's action type must implement `serde::Serialize`",
    label = "this action type is not `Serialize`",
    note = "add `#[derive(serde::Serialize)]` to `{Self}`",
    note = "the `Action` trait does not require `Serialize` — only the recording tier does, so environments that are never recorded pay nothing for it"
)]
pub trait RecordableAction: Serialize {}

#[diagnostic::do_not_recommend]
impl<T: Serialize> RecordableAction for T {}
