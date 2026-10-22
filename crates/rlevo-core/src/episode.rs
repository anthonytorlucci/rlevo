//! The episode-lifecycle guard an [`Environment`] holds.
//!
//! An episode that has emitted a snapshot with
//! [`EpisodeStatus::is_done`](crate::environment::EpisodeStatus::is_done) `== true` is
//! over. Calling [`step`](crate::environment::Environment::step) again must
//! not silently resurrect it: the contract on `Environment::step` is that the
//! second call returns [`EnvironmentError::StepAfterEpisodeEnd`].
//!
//! [`EpisodeGuard`] is the one-field state machine that implements that
//! contract. It lives beside the contract, in this crate, because every
//! implementor of [`Environment`] owes it: an environment written against
//! `rlevo-core` alone holds the same guard as the ones `rlevo-environments`
//! ships, and a wrapper that ends an episode itself (a time limit) holds its
//! own.
//!
//! # Usage
//!
//! Hold a guard in the environment struct, `check()` at the very top of `step()`
//! (before any state mutation or RNG draw, so a rejected call leaves the
//! environment untouched — including its RNG stream, which ADR 0029 makes a
//! persistent, observable part of the environment's state), `record()` the
//! status of every snapshot you emit, and `reset()` the guard in
//! `Environment::reset`:
//!
//! ```rust,ignore
//! fn reset(&mut self) -> Result<Snapshot<Self::ObservationType>, EnvironmentError> {
//!     self.guard.reset();
//!     // ... rebuild initial state ...
//!     Ok(Snapshot::running(obs, ScalarReward(0.0)))
//! }
//!
//! fn step(&mut self, action: Self::ActionType) -> Result<Snapshot<Self::ObservationType>, EnvironmentError> {
//!     self.guard.check()?;
//!     // ... apply the action, build `snapshot` ...
//!     self.guard.record(snapshot.status);
//!     Ok(snapshot)
//! }
//! ```
//!
//! The guard stores an [`EpisodeStatus`], never a `done: bool`: `EpisodeStatus`
//! is the single source of truth for episode termination (`rules.md`, section
//! 10), and the stored status is what the error carries back to the caller.
//!
//! [`Environment`]: crate::environment::Environment
//! [`EnvironmentError::StepAfterEpisodeEnd`]: crate::environment::EnvironmentError::StepAfterEpisodeEnd
//! [`EpisodeGuard`]: crate::episode::EpisodeGuard
//! [`EpisodeStatus`]: crate::environment::EpisodeStatus

use crate::environment::{EnvironmentError, EpisodeStatus};

/// Tracks whether the current episode has ended, and rejects a post-terminal
/// [`step`](crate::environment::Environment::step).
///
/// A fresh guard is [`EpisodeStatus::Running`]. [`record`](Self::record) stores
/// the status of the snapshot the environment just emitted;
/// [`check`](Self::check) fails once that stored status is done;
/// [`reset`](Self::reset) re-opens the guard for a new episode.
///
/// The status field is private: it can only be advanced through `record` and
/// cleared through `reset`, so an environment cannot accidentally hand-write a
/// state the snapshot it emitted does not agree with.
///
/// # Examples
///
/// ```
/// use rlevo_core::environment::{EnvironmentError, EpisodeStatus};
/// use rlevo_core::episode::EpisodeGuard;
///
/// let mut guard = EpisodeGuard::new();
/// assert!(guard.check().is_ok(), "a fresh episode is steppable");
///
/// guard.record(EpisodeStatus::Terminated);
/// let err = guard.check().expect_err("a finished episode is not steppable");
/// assert!(matches!(
///     err,
///     EnvironmentError::StepAfterEpisodeEnd {
///         status: EpisodeStatus::Terminated
///     }
/// ));
///
/// guard.reset();
/// assert!(guard.check().is_ok(), "reset() begins a new episode");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EpisodeGuard {
    /// Status of the most recently emitted snapshot; `Running` before the first
    /// step of an episode.
    status: EpisodeStatus,
}

impl EpisodeGuard {
    /// Creates a guard for a fresh, running episode.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            status: EpisodeStatus::Running,
        }
    }

    /// Re-opens the guard for a new episode.
    ///
    /// Call from [`Environment::reset`](crate::environment::Environment::reset).
    pub const fn reset(&mut self) {
        self.status = EpisodeStatus::Running;
    }

    /// Rejects a step taken after the episode already ended.
    ///
    /// Call at the **top** of
    /// [`Environment::step`](crate::environment::Environment::step), before
    /// any state mutation or RNG draw, so a rejected call leaves the environment
    /// untouched — including its RNG stream, which ADR 0029 makes a persistent,
    /// observable part of the environment's state.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`], carrying the
    /// [`EpisodeStatus`] that ended the episode, when the recorded status is
    /// done ([`Terminated`](EpisodeStatus::Terminated) or
    /// [`Truncated`](EpisodeStatus::Truncated)).
    pub const fn check(&self) -> Result<(), EnvironmentError> {
        if self.status.is_done() {
            return Err(EnvironmentError::StepAfterEpisodeEnd {
                status: self.status,
            });
        }
        Ok(())
    }

    /// Records the status of the snapshot the environment just emitted.
    ///
    /// Call once per emitted snapshot, with that snapshot's own status — passing
    /// anything else breaks the invariant that the guard and the snapshot agree.
    pub const fn record(&mut self, status: EpisodeStatus) {
        self.status = status;
    }

    /// Status of the most recently emitted snapshot.
    #[must_use]
    pub const fn status(&self) -> EpisodeStatus {
        self.status
    }
}

impl Default for EpisodeGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::EpisodeGuard;
    use crate::environment::{EnvironmentError, EpisodeStatus};

    #[test]
    fn test_episode_guard_check_passes_when_fresh() {
        let guard = EpisodeGuard::new();
        assert_eq!(
            guard.status(),
            EpisodeStatus::Running,
            "a fresh guard must start Running"
        );
        assert!(
            guard.check().is_ok(),
            "a fresh guard must permit the first step"
        );
    }

    #[test]
    fn test_episode_guard_default_matches_new() {
        assert_eq!(
            EpisodeGuard::default().status(),
            EpisodeGuard::new().status(),
            "Default must agree with new(): a running episode"
        );
    }

    #[test]
    fn test_episode_guard_check_rejects_after_terminated() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Terminated);

        let err = guard
            .check()
            .expect_err("check() must fail once the episode has terminated");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                EpisodeStatus::Terminated,
                "the error must carry Terminated, the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }
    }

    #[test]
    fn test_episode_guard_check_rejects_after_truncated() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Truncated);

        let err = guard
            .check()
            .expect_err("check() must fail once the episode has been truncated");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                EpisodeStatus::Truncated,
                "the error must carry Truncated, so the caller can tell it from a termination"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }
    }

    #[test]
    fn test_episode_guard_check_passes_after_running() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Running);
        assert!(
            guard.check().is_ok(),
            "recording a Running snapshot must leave the episode steppable"
        );
    }

    #[test]
    fn test_episode_guard_reset_reopens_done_guard() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Terminated);
        guard.reset();

        assert_eq!(
            guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );
        assert!(
            guard.check().is_ok(),
            "reset() must re-open a guard that had ended"
        );
    }
}
