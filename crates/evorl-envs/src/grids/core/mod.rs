//! Shared primitives for the gridworld environments in [`super`].
//!
//! Every concrete grid environment is built from the same small set of
//! building blocks — a [`Grid`] of [`Entity`] cells, an [`AgentState`]
//! tracking position/direction/carried item, a fixed 7-action
//! [`GridAction`] space, and a 7×7×3 egocentric [`GridObservation`]. The
//! [`apply_action`] function is the single source of truth for grid
//! mechanics: every environment's `step` delegates to it and then maps the
//! returned [`StepOutcome`] to env-specific reward + termination logic.

pub mod action;
pub mod agent;
pub mod color;
pub mod direction;
pub mod dynamics;
pub mod entity;
pub mod grid;
pub mod observation;
pub mod render;
pub mod reward;
pub mod state;

pub use action::GridAction;
pub use agent::AgentState;
pub use color::Color;
pub use direction::Direction;
pub use dynamics::{apply_action, StepOutcome};
pub use entity::{DoorState, Entity};
pub use grid::{egocentric_view, Grid};
pub use observation::{GridObservation, OBS_CHANNELS, VIEW_SIZE};
pub use render::render_ascii;
pub use reward::success_reward;
pub use state::GridState;

use evorl_core::environment::SnapshotBase;
use evorl_core::reward::ScalarReward;

/// Canonical snapshot type produced by every grid environment's
/// [`Environment::step`](evorl_core::environment::Environment::step).
///
/// All grid envs use a 3-D observation (`[7, 7, 3]`) and a scalar reward,
/// so this alias saves typing in every per-env `impl`.
pub type GridSnapshot = SnapshotBase<3, GridObservation, ScalarReward>;

/// Build a [`GridSnapshot`] from a borrowed [`GridState`] plus a raw reward
/// and done flag. Every env's `step()` eventually calls this.
#[must_use]
pub fn build_snapshot(state: &GridState, reward: f32, done: bool) -> GridSnapshot {
    use evorl_core::base::State as _;
    if done {
        SnapshotBase::terminated(state.observe(), ScalarReward::new(reward))
    } else {
        SnapshotBase::running(state.observe(), ScalarReward::new(reward))
    }
}
