//! Shared primitives for the gridworld environments in [`super`].
//!
//! Every concrete grid environment is built from the same small set of
//! building blocks — a [`Grid`] of [`Entity`] cells, an [`AgentState`]
//! tracking position/direction/carried item, a fixed 7-action
//! [`GridAction`] space, and a 7×7×3 egocentric [`GridObservation`]. The
//! [`apply_action`] function is the single source of truth for grid
//! mechanics: every environment's `step` delegates to it and then maps the
//! returned [`StepOutcome`] to env-specific reward + termination logic.
//!
//! The observation is the one block that is not universal:
//! [`GoToDoorEnv`](crate::grids::go_to_door::GoToDoorEnv) is goal-conditioned and
//! emits a `7×7×4` view whose fourth channel carries the episode mission, so it
//! uses neither [`GridObservation`] nor [`GridSnapshot`]. Everything else on this
//! list — grid, agent, actions, dynamics — it shares. See ADR 0043
//! (`docs/adr/0043-grid-observation-contract.md`).

pub mod action;
pub mod agent;
pub mod color;
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
// `Direction` was lifted to the crate root (`crate::direction`); re-export both
// the module and the type so existing `grids::core::{direction, Direction}`
// paths keep resolving.
pub use crate::direction::{self, Direction};
pub use dynamics::{StepOutcome, apply_action};
pub use entity::{DoorState, Entity};
pub use grid::{Grid, egocentric_view};
pub use observation::{GridObservation, OBS_CHANNELS, VIEW_SIZE};
pub use render::render_ascii;
pub use reward::success_reward;
pub use state::GridState;

use rlevo_core::environment::SnapshotBase;
use rlevo_core::reward::ScalarReward;

/// Canonical snapshot type produced by the
/// [`Environment::step`](rlevo_core::environment::Environment::step) of every
/// grid environment **except
/// [`GoToDoorEnv`](crate::grids::go_to_door::GoToDoorEnv)**.
///
/// Those envs pair the shared 3-D [`GridObservation`] (`[7, 7, 3]`) with a
/// scalar reward, so this alias saves typing in every per-env `impl`.
///
/// `GoToDoorEnv` is the one exception: it is goal-conditioned, so it carries the
/// mission colour in a fourth observation channel
/// ([`GoToDoorObservation`](crate::grids::go_to_door::GoToDoorObservation),
/// `[7, 7, 4]`) and emits a
/// [`GoToDoorSnapshot`](crate::grids::go_to_door::GoToDoorSnapshot) instead. The
/// rank stays 3, so its `Environment<3, 3, 1>` bound is unchanged; only the
/// channel count differs. ADR 0043 (`docs/adr/0043-grid-observation-contract.md`)
/// records why the shared observation was not widened for all twelve envs.
pub type GridSnapshot = SnapshotBase<3, GridObservation, ScalarReward>;

/// Build a [`GridSnapshot`] from a borrowed [`GridState`] plus a raw reward
/// and done flag.
///
/// Every env whose snapshot is a [`GridSnapshot`] routes its `step()` through
/// here; `GoToDoorEnv` builds its own [`SnapshotBase`] because its observation
/// needs the episode mission (see the [`GridSnapshot`] docs and ADR 0043).
#[must_use]
pub fn build_snapshot(state: &GridState, reward: f32, done: bool) -> GridSnapshot {
    use rlevo_core::base::State as _;
    if done {
        SnapshotBase::terminated(state.observe(), ScalarReward::new(reward))
    } else {
        SnapshotBase::running(state.observe(), ScalarReward::new(reward))
    }
}
