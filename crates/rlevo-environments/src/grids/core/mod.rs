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
//! Per-episode layout randomization is shared the same way: [`placement`]
//! holds the free-cell predicate, the uniform position sampler, and the
//! agent-pose draw, so no environment hand-rolls a rejection loop.
//!
//! The observation is the one block that is not universal:
//! [`GoToDoorEnv`](crate::grids::go_to_door::GoToDoorEnv) is goal-conditioned and
//! emits a `7×7×4` view whose fourth channel carries the episode mission, so it
//! uses neither [`GridObservation`] nor [`GridSnapshot`]. Everything else on this
//! list — grid, agent, actions, dynamics — it shares. See ADR 0043
//! (`docs/adr/0043-grid-observation-contract.md`).
//!
//! Emission is parameterised by [`Visibility`], the per-environment analogue of
//! canonical Minigrid's `see_through_walls` constructor argument. Every grid
//! environment produces its observation through [`observe_grid`] and hands the
//! result to [`build_snapshot`], so the visibility policy is chosen by the
//! environment — the emission model belongs to the environment, not to the
//! state (ADR 0047).

pub mod action;
pub mod agent;
pub mod color;
pub mod dynamics;
pub mod entity;
pub mod grid;
pub mod observation;
pub mod placement;
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
pub use placement::{
    PlacementError, Rect, is_free, no_reject, place_agent, place_obj, random_direction, sample_pos,
};
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

/// Per-environment emission-model policy: does the agent see through opaque
/// cells?
///
/// This is the rlevo spelling of canonical Minigrid's `see_through_walls`
/// constructor argument ([`Occluded`](Self::Occluded) ==
/// `see_through_walls=False`, which is canonical's default, and
/// [`SeeThrough`](Self::SeeThrough) == `see_through_walls=True`). Minigrid sets
/// it per environment rather than per family, which is why it is a value an env
/// supplies to [`observe_grid`] and not a property of [`GridState`].
///
/// There is deliberately **no** `Default` impl. A default would be wrong for a
/// third of the family — canonical marks four of the twelve environments
/// see-through and the remaining eight occluded — so every environment must
/// state its own value at the call site rather than inherit one silently.
///
/// # Examples
///
/// ```
/// use rlevo_environments::grids::core::{
///     AgentState, Grid, GridState, Visibility, observe_grid,
/// };
/// use rlevo_environments::direction::Direction;
///
/// let mut grid = Grid::new(5, 5);
/// grid.draw_walls();
/// let state = GridState::new(grid, AgentState::new(1, 1, Direction::East));
/// let obs = observe_grid(&state, Visibility::SeeThrough);
/// assert_eq!(obs.agent_direction, Some(Direction::East));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Opaque cells (walls, closed doors) hide what lies behind them.
    ///
    /// Canonical `see_through_walls=False` — Minigrid's default.
    Occluded,
    /// Every cell of the view window is reported, regardless of what stands
    /// between it and the agent.
    ///
    /// Canonical `see_through_walls=True`.
    SeeThrough,
}

/// Produce the egocentric [`GridObservation`] for `state` under the emission
/// policy `visibility`.
///
/// This is the single entry point for grid observation production: every
/// environment in the family calls it (directly, or by feeding its result to
/// [`build_snapshot`]) so that the per-env visibility policy has exactly one
/// place to take effect.
///
/// # Arguments
///
/// * `state` — the world state to observe.
/// * `visibility` — the environment's emission policy; see [`Visibility`].
///
/// # Returns
///
/// The `7×7×3` view rotated into the agent's frame, tagged with the agent's
/// absolute facing.
#[must_use]
pub fn observe_grid(state: &GridState, visibility: Visibility) -> GridObservation {
    let view = egocentric_view(&state.grid, &state.agent);
    let view = match visibility {
        // TODO(#281): occlusion lands in T2 — `occlude` is the identity today,
        // so both arms emit the same view and this seam changes no observation
        // byte.
        Visibility::Occluded => occlude(view),
        Visibility::SeeThrough => view,
    };
    GridObservation::from_entity_view(view, state.agent.direction)
}

/// Hide the cells of an egocentric `view` that an opaque cell stands in front
/// of.
///
/// The input is the rotated view window, so the agent is at
/// `view[VIEW_SIZE - 1][VIEW_SIZE / 2]` looking toward row `0` — the same frame
/// canonical Minigrid's `Grid.process_vis` runs in.
///
/// **This is the identity function today.** Introducing the [`Visibility`]
/// seam is deliberately a no-op on every emitted observation; the shadow cast
/// itself is a separate change, so that a behavioural diff is attributable to
/// one commit.
fn occlude(view: [[Entity; VIEW_SIZE]; VIEW_SIZE]) -> [[Entity; VIEW_SIZE]; VIEW_SIZE] {
    // TODO(#281): port `Grid.process_vis` here in T2 — two horizontal sweeps
    // per row from the agent's row outward, masking behind any cell whose
    // `see_behind` is false (walls, and doors that are not open).
    view
}

/// Build a [`GridSnapshot`] from an already-produced [`GridObservation`] plus a
/// raw reward and done flag.
///
/// The observation is passed in rather than projected here so that the
/// [`Visibility`] decision stays with the environment that owns it; callers
/// produce it with [`observe_grid`].
///
/// Every env whose snapshot is a [`GridSnapshot`] routes its `step()` through
/// here; `GoToDoorEnv` builds its own [`SnapshotBase`] because its observation
/// needs the episode mission (see the [`GridSnapshot`] docs and ADR 0043).
#[must_use]
pub fn build_snapshot(observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
    if done {
        SnapshotBase::terminated(observation, ScalarReward::new(reward))
    } else {
        SnapshotBase::running(observation, ScalarReward::new(reward))
    }
}
