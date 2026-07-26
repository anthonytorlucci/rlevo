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
//! environment declares its policy as an inherent `const VISIBILITY` and reads
//! it from its own
//! [`Sensor`](rlevo_core::environment::Sensor) impl, which produces the
//! observation through [`observe_grid`] (or, for `GoToDoorEnv`, [`mask_view`]
//! plus the wider mission-carrying encoder) and hands the result to
//! [`build_snapshot`]. The visibility policy is therefore chosen by the
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
// `egocentric_view` is deliberately **not** re-exported: it is the raw,
// unoccluded window, which is only a correct emission model for an environment
// that chose `Visibility::SeeThrough`. `observe_grid` is the public entry point
// so the visibility policy cannot be bypassed from outside the crate.
pub use grid::Grid;
pub use observation::{GridObservation, OBS_CHANNELS, UNSEEN_TYPE, VIEW_SIZE};
pub use placement::{
    PlacementError, Rect, is_free, no_reject, place_agent, place_obj, random_direction, sample_pos,
};
pub use render::render_ascii;
pub use reward::success_reward;
pub use state::GridState;

use grid::egocentric_view;
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
    /// Canonical `see_through_walls=False` — Minigrid's default. The shadow
    /// cast is a port of `Grid.process_vis`; hidden cells encode as
    /// [`UNSEEN_TYPE`].
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
/// absolute facing. Under [`Visibility::Occluded`] the cells the agent cannot
/// see carry the [`UNSEEN_TYPE`] byte triple; under
/// [`Visibility::SeeThrough`] every cell is reported.
#[must_use]
pub fn observe_grid(state: &GridState, visibility: Visibility) -> GridObservation {
    let masked = mask_view(&state.grid, &state.agent, visibility);
    GridObservation::from_masked_view(masked, state.agent.direction)
}

/// Cut the egocentric window for `agent` and apply the emission policy
/// `visibility` to it, yielding a view whose hidden cells are `None`.
///
/// The single place the [`Visibility`] dispatch happens. [`observe_grid`] is the
/// entry point for the eleven environments emitting a [`GridObservation`];
/// [`GoToDoorEnv`](crate::grids::go_to_door::GoToDoorEnv) needs the same masked
/// view but a wider encoder (its fourth channel carries the mission), so it
/// calls this and feeds
/// [`GoToDoorObservation::from_masked_view`](crate::grids::go_to_door::GoToDoorObservation::from_masked_view).
/// Keeping the dispatch here means there is one `match Visibility`, not two.
pub(super) fn mask_view(
    grid: &Grid,
    agent: &AgentState,
    visibility: Visibility,
) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
    let view = egocentric_view(grid, agent);
    // Both arms feed the *same* encoder, so the only difference between the two
    // policies is which cells arrive as `None`.
    match visibility {
        Visibility::Occluded => grid::process_vis(view),
        Visibility::SeeThrough => view.map(|row| row.map(Some)),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A corridor with a solid wall run beside the agent — enough geometry for
    /// the two visibility policies to disagree.
    fn walled_state() -> GridState {
        let mut grid = Grid::new(9, 9);
        grid.draw_walls();
        for y in 1..8 {
            grid.set(3, y, Entity::Wall);
        }
        grid.set(5, 1, Entity::Goal);
        GridState::new(grid, AgentState::new(5, 5, Direction::North))
    }

    #[test]
    fn see_through_reports_every_cell() {
        let state = walled_state();
        let obs = observe_grid(&state, Visibility::SeeThrough);

        // `UNSEEN_TYPE` currently collides with `Entity::Empty` in the byte
        // encoding (see the `UNSEEN_TYPE` docs), so assert against the *source*
        // of truth instead: SeeThrough must reproduce the raw view exactly.
        let raw = GridObservation::from_entity_view(
            egocentric_view(&state.grid, &state.agent),
            state.agent.direction,
        );
        assert_eq!(
            obs, raw,
            "SeeThrough must emit the unoccluded view byte for byte — no cell may be masked"
        );
    }

    #[test]
    fn see_through_is_byte_identical_to_the_pre_occlusion_encoder() {
        // The end-to-end no-op guard for this commit: every environment is
        // still `Visibility::SeeThrough`, so no observation byte may change.
        for direction in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            for (x, y) in [(1, 1), (5, 5), (7, 7), (2, 6)] {
                let mut grid = Grid::new(9, 9);
                grid.draw_walls();
                for gy in 1..8 {
                    grid.set(3, gy, Entity::Wall);
                }
                grid.set(6, 2, Entity::Door(Color::Blue, DoorState::Closed));
                grid.set(6, 6, Entity::Lava);
                let state = GridState::new(grid, AgentState::new(x, y, direction));

                let before = GridObservation::from_entity_view(
                    egocentric_view(&state.grid, &state.agent),
                    direction,
                );
                assert_eq!(
                    observe_grid(&state, Visibility::SeeThrough),
                    before,
                    "SeeThrough at ({x}, {y}) facing {direction:?} must be unchanged"
                );
            }
        }
    }

    #[test]
    fn occluded_hides_cells_that_see_through_reports() {
        let state = walled_state();
        let open = observe_grid(&state, Visibility::SeeThrough);
        let shut = observe_grid(&state, Visibility::Occluded);

        assert_ne!(
            open, shut,
            "with a wall run beside the agent the two policies must disagree"
        );

        // The agent faces North from (5, 5); the wall column at x == 3 is two
        // cells to its left, so the world beyond it is unseen.
        let masked = process_vis_of(&state);
        assert!(
            masked.iter().flatten().any(Option::is_none),
            "the occluded view must mask at least one cell"
        );
    }

    fn process_vis_of(state: &GridState) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
        grid::process_vis(egocentric_view(&state.grid, &state.agent))
    }
}
