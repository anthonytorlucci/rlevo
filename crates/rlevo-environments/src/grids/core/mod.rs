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
//! What an observation carries is the masked window **plus the carried item
//! stamped at the agent's own cell** (or [`Entity::Empty`] when the hand is
//! empty), matching canonical Minigrid's `gen_obs_grid`.
//!
//! Emission is parameterised by [`Visibility`], the per-environment analogue of
//! canonical Minigrid's `see_through_walls` constructor argument. Every grid
//! environment declares its policy as an inherent `const VISIBILITY` and reads
//! it from its own
//! [`Sensor`](rlevo_core::environment::Sensor) impl, which produces the
//! observation through [`observe_grid`] (or, for `GoToDoorEnv`, the crate-private
//! `mask_view` plus the wider mission-carrying encoder) and hands the result to
//! [`build_snapshot`]. The visibility policy is therefore chosen by the
//! environment — the emission model belongs to the environment, not to the
//! state (ADR 0047).
//!
//! The twelve values follow canonical Minigrid exactly — eight `Occluded`, four
//! `SeeThrough`; upstream's own default is occlusion, and an environment opts
//! *out*. `grep -rn "const VISIBILITY"` is the audit surface: each site's doc
//! comment names the canonical file it was read from. ADR 0063
//! (`docs/adr/0063-grid-visibility-occlusion.md`) holds the full table.

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

use grid::{egocentric_view, stamp_carried};
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
///
/// The agent's own cell (`view[VIEW_SIZE - 1][VIEW_SIZE / 2]`) reports
/// [`AgentState::carrying`], or [`Entity::Empty`] when the hand is empty —
/// **never** the world entity the agent is standing on. Two consequences follow:
///
/// * An [`Entity::Goal`] or [`Entity::Lava`] cell *under* the agent does not
///   leak into the observation; the terminal tile is not a readable channel.
/// * `carrying` is now the POMDP-visible channel that `dynamics::toggle`
///   (unlocking a locked door) and `UnlockPickupEnv::has_target` (episode
///   success) gate on, so a policy can tell "holding the key" from
///   "empty-handed".
#[must_use]
pub fn observe_grid(state: &GridState, visibility: Visibility) -> GridObservation {
    let masked = mask_view(&state.grid, &state.agent, visibility);
    GridObservation::from_masked_view(masked, state.agent.direction)
}

/// Cut the egocentric window for `agent`, apply the emission policy
/// `visibility` to it (hidden cells become `None`), and stamp the carried item
/// onto the agent's own cell.
///
/// The single place the [`Visibility`] dispatch happens. [`observe_grid`] is the
/// entry point for the eleven environments emitting a [`GridObservation`];
/// [`GoToDoorEnv`](crate::grids::go_to_door::GoToDoorEnv) needs the same masked
/// view but a wider encoder (its fourth channel carries the mission), so it
/// calls this and feeds
/// [`GoToDoorObservation::from_masked_view`](crate::grids::go_to_door::GoToDoorObservation::from_masked_view).
/// Keeping the dispatch here means there is one `match Visibility`, not two.
///
/// # Agent cell
///
/// The order is mask *then* stamp, matching canonical `gen_obs_grid`:
/// [`grid::stamp_carried`] overwrites `view[VIEW_SIZE - 1][VIEW_SIZE / 2]` with
/// [`AgentState::carrying`] (or [`Entity::Empty`] for an empty hand) after the
/// visibility policy has run, so the carried item is unmaskable under either
/// policy. A caller reading `masked[VIEW_SIZE - 1][VIEW_SIZE / 2]` is reading
/// the hand, not the board.
pub(super) fn mask_view(
    grid: &Grid,
    agent: &AgentState,
    visibility: Visibility,
) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
    let view = egocentric_view(grid, agent);
    // Both arms feed the *same* encoder, so the only difference between the two
    // policies is which cells arrive as `None`.
    let masked = match visibility {
        Visibility::Occluded => grid::process_vis(view),
        Visibility::SeeThrough => view.map(|row| row.map(Some)),
    };
    // Canonical stamps the carried item *after* `process_vis`, and the mask is
    // seeded visible at the agent's own cell, so this is unmaskable under both
    // policies — see `stamp_carried`.
    stamp_carried(masked, agent)
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

    /// View row the agent occupies. `grid`'s own `AGENT_VIEW_ROW` is private to
    /// that module, so it is restated here from [`VIEW_SIZE`] — same derivation,
    /// not a hardcoded `6`.
    const AGENT_ROW: usize = VIEW_SIZE - 1;
    /// View column the agent occupies; see [`AGENT_ROW`].
    const AGENT_COL: usize = VIEW_SIZE / 2;

    /// The byte triple an empty hand stamps: canonical
    /// `OBJECT_TO_IDX["empty"] == 1`, no colour, no state.
    ///
    /// Written as a literal rather than as `Entity::Empty.type_u8()` so that a
    /// renumbering of the canonical table has to be acknowledged here.
    const EMPTY_HAND: [u8; OBS_CHANNELS] = [1, 0, 0];

    /// The byte triple a masked cell encodes to — `[UNSEEN_TYPE, 0, 0]`.
    const UNSEEN: [u8; OBS_CHANNELS] = [UNSEEN_TYPE, 0, 0];

    /// The four facings, swept by every stamp invariant below.
    const FACINGS: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

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

    /// The encoded triple at the agent's own cell — the one `stamp_carried`
    /// owns.
    fn agent_cell(obs: &GridObservation) -> [u8; OBS_CHANNELS] {
        obs.view[AGENT_ROW][AGENT_COL]
    }

    /// View positions at which two observations' encoded bytes differ.
    fn differing_cells(a: &GridObservation, b: &GridObservation) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for row in 0..VIEW_SIZE {
            for col in 0..VIEW_SIZE {
                if a.view[row][col] != b.view[row][col] {
                    out.push((row, col));
                }
            }
        }
        out
    }

    /// Assert that `obs` reproduces `raw` everywhere **except** the agent's own
    /// cell, which `mask_view` overwrites with the hand.
    fn assert_agrees_outside_the_agent_cell(
        obs: &GridObservation,
        raw: &GridObservation,
        context: &str,
    ) {
        assert_eq!(
            obs.agent_direction, raw.agent_direction,
            "{context}: the facing carried beside the view must be untouched"
        );
        let differing: Vec<_> = differing_cells(obs, raw)
            .into_iter()
            .filter(|&pos| pos != (AGENT_ROW, AGENT_COL))
            .collect();
        assert!(
            differing.is_empty(),
            "{context}: only the agent's own cell may move; these did too: {differing:?}"
        );
    }

    #[test]
    fn see_through_reports_every_cell() {
        let state = walled_state();
        let obs = observe_grid(&state, Visibility::SeeThrough);

        // Assert against the *source* of truth rather than against the absence
        // of `UNSEEN_TYPE` bytes: SeeThrough must reproduce the raw view
        // exactly, which is the stronger claim and does not depend on which
        // entities happen to sit in the window.
        //
        // Exactly one cell is exempt: `mask_view` stamps `AgentState::carrying`
        // onto the agent's own cell after the visibility policy has run, so the
        // raw window's world entity there is never emitted. This fixture's agent
        // happens to stand on `Entity::Empty` empty-handed, so the two agree at
        // that cell too — but agreeing by coincidence is not the contract, so
        // the cell is asserted separately against the stamp.
        let raw = GridObservation::from_entity_view(
            egocentric_view(&state.grid, &state.agent),
            state.agent.direction,
        );
        assert_agrees_outside_the_agent_cell(&obs, &raw, "SeeThrough over `walled_state`");
        assert_eq!(
            state.agent.carrying, None,
            "fixture must be empty-handed for the expectation below"
        );
        assert_eq!(
            agent_cell(&obs),
            EMPTY_HAND,
            "the agent's own cell reports the empty hand, not the world cell under it"
        );
    }

    #[test]
    fn see_through_masks_nothing_outside_the_agent_cell() {
        // `SeeThrough` masks nothing, so what it emits is the raw window — with
        // the agent's own cell replaced by the hand. Eight of the twelve
        // environments are now `Occluded`, but for the four that opt out no
        // observation byte may move merely because the occlusion machinery
        // exists: outside that one cell, only the `Occluded` arm may differ from
        // this baseline.
        //
        // The last two poses are the red-green teeth. One carries a `Key` (raw
        // window says `Empty`, the emission must say `Key`); one stands on
        // `Entity::Goal` (raw window says `Goal`, the emission must say the
        // empty hand). Revert the stamp and both fail.
        for direction in FACINGS {
            for (x, y, carrying, under) in [
                (1, 1, None, None),
                (5, 5, None, None),
                (7, 7, None, None),
                (2, 6, None, None),
                (5, 5, Some(Entity::Key(Color::Yellow)), None),
                (5, 5, None, Some(Entity::Goal)),
            ] {
                let mut grid = Grid::new(9, 9);
                grid.draw_walls();
                for gy in 1..8 {
                    grid.set(3, gy, Entity::Wall);
                }
                grid.set(6, 2, Entity::Door(Color::Blue, DoorState::Closed));
                grid.set(6, 6, Entity::Lava);
                if let Some(entity) = under {
                    grid.set(x, y, entity);
                }
                let mut agent = AgentState::new(x, y, direction);
                agent.carrying = carrying;
                let state = GridState::new(grid, agent);

                let before = GridObservation::from_entity_view(
                    egocentric_view(&state.grid, &state.agent),
                    direction,
                );
                let obs = observe_grid(&state, Visibility::SeeThrough);
                assert_agrees_outside_the_agent_cell(
                    &obs,
                    &before,
                    &format!(
                        "SeeThrough at ({x}, {y}) facing {direction:?} \
                         carrying {carrying:?} standing on {under:?}"
                    ),
                );

                let expected =
                    carrying.map_or(EMPTY_HAND, |e| [e.type_u8(), e.color_u8(), e.state_u8()]);
                assert_eq!(
                    agent_cell(&obs),
                    expected,
                    "at ({x}, {y}) facing {direction:?} the agent's own cell must \
                     report the hand ({carrying:?}), not the world entity under it"
                );
            }
        }
    }

    #[test]
    fn carried_item_is_stamped_at_the_agent_cell_for_every_facing() {
        for entity in [
            Entity::Key(Color::Yellow),
            Entity::Ball(Color::Red),
            Entity::Box(Color::Green),
        ] {
            for direction in FACINGS {
                let mut state = walled_state();
                state.agent.direction = direction;
                state.agent.carrying = Some(entity);

                for visibility in [Visibility::Occluded, Visibility::SeeThrough] {
                    let obs = observe_grid(&state, visibility);
                    assert_eq!(
                        agent_cell(&obs),
                        // The third byte is a literal `0`, not
                        // `entity.state_u8()`: carryables inherit
                        // `WorldObj.encode`, and only `Door` has a state byte —
                        // and a `Door` cannot be picked up. Pinning the literal
                        // means a future `state_u8` that starts reporting
                        // something for keys/balls/boxes is caught here rather
                        // than silently mirrored by the expectation.
                        [entity.type_u8(), entity.color_u8(), 0],
                        "{entity:?} facing {direction:?} under {visibility:?} must be \
                         stamped at the agent's own cell"
                    );
                }
            }
        }
    }

    #[test]
    fn an_empty_hand_stamps_the_empty_triple() {
        for direction in FACINGS {
            let mut state = walled_state();
            state.agent.direction = direction;
            state.agent.carrying = None;

            for visibility in [Visibility::Occluded, Visibility::SeeThrough] {
                let obs = observe_grid(&state, visibility);
                assert_eq!(
                    agent_cell(&obs),
                    EMPTY_HAND,
                    "an empty hand is canonical's `grid.set(*agent_pos, None)`, which \
                     `Grid.encode` turns into `OBJECT_TO_IDX[\"empty\"] == 1` — \
                     facing {direction:?} under {visibility:?}"
                );
                // The likeliest mis-transcription: stamping rlevo's `None`,
                // which means *masked*, would tell the policy that the one cell
                // it always sees is unobserved.
                assert_ne!(
                    agent_cell(&obs),
                    UNSEEN,
                    "the agent's own cell must never encode as unseen — \
                     facing {direction:?} under {visibility:?}"
                );
            }
        }
    }

    #[test]
    fn the_world_entity_under_the_agent_does_not_leak() {
        // The regression guard for the terminal-frame behaviour change: before
        // the stamp, the agent's own cell reported the world entity, so on every
        // terminal frame it read `[8, 0, 0]` (goal) or `[9, 0, 0]` (lava) and the
        // terminal tile was a channel the policy could read.
        for entity in [Entity::Goal, Entity::Lava, Entity::Floor] {
            for visibility in [Visibility::Occluded, Visibility::SeeThrough] {
                let mut state = walled_state();
                state.agent.carrying = None;
                state.grid.set(state.agent.x, state.agent.y, entity);
                assert_eq!(
                    state.grid.get(state.agent.x, state.agent.y),
                    entity,
                    "fixture must actually place {entity:?} under the agent"
                );

                let obs = observe_grid(&state, visibility);
                assert_eq!(
                    agent_cell(&obs),
                    EMPTY_HAND,
                    "standing on {entity:?} under {visibility:?}, the agent's own cell \
                     must still report the empty hand"
                );
                assert_ne!(
                    agent_cell(&obs)[0],
                    entity.type_u8(),
                    "the {entity:?} under the agent must not leak into channel 0 \
                     under {visibility:?}"
                );
            }
        }
    }

    #[test]
    fn the_carried_item_is_never_masked() {
        // `process_vis` seeds the agent's own cell visible, and the stamp runs
        // after the mask, so occlusion cannot reach the hand.
        let key = Entity::Key(Color::Blue);
        let mut grid = Grid::new(9, 9);
        grid.draw_walls();
        let (ax, ay) = (4, 4);
        for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
            grid.set(ax + dx, ay + dy, Entity::Wall);
        }
        let mut agent = AgentState::new(ax, ay, Direction::North);
        agent.carrying = Some(key);
        let state = GridState::new(grid, agent);

        let obs = observe_grid(&state, Visibility::Occluded);
        assert_eq!(
            agent_cell(&obs),
            [key.type_u8(), key.color_u8(), 0],
            "walled in on all four sides, the agent still sees its own hand"
        );
        assert_ne!(
            agent_cell(&obs),
            UNSEEN,
            "the hand must not encode as an unseen cell"
        );

        // Non-vacuity: the occluder really is occluding, so the assertion above
        // is not passing merely because nothing was masked.
        let masked_cells = obs
            .view
            .iter()
            .flatten()
            .filter(|&&cell| cell == UNSEEN)
            .count();
        assert!(
            masked_cells > 0,
            "fixture must actually mask something, or the claim above is vacuous"
        );
    }

    #[test]
    fn carrying_changes_the_observation_and_nothing_else() {
        // The generic form of issue #1027's reproducer: two states that differ
        // only in `AgentState::carrying` must produce different observations,
        // and the difference must be confined to the agent's own cell.
        for visibility in [Visibility::Occluded, Visibility::SeeThrough] {
            let empty_handed = walled_state();
            let mut holding = walled_state();
            holding.agent.carrying = Some(Entity::Key(Color::Yellow));
            assert_eq!(
                empty_handed.agent.carrying, None,
                "the two states must differ in exactly one field"
            );

            let obs_a = observe_grid(&empty_handed, visibility);
            let obs_b = observe_grid(&holding, visibility);

            assert_ne!(
                obs_a, obs_b,
                "picking an object up must be visible to the policy under {visibility:?}"
            );
            assert_eq!(
                differing_cells(&obs_a, &obs_b),
                vec![(AGENT_ROW, AGENT_COL)],
                "the hand is the only channel `carrying` may move, under {visibility:?}"
            );
        }
    }

    #[test]
    fn the_two_visibility_policies_agree_at_the_agent_cell() {
        // The stamp lives outside the `match Visibility` in `mask_view`. Moving
        // it into one arm would make the hand policy-dependent; this pins that
        // it is not.
        for carrying in [
            None,
            Some(Entity::Ball(Color::Red)),
            Some(Entity::Box(Color::Green)),
        ] {
            for direction in FACINGS {
                let mut state = walled_state();
                state.agent.direction = direction;
                state.agent.carrying = carrying;

                assert_eq!(
                    agent_cell(&observe_grid(&state, Visibility::Occluded)),
                    agent_cell(&observe_grid(&state, Visibility::SeeThrough)),
                    "the agent's own cell is policy-independent — carrying \
                     {carrying:?} facing {direction:?}"
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
