//! Test-only shortest-path planner for the `grids` family.
//!
//! # Why this exists
//!
//! A hard-coded action script proves that **one** board is solvable. The moment
//! an environment samples its layout per episode, that script is either right by
//! luck or silently wrong — and worse, it stops being a statement about the env
//! at all. This planner replaces the script with a search: give it the board the
//! environment actually built and a success predicate, and it returns a genuine
//! action sequence, or `None` if the board is unsolvable. The caller then feeds
//! that sequence through the public [`Environment::step`] API, so the end-to-end
//! property the scripted oracles had is preserved exactly.
//!
//! [`Environment::step`]: rlevo_core::environment::Environment::step
//!
//! # Placement
//!
//! This is deliberately a **duplicated test helper**, not public API:
//!
//! - `rlevo-test-support` is not an option — it already depends on
//!   `rlevo-environments`, so a dev-dependency back would be a cycle.
//! - A public `grids::core::plan` was considered and rejected for now: shipping a
//!   solver is a one-way door on the crate's public surface. Start duplicated;
//!   promote if it earns it.
//!
//! Files under `tests/common/` are not test targets themselves (Cargo only picks
//! up `tests/*.rs`), so this compiles once, inside whichever test binary
//! `mod common;`-es it.
//!
//! # The search
//!
//! Breadth-first over the state that [`apply_action`] can mutate, expanding every
//! [`GridAction`] against a **cloned** grid so the live environment is never
//! touched. BFS (not DFS, not greedy) because a shortest plan keeps the step
//! count well inside `max_steps`, which the `success_reward` formula needs in
//! order to pay anything at all.

use std::collections::{HashSet, VecDeque};

use rlevo_core::action::DiscreteAction as _;
use rlevo_environments::grids::core::{
    AgentState, Direction, Entity, Grid, GridAction, GridState, StepOutcome, apply_action,
};

/// The minimal sufficient BFS key, derived from what [`apply_action`] mutates.
///
/// `apply_action` writes to exactly two places: the [`AgentState`] (`x`, `y`,
/// `direction`, `carrying` — every field it has) and the [`Grid`] cells
/// (`Pickup` clears a cell, `Drop` fills one, `Toggle` rewrites a door's
/// [`DoorState`]). Since `Drop` can place a carried object on *any* empty cell,
/// there is no smaller honest key than the whole cell array: restricting it to
/// "the cells that started as doors or objects" would collapse two genuinely
/// different boards into one and let the search return a plan that does not
/// replay.
///
/// [`DoorState`]: rlevo_environments::grids::core::DoorState
#[derive(Debug, PartialEq, Eq, Hash)]
struct BoardKey {
    /// Agent world x.
    x: i32,
    /// Agent world y.
    y: i32,
    /// Agent facing.
    direction: Direction,
    /// Item in the agent's hand, if any.
    carrying: Option<Entity>,
    /// Every grid cell, row-major.
    cells: Vec<Entity>,
}

impl BoardKey {
    /// Snapshot `(grid, agent)` into a hashable key.
    fn of(grid: &Grid, agent: &AgentState) -> Self {
        let mut cells = Vec::with_capacity(grid.width() * grid.height());
        for y in 0..grid.height() {
            let gy = i32::try_from(y).expect("grid height fits in i32");
            for x in 0..grid.width() {
                let gx = i32::try_from(x).expect("grid width fits in i32");
                cells.push(grid.get(gx, gy));
            }
        }
        Self {
            x: agent.x,
            y: agent.y,
            direction: agent.direction,
            carrying: agent.carrying,
            cells,
        }
    }
}

/// One node of the BFS tree: a reachable board plus the edge that reached it.
#[derive(Debug)]
struct Node {
    /// The cloned grid at this node.
    grid: Grid,
    /// The agent at this node.
    agent: AgentState,
    /// `(parent index, action taken)`; `None` for the root.
    parent: Option<(usize, GridAction)>,
}

/// Find a shortest action sequence from `state` to a board satisfying
/// `is_solved`, or `None` if no such sequence exists.
///
/// The search never plans *through* lava: every env in this family that places
/// [`Entity::Lava`] treats stepping onto it as terminal with reward `0.0`, so a
/// plan that walked over lava would not be a solution even though the resulting
/// board might satisfy the predicate.
///
/// [`GridAction::Done`] is expanded like any other action, but can never appear
/// in a returned plan: it leaves the board identical, so its successor is always
/// already visited.
///
/// Returns an empty plan when `state` is already solved — callers that require
/// real work should assert the plan is non-empty.
pub fn plan<F>(state: &GridState, is_solved: F) -> Option<Vec<GridAction>>
where
    F: Fn(&Grid, &AgentState) -> bool,
{
    if is_solved(&state.grid, &state.agent) {
        return Some(Vec::new());
    }

    let actions = GridAction::enumerate();
    let mut nodes = vec![Node {
        grid: state.grid.clone(),
        agent: state.agent,
        parent: None,
    }];
    let mut seen = HashSet::from([BoardKey::of(&state.grid, &state.agent)]);
    let mut frontier = VecDeque::from([0usize]);

    while let Some(idx) = frontier.pop_front() {
        for &action in &actions {
            let mut grid = nodes[idx].grid.clone();
            let mut agent = nodes[idx].agent;
            if apply_action(&mut grid, &mut agent, action) == StepOutcome::HitLava {
                continue;
            }
            if !seen.insert(BoardKey::of(&grid, &agent)) {
                continue;
            }
            let solved = is_solved(&grid, &agent);
            nodes.push(Node {
                grid,
                agent,
                parent: Some((idx, action)),
            });
            let next = nodes.len() - 1;
            if solved {
                return Some(unwind(&nodes, next));
            }
            frontier.push_back(next);
        }
    }
    None
}

/// Walk `parent` links from `idx` back to the root and reverse into a plan.
fn unwind(nodes: &[Node], mut idx: usize) -> Vec<GridAction> {
    let mut plan = Vec::new();
    while let Some((parent, action)) = nodes[idx].parent {
        plan.push(action);
        idx = parent;
    }
    plan.reverse();
    plan
}

/// Success predicate: the agent is standing on the goal cell.
///
/// [`apply_action`] moves the agent *onto* [`Entity::Goal`] without clearing the
/// cell, so reading the agent's own cell back is the honest test.
pub fn on_goal(grid: &Grid, agent: &AgentState) -> bool {
    grid.get(agent.x, agent.y) == Entity::Goal
}

/// Success predicate factory: the agent has `target` in hand.
pub fn carrying(target: Entity) -> impl Fn(&Grid, &AgentState) -> bool {
    move |_grid, agent| agent.carrying == Some(target)
}

/// Self-tests for the planner itself.
///
/// These are the negative controls the oracle tests structurally cannot provide:
/// every oracle asserts `plan(..).is_some()`, so a planner that always returned
/// `Some` — or one that happily walked over lava — would pass all of them. The
/// synthetic boards below are the one place in this test tree where literal
/// coordinates are appropriate, because the board *is* the fixture rather than
/// an environment's layout being read back.
///
/// Cargo compiles integration-test targets with `--test`, so `cfg(test)` holds
/// here and these run inside whichever binary declares `mod common;`.
#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_environments::grids::core::DoorState;
    use rlevo_environments::grids::core::color::Color;

    /// A 5×5 walled room with the agent at `(1, 1)` facing East.
    fn room() -> GridState {
        let mut grid = Grid::new(5, 5);
        grid.draw_walls();
        GridState::new(grid, AgentState::new(1, 1, Direction::East))
    }

    #[test]
    fn reports_none_when_the_goal_is_walled_off() {
        let mut state = room();
        // Seal (3, 1..=3) into its own cell and drop the goal inside it.
        state.grid.set(2, 1, Entity::Wall);
        state.grid.set(2, 2, Entity::Wall);
        state.grid.set(2, 3, Entity::Wall);
        state.grid.set(3, 2, Entity::Wall);
        state.grid.set(3, 1, Entity::Wall);
        state.grid.set(3, 3, Entity::Goal);
        assert!(
            plan(&state, on_goal).is_none(),
            "an unreachable goal must yield no plan; a planner that always \
             answers `Some` would pass every oracle in this crate"
        );
    }

    #[test]
    fn refuses_to_route_through_lava() {
        let mut state = room();
        // Lava fills the only column between the agent and the goal.
        for y in 1..4 {
            state.grid.set(2, y, Entity::Lava);
        }
        state.grid.set(3, 3, Entity::Goal);
        assert!(
            plan(&state, on_goal).is_none(),
            "lava is terminal with reward 0.0, so it is not a corridor"
        );
    }

    #[test]
    fn finds_a_route_and_the_route_replays() {
        let mut state = room();
        state.grid.set(3, 3, Entity::Goal);
        let route = plan(&state, on_goal).expect("an open room must be solvable");
        assert!(!route.is_empty(), "the agent does not start on the goal");

        let (mut grid, mut agent) = (state.grid.clone(), state.agent);
        for action in route {
            apply_action(&mut grid, &mut agent, action);
        }
        assert!(
            on_goal(&grid, &agent),
            "replaying the plan must land on the goal"
        );
    }

    #[test]
    fn plans_the_pickup_ordering_a_full_hand_forbids() {
        let mut state = room();
        // Both objects sit in front of the agent's start row; `pickup` is a no-op
        // with a full hand, so reaching the ball requires dropping the key.
        state.grid.set(2, 1, Entity::Key(Color::Yellow));
        state.grid.set(2, 2, Entity::Ball(Color::Green));
        let route = plan(&state, carrying(Entity::Ball(Color::Green)))
            .expect("the ball is reachable once the hand is free");

        let (mut grid, mut agent) = (state.grid.clone(), state.agent);
        for action in route {
            apply_action(&mut grid, &mut agent, action);
        }
        assert_eq!(
            agent.carrying,
            Some(Entity::Ball(Color::Green)),
            "the plan must end with the ball in hand"
        );
    }

    #[test]
    fn plans_the_unlock_ordering_for_a_locked_door() {
        let mut state = room();
        state.grid.set(2, 1, Entity::Wall);
        state.grid.set(2, 3, Entity::Wall);
        state
            .grid
            .set(2, 2, Entity::Door(Color::Blue, DoorState::Locked));
        state.grid.set(1, 2, Entity::Key(Color::Blue));
        state.grid.set(3, 3, Entity::Goal);
        let route = plan(&state, on_goal).expect("key → unlock → open → goal is a route");

        let (mut grid, mut agent) = (state.grid.clone(), state.agent);
        for action in route {
            apply_action(&mut grid, &mut agent, action);
        }
        assert!(
            on_goal(&grid, &agent),
            "the plan must get through the locked door"
        );
    }
}
