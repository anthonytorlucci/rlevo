//! Shared grid mechanics: every environment's `step()` delegates to
//! [`apply_action`], then maps the returned [`StepOutcome`] to its own
//! reward / termination logic.

use super::action::GridAction;
use super::agent::AgentState;
use super::entity::{DoorState, Entity};
use super::grid::Grid;

/// Classification of what happened when an action was applied to the grid.
///
/// Environments consume this to decide reward and termination — the
/// dynamics layer intentionally knows nothing about either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// Agent moved onto an ordinary walkable cell.
    Moved,
    /// `Forward` blocked by a wall, closed door, or object.
    Bumped,
    /// Agent picked up the entity in front (now stored in `agent.carrying`).
    PickedUp(Entity),
    /// Agent dropped the carried entity into the cell in front.
    Dropped,
    /// Door opened or closed via `Toggle`.
    Toggled,
    /// Locked door was unlocked (moved from `Locked` to `Closed`).
    Unlocked,
    /// Agent stepped onto lava — typically terminal.
    HitLava,
    /// Agent stepped onto the goal — typically terminal.
    ReachedGoal,
    /// Agent issued the `Done` action.
    DoneAction,
    /// The action had no effect (e.g. pickup with empty hand over empty cell).
    NoOp,
}

/// Apply `action` to `(grid, agent)`, mutating both in place, and return
/// a classification of what happened.
pub fn apply_action(grid: &mut Grid, agent: &mut AgentState, action: GridAction) -> StepOutcome {
    match action {
        GridAction::TurnLeft => {
            agent.direction = agent.direction.left();
            StepOutcome::NoOp
        }
        GridAction::TurnRight => {
            agent.direction = agent.direction.right();
            StepOutcome::NoOp
        }
        GridAction::Forward => step_forward(grid, agent),
        GridAction::Pickup => pickup(grid, agent),
        GridAction::Drop => drop_item(grid, agent),
        GridAction::Toggle => toggle(grid, agent),
        GridAction::Done => StepOutcome::DoneAction,
    }
}

fn step_forward(grid: &Grid, agent: &mut AgentState) -> StepOutcome {
    let (fx, fy) = agent.front();
    let front = grid.get(fx, fy);
    if !front.is_passable() {
        return StepOutcome::Bumped;
    }
    agent.x = fx;
    agent.y = fy;
    match front {
        Entity::Goal => StepOutcome::ReachedGoal,
        Entity::Lava => StepOutcome::HitLava,
        _ => StepOutcome::Moved,
    }
}

fn pickup(grid: &mut Grid, agent: &mut AgentState) -> StepOutcome {
    if agent.carrying.is_some() {
        return StepOutcome::NoOp;
    }
    let (fx, fy) = agent.front();
    if !grid.in_bounds(fx, fy) {
        return StepOutcome::NoOp;
    }
    let front = grid.get(fx, fy);
    if front.is_pickable() {
        agent.carrying = Some(front);
        grid.set(fx, fy, Entity::Empty);
        StepOutcome::PickedUp(front)
    } else {
        StepOutcome::NoOp
    }
}

fn drop_item(grid: &mut Grid, agent: &mut AgentState) -> StepOutcome {
    let Some(item) = agent.carrying else {
        return StepOutcome::NoOp;
    };
    let (fx, fy) = agent.front();
    if !grid.in_bounds(fx, fy) {
        return StepOutcome::NoOp;
    }
    let front = grid.get(fx, fy);
    if matches!(front, Entity::Empty | Entity::Floor) {
        grid.set(fx, fy, item);
        agent.carrying = None;
        StepOutcome::Dropped
    } else {
        StepOutcome::NoOp
    }
}

fn toggle(grid: &mut Grid, agent: &mut AgentState) -> StepOutcome {
    let (fx, fy) = agent.front();
    if !grid.in_bounds(fx, fy) {
        return StepOutcome::NoOp;
    }
    let front = grid.get(fx, fy);
    match front {
        Entity::Door(color, DoorState::Locked) => {
            if agent.carrying == Some(Entity::Key(color)) {
                grid.set(fx, fy, Entity::Door(color, DoorState::Closed));
                StepOutcome::Unlocked
            } else {
                StepOutcome::NoOp
            }
        }
        Entity::Door(color, DoorState::Closed) => {
            grid.set(fx, fy, Entity::Door(color, DoorState::Open));
            StepOutcome::Toggled
        }
        Entity::Door(color, DoorState::Open) => {
            grid.set(fx, fy, Entity::Door(color, DoorState::Closed));
            StepOutcome::Toggled
        }
        _ => StepOutcome::NoOp,
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::Color;
    use super::super::direction::Direction;
    use super::*;

    fn setup() -> (Grid, AgentState) {
        let mut g = Grid::new(5, 5);
        g.draw_walls();
        let a = AgentState::new(1, 1, Direction::East);
        (g, a)
    }

    #[test]
    fn turn_left_rotates_ccw() {
        let (mut g, mut a) = setup();
        apply_action(&mut g, &mut a, GridAction::TurnLeft);
        assert_eq!(a.direction, Direction::North);
    }

    #[test]
    fn turn_right_rotates_cw() {
        let (mut g, mut a) = setup();
        apply_action(&mut g, &mut a, GridAction::TurnRight);
        assert_eq!(a.direction, Direction::South);
    }

    #[test]
    fn forward_onto_empty_moves_agent() {
        let (mut g, mut a) = setup();
        let outcome = apply_action(&mut g, &mut a, GridAction::Forward);
        assert_eq!(outcome, StepOutcome::Moved);
        assert_eq!((a.x, a.y), (2, 1));
    }

    #[test]
    fn forward_into_wall_bumps_and_holds_position() {
        let (mut g, mut a) = setup();
        a.direction = Direction::North; // wall directly above
        let outcome = apply_action(&mut g, &mut a, GridAction::Forward);
        assert_eq!(outcome, StepOutcome::Bumped);
        assert_eq!((a.x, a.y), (1, 1));
    }

    #[test]
    fn forward_onto_goal_reports_reached_goal() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Goal);
        let outcome = apply_action(&mut g, &mut a, GridAction::Forward);
        assert_eq!(outcome, StepOutcome::ReachedGoal);
        assert_eq!((a.x, a.y), (2, 1));
    }

    #[test]
    fn forward_onto_lava_reports_hit_lava() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Lava);
        let outcome = apply_action(&mut g, &mut a, GridAction::Forward);
        assert_eq!(outcome, StepOutcome::HitLava);
    }

    #[test]
    fn pickup_grabs_key_and_clears_cell() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Key(Color::Yellow));
        let outcome = apply_action(&mut g, &mut a, GridAction::Pickup);
        assert_eq!(outcome, StepOutcome::PickedUp(Entity::Key(Color::Yellow)));
        assert_eq!(a.carrying, Some(Entity::Key(Color::Yellow)));
        assert_eq!(g.get(2, 1), Entity::Empty);
    }

    #[test]
    fn pickup_with_full_hand_is_noop() {
        let (mut g, mut a) = setup();
        a.carrying = Some(Entity::Key(Color::Red));
        g.set(2, 1, Entity::Key(Color::Yellow));
        let outcome = apply_action(&mut g, &mut a, GridAction::Pickup);
        assert_eq!(outcome, StepOutcome::NoOp);
        // Hand still holds the original key; grid still has the yellow one.
        assert_eq!(a.carrying, Some(Entity::Key(Color::Red)));
        assert_eq!(g.get(2, 1), Entity::Key(Color::Yellow));
    }

    #[test]
    fn drop_places_carried_item_in_front() {
        let (mut g, mut a) = setup();
        a.carrying = Some(Entity::Ball(Color::Green));
        let outcome = apply_action(&mut g, &mut a, GridAction::Drop);
        assert_eq!(outcome, StepOutcome::Dropped);
        assert_eq!(a.carrying, None);
        assert_eq!(g.get(2, 1), Entity::Ball(Color::Green));
    }

    #[test]
    fn drop_with_empty_hand_is_noop() {
        let (mut g, mut a) = setup();
        let outcome = apply_action(&mut g, &mut a, GridAction::Drop);
        assert_eq!(outcome, StepOutcome::NoOp);
    }

    #[test]
    fn toggle_closes_and_opens_closed_door() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Door(Color::Grey, DoorState::Closed));
        let outcome = apply_action(&mut g, &mut a, GridAction::Toggle);
        assert_eq!(outcome, StepOutcome::Toggled);
        assert_eq!(g.get(2, 1), Entity::Door(Color::Grey, DoorState::Open));
        // Toggling again closes the door.
        let outcome = apply_action(&mut g, &mut a, GridAction::Toggle);
        assert_eq!(outcome, StepOutcome::Toggled);
        assert_eq!(g.get(2, 1), Entity::Door(Color::Grey, DoorState::Closed));
    }

    #[test]
    fn toggle_locked_without_key_is_noop() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Door(Color::Blue, DoorState::Locked));
        let outcome = apply_action(&mut g, &mut a, GridAction::Toggle);
        assert_eq!(outcome, StepOutcome::NoOp);
        assert_eq!(g.get(2, 1), Entity::Door(Color::Blue, DoorState::Locked));
    }

    #[test]
    fn toggle_locked_with_matching_key_unlocks() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Door(Color::Blue, DoorState::Locked));
        a.carrying = Some(Entity::Key(Color::Blue));
        let outcome = apply_action(&mut g, &mut a, GridAction::Toggle);
        assert_eq!(outcome, StepOutcome::Unlocked);
        assert_eq!(g.get(2, 1), Entity::Door(Color::Blue, DoorState::Closed));
    }

    #[test]
    fn toggle_locked_with_wrong_key_is_noop() {
        let (mut g, mut a) = setup();
        g.set(2, 1, Entity::Door(Color::Blue, DoorState::Locked));
        a.carrying = Some(Entity::Key(Color::Red));
        let outcome = apply_action(&mut g, &mut a, GridAction::Toggle);
        assert_eq!(outcome, StepOutcome::NoOp);
    }

    #[test]
    fn done_action_reports_done_action() {
        let (mut g, mut a) = setup();
        let outcome = apply_action(&mut g, &mut a, GridAction::Done);
        assert_eq!(outcome, StepOutcome::DoneAction);
    }
}
