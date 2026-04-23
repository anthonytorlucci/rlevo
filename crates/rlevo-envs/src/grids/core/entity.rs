//! World-object types that populate grid cells.

use super::color::Color;
use serde::{Deserialize, Serialize};

/// Open/closed/locked state of a door.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DoorState {
    Open,
    #[default]
    Closed,
    Locked,
}

impl DoorState {
    /// Encode as a byte for observation channels.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Open => 0,
            Self::Closed => 1,
            Self::Locked => 2,
        }
    }

    /// A door only blocks movement when it is not open.
    #[must_use]
    pub const fn is_passable(self) -> bool {
        matches!(self, Self::Open)
    }
}

/// Any object that can occupy a grid cell.
///
/// Box contents are intentionally not stored here — environments that need
/// containers can track them in a side table keyed by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Entity {
    /// Empty walkable cell.
    #[default]
    Empty,
    /// Impassable wall.
    Wall,
    /// Walkable floor (semantically identical to [`Empty`] but drawn
    /// differently by the renderer).
    ///
    /// [`Empty`]: Self::Empty
    Floor,
    /// Terminal goal cell; stepping here wins the episode.
    Goal,
    /// Hazard cell; stepping here ends the episode in failure.
    Lava,
    /// Door of the given color and state.
    Door(Color, DoorState),
    /// Colored key. Picking one up enables unlocking matching doors.
    Key(Color),
    /// Colored ball (pickable, pushable).
    Ball(Color),
    /// Colored box (pickable container).
    Box(Color),
}

impl Entity {
    /// Whether the agent can walk onto this cell.
    #[must_use]
    pub const fn is_passable(self) -> bool {
        match self {
            Self::Empty | Self::Floor | Self::Goal | Self::Lava => true,
            Self::Door(_, state) => state.is_passable(),
            Self::Wall | Self::Key(_) | Self::Ball(_) | Self::Box(_) => false,
        }
    }

    /// Whether the agent can pick this entity up with [`Pickup`].
    ///
    /// [`Pickup`]: super::action::GridAction::Pickup
    #[must_use]
    pub const fn is_pickable(self) -> bool {
        matches!(self, Self::Key(_) | Self::Ball(_) | Self::Box(_))
    }

    /// Entity-type byte for observation channel 0.
    ///
    /// `0` is reserved for [`Empty`]; every other variant maps to a
    /// distinct positive byte.
    ///
    /// [`Empty`]: Self::Empty
    #[must_use]
    pub const fn type_u8(self) -> u8 {
        match self {
            Self::Empty => 0,
            Self::Wall => 1,
            Self::Floor => 2,
            Self::Goal => 3,
            Self::Lava => 4,
            Self::Door(_, _) => 5,
            Self::Key(_) => 6,
            Self::Ball(_) => 7,
            Self::Box(_) => 8,
        }
    }

    /// Color byte for observation channel 1. Returns `0` for entities that
    /// do not carry a color.
    #[must_use]
    pub const fn color_u8(self) -> u8 {
        match self {
            Self::Door(c, _) | Self::Key(c) | Self::Ball(c) | Self::Box(c) => c.to_u8(),
            _ => 0,
        }
    }

    /// Entity-state byte for observation channel 2. Currently only
    /// meaningful for doors.
    #[must_use]
    pub const fn state_u8(self) -> u8 {
        match self {
            Self::Door(_, s) => s.to_u8(),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passability_matches_spec() {
        assert!(Entity::Empty.is_passable());
        assert!(Entity::Floor.is_passable());
        assert!(Entity::Goal.is_passable());
        assert!(Entity::Lava.is_passable());
        assert!(!Entity::Wall.is_passable());
        assert!(!Entity::Key(Color::Yellow).is_passable());
        assert!(!Entity::Ball(Color::Red).is_passable());
        assert!(!Entity::Box(Color::Green).is_passable());
        assert!(Entity::Door(Color::Blue, DoorState::Open).is_passable());
        assert!(!Entity::Door(Color::Blue, DoorState::Closed).is_passable());
        assert!(!Entity::Door(Color::Blue, DoorState::Locked).is_passable());
    }

    #[test]
    fn only_objects_are_pickable() {
        assert!(Entity::Key(Color::Yellow).is_pickable());
        assert!(Entity::Ball(Color::Red).is_pickable());
        assert!(Entity::Box(Color::Green).is_pickable());
        assert!(!Entity::Empty.is_pickable());
        assert!(!Entity::Wall.is_pickable());
        assert!(!Entity::Goal.is_pickable());
        assert!(!Entity::Lava.is_pickable());
        assert!(!Entity::Door(Color::Red, DoorState::Open).is_pickable());
    }

    #[test]
    fn type_u8_is_unique_per_variant() {
        let codes = [
            Entity::Empty.type_u8(),
            Entity::Wall.type_u8(),
            Entity::Floor.type_u8(),
            Entity::Goal.type_u8(),
            Entity::Lava.type_u8(),
            Entity::Door(Color::Red, DoorState::Open).type_u8(),
            Entity::Key(Color::Red).type_u8(),
            Entity::Ball(Color::Red).type_u8(),
            Entity::Box(Color::Red).type_u8(),
        ];
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len());
    }

    #[test]
    fn door_channel_bytes_reflect_state_and_color() {
        let e = Entity::Door(Color::Blue, DoorState::Locked);
        assert_eq!(e.color_u8(), Color::Blue.to_u8());
        assert_eq!(e.state_u8(), DoorState::Locked.to_u8());
    }

    #[test]
    fn non_door_has_zero_state() {
        assert_eq!(Entity::Key(Color::Red).state_u8(), 0);
        assert_eq!(Entity::Wall.state_u8(), 0);
    }
}
