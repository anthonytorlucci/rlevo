//! Cardinal direction the agent is facing.
//!
//! World coordinates are oriented with `+x` pointing right (East) and
//! `+y` pointing *down* (South). `North` is `-y`, matching the convention
//! used by every other tile in this module.

use serde::{Deserialize, Serialize};

/// Cardinal direction the agent is facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Direction {
    #[default]
    East,
    South,
    West,
    North,
}

impl Direction {
    /// Rotate 90° counter-clockwise.
    #[must_use]
    pub const fn left(self) -> Self {
        match self {
            Self::East => Self::North,
            Self::North => Self::West,
            Self::West => Self::South,
            Self::South => Self::East,
        }
    }

    /// Rotate 90° clockwise.
    #[must_use]
    pub const fn right(self) -> Self {
        match self {
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
            Self::North => Self::East,
        }
    }

    /// `(dx, dy)` unit vector one step in this direction.
    ///
    /// `+x` is East, `+y` is South (world y increases downward).
    #[must_use]
    pub const fn delta(self) -> (i32, i32) {
        match self {
            Self::East => (1, 0),
            Self::South => (0, 1),
            Self::West => (-1, 0),
            Self::North => (0, -1),
        }
    }

    /// Encode as a compact byte for observation channels.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::East => 0,
            Self::South => 1,
            Self::West => 2,
            Self::North => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Direction; 4] = [
        Direction::East,
        Direction::South,
        Direction::West,
        Direction::North,
    ];

    #[test]
    fn left_is_inverse_of_right() {
        for d in ALL {
            assert_eq!(d.left().right(), d);
            assert_eq!(d.right().left(), d);
        }
    }

    #[test]
    fn four_turns_return_to_start() {
        for d in ALL {
            assert_eq!(d.left().left().left().left(), d);
            assert_eq!(d.right().right().right().right(), d);
        }
    }

    #[test]
    fn delta_is_unit_vector() {
        for d in ALL {
            let (dx, dy) = d.delta();
            assert_eq!(dx.abs() + dy.abs(), 1);
        }
    }

    #[test]
    fn opposite_direction_delta_negates() {
        assert_eq!(Direction::East.delta(), (1, 0));
        assert_eq!(Direction::West.delta(), (-1, 0));
        assert_eq!(Direction::North.delta(), (0, -1));
        assert_eq!(Direction::South.delta(), (0, 1));
    }
}
