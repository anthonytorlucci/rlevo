//! The 7-action discrete action space shared by every grid environment.

use rlevo_core::action::DiscreteAction;
use rlevo_core::base::Action;

/// Actions available to a grid agent. Matches Minigrid's canonical 7-action set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GridAction {
    /// Rotate 90° counter-clockwise.
    TurnLeft = 0,
    /// Rotate 90° clockwise.
    TurnRight = 1,
    /// Move one cell forward if passable.
    Forward = 2,
    /// Pick up the object directly in front, if any.
    Pickup = 3,
    /// Drop the carried object in the cell directly in front.
    Drop = 4,
    /// Toggle (open/close/unlock) the object in front.
    Toggle = 5,
    /// Signal episode end (used by mission-conditioned environments).
    Done = 6,
}

impl Action<1> for GridAction {
    fn shape() -> [usize; 1] {
        [Self::ACTION_COUNT]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for GridAction {
    const ACTION_COUNT: usize = 7;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::TurnLeft,
            1 => Self::TurnRight,
            2 => Self::Forward,
            3 => Self::Pickup,
            4 => Self::Drop,
            5 => Self::Toggle,
            6 => Self::Done,
            _ => panic!("GridAction index out of bounds: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        *self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_count_is_seven() {
        assert_eq!(GridAction::ACTION_COUNT, 7);
    }

    #[test]
    fn index_roundtrip_all_variants() {
        for i in 0..GridAction::ACTION_COUNT {
            let a = GridAction::from_index(i);
            assert_eq!(a.to_index(), i);
        }
    }

    #[test]
    fn enumerate_returns_all_actions() {
        let all = GridAction::enumerate();
        assert_eq!(all.len(), GridAction::ACTION_COUNT);
    }

    #[test]
    fn every_action_is_valid() {
        for a in GridAction::enumerate() {
            assert!(a.is_valid());
        }
    }

    #[test]
    #[should_panic(expected = "GridAction index out of bounds")]
    fn from_index_out_of_bounds_panics() {
        let _ = GridAction::from_index(7);
    }

    #[test]
    fn shape_matches_action_count() {
        assert_eq!(
            <GridAction as Action<1>>::shape(),
            [GridAction::ACTION_COUNT]
        );
    }
}
