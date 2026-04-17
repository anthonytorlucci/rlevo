//! Discrete action type for LunarLander (design decision D1).

use evorl_core::action::DiscreteAction;
use evorl_core::base::Action;
use serde::{Deserialize, Serialize};

/// 4-way discrete action for [`super::LunarLanderDiscrete`] (D1).
///
/// Maps to engine firing commands:
/// - `DoNothing`  (0): no thrust
/// - `LeftEngine` (1): fire left side engine
/// - `MainEngine` (2): fire main (downward) engine
/// - `RightEngine`(3): fire right side engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LunarLanderDiscreteAction {
    DoNothing = 0,
    LeftEngine = 1,
    MainEngine = 2,
    RightEngine = 3,
}

impl Action<1> for LunarLanderDiscreteAction {
    fn shape() -> [usize; 1] {
        [4]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for LunarLanderDiscreteAction {
    const ACTION_COUNT: usize = 4;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::DoNothing,
            1 => Self::LeftEngine,
            2 => Self::MainEngine,
            3 => Self::RightEngine,
            _ => panic!("LunarLanderDiscreteAction index out of range: {index}"),
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
    fn test_action_count() {
        assert_eq!(LunarLanderDiscreteAction::ACTION_COUNT, 4);
    }

    #[test]
    fn test_shape() {
        assert_eq!(LunarLanderDiscreteAction::shape(), [4]);
    }

    #[test]
    fn test_roundtrip() {
        for i in 0..4 {
            let a = LunarLanderDiscreteAction::from_index(i);
            assert_eq!(a.to_index(), i);
        }
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_panics() {
        LunarLanderDiscreteAction::from_index(4);
    }
}
