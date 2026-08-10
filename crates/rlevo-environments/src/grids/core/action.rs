//! The 7-action discrete action space shared by every grid environment.

use rlevo_core::action::{DiscreteAction, InvalidActionError};
use rlevo_core::base::Action;
use serde::{Deserialize, Serialize};

/// Actions available to a grid agent. Matches Minigrid's canonical 7-action set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

// The explicit discriminants above are a wire format: they are the indices a
// policy head emits and the values `to_index` returns. Pin them at compile
// time so a reordering of the variants cannot silently renumber the action
// space. The final assertion ties the enum's cardinality to `ACTION_COUNT`, so
// adding a variant without bumping the constant fails to compile.
const _: () = {
    assert!(GridAction::TurnLeft as usize == 0);
    assert!(GridAction::TurnRight as usize == 1);
    assert!(GridAction::Forward as usize == 2);
    assert!(GridAction::Pickup as usize == 3);
    assert!(GridAction::Drop as usize == 4);
    assert!(GridAction::Toggle as usize == 5);
    assert!(GridAction::Done as usize == 6);
    assert!(GridAction::Done as usize + 1 == <GridAction as DiscreteAction<1>>::ACTION_COUNT);
};

impl GridAction {
    /// Constructs an action from its zero-based index, returning an error if
    /// `index` is out of range.
    ///
    /// This is the fallible counterpart to
    /// [`DiscreteAction::from_index`], for callers whose index comes from data
    /// (a config file, a replay log, a deserialized trajectory) rather than
    /// from a trusted in-range computation.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidActionError`] if `index >= ACTION_COUNT`; the message
    /// names the offending index and the valid range.
    pub fn try_from_index(index: usize) -> Result<Self, InvalidActionError> {
        match index {
            0 => Ok(Self::TurnLeft),
            1 => Ok(Self::TurnRight),
            2 => Ok(Self::Forward),
            3 => Ok(Self::Pickup),
            4 => Ok(Self::Drop),
            5 => Ok(Self::Toggle),
            6 => Ok(Self::Done),
            _ => Err(InvalidActionError {
                message: format!(
                    "GridAction index {index} out of range [0, {})",
                    Self::ACTION_COUNT
                ),
            }),
        }
    }
}

impl TryFrom<usize> for GridAction {
    type Error = InvalidActionError;

    /// Delegates to [`GridAction::try_from_index`].
    ///
    /// # Errors
    ///
    /// Returns [`InvalidActionError`] if `index >= ACTION_COUNT`.
    fn try_from(index: usize) -> Result<Self, Self::Error> {
        Self::try_from_index(index)
    }
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

    /// Constructs an action from its zero-based index.
    ///
    /// Use [`Self::try_from_index`] (or the equivalent [`TryFrom<usize>`]
    /// implementation) when the index originates from data rather than from a
    /// trusted in-range computation such as an argmax over `ACTION_COUNT`
    /// logits.
    ///
    /// # Panics
    ///
    /// Panics if `index >= ACTION_COUNT`, per the
    /// [`DiscreteAction::from_index`] contract.
    fn from_index(index: usize) -> Self {
        // The error's own `Display` repeats the index and prefixes "Invalid
        // action:", both of which would read as noise after this panic's
        // pinned prefix; only the valid range is new information.
        Self::try_from_index(index).unwrap_or_else(|_| {
            panic!(
                "GridAction index out of bounds: {index}, valid range is [0, {})",
                Self::ACTION_COUNT
            )
        })
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
    fn index_bindings_are_canonical() {
        // Literal, not a loop: `index_roundtrip_all_variants` is self-referential
        // and still passes if the variant order is shuffled. These pin each
        // index to a named variant.
        assert_eq!(GridAction::from_index(0), GridAction::TurnLeft);
        assert_eq!(GridAction::from_index(1), GridAction::TurnRight);
        assert_eq!(GridAction::from_index(2), GridAction::Forward);
        assert_eq!(GridAction::from_index(3), GridAction::Pickup);
        assert_eq!(GridAction::from_index(4), GridAction::Drop);
        assert_eq!(GridAction::from_index(5), GridAction::Toggle);
        assert_eq!(GridAction::from_index(6), GridAction::Done);
    }

    #[test]
    fn to_index_values_are_canonical() {
        assert_eq!(GridAction::TurnLeft.to_index(), 0);
        assert_eq!(GridAction::TurnRight.to_index(), 1);
        assert_eq!(GridAction::Forward.to_index(), 2);
        assert_eq!(GridAction::Pickup.to_index(), 3);
        assert_eq!(GridAction::Drop.to_index(), 4);
        assert_eq!(GridAction::Toggle.to_index(), 5);
        assert_eq!(GridAction::Done.to_index(), 6);
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

    // A separate test from the one above: a `#[should_panic]` body unwinds at
    // the first panic, so one body cannot exercise both inputs.
    #[test]
    #[should_panic(expected = "GridAction index out of bounds")]
    fn from_index_usize_max_panics() {
        let _ = GridAction::from_index(usize::MAX);
    }

    #[test]
    fn try_from_index_accepts_every_valid_index() {
        for i in 0..GridAction::ACTION_COUNT {
            assert_eq!(GridAction::try_from_index(i), Ok(GridAction::from_index(i)));
        }
    }

    #[test]
    fn try_from_index_rejects_out_of_range() {
        let err = GridAction::try_from_index(7).expect_err("index 7 is out of range");
        assert!(err.to_string().contains("index 7"), "message was {err}");

        let err = GridAction::try_from_index(usize::MAX).expect_err("usize::MAX is out of range");
        assert!(
            err.to_string().contains(&usize::MAX.to_string()),
            "message was {err}"
        );
    }

    #[test]
    fn try_from_matches_inherent_method() {
        assert_eq!(GridAction::try_from(3usize), GridAction::try_from_index(3));
        assert_eq!(GridAction::try_from(3usize), Ok(GridAction::Pickup));
        assert_eq!(
            GridAction::try_from(usize::MAX),
            GridAction::try_from_index(usize::MAX)
        );
        assert!(GridAction::try_from(7usize).is_err());
    }

    #[test]
    fn shape_matches_action_count() {
        assert_eq!(
            <GridAction as Action<1>>::shape(),
            [GridAction::ACTION_COUNT]
        );
    }
}
