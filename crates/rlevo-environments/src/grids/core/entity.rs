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

    /// Whether the agent's line of sight passes *through* this cell.
    ///
    /// This is the rlevo spelling of canonical Minigrid's
    /// `WorldObj.see_behind()`, the sole predicate driving the shadow cast in
    /// `grid::process_vis`. Canonical's base
    /// implementation returns `true` and exactly two subclasses override it:
    /// `Wall` returns `false`, and `Door` returns `self.is_open`. Everything
    /// else — `Goal`, `Floor`, `Lava`, `Key`, `Ball`, `Box` — inherits the
    /// transparent default.
    ///
    /// # This is deliberately *not* [`is_passable`](Self::is_passable)
    ///
    /// The two predicates answer different questions and disagree on four
    /// variants. A key, ball, or box blocks movement but not sight; lava
    /// permits movement (fatally) and also sight. Defining transparency in
    /// terms of passability would make an agent blind to everything behind a
    /// ball on the floor, which canonical Minigrid does not do — and it would
    /// compile, look plausible, and only show up as a silently harder task.
    ///
    /// | Variant | `is_passable` | `see_behind` |
    /// |---------|---------------|--------------|
    /// | `Empty` / `Floor` / `Goal` / `Lava` | `true` | `true` |
    /// | `Key` / `Ball` / `Box` | `false` | **`true`** |
    /// | `Wall` | `false` | `false` |
    /// | `Door(_, Open)` | `true` | `true` |
    /// | `Door(_, Closed \| Locked)` | `false` | `false` |
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_environments::grids::core::{Color, DoorState, Entity};
    ///
    /// assert!(!Entity::Wall.see_behind());
    /// assert!(Entity::Lava.see_behind());
    /// assert!(Entity::Ball(Color::Red).see_behind());
    /// assert!(Entity::Door(Color::Blue, DoorState::Open).see_behind());
    /// assert!(!Entity::Door(Color::Blue, DoorState::Closed).see_behind());
    /// ```
    #[must_use]
    pub const fn see_behind(self) -> bool {
        match self {
            // Canonical `Wall.see_behind()` -> False.
            Self::Wall => false,
            // Canonical `Door.see_behind()` -> `self.is_open`. For a
            // `DoorState` alone, "passable" and "open" are the same predicate
            // (`DoorState::is_passable` is `matches!(self, Open)`); it is only
            // at the `Entity` level that passability and transparency diverge.
            Self::Door(_, state) => state.is_passable(),
            // Canonical's base `WorldObj.see_behind()` -> True. Listed
            // exhaustively rather than with `_` so that a new entity variant
            // must make a deliberate transparency choice.
            Self::Empty
            | Self::Floor
            | Self::Goal
            | Self::Lava
            | Self::Key(_)
            | Self::Ball(_)
            | Self::Box(_) => true,
        }
    }

    /// Entity-type byte for observation channel 0, in canonical Minigrid's
    /// `OBJECT_TO_IDX` numbering.
    ///
    /// **`0` is never returned.** It is reserved for
    /// [`UNSEEN_TYPE`](super::observation::UNSEEN_TYPE), the byte a cell the
    /// agent cannot see carries; every `Entity` maps to a distinct byte in
    /// `1..=9`.
    ///
    /// | Byte | Meaning |
    /// |------|---------|
    /// | 0 | *unseen* — not an `Entity`; see [`UNSEEN_TYPE`](super::observation::UNSEEN_TYPE) |
    /// | 1 | [`Empty`](Self::Empty) |
    /// | 2 | [`Wall`](Self::Wall) |
    /// | 3 | [`Floor`](Self::Floor) |
    /// | 4 | [`Door`](Self::Door) |
    /// | 5 | [`Key`](Self::Key) |
    /// | 6 | [`Ball`](Self::Ball) |
    /// | 7 | [`Box`](Self::Box) |
    /// | 8 | [`Goal`](Self::Goal) |
    /// | 9 | [`Lava`](Self::Lava) |
    ///
    /// This is `OBJECT_TO_IDX` from `minigrid/core/constants.py` verbatim, for
    /// every key `Entity` has (ADR 0063 Decision 4). Two reasons it is not an
    /// arbitrary ordering:
    ///
    /// - **Parity.** ADR 0043 treats Minigrid as the authoritative reference for
    ///   this family. A numbering that *resembles* canonical's while silently
    ///   disagreeing — rlevo's pre-ADR-0063 table, which had `Goal = 3` and
    ///   `Lava = 4` — is worse than either matching it or making no claim at
    ///   all, because it invites a reader to assume an equivalence that is not
    ///   there.
    /// - **Zero means unknown.** With `0` reserved, an all-zero observation
    ///   tensor decodes as *every cell unseen* — the correct reading of a
    ///   zero-padded or attention-masked sequence, and what the `TensorConvertible`
    ///   no-fabrication clause requires. Were `Empty` still `0`, the same tensor
    ///   would decode as *every cell confirmed empty floor*: a materially
    ///   different and false claim, in exactly the recurrent-POMDP workload these
    ///   environments exist to train.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_environments::grids::core::{Color, DoorState, Entity, UNSEEN_TYPE};
    ///
    /// assert_eq!(Entity::Empty.type_u8(), 1);
    /// assert_eq!(Entity::Wall.type_u8(), 2);
    /// assert_eq!(Entity::Goal.type_u8(), 8);
    /// assert_eq!(Entity::Lava.type_u8(), 9);
    ///
    /// // No entity is ever confusable with a masked cell.
    /// assert_ne!(Entity::Empty.type_u8(), UNSEEN_TYPE);
    /// ```
    #[must_use]
    pub const fn type_u8(self) -> u8 {
        match self {
            Self::Empty => 1,
            Self::Wall => 2,
            Self::Floor => 3,
            Self::Door(_, _) => 4,
            Self::Key(_) => 5,
            Self::Ball(_) => 6,
            Self::Box(_) => 7,
            Self::Goal => 8,
            Self::Lava => 9,
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
    use super::super::observation::UNSEEN_TYPE;
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
    fn only_walls_and_shut_doors_block_sight() {
        // Canonical Minigrid overrides `see_behind` on exactly two classes.
        assert!(!Entity::Wall.see_behind(), "a wall is opaque");
        assert!(
            !Entity::Door(Color::Blue, DoorState::Closed).see_behind(),
            "a closed door is opaque"
        );
        assert!(
            !Entity::Door(Color::Blue, DoorState::Locked).see_behind(),
            "a locked door is opaque"
        );
        assert!(
            Entity::Door(Color::Blue, DoorState::Open).see_behind(),
            "an open door is transparent"
        );

        // Everything else inherits the transparent base implementation.
        for transparent in [
            Entity::Empty,
            Entity::Floor,
            Entity::Goal,
            Entity::Lava,
            Entity::Key(Color::Yellow),
            Entity::Ball(Color::Red),
            Entity::Box(Color::Green),
        ] {
            assert!(
                transparent.see_behind(),
                "{transparent:?} does not override canonical WorldObj.see_behind"
            );
        }
    }

    #[test]
    fn see_behind_is_not_passability() {
        // The four variants where the two predicates disagree. Conflating them
        // is the single most likely way to mis-port the shadow cast, and it
        // would compile.
        assert!(Entity::Lava.is_passable(), "lava is walkable (fatally)");
        assert!(Entity::Lava.see_behind(), "lava is also transparent");
        for blocking_but_transparent in [
            Entity::Key(Color::Yellow),
            Entity::Ball(Color::Red),
            Entity::Box(Color::Green),
        ] {
            assert!(
                !blocking_but_transparent.is_passable(),
                "{blocking_but_transparent:?} blocks movement"
            );
            assert!(
                blocking_but_transparent.see_behind(),
                "{blocking_but_transparent:?} must NOT block sight — see_behind is \
                 not is_passable"
            );
        }
    }

    /// Every variant, paired with its canonical `OBJECT_TO_IDX` byte.
    ///
    /// Written out as literals rather than derived from `type_u8` so the table
    /// is pinned against the upstream constant, not against itself.
    const CANONICAL_TABLE: [(Entity, u8); 9] = [
        (Entity::Empty, 1),
        (Entity::Wall, 2),
        (Entity::Floor, 3),
        (Entity::Door(Color::Red, DoorState::Open), 4),
        (Entity::Key(Color::Red), 5),
        (Entity::Ball(Color::Red), 6),
        (Entity::Box(Color::Red), 7),
        (Entity::Goal, 8),
        (Entity::Lava, 9),
    ];

    #[test]
    fn type_u8_matches_canonical_object_to_idx() {
        // `minigrid/core/constants.py`: unseen=0, empty=1, wall=2, floor=3,
        // door=4, key=5, ball=6, box=7, goal=8, lava=9 (ADR 0063 Decision 4).
        for (entity, expected) in CANONICAL_TABLE {
            assert_eq!(
                entity.type_u8(),
                expected,
                "{entity:?} must carry canonical OBJECT_TO_IDX byte {expected}"
            );
        }
    }

    #[test]
    fn type_u8_is_unique_per_variant() {
        let codes: Vec<u8> = CANONICAL_TABLE.iter().map(|&(e, _)| e.type_u8()).collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len());
    }

    #[test]
    fn no_entity_claims_the_unseen_byte() {
        // The load-bearing property: `0` means *unknown*, so it must not be
        // reachable from any entity. If it ever is, a masked cell and a real
        // cell encode identically and the whole shadow cast is invisible to a
        // policy at that cell (ADR 0063 Decision 4).
        for (entity, _) in CANONICAL_TABLE {
            assert_ne!(
                entity.type_u8(),
                UNSEEN_TYPE,
                "{entity:?} collides with the reserved unseen byte"
            );
        }
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
