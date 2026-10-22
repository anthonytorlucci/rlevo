//! `$7 \times 7 \times 3$` egocentric observation emitted by every grid environment.
//!
//! The agent sits at the bottom-center of its view window and looks toward
//! the top. Every visible cell is encoded into three bytes — entity type,
//! color, and door state — laid out as a `[VIEW_SIZE][VIEW_SIZE][OBS_CHANNELS]`
//! array. The agent's facing is carried *beside* the view as a [`Direction`]
//! and is deliberately **not** written into the host row — the view is already rotated into the agent's own frame, so the
//! absolute heading is redundant for a policy (see
//! [`GridObservation::agent_direction`]).

use super::entity::Entity;
use crate::direction::Direction;
use rlevo_core::base::{HostRow, Observation};
use serde::{Deserialize, Serialize};

/// Side length (height and width) of the agent's local view window in cells.
///
/// Matches the Minigrid default of 7, giving the agent a `$7 \times 7$` field of view centered one
/// cell in front of its current position.
pub const VIEW_SIZE: usize = 7;

/// Number of per-cell encoding channels: entity type, color index, and door state.
///
/// Maps to the three `Entity` methods: [`Entity::type_u8`], [`Entity::color_u8`],
/// and [`Entity::state_u8`].
///
/// [`Entity::type_u8`]: super::entity::Entity::type_u8
/// [`Entity::color_u8`]: super::entity::Entity::color_u8
/// [`Entity::state_u8`]: super::entity::Entity::state_u8
pub const OBS_CHANNELS: usize = 3;

/// Channel-0 byte written for a cell the agent cannot see.
///
/// Canonical Minigrid's `Grid.encode` writes `[OBJECT_TO_IDX["unseen"], 0, 0]`
/// for a masked cell, and `OBJECT_TO_IDX["unseen"] == 0`.
///
/// # Zero means unknown — read this before consuming channel 0
///
/// Canonical numbers `unseen = 0` and `empty = 1` as *distinct* type indices,
/// precisely so a masked cell cannot be confused with a confirmed-empty one, and
/// [`Entity::type_u8`] now follows that table exactly (ADR 0063 Decision 4). No
/// `Entity` returns `0`, so this byte is unambiguous: a channel-0 zero means
/// *the agent could not see this cell*, never *the agent saw floor here*.
///
/// This is load-bearing rather than decorative. Eight of the twelve grid
/// environments select [`Visibility::Occluded`](super::Visibility::Occluded), so
/// masked cells are produced on every step of those environments; before the
/// renumbering, a masked `Empty` floor encoded identically to a confirmed-empty
/// one and the occlusion signal was erased wherever the hidden cells happened to
/// be empty. At `MemoryEnv`'s fork junction facing West — the pose at which the
/// agent must answer — *every* hidden cell is `Empty`, so the occluded
/// observation was byte-identical to the unoccluded one. It no longer is; see
/// `memory::tests::test_memory_env_masked_empty_cells_are_encoded_as_unseen`.
///
/// It also gives the host row the right reading at zero: an all-zero row means
/// "every cell unseen", which is what a zero-padded or attention-masked
/// sequence actually means, rather than "every cell is confirmed empty floor".
///
/// **Do not "fix" a future collision by picking a different value here.** `0` is
/// the canonical unseen index; it is `type_u8` that must stay clear of it.
///
/// [`Entity::type_u8`]: super::entity::Entity::type_u8
/// [`Entity::Empty`]: super::entity::Entity::Empty
pub const UNSEEN_TYPE: u8 = 0;

/// Egocentric observation of the `$7 \times 7$` cells around the agent.
///
/// The agent sits at view row `VIEW_SIZE - 1`, column `VIEW_SIZE / 2`, and
/// faces toward row `0`. Cells that fall outside the world decode as
/// [`Entity::Wall`]. Each cell is encoded into three bytes:
///
/// | Channel | Meaning                                            |
/// |---------|----------------------------------------------------|
/// | 0       | Entity type ([`Entity::type_u8`])                  |
/// | 1       | Color ([`Entity::color_u8`], `0` if no color)      |
/// | 2       | Door state ([`Entity::state_u8`], `0` if no state) |
///
/// # The agent's own cell carries the hand, not the terrain
///
/// `view[VIEW_SIZE - 1][VIEW_SIZE / 2]` is the agent's own cell, and it encodes
/// [`AgentState::carrying`](super::agent::AgentState::carrying) — `[1, 0, 0]`
/// (canonical `OBJECT_TO_IDX["empty"]`) when the hand is empty. It is **not**
/// the world entity the agent is standing on: that cell is overwritten by
/// `grid::stamp_carried` inside `mask_view`, upstream of this type, matching the
/// final step of canonical Minigrid's `gen_obs_grid`.
///
/// Two consequences for anyone reading channel 0 at that position:
///
/// * A [`Entity::Goal`] or [`Entity::Lava`] tile *under* the agent never
///   appears, so the terminal tile is not a readable channel on the terminal
///   frame.
/// * The cell is unmaskable — under either
///   [`Visibility`](super::Visibility) policy it is `Some`, so it never encodes
///   as [`UNSEEN_TYPE`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridObservation {
    /// Encoded view, indexed as `view[row][col][channel]`.
    pub view: [[[u8; OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE],
    /// Agent's absolute facing when the observation was produced.
    ///
    /// Typed as [`Direction`] rather than a raw byte (ADR 0061 Decision 1):
    /// the byte form let illegal values like `200` through unchecked, with
    /// no `from_u8` to decode one back. [`Direction::to_u8`] still yields
    /// the canonical Minigrid byte order for callers that want it.
    ///
    /// # Why it is not in the row
    ///
    /// The facing is deliberately **absent** from the host row
    /// ([`HostRow::write_host_row`]). [`view`](Self::view) is already
    /// rotated into the agent's own frame, so an absolute heading is
    /// decision-vestigial for a policy: canonical Minigrid ships `direction`
    /// as a separate entry of the observation dict and never inside the
    /// image, and every published baseline (`ImgObsWrapper`,
    /// `rl-starter-files`, RIDE) discards it before the network.
    pub agent_direction: Direction,
}

impl GridObservation {
    /// Encode a visibility-masked `$7 \times 7$` entity view and the agent's facing into an
    /// observation.
    ///
    /// This is the single encoder behind every grid observation. A `Some` cell
    /// is encoded from its [`Entity`]; a `None` cell — one the shadow cast in
    /// `grid::process_vis` hid — becomes the byte triple
    /// `[UNSEEN_TYPE, 0, 0]`, matching canonical Minigrid's
    /// `Grid.encode(vis_mask)`.
    ///
    /// Callers holding an unmasked view use
    /// [`from_entity_view`](Self::from_entity_view), which delegates here.
    ///
    /// # A pure encoder — it does not stamp
    ///
    /// This function encodes the `view` it is handed and nothing else. In
    /// particular it does **not** write the agent's carried item onto
    /// `view[VIEW_SIZE - 1][VIEW_SIZE / 2]`; that stamp lives upstream in
    /// `mask_view`, which applies it after the visibility policy has run (the
    /// canonical `gen_obs_grid` order). Every production path therefore arrives
    /// here with the agent's cell already carrying the hand.
    ///
    /// The split is deliberate: it keeps the encoder total over its input, so a
    /// test may pass an arbitrary entity — or `None` — at the agent's cell and
    /// get it encoded verbatim, and so the stamp has exactly one home rather
    /// than being reapplied by each of the two encoders.
    ///
    /// # Arguments
    ///
    /// * `view` — the rotated view window, `None` where the agent's sight is
    ///   blocked.
    /// * `direction` — the agent's absolute facing, carried beside the view
    ///   (see [`agent_direction`](Self::agent_direction)).
    #[must_use]
    pub fn from_masked_view(
        view: [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE],
        direction: Direction,
    ) -> Self {
        let mut encoded = [[[0u8; OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE];
        for (r, row) in view.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                encoded[r][c] = match cell {
                    Some(entity) => [entity.type_u8(), entity.color_u8(), entity.state_u8()],
                    // Canonical: `array[i, j, :] = [OBJECT_TO_IDX["unseen"], 0, 0]`.
                    None => [UNSEEN_TYPE, 0, 0],
                };
            }
        }
        Self {
            view: encoded,
            agent_direction: direction,
        }
    }

    /// Encode a fully visible `$7 \times 7$` entity view and the agent's facing into an
    /// observation.
    ///
    /// Equivalent to wrapping every cell in `Some` and calling
    /// [`from_masked_view`](Self::from_masked_view) — which is exactly what it
    /// does, so there is one encoder rather than two.
    #[must_use]
    pub fn from_entity_view(view: [[Entity; VIEW_SIZE]; VIEW_SIZE], direction: Direction) -> Self {
        Self::from_masked_view(view.map(|row| row.map(Some)), direction)
    }
}

impl Observation<3> for GridObservation {
    fn shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]
    }
}

impl HostRow<3> for GridObservation {
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        for row in &self.view {
            for cell in row {
                for &channel in cell {
                    buf.push(f32::from(channel));
                }
            }
        }
    }

    /// Always `true`: this row is structurally incapable of carrying a
    /// non-finite value (ADR 0067 §Decision 2).
    ///
    /// This override is *both* a structural assertion and a performance
    /// decision: the default body would materialize all
    /// `VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS` values as `f32` on every call to
    /// prove something the payload type already guarantees.
    fn row_is_finite(&self, _scratch: &mut Vec<f32>) -> bool {
        // Compile-time witness for the structural claim below. If the payload
        // type ever changes, THIS LINE fails to compile and the override must
        // be re-derived, not re-asserted. The ascription names the full nested
        // array, so a change to the element type, the channel count, or the
        // view extent all break it.
        let _: &[[[u8; OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE] = &self.view;
        // `u8 -> f32` is total: no element of this row can be `NaN` or `$\pm\infty$`.
        // `agent_direction` is not written into the row at all.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::Color;
    use super::super::entity::DoorState;
    use super::*;

    #[test]
    fn shape_is_7x7x3() {
        assert_eq!(
            <GridObservation as Observation<3>>::shape(),
            [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]
        );
    }

    #[test]
    fn encodes_entities_by_channel() {
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        view[6][3] = Entity::Goal;

        let obs = GridObservation::from_entity_view(view, Direction::North);

        // Literals, not `type_u8()` calls: these pin the canonical
        // `OBJECT_TO_IDX` values, so re-deriving them from the code under test
        // would make the assertion vacuous.
        assert_eq!(obs.view[0][0][0], 2); // Wall type byte
        assert_eq!(obs.view[3][3][0], 4); // Door type byte
        assert_eq!(obs.view[3][3][1], Color::Blue.to_u8());
        assert_eq!(obs.view[3][3][2], DoorState::Locked.to_u8());
        assert_eq!(obs.view[6][3][0], 8); // Goal type byte
        assert_eq!(obs.agent_direction, Direction::North);
    }

    #[test]
    fn unseen_not_empty_is_the_zero_type_byte() {
        // Was `empty_cells_encode_as_zero`, which pinned the pre-ADR-0063
        // numbering. Zero is now reserved for *unseen*: a confirmed-empty cell
        // carries `1`, and only a masked cell carries `0`. The two views below
        // are the same board under the two readings, so a regression that
        // re-collided them would fail both halves.
        let confirmed_empty = GridObservation::from_entity_view(
            [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE],
            Direction::East,
        );
        for row in &confirmed_empty.view {
            for cell in row {
                assert_eq!(
                    cell,
                    &[1, 0, 0],
                    "a seen-but-empty cell is canonical `empty` = 1, not zero"
                );
            }
        }

        let all_masked =
            GridObservation::from_masked_view([[None; VIEW_SIZE]; VIEW_SIZE], Direction::East);
        for row in &all_masked.view {
            for cell in row {
                assert_eq!(cell, &[0, 0, 0], "only an unseen cell encodes as all-zero");
            }
        }

        assert_ne!(
            confirmed_empty.view, all_masked.view,
            "an all-empty view and an all-masked view must not encode identically"
        );
    }

    /// The row is the view in `[row][col][channel]` order, one `f32` per byte.
    #[test]
    fn host_row_is_the_view_in_row_column_channel_order() {
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        view[6][3] = Entity::Goal;
        let obs = GridObservation::from_entity_view(view, Direction::East);

        let mut row = Vec::new();
        obs.write_host_row(&mut row);

        assert_eq!(
            row.len(),
            GridObservation::shape().iter().product::<usize>()
        );
        for r in 0..VIEW_SIZE {
            for c in 0..VIEW_SIZE {
                for ch in 0..OBS_CHANNELS {
                    assert_eq!(
                        row[(r * VIEW_SIZE + c) * OBS_CHANNELS + ch],
                        f32::from(obs.view[r][c][ch]),
                        "cell ({r}, {c}), channel {ch}"
                    );
                }
            }
        }
        // Literals pin the stride: the wall's type byte opens the row, and the
        // door's color sits at cell (3, 3), channel 1.
        assert_eq!(row[0], 2.0);
        assert_eq!(
            row[(3 * VIEW_SIZE + 3) * OBS_CHANNELS + 1],
            f32::from(Color::Blue.to_u8())
        );
    }

    /// The facing is not in the row: every facing writes the same row.
    #[test]
    fn host_row_ignores_agent_direction() {
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        let obs = GridObservation::from_entity_view(view, Direction::North);
        let mut expected = Vec::new();
        obs.write_host_row(&mut expected);

        for agent_direction in [Direction::East, Direction::South, Direction::West] {
            let mut row = Vec::new();
            GridObservation {
                agent_direction,
                ..obs
            }
            .write_host_row(&mut row);
            assert_eq!(row, expected, "facing {agent_direction:?}");
        }
    }

    #[test]
    fn masked_cells_encode_as_the_unseen_triple() {
        let mut view = [[Some(Entity::Empty); VIEW_SIZE]; VIEW_SIZE];
        view[2][5] = None;
        view[6][3] = Some(Entity::Goal);

        let obs = GridObservation::from_masked_view(view, Direction::South);

        assert_eq!(
            obs.view[2][5],
            [UNSEEN_TYPE, 0, 0],
            "a masked cell encodes as canonical's [unseen, 0, 0] triple"
        );
        assert_eq!(obs.view[6][3][0], Entity::Goal.type_u8());
        assert_eq!(obs.agent_direction, Direction::South);
    }

    #[test]
    fn from_entity_view_delegates_to_from_masked_view() {
        // One encoder, not two: wrapping every cell in `Some` must reproduce
        // the unmasked encoding byte for byte.
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        view[6][3] = Entity::Goal;

        let unmasked = GridObservation::from_entity_view(view, Direction::North);
        let wrapped =
            GridObservation::from_masked_view(view.map(|row| row.map(Some)), Direction::North);

        assert_eq!(
            unmasked, wrapped,
            "from_entity_view must be from_masked_view with every cell Some"
        );
    }

    #[test]
    fn unseen_type_is_the_canonical_index() {
        // Canonical `OBJECT_TO_IDX["unseen"] == 0`, and `type_u8` reserves it
        // (ADR 0063 Decision 4).
        assert_eq!(UNSEEN_TYPE, 0);
    }

    #[test]
    fn unseen_and_empty_do_not_share_a_byte() {
        // Was `unseen_and_empty_still_share_a_byte`, which pinned the defect:
        // eight environments emit masked cells, and while `Entity::Empty` was
        // also `0` a masked cell was indistinguishable from a confirmed-empty
        // one. ADR 0063 Decision 4 renumbered `type_u8` to canonical
        // `OBJECT_TO_IDX`, so this is now the regression guard for the fix —
        // if it ever fails again, the occlusion signal is being silently
        // erased at the encoder for every masked floor cell.
        assert_ne!(
            UNSEEN_TYPE,
            Entity::Empty.type_u8(),
            "a masked cell and a confirmed-empty cell must not encode alike"
        );
        assert_eq!(
            Entity::Empty.type_u8(),
            1,
            "canonical `OBJECT_TO_IDX[\"empty\"] == 1`"
        );
    }

    #[test]
    fn every_entity_and_a_masked_cell_reach_the_row() {
        // Every variant in one view, plus a masked cell, so the entity bytes
        // and the absence reading are checked against the same row.
        let entities = [
            Entity::Empty,
            Entity::Wall,
            Entity::Floor,
            Entity::Door(Color::Blue, DoorState::Locked),
            Entity::Key(Color::Yellow),
            Entity::Ball(Color::Red),
            Entity::Box(Color::Green),
            Entity::Goal,
            Entity::Lava,
        ];
        // Nine variants do not fit down one column of a `$7 \times 7$` window, so lay them out in
        // row-major order and keep the mapping for the assertions.
        let cell_of = |i: usize| (i / VIEW_SIZE, i % VIEW_SIZE);
        assert!(
            entities.len() < VIEW_SIZE * VIEW_SIZE,
            "the variant list must fit in the view window"
        );
        let masked_cell = cell_of(entities.len());

        let mut view = [[Some(Entity::Empty); VIEW_SIZE]; VIEW_SIZE];
        for (i, &e) in entities.iter().enumerate() {
            let (r, c) = cell_of(i);
            view[r][c] = Some(e);
        }
        view[masked_cell.0][masked_cell.1] = None;

        let obs = GridObservation::from_masked_view(view, Direction::North);
        let mut row = Vec::new();
        obs.write_host_row(&mut row);
        let cell = |r: usize, c: usize| {
            let at = (r * VIEW_SIZE + c) * OBS_CHANNELS;
            [row[at], row[at + 1], row[at + 2]]
        };

        for (i, &e) in entities.iter().enumerate() {
            let (r, c) = cell_of(i);
            assert_eq!(
                cell(r, c),
                [
                    f32::from(e.type_u8()),
                    f32::from(e.color_u8()),
                    f32::from(e.state_u8())
                ],
                "{e:?} must reach the row unchanged"
            );
        }
        let (mr, mc) = masked_cell;
        assert_eq!(
            cell(mr, mc),
            [f32::from(UNSEEN_TYPE), 0.0, 0.0],
            "a masked cell writes absence, not a plausible entity"
        );
        assert_ne!(
            cell(mr, mc)[0],
            f32::from(Entity::Empty.type_u8()),
            "a masked cell must not read as confirmed-empty floor"
        );
    }

    #[test]
    fn an_all_masked_view_writes_an_all_zero_row() {
        // The reason the renumbering is load-bearing rather than cosmetic:
        // these environments train recurrent POMDP policies, so zero-padded and
        // attention-masked rows are the normal case. A zero row must be what
        // "nothing known" writes, and a seen-but-empty board must not write it.
        let mut row = Vec::new();
        GridObservation::from_masked_view([[None; VIEW_SIZE]; VIEW_SIZE], Direction::East)
            .write_host_row(&mut row);
        assert_eq!(row, vec![0.0_f32; VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS]);

        let mut empty_row = Vec::new();
        GridObservation::from_entity_view([[Entity::Empty; VIEW_SIZE]; VIEW_SIZE], Direction::East)
            .write_host_row(&mut empty_row);
        for channel_0 in empty_row.iter().step_by(OBS_CHANNELS) {
            assert_eq!(*channel_0, 1.0, "a seen-but-empty cell writes `empty` = 1");
        }
    }

    #[test]
    fn from_entity_view_records_facing() {
        let view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        for direction in [
            Direction::East,
            Direction::South,
            Direction::West,
            Direction::North,
        ] {
            let obs = GridObservation::from_entity_view(view, direction);
            assert_eq!(
                obs.agent_direction, direction,
                "from_entity_view must record the facing it was handed"
            );
        }
    }
}
