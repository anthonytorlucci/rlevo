//! 7×7×3 egocentric observation emitted by every grid environment.
//!
//! The agent sits at the bottom-center of its view window and looks toward
//! the top. Every visible cell is encoded into three bytes — entity type,
//! color, and door state — laid out as a `[VIEW_SIZE][VIEW_SIZE][OBS_CHANNELS]`
//! array. The agent's facing is carried *beside* the view as an
//! `Option<`[`Direction`]`>` and is deliberately **not** encoded into the
//! tensor — the view is already rotated into the agent's own frame, so the
//! absolute heading is redundant for a policy. Decoding a tensor therefore
//! yields `None` rather than a fabricated facing (see
//! [`GridObservation::agent_direction`] and
//! [`TensorConvertible::from_tensor`]).

use super::entity::Entity;
use crate::direction::Direction;
use burn::tensor::{Tensor, backend::Backend};
use rlevo_core::base::{HostRow, Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// Side length (height and width) of the agent's local view window in cells.
///
/// Matches the Minigrid default of 7, giving the agent a `7 × 7` field of
/// view centered one cell in front of its current position.
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
/// It also buys the [`TensorConvertible`] no-fabrication clause a stronger
/// reading at the tensor level: an all-zero tensor decodes as "every cell
/// unseen", which is what a zero-padded or attention-masked sequence actually
/// means, rather than as "every cell is confirmed empty floor".
///
/// **Do not "fix" a future collision by picking a different value here.** `0` is
/// the canonical unseen index; it is `type_u8` that must stay clear of it.
///
/// [`Entity::type_u8`]: super::entity::Entity::type_u8
/// [`Entity::Empty`]: super::entity::Entity::Empty
pub const UNSEEN_TYPE: u8 = 0;

/// Egocentric observation of the 7×7 cells around the agent.
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
    /// Agent's absolute facing when the observation was produced, or `None`
    /// when the facing is genuinely unknown.
    ///
    /// Typed as [`Direction`] rather than a raw byte so that no illegal
    /// encoding is representable (issue #844); [`Direction::to_u8`] still
    /// yields the canonical Minigrid byte order for callers that want it.
    ///
    /// # Why it is not in the tensor
    ///
    /// The facing is deliberately **absent** from the tensor produced by
    /// [`TensorConvertible::to_tensor`]. [`view`](Self::view) is already
    /// rotated into the agent's own frame, so an absolute heading is
    /// decision-vestigial for a policy: canonical Minigrid ships `direction`
    /// as a separate entry of the observation dict and never inside the
    /// image, and every published baseline (`ImgObsWrapper`,
    /// `rl-starter-files`, RIDE) discards it before the network.
    ///
    /// # What `None` means
    ///
    /// `None` means *this observation was decoded from a tensor, so its facing
    /// is genuinely unknown* — **not** "the agent faces some default". A value
    /// produced by [`from_entity_view`](Self::from_entity_view) — i.e. by any
    /// environment's `project()` — is always `Some`. When the true facing is
    /// needed, read it from the full state
    /// ([`GridState::agent`](super::state::GridState::agent)), which is where
    /// the allocentric heading lives.
    pub agent_direction: Option<Direction>,
}

impl GridObservation {
    /// Encode a visibility-masked 7×7 entity view and the agent's facing into
    /// an observation.
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
    /// * `direction` — the agent's absolute facing, carried beside the tensor
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
            agent_direction: Some(direction),
        }
    }

    /// Encode a fully visible 7×7 entity view and the agent's facing into an
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
    fn row_shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]
    }

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
        // `u8 -> f32` is total: no element of this row can be NaN or ±Inf.
        // `agent_direction` is not written into the row at all.
        true
    }
}

impl<B: Backend> TensorConvertible<3, B> for GridObservation {
    /// Reconstructs the 7×7×3 view from a tensor.
    ///
    /// The tensor carries only the view channels, because the view is already
    /// rotated into the agent's frame and the absolute facing adds nothing a
    /// policy consumes (see [`agent_direction`](GridObservation::agent_direction)).
    /// The decoded observation therefore reports
    /// `agent_direction == None` — the facing is *unknown*, and this method
    /// will never invent a plausible one. Callers needing the true facing must
    /// carry it out-of-band, from [`GridState`](super::state::GridState).
    ///
    /// This satisfies the [`TensorConvertible`] contract's two clauses:
    /// decode-then-re-encode reproduces the tensor exactly, and the one field
    /// the tensor omits decodes to an explicit absence.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape does not equal
    /// `[VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]` or the backend fails to
    /// materialize its data.
    fn from_tensor(tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS] {
            return Err(TensorConversionError {
                message: format!(
                    "expected shape [{VIEW_SIZE}, {VIEW_SIZE}, {OBS_CHANNELS}], got {dims:?}"
                ),
            });
        }
        let flat = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: format!("failed to read tensor data: {e:?}"),
            })?;
        let mut view = [[[0u8; OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE];
        let mut idx = 0;
        for row in &mut view {
            for cell in row {
                for channel in cell {
                    let value = flat[idx];
                    if !value.is_finite() || value < 0.0 || value > f32::from(u8::MAX) {
                        return Err(TensorConversionError {
                            message: format!("value at index {idx} out of u8 range: {value}"),
                        });
                    }
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        *channel = value as u8;
                    }
                    idx += 1;
                }
            }
        }
        Ok(Self {
            view,
            // The tensor carries no facing, so a decode must not invent one.
            agent_direction: None,
        })
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
        assert_eq!(obs.agent_direction, Some(Direction::North));
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

    #[test]
    fn view_round_trips_through_tensor() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        view[6][3] = Entity::Goal;
        let obs = GridObservation::from_entity_view(view, Direction::East);

        let tensor =
            <GridObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let round_tripped =
            <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor).unwrap();

        assert_eq!(round_tripped.view, obs.view);
        // The tensor carries no facing, so a decode must report the facing as
        // unknown rather than inventing a plausible one.
        assert_eq!(
            round_tripped.agent_direction, None,
            "a decoded observation must not fabricate a facing"
        );
    }

    #[test]
    fn decoded_observation_re_encodes_to_the_same_tensor() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        view[6][3] = Entity::Goal;
        let obs = GridObservation::from_entity_view(view, Direction::West);

        let tensor =
            <GridObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let decoded =
            <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor.clone())
                .expect("decode of a self-produced tensor must succeed");

        // Clause 2 of the TensorConvertible contract: an unwritten field
        // decodes to an explicit absence, never to a plausible value.
        assert_eq!(
            decoded.agent_direction, None,
            "the facing is absent from the tensor, so it must decode as unknown"
        );

        // Clause 1: decode-then-re-encode is a no-op on the tensor.
        let re_encoded =
            <GridObservation as TensorConvertible<3, TestBackend>>::to_tensor(&decoded, &device);
        assert_eq!(
            re_encoded.to_data().to_vec::<f32>().unwrap(),
            tensor.to_data().to_vec::<f32>().unwrap(),
            "re-encoding a decoded observation must reproduce the same tensor row"
        );
    }

    #[test]
    fn from_tensor_rejects_wrong_shape() {
        use burn::backend::Flex;
        use burn::tensor::TensorData as TD;
        type TestBackend = Flex;
        let device = Default::default();

        let flat = vec![0.0f32; VIEW_SIZE * VIEW_SIZE * 2];
        let data = TD::new(flat, [VIEW_SIZE, VIEW_SIZE, 2]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let err = <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
            .unwrap_err();
        assert!(err.message.contains("expected shape"));
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
        assert_eq!(obs.agent_direction, Some(Direction::South));
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
    fn every_entity_survives_a_tensor_round_trip_and_a_mask_decodes_to_absence() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        // Every variant in one view, plus a masked cell, so the round trip and
        // the absence reading are checked against the same tensor.
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
        // Nine variants do not fit down one column of a 7×7 window, so lay them
        // out in row-major order and keep the mapping for the assertions.
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
        let tensor =
            <GridObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let decoded = <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
            .expect("decode of a self-produced tensor must succeed");

        for (i, &e) in entities.iter().enumerate() {
            let (r, c) = cell_of(i);
            assert_eq!(
                decoded.view[r][c],
                [e.type_u8(), e.color_u8(), e.state_u8()],
                "{e:?} must survive encode -> tensor -> decode unchanged"
            );
        }
        let (mr, mc) = masked_cell;
        assert_eq!(
            decoded.view[mr][mc],
            [UNSEEN_TYPE, 0, 0],
            "a masked cell must decode to absence, not to a plausible entity"
        );
        assert_ne!(
            decoded.view[mr][mc][0],
            Entity::Empty.type_u8(),
            "a masked cell must not read back as confirmed-empty floor"
        );
    }

    #[test]
    fn an_all_zero_tensor_decodes_as_every_cell_unseen() {
        use burn::backend::Flex;
        use burn::tensor::TensorData as TD;
        type TestBackend = Flex;
        let device = Default::default();

        // The reason the renumbering is load-bearing rather than cosmetic:
        // these environments train recurrent POMDP policies, so zero-padded and
        // attention-masked rows are the normal case. Such a row must read as
        // "nothing known", not as a fabricated board of empty floor.
        let flat = vec![0.0f32; VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS];
        let data = TD::new(flat, [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);

        let decoded = <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
            .expect("an in-range zero tensor must decode");

        for row in &decoded.view {
            for cell in row {
                assert_eq!(
                    cell[0], UNSEEN_TYPE,
                    "a zero channel-0 byte must mean unseen, never a real entity"
                );
            }
        }
        assert_eq!(
            decoded.agent_direction, None,
            "the facing is absent from the tensor, so it must decode as unknown"
        );
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
                obs.agent_direction,
                Some(direction),
                "from_entity_view must record the facing it was handed"
            );
        }
    }
}
