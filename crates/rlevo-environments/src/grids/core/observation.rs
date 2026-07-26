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
/// # Live collision with `Entity::Empty` — read this before consuming channel 0
///
/// Canonical numbers `unseen = 0` and `empty = 1` as *distinct* type indices,
/// precisely so a masked cell cannot be confused with a confirmed-empty one.
/// rlevo's [`Entity::type_u8`] does not yet use canonical indices —
/// [`Entity::Empty`] is also `0` — so the two encode identically.
///
/// This **is now load-bearing**, where it previously was not. Eight of the
/// twelve grid environments select
/// [`Visibility::Occluded`](super::Visibility::Occluded), so `None` cells are
/// produced on every step of those environments, and every masked cell that was
/// an `Empty` floor is *indistinguishable* in the encoding from a cell the agent
/// confirmed to be empty. A masked `Wall`, `Goal`, `Key`, `Door`, … still
/// changes its byte and so still carries the occlusion signal; a masked `Empty`
/// silently does not. In practice most masked cells are non-`Empty` (they are
/// behind an occluder, so they are usually wall or the far side of a room), but
/// "most" is not "all", and the distinction the whole shadow cast exists to draw
/// — *unknown* versus *confirmed floor* — is partially collapsed until the
/// renumbering lands.
///
/// The fix is ADR 0063 Decision 4 (`Empty` becomes `1`, and the whole table
/// moves to canonical `OBJECT_TO_IDX`), deferred out of the visibility change so
/// that a behaviour change and an encoding break do not land in one diff. It is
/// tracked under issue #281 and must land before any grid observation baseline
/// is ever recorded.
///
/// **Do not "fix" the collision by picking a different value here.** `0` is the
/// canonical unseen index; it is `type_u8` that has to move.
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

        assert_eq!(obs.view[0][0][0], Entity::Wall.type_u8());
        assert_eq!(obs.view[3][3][0], 5); // Door type byte
        assert_eq!(obs.view[3][3][1], Color::Blue.to_u8());
        assert_eq!(obs.view[3][3][2], DoorState::Locked.to_u8());
        assert_eq!(obs.view[6][3][0], 3); // Goal type byte
        assert_eq!(obs.agent_direction, Some(Direction::North));
    }

    #[test]
    fn empty_cells_encode_as_zero() {
        let view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        let obs = GridObservation::from_entity_view(view, Direction::East);
        for row in &obs.view {
            for cell in row {
                assert_eq!(cell, &[0, 0, 0]);
            }
        }
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
        // Canonical `OBJECT_TO_IDX["unseen"] == 0`. The collision with
        // `Entity::Empty` is tracked in #281 — it is `type_u8` that moves to
        // canonical indices, not this constant.
        assert_eq!(UNSEEN_TYPE, 0);
    }

    #[test]
    fn unseen_and_empty_still_share_a_byte() {
        // Pins the *known defect*, not a desired property: eight environments now
        // emit masked cells, and a masked `Empty` is currently indistinguishable
        // from a confirmed-empty one. When ADR 0063 Decision 4 renumbers
        // `type_u8` this assertion flips to `assert_ne!` and the `UNSEEN_TYPE`
        // docs lose their collision section.
        assert_eq!(
            UNSEEN_TYPE,
            Entity::Empty.type_u8(),
            "if these have diverged, the canonical renumbering landed — update the \
             UNSEEN_TYPE docs and this test together"
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
