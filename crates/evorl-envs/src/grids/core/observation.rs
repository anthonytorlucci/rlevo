//! 7×7×3 egocentric observation emitted by every grid environment.

use super::direction::Direction;
use super::entity::Entity;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use evorl_core::base::{Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// Side length (height and width) of the agent's local view window.
pub const VIEW_SIZE: usize = 7;
/// Number of channels in the observation: `(type, color, state)`.
pub const OBS_CHANNELS: usize = 3;

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
    /// Agent's current facing, encoded via [`Direction::to_u8`].
    pub agent_direction: u8,
}

impl GridObservation {
    /// Encode a decoded 7×7 entity view and the agent's facing into an
    /// observation.
    #[must_use]
    pub fn from_entity_view(
        view: [[Entity; VIEW_SIZE]; VIEW_SIZE],
        direction: Direction,
    ) -> Self {
        let mut encoded = [[[0u8; OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE];
        for (r, row) in view.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                encoded[r][c] = [cell.type_u8(), cell.color_u8(), cell.state_u8()];
            }
        }
        Self {
            view: encoded,
            agent_direction: direction.to_u8(),
        }
    }
}

impl Observation<3> for GridObservation {
    fn shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]
    }
}

impl<B: Backend> TensorConvertible<3, B> for GridObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 3> {
        let mut flat = Vec::with_capacity(VIEW_SIZE * VIEW_SIZE * OBS_CHANNELS);
        for row in &self.view {
            for cell in row {
                for &channel in cell {
                    flat.push(f32::from(channel));
                }
            }
        }
        let data = TensorData::new(flat, [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]);
        Tensor::<B, 3>::from_data(data, device)
    }

    /// Reconstructs the 7×7×3 view from a tensor.
    ///
    /// The tensor contains only the view channels. `agent_direction` is not
    /// encoded in the tensor representation and is defaulted to
    /// [`Direction::North`]; callers that need round-trip fidelity for the
    /// direction must carry it out-of-band.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape does not equal
    /// `[VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]` or the backend fails to
    /// materialize its data.
    fn from_tensor(tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS] {
            return Err(TensorConversionError {
                message: format!(
                    "expected shape [{VIEW_SIZE}, {VIEW_SIZE}, {OBS_CHANNELS}], got {dims:?}"
                ),
            });
        }
        let flat = tensor.into_data().into_vec::<f32>().map_err(|e| {
            TensorConversionError {
                message: format!("failed to read tensor data: {e:?}"),
            }
        })?;
        let mut view = [[[0u8; OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE];
        let mut idx = 0;
        for row in &mut view {
            for cell in row {
                for channel in cell {
                    let value = flat[idx];
                    if !value.is_finite() || value < 0.0 || value > f32::from(u8::MAX) {
                        return Err(TensorConversionError {
                            message: format!(
                                "value at index {idx} out of u8 range: {value}"
                            ),
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
            agent_direction: Direction::North.to_u8(),
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
        assert_eq!(obs.agent_direction, Direction::North.to_u8());
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
        use burn::backend::NdArray;
        type TestBackend = NdArray;
        let device = Default::default();

        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[0][0] = Entity::Wall;
        view[3][3] = Entity::Door(Color::Blue, DoorState::Locked);
        view[6][3] = Entity::Goal;
        let obs = GridObservation::from_entity_view(view, Direction::East);

        let tensor = <GridObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let round_tripped =
            <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor).unwrap();

        assert_eq!(round_tripped.view, obs.view);
        // agent_direction is not encoded in the tensor; defaults to North.
        assert_eq!(round_tripped.agent_direction, Direction::North.to_u8());
    }

    #[test]
    fn from_tensor_rejects_wrong_shape() {
        use burn::backend::NdArray;
        use burn::tensor::TensorData as TD;
        type TestBackend = NdArray;
        let device = Default::default();

        let flat = vec![0.0f32; VIEW_SIZE * VIEW_SIZE * 2];
        let data = TD::new(flat, [VIEW_SIZE, VIEW_SIZE, 2]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let err =
            <GridObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor).unwrap_err();
        assert!(err.message.contains("expected shape"));
    }

    #[test]
    fn direction_is_encoded_per_byte() {
        let view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        let obs = GridObservation::from_entity_view(view, Direction::South);
        assert_eq!(obs.agent_direction, Direction::South.to_u8());
    }
}
