//! 7×7×3 egocentric observation emitted by every grid environment.

use super::direction::Direction;
use super::entity::Entity;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use evorl_core::base::{Observation, TensorConvertible};
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
    fn direction_is_encoded_per_byte() {
        let view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        let obs = GridObservation::from_entity_view(view, Direction::South);
        assert_eq!(obs.agent_direction, Direction::South.to_u8());
    }
}
