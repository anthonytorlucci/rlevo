use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use evorl_core::action::{Action, DiscreteAction};
use evorl_core::base::TensorConvertible;
use evorl_core::dynamics::{ExperienceTuple, History, HistoryRepresentation};
use evorl_core::state::{Observation, State};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridPositionObservation {
    x: i32,
    y: i32,
}

impl GridPositionObservation {
    /// Creates a new GridPositionObservation with the given coordinates.
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl Default for GridPositionObservation {
    /// Returns a GridPositionObservation at the origin (0, 0).
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}

impl Observation<1> for GridPositionObservation {
    fn shape() -> [usize; 1] {
        [2]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GridPosition {
    x: i32,
    y: i32,
    max_x: i32,
    max_y: i32,
}

impl State<1> for GridPosition {
    type Observation = GridPositionObservation;
    fn is_valid(&self) -> bool {
        self.x >= 0 && self.y >= 0 && self.x < self.max_x && self.y < self.max_y
    }

    fn numel(&self) -> usize {
        2 // x and y coordinates
    }

    fn shape() -> [usize; 1] {
        [2]
    }

    fn observe(&self) -> Self::Observation {
        GridPositionObservation::new(self.x, self.y)
    }
}

impl<const D: usize, B: Backend> TensorConvertible<D, B> for GridPosition {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, D> {
        todo!("Implement conversion to tensor")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GridAction {
    Up,    // 0
    Down,  // 1
    Left,  // 2
    Right, // 3
}

impl Action<1> for GridAction {
    fn is_valid(&self) -> bool {
        true // no type specific contratints
    }

    fn shape() -> [usize; 1] {
        [4] // Up, Down, Left, Right
    }
}

impl DiscreteAction<1> for GridAction {
    const ACTION_COUNT: usize = 4;
    fn from_index(index: usize) -> Self {
        match index {
            0 => GridAction::Up,
            1 => GridAction::Down,
            2 => GridAction::Left,
            3 => GridAction::Right,
            _ => panic!("Unknown action index: {}", index),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            GridAction::Up => 0,
            GridAction::Down => 1,
            GridAction::Left => 2,
            GridAction::Right => 3,
        }
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------------------------------------------------
    // 1. Create a few valid GridPosition instances
    // -----------------------------------------------------------------------
    let mut pos1 = GridPosition {
        x: 3,
        y: 7,
        max_x: 10,
        max_y: 10,
    };
    let mut pos2 = GridPosition {
        x: 0,
        y: 0,
        max_x: 10,
        max_y: 10,
    };
    let mut pos3 = GridPosition {
        x: 9,
        y: 9,
        max_x: 6,
        max_y: 10,
    };

    // -----------------------------------------------------------------------
    // 2. Validate them using the `State` implementation
    // -----------------------------------------------------------------------
    println!("pos1 is valid? {}", pos1.is_valid()); // true
    println!("pos2 is valid? {}", pos2.is_valid()); // true
    println!("pos3 is valid? {}", pos3.is_valid()); // false

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
