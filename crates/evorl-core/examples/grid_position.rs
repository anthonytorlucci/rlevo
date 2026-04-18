use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use evorl_core::action::DiscreteAction;
use evorl_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridPositionObservation {
    x: i32,
    y: i32,
}

impl GridPositionObservation {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl Default for GridPositionObservation {
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
        2
    }

    fn shape() -> [usize; 1] {
        [2]
    }

    fn observe(&self) -> Self::Observation {
        GridPositionObservation::new(self.x, self.y)
    }
}

impl<const D: usize, B: Backend> TensorConvertible<D, B> for GridPosition {
    fn to_tensor(&self, _device: &B::Device) -> Tensor<B, D> {
        todo!("Implement conversion to tensor")
    }

    fn from_tensor(_tensor: Tensor<B, D>) -> Result<Self, TensorConversionError> {
        todo!("Implement conversion from tensor")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GridAction {
    Up,
    Down,
    Left,
    Right,
}

impl Action<1> for GridAction {
    fn is_valid(&self) -> bool {
        true
    }

    fn shape() -> [usize; 1] {
        [4]
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

/// Applies an action to a grid position, returning the candidate next position.
/// The caller should check `is_valid()` on the result before accepting it.
fn apply_action(pos: &GridPosition, action: GridAction) -> GridPosition {
    let (dx, dy) = match action {
        GridAction::Up => (0, 1),
        GridAction::Down => (0, -1),
        GridAction::Left => (-1, 0),
        GridAction::Right => (1, 0),
    };
    GridPosition {
        x: pos.x + dx,
        y: pos.y + dy,
        max_x: pos.max_x,
        max_y: pos.max_y,
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------------------------------------------------
    // 1. State: shape, DIM, numel, and is_valid
    // -----------------------------------------------------------------------
    println!("=== State ===");

    let pos_a = GridPosition { x: 3, y: 7, max_x: 10, max_y: 10 };
    let pos_b = GridPosition { x: 0, y: 0, max_x: 10, max_y: 10 };
    let pos_oob = GridPosition { x: 9, y: 9, max_x: 6, max_y: 10 }; // x exceeds max_x

    println!("GridPosition::DIM   = {}", GridPosition::DIM);
    println!("GridPosition::shape = {:?}", GridPosition::shape());
    println!("pos_a.numel()       = {}", pos_a.numel());
    println!("pos_a.is_valid()    = {}", pos_a.is_valid()); // true
    println!("pos_b.is_valid()    = {}", pos_b.is_valid()); // true
    println!("pos_oob.is_valid()  = {}", pos_oob.is_valid()); // false (x=9 >= max_x=6)

    // -----------------------------------------------------------------------
    // 2. Observation: derived from state via observe()
    // -----------------------------------------------------------------------
    println!("\n=== Observation ===");

    let obs_a = pos_a.observe();
    println!(
        "GridPositionObservation::DIM   = {}",
        GridPositionObservation::DIM
    );
    println!(
        "GridPositionObservation::shape = {:?}",
        GridPositionObservation::shape()
    );
    println!("obs_a (from pos_a)  = {:?}", obs_a);

    // The observation exposes only what the agent can perceive —
    // max_x / max_y are internal to the state and not included.
    let obs_b = pos_b.observe();
    println!("obs_b (from pos_b)  = {:?}", obs_b);

    // -----------------------------------------------------------------------
    // 3. Action: shape, DIM, ACTION_COUNT, from_index / to_index round-trip
    // -----------------------------------------------------------------------
    println!("\n=== Action ===");

    println!("GridAction::DIM          = {}", GridAction::DIM);
    println!("GridAction::shape        = {:?}", GridAction::shape());
    println!("GridAction::ACTION_COUNT = {}", GridAction::ACTION_COUNT);

    // Index ↔ variant round-trip
    for i in 0..GridAction::ACTION_COUNT {
        let action = GridAction::from_index(i);
        assert_eq!(action.to_index(), i); // round-trips
        println!("  index {} → {:?} → index {}", i, action, action.to_index());
    }

    // All actions are structurally valid (no bounds or range constraints)
    let up = GridAction::Up;
    println!("GridAction::Up.is_valid() = {}", up.is_valid());

    // -----------------------------------------------------------------------
    // 4. Action: enumerate and random sampling
    // -----------------------------------------------------------------------
    println!("\n=== Action enumeration and random sampling ===");

    let all_actions = GridAction::enumerate();
    println!("All actions ({} total): {:?}", all_actions.len(), all_actions);

    let random_action = GridAction::random();
    println!("Random action: {:?} (index {})", random_action, random_action.to_index());

    // -----------------------------------------------------------------------
    // 5. State + Action + Observation interplay: manual transition loop
    // -----------------------------------------------------------------------
    println!("\n=== Transition loop ===");

    let mut pos = GridPosition { x: 2, y: 2, max_x: 5, max_y: 5 };

    // Walk a fixed sequence of actions, printing state and observation each step.
    let sequence = [
        GridAction::Right,
        GridAction::Right,
        GridAction::Up,
        GridAction::Left,
        GridAction::Down,
        GridAction::Down, // walks below y=0, landing out-of-bounds
    ];

    for action in sequence {
        let obs_before = pos.observe();
        let next = apply_action(&pos, action);
        let valid = next.is_valid();

        println!(
            "  obs={:?}  + {:?}  →  ({},{}) valid={}",
            obs_before, action, next.x, next.y, valid
        );

        if valid {
            pos = next;
        } else {
            println!("  (out-of-bounds — state unchanged)");
        }
    }

    let final_obs = pos.observe();
    println!("Final observation: {:?}", final_obs);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
