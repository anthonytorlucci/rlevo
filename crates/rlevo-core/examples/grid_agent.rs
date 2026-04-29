//! Introduces `State`, `Observation`, `DiscreteAction`, and `MultiDiscreteAction`
//! through a minimal egocentric grid agent.
//!
//! The model here mirrors the architecture of the Minigrid-style environments in
//! `rlevo-environments::grids`:
//!
//! - The agent has a **facing direction** — it turns and then steps forward rather
//!   than moving in absolute coordinates.
//! - Observations encode `(x, y, facing)` and support full tensor round-trips via
//!   [`TensorConvertible`].
//! - Two action types are shown: a [`DiscreteAction`] for egocentric movement and a
//!   [`MultiDiscreteAction`] for compound (move + interact) control.
//!
//! # What you will learn
//!
//! - Wiring `State`, `Observation`, and `TensorConvertible` together
//! - Implementing `DiscreteAction` with index round-trips
//! - Implementing `MultiDiscreteAction` for independent sub-dimensions
//! - Writing an egocentric transition loop with boundary enforcement
//!
//! # Running
//!
//! ```bash
//! cargo run -p rlevo-core --example grid_agent
//! ```

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo_core::action::{DiscreteAction, MultiDiscreteAction};
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// Concrete backend used for tensor demonstrations in `main`.
type DemoBackend = NdArray;

// ─── Facing ──────────────────────────────────────────────────────────────────

/// Compass direction the agent is currently facing.
///
/// Encoded as a byte (`North=0`, `East=1`, `South=2`, `West=3`) for tensor
/// storage. Rotation never moves the position; forward movement uses
/// [`delta`](Facing::delta).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Facing {
    North,
    East,
    South,
    West,
}

impl Facing {
    /// Returns the facing after a 90° counter-clockwise rotation.
    fn turn_left(self) -> Self {
        match self {
            Self::North => Self::West,
            Self::West => Self::South,
            Self::South => Self::East,
            Self::East => Self::North,
        }
    }

    /// Returns the facing after a 90° clockwise rotation.
    fn turn_right(self) -> Self {
        match self {
            Self::North => Self::East,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
        }
    }

    /// Returns the `(dx, dy)` unit displacement for a forward step.
    fn delta(self) -> (i32, i32) {
        match self {
            Self::North => (0, -1),
            Self::East => (1, 0),
            Self::South => (0, 1),
            Self::West => (-1, 0),
        }
    }

    fn to_u8(self) -> u8 {
        self as u8
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::North),
            1 => Some(Self::East),
            2 => Some(Self::South),
            3 => Some(Self::West),
            _ => None,
        }
    }
}

// ─── Observation ─────────────────────────────────────────────────────────────

/// Agent-visible snapshot: grid position and current facing.
///
/// Grid bounds are not exposed — they belong to the environment state and are
/// invisible to the agent, matching the pattern used in the Minigrid ports in
/// `rlevo-environments`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct AgentObservation {
    /// Column (x-axis, zero-based).
    x: i32,
    /// Row (y-axis, zero-based, increasing downward).
    y: i32,
    /// Current facing direction.
    facing: Facing,
}

impl Observation<1> for AgentObservation {
    fn shape() -> [usize; 1] {
        [3] // [x, y, facing_index]
    }
}

impl<B: Backend> TensorConvertible<1, B> for AgentObservation {
    #[allow(clippy::cast_precision_loss)]
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        let data = TensorData::new(
            vec![self.x as f32, self.y as f32, f32::from(self.facing.to_u8())],
            [3],
        );
        Tensor::from_data(data, device)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims[0] != 3 {
            return Err(TensorConversionError {
                message: format!("expected shape [3], got {dims:?}"),
            });
        }
        let vals = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: format!("failed to read tensor data: {e:?}"),
            })?;
        let facing_byte = vals[2] as u8;
        let facing = Facing::from_u8(facing_byte).ok_or_else(|| TensorConversionError {
            message: format!("invalid facing byte: {facing_byte}"),
        })?;
        Ok(Self {
            x: vals[0] as i32,
            y: vals[1] as i32,
            facing,
        })
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Full environment state: position, facing, and grid bounds.
///
/// `width` and `height` are exclusive upper bounds. They are hidden from
/// the agent — [`observe`](AgentState::observe) omits them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentState {
    x: i32,
    y: i32,
    facing: Facing,
    /// Exclusive upper bound for `x`.
    width: i32,
    /// Exclusive upper bound for `y`.
    height: i32,
}

impl State<1> for AgentState {
    type Observation = AgentObservation;

    fn shape() -> [usize; 1] {
        [3]
    }

    fn observe(&self) -> AgentObservation {
        AgentObservation {
            x: self.x,
            y: self.y,
            facing: self.facing,
        }
    }

    fn is_valid(&self) -> bool {
        self.x >= 0 && self.y >= 0 && self.x < self.width && self.y < self.height
    }
}

// ─── Discrete action ─────────────────────────────────────────────────────────

/// Egocentric movement: turn left, turn right, or step forward.
///
/// Mirrors the movement subset of `GridAction` in `rlevo-environments::grids`. The
/// agent never specifies an absolute direction — it rotates relative to its
/// current facing, then steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveAction {
    /// Rotate 90° counter-clockwise.
    TurnLeft,
    /// Rotate 90° clockwise.
    TurnRight,
    /// Step one cell in the current facing direction.
    Forward,
}

impl Action<1> for MoveAction {
    fn is_valid(&self) -> bool {
        true
    }

    fn shape() -> [usize; 1] {
        [Self::ACTION_COUNT]
    }
}

impl DiscreteAction<1> for MoveAction {
    const ACTION_COUNT: usize = 3;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::TurnLeft,
            1 => Self::TurnRight,
            2 => Self::Forward,
            _ => panic!("MoveAction index out of bounds: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Self::TurnLeft => 0,
            Self::TurnRight => 1,
            Self::Forward => 2,
        }
    }
}

// ─── Multi-discrete action ────────────────────────────────────────────────────

/// Optional interaction the agent performs after moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interact {
    /// Index 0 — do nothing after moving.
    Skip,
    /// Index 1 — toggle the object directly in front (open a door, flip a switch).
    Toggle,
}

impl Interact {
    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Skip,
            1 => Self::Toggle,
            _ => panic!("Interact index out of bounds: {i}"),
        }
    }

    fn to_index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Toggle => "toggle",
        }
    }
}

/// Compound action: movement paired with an optional interaction.
///
/// Implements [`MultiDiscreteAction<2>`] — the two sub-dimensions are
/// independent. The index array is `[movement_index, interact_index]` and
/// the full space has shape `[3, 2]` (3 moves × 2 interactions = 6 combinations).
///
/// This is the `MultiDiscreteAction` counterpart to [`MoveAction`]: use it
/// when the agent must simultaneously decide movement *and* what to do with
/// an object in front of it.
#[derive(Debug, Clone, Copy)]
struct CompoundAction {
    movement: MoveAction,
    interact: Interact,
}

impl Action<2> for CompoundAction {
    fn is_valid(&self) -> bool {
        true
    }

    fn shape() -> [usize; 2] {
        [3, 2]
    }
}

impl MultiDiscreteAction<2> for CompoundAction {
    fn from_indices(indices: [usize; 2]) -> Self {
        Self {
            movement: MoveAction::from_index(indices[0]),
            interact: Interact::from_index(indices[1]),
        }
    }

    fn to_indices(&self) -> [usize; 2] {
        [self.movement.to_index(), self.interact.to_index()]
    }
}

// ─── Transition ──────────────────────────────────────────────────────────────

/// Returns the candidate next state after applying `action` to `state`.
///
/// The result may be out-of-bounds. Callers must call [`State::is_valid`] on
/// the successor and decide whether to accept or discard it.
fn step(state: &AgentState, action: MoveAction) -> AgentState {
    match action {
        MoveAction::TurnLeft => AgentState {
            facing: state.facing.turn_left(),
            ..state.clone()
        },
        MoveAction::TurnRight => AgentState {
            facing: state.facing.turn_right(),
            ..state.clone()
        },
        MoveAction::Forward => {
            let (dx, dy) = state.facing.delta();
            AgentState {
                x: state.x + dx,
                y: state.y + dy,
                ..state.clone()
            }
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. State ────────────────────────────────────────────────────────────
    println!("=== State ===");
    let state = AgentState {
        x: 2,
        y: 3,
        facing: Facing::East,
        width: 5,
        height: 5,
    };
    let oob = AgentState {
        x: 5,
        y: 0,
        facing: Facing::North,
        width: 5,
        height: 5,
    };
    println!("AgentState::DIM     = {}", AgentState::DIM);
    println!("AgentState::shape() = {:?}", AgentState::shape());
    println!("state.numel()       = {}", state.numel());
    println!("state.is_valid()    = {}", state.is_valid()); // true
    println!("oob.is_valid()      = {}", oob.is_valid()); // false (x=5 >= width=5)

    // ── 2. Observation + tensor round-trip ──────────────────────────────────
    println!("\n=== Observation + tensor round-trip ===");
    let obs = state.observe();
    println!("obs                          = {:?}", obs);
    println!("AgentObservation::DIM        = {}", AgentObservation::DIM);
    println!(
        "AgentObservation::shape()    = {:?}",
        AgentObservation::shape()
    );

    let device: <DemoBackend as Backend>::Device = Default::default();
    let tensor = <AgentObservation as TensorConvertible<1, DemoBackend>>::to_tensor(&obs, &device);
    println!(
        "tensor                       = {:?}",
        tensor.clone().into_data()
    );
    let recovered = <AgentObservation as TensorConvertible<1, DemoBackend>>::from_tensor(tensor)?;
    println!("round-trip match             = {}", recovered == obs);

    // ── 3. DiscreteAction ───────────────────────────────────────────────────
    println!("\n=== DiscreteAction ===");
    println!("MoveAction::ACTION_COUNT = {}", MoveAction::ACTION_COUNT);
    for i in 0..MoveAction::ACTION_COUNT {
        let a = MoveAction::from_index(i);
        assert_eq!(a.to_index(), i);
        println!("  index {} → {:?} → index {}", i, a, a.to_index());
    }
    println!("enumerate count = {}", MoveAction::enumerate().len());
    println!("random sample   = {:?}", MoveAction::random());

    // ── 4. MultiDiscreteAction ──────────────────────────────────────────────
    println!("\n=== MultiDiscreteAction ===");
    println!("CompoundAction::shape() = {:?}", CompoundAction::shape());
    for movement in MoveAction::enumerate() {
        for interact in [Interact::Skip, Interact::Toggle] {
            let action = CompoundAction { movement, interact };
            let indices = action.to_indices();
            assert_eq!(CompoundAction::from_indices(indices).to_indices(), indices);
            println!("  {:?} + {} → {:?}", movement, interact.name(), indices);
        }
    }

    // ── 5. Egocentric transition loop ───────────────────────────────────────
    println!("\n=== Egocentric transition loop ===");
    let mut s = AgentState {
        x: 1,
        y: 3,
        facing: Facing::North,
        width: 5,
        height: 5,
    };
    let sequence = [
        MoveAction::Forward,   // y: 3→2
        MoveAction::Forward,   // y: 2→1
        MoveAction::TurnRight, // now facing East
        MoveAction::Forward,   // x: 1→2
        MoveAction::TurnLeft,  // back to North
        MoveAction::Forward,   // y: 1→0
        MoveAction::Forward,   // y: 0→-1  out-of-bounds
    ];
    for action in sequence {
        let obs_before = s.observe();
        let next = step(&s, action);
        let valid = next.is_valid();
        println!(
            "  ({},{}) {:?}  +  {:?}  →  ({},{}) valid={}",
            obs_before.x, obs_before.y, obs_before.facing, action, next.x, next.y, valid,
        );
        if valid {
            s = next;
        } else {
            println!("  (out-of-bounds — state unchanged)");
        }
    }

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
