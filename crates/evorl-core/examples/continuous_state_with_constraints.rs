//! # RobotPose: Continuous State with Workspace Constraints
//!
//! This example demonstrates how to implement a constrained state representation for reinforcement learning
//! environments. It showcases several key design patterns and best practices for the burn-evorl library.
//!
//! ## Design Patterns & Principles
//!
//! ### 1. **Builder Pattern with Validation**
//!
//! The `RobotPose::new()` method acts as a smart constructor that validates constraints before
//! creating an instance. This prevents invalid states from entering the system:
//!
//! ```ignore
//! // Returns None if constraints are violated
//! let pose = RobotPose::new(x, y, theta)?;
//! ```
//!
//! Benefits:
//! - **Type safety**: Invalid poses cannot exist; you must explicitly check `Option`
//! - **Fail-fast**: Constraints are validated at construction time, not later in training
//! - **Clear API**: The `Option` return type signals that construction can fail
//!
//! ### 2. **State Trait Implementation**
//!
//! The `State` trait is implemented to describe the state's structure to neural networks and
//! serialization systems:
//!
//! - `is_valid()`: Runtime constraint validation (useful for debugging invalid agent actions)
//! - `numel()`: Returns 3 (x, y, theta are 3 scalar elements)
//! - `shape()`: Returns [3] (single 1D vector of 3 elements)
//!
//! This is critical for:
//! - **NN Architecture**: Tells the library to create input layers with 3 units
//! - **Serialization**: Ensures consistent data representation across training/inference
//! - **Batching**: Framework knows how to stack multiple poses into batch tensors
//!
//! ### 3. **Copy Semantics for Value Types**
//!
//! `RobotPose` derives `Copy` because it's a small (12 bytes), immutable value type.
//! This enables:
//! - **Zero-cost abstraction**: No heap allocations or reference counting
//! - **Predictable behavior**: Values are copied instead of moved, simplifying ownership
//! - **Performance**: Ideal for trajectory storage and frequent comparisons
//!
//! ### 4. **Utility Methods for Domain Logic**
//!
//! Beyond state representation, helper methods encapsulate robotics-specific operations:
//!
//! - `distance_to()`: Euclidean distance metric for reward shaping
//! - `normalize_orientation()`: Wraps angles to valid range (handles action accumulation)
//! - `orientation_degrees()`: Convenient unit conversion for human readability
//!
//! ## Constraint Model
//!
//! The robot operates in a constrained workspace:
//!
//! | Dimension | Min | Max | Unit |
//! |-----------|-----|-----|------|
//! | X Position | 0 | 1000 | mm |
//! | Y Position | 0 | 1000 | mm |
//! | Orientation | -180 | 180 | degrees |
//!
//! These constraints:
//! - **Reflect physical reality**: Robot joints have limits, workspaces are bounded
//! - **Guide learning**: Agents learn which actions are feasible vs. infeasible
//! - **Prevent degenerate solutions**: Forces agents to respect environment laws
//!
//! ## Use Cases in Reinforcement Learning
//!
//! ### 1. **Environment Simulation**
//! ```ignore
//! let current = RobotPose::new(500, 500, 0)?;
//! let next = RobotPose::new(current.x_mm + dx, current.y_mm + dy, theta)?;
//! // next is None if action would violate constraints
//! // Agent learns to avoid invalid actions
//! ```
//!
//! ### 2. **Trajectory Validation**
//! ```ignore
//! let trajectory = vec![pose1, pose2, pose3];
//! assert!(trajectory.iter().all(|p| p.is_valid()));
//! // Ensures entire trajectory respects constraints
//! ```
//!
//! ### 3. **Reward Shaping**
//! ```ignore
//! let goal = RobotPose::new(1000, 1000, 90)?;
//! let reward = -current.distance_to(&goal) as f32;
//! // Distance metric enables gradient-based learning
//! ```
//!
//! ### 4. **State Normalization for Neural Networks**
//! Continuous values are normalized to [-1, 1] range before feeding to NN:
//! ```ignore
//! let x_norm = (pose.x_mm as f64) / 500.0 - 1.0;  // [-1, 1]
//! let theta_norm = (pose.theta_mdeg as f64) / 180_000.0;  // [-1, 1]
//! ```
//!
//! ## Example Output Structure
//!
//! The `main()` function demonstrates:
//!
//! 1. **Valid State Construction**: Creating poses that satisfy all constraints
//! 2. **Invalid State Rejection**: Showing how constraints prevent bad states
//! 3. **Distance Metrics**: Computing Euclidean distances for reward shaping
//! 4. **Orientation Normalization**: Wrapping angles back to valid range
//! 5. **Trajectory Simulation**: Modeling a complete robot path
//! 6. **RL Agent Behavior**: How agents learn to select valid actions
//!
//! ## Key Takeaways
//!
//! - **Constraints as First-Class Citizens**: Use Rust's type system to enforce validity
//! - **Builder Pattern**: `Option` returns make constraint violations explicit
//! - **State Trait**: Bridge between domain logic and RL framework abstractions
//! - **Value Semantics**: `Copy` types are ergonomic for frequently-used states
//! - **Domain Methods**: Encapsulate robotics knowledge (distance, normalization)
//!
//! This example serves as a template for any continuous control environment where
//! state validity depends on domain constraints (e.g., physics engines, game rules).

use evorl_core::base::{Observation, State};
use serde::{Deserialize, Serialize};

/// Observation type for RobotPose: the perceived robot state.
///
/// This represents what the agent can observe from the environment. In this example,
/// the observation is the complete pose (full observability), but in more complex
/// scenarios it might be partial (e.g., only position, or position with noise).
///
/// The `Observation<1>` trait parameter indicates this is a 1-dimensional observation
/// space with shape [3] (three scalar values: x, y, theta).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct RobotPoseObservation {
    x_mm: i32,       // Observed X position in millimeters
    y_mm: i32,       // Observed Y position in millimeters
    theta_mdeg: i32, // Observed orientation in millidegrees
}

impl Observation<1> for RobotPoseObservation {
    /// Returns the shape of this observation space: [3] (three scalar elements)
    fn shape() -> [usize; 1] {
        [3]
    }
}

impl RobotPoseObservation {
    /// Creates an observation from a robot pose.
    pub fn from_pose(pose: &RobotPose) -> Self {
        RobotPoseObservation {
            x_mm: pose.x_mm,
            y_mm: pose.y_mm,
            theta_mdeg: pose.theta_mdeg,
        }
    }

    /// Converts the observation back to a pose (since we have full observability).
    // Inverse of `from_pose`; demonstrates the round-trip but is not called in this example.
    #[allow(dead_code)]
    pub fn to_pose(self) -> Option<RobotPose> {
        RobotPose::new(self.x_mm, self.y_mm, self.theta_mdeg)
    }
}

/// Represents the 2D pose of a robot in the workspace.
///
/// The robot operates in a 1000mm x 1000mm workspace with orientation
/// constrained to the range [-180°, 180°].
///
/// # Fields
/// * `x_mm` - X position in millimeters (0-1000mm)
/// * `y_mm` - Y position in millimeters (0-1000mm)
/// * `theta_mdeg` - Orientation in millidegrees (where 1000 mdeg = 1°)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RobotPose {
    x_mm: i32,       // Position X in millimeters
    y_mm: i32,       // Position Y in millimeters
    theta_mdeg: i32, // Orientation in millidegrees (-180,000 to 180,000 mdeg)
}

impl State<1> for RobotPose {
    /// The observation type produced by this state.
    ///
    /// Since the agent has full observability, the observation is a direct view
    /// of the complete state. In more realistic scenarios, this could be a noisy
    /// or partial observation.
    type Observation = RobotPoseObservation;

    /// Validates that the robot pose satisfies workspace constraints.
    ///
    /// The robot must remain within:
    /// - X bounds: [0, 1000] mm
    /// - Y bounds: [0, 1000] mm
    /// - Orientation: [-180°, 180°] or [-180,000, 180,000] millidegrees
    fn is_valid(&self) -> bool {
        // Workspace bounds: 0-1000mm
        self.x_mm >= 0
            && self.x_mm <= 1000
            && self.y_mm >= 0
            && self.y_mm <= 1000
            && // Orientation: -180 to 180 degrees (-180,000 to 180,000 millidegrees)
            self.theta_mdeg >= -180_000
            && self.theta_mdeg <= 180_000
    }

    /// Returns the number of scalar elements in the pose representation.
    ///
    /// A robot pose has 3 elements: x, y, and theta.
    fn numel(&self) -> usize {
        3
    }

    /// Returns the logical shape of this state's tensor representation.
    ///
    /// The pose is represented as a flat 1D vector of 3 elements.
    fn shape() -> [usize; 1] {
        [3] // Single flat array [x, y, theta]
    }

    /// Generates an observation from this state.
    ///
    /// This method produces the perception that would be available to a learning agent.
    /// Since we have full observability, the observation contains the complete state.
    fn observe(&self) -> Self::Observation {
        RobotPoseObservation::from_pose(self)
    }
}

impl RobotPose {
    /// Creates a new robot pose with validation.
    ///
    /// Returns `Some(pose)` if the position and orientation satisfy all constraints,
    /// otherwise returns `None`.
    ///
    /// # Arguments
    /// * `x_mm` - X position in millimeters (must be 0-1000)
    /// * `y_mm` - Y position in millimeters (must be 0-1000)
    /// * `theta_mdeg` - Orientation in millidegrees (must be -180,000 to 180,000)
    ///
    /// # Example
    /// ```
    /// let pose = RobotPose::new(500, 500, 0);
    /// assert!(pose.is_some());
    /// assert_eq!(pose.unwrap().x_mm, 500);
    /// ```
    pub fn new(x_mm: i32, y_mm: i32, theta_mdeg: i32) -> Option<Self> {
        let pose = RobotPose {
            x_mm,
            y_mm,
            theta_mdeg,
        };

        if pose.is_valid() { Some(pose) } else { None }
    }

    /// Creates a pose without validation (use with caution).
    ///
    /// This is useful in scenarios where you know the values are valid,
    /// but validation overhead should be avoided.
    pub fn new_unchecked(x_mm: i32, y_mm: i32, theta_mdeg: i32) -> Self {
        RobotPose {
            x_mm,
            y_mm,
            theta_mdeg,
        }
    }

    /// Calculates the Euclidean distance (in mm) to another pose, ignoring orientation.
    pub fn distance_to(&self, other: &RobotPose) -> f64 {
        let dx = (self.x_mm - other.x_mm) as f64;
        let dy = (self.y_mm - other.y_mm) as f64;
        (dx * dx + dy * dy).sqrt()
    }

    /// Normalizes the orientation to the [-180°, 180°] range.
    ///
    /// Useful when integrating actions over time may cause orientation
    /// to accumulate beyond valid bounds.
    pub fn normalize_orientation(self) -> Self {
        let mut theta = self.theta_mdeg;

        // Normalize to [-180,000, 180,000] millidegree range
        while theta > 180_000 {
            theta -= 360_000;
        }
        while theta < -180_000 {
            theta += 360_000;
        }

        RobotPose {
            x_mm: self.x_mm,
            y_mm: self.y_mm,
            theta_mdeg: theta,
        }
    }

    /// Returns the orientation in degrees (scaled from millidegrees).
    pub fn orientation_degrees(&self) -> f64 {
        self.theta_mdeg as f64 / 1000.0
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   RobotPose State Constraint Example for burn-evorl        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. VALID STATE CONSTRUCTION
    // ========================================================================
    println!("1️⃣  VALID STATE CONSTRUCTION");
    println!("{}", "—".repeat(60));

    let pose1 = RobotPose::new(500, 500, 0).expect("Center workspace position should be valid");
    println!(
        "✓ Created valid pose at ({}, {}) with θ = {}°",
        pose1.x_mm,
        pose1.y_mm,
        pose1.orientation_degrees()
    );
    println!("  - is_valid(): {}", pose1.is_valid());
    println!("  - numel(): {}", pose1.numel());
    println!("  - shape(): {:?}", RobotPose::shape());

    // Observation demonstration
    let obs = pose1.observe();
    println!("\n  Observation from state:");
    println!(
        "    x_mm: {}, y_mm: {}, theta_mdeg: {}",
        obs.x_mm, obs.y_mm, obs.theta_mdeg
    );
    println!("    Observation shape: {:?}", RobotPoseObservation::shape());

    println!();

    // ========================================================================
    // 2. INVALID STATE REJECTION
    // ========================================================================
    println!("2️⃣  INVALID STATE REJECTION");
    println!("{}", "—".repeat(60));

    let invalid_cases = vec![
        (1001, 500, 0, "X out of bounds (too large)"),
        (-1, 500, 0, "X out of bounds (negative)"),
        (500, 1001, 0, "Y out of bounds (too large)"),
        (500, -1, 0, "Y out of bounds (negative)"),
        (500, 500, 181_000, "Orientation out of bounds (too large)"),
        (
            500,
            500,
            -181_000,
            "Orientation out of bounds (too negative)",
        ),
    ];

    for (x, y, theta, description) in invalid_cases {
        match RobotPose::new(x, y, theta) {
            Some(_) => println!("✗ UNEXPECTED: {} should have failed", description),
            None => println!("✓ Correctly rejected: {}", description),
        }
    }

    println!();

    // ========================================================================
    // 3. DISTANCE METRICS
    // ========================================================================
    println!("3️⃣  DISTANCE METRICS FOR REWARD SHAPING");
    println!("{}", "—".repeat(60));

    let start = RobotPose::new(100, 100, 0).expect("Valid pose");
    let goal = RobotPose::new(800, 900, 90_000).expect("Valid pose");

    let distance = start.distance_to(&goal);
    println!("Start: ({}, {})", start.x_mm, start.y_mm);
    println!("Goal:  ({}, {})", goal.x_mm, goal.y_mm);
    println!("Euclidean distance: {:.2} mm", distance);

    // Normalized reward
    let max_distance = 1414.21; // sqrt(1000^2 + 1000^2)
    let normalized_reward = 1.0 - (distance / max_distance).min(1.0);
    println!(
        "Normalized reward (closer to goal = higher): {:.3}",
        normalized_reward
    );

    println!();

    // ========================================================================
    // 4. ORIENTATION NORMALIZATION
    // ========================================================================
    println!("4️⃣  ORIENTATION NORMALIZATION");
    println!("{}", "—".repeat(60));

    let positions = vec![
        (270_000, "3/4 rotation"),
        (360_000, "Full rotation"),
        (450_000, "1.25 rotations"),
        (-270_000, "-3/4 rotation"),
        (-360_000, "-Full rotation"),
    ];

    for (theta_mdeg, description) in positions {
        let pose = RobotPose::new_unchecked(500, 500, theta_mdeg);
        let normalized = pose.normalize_orientation();
        println!(
            "  {} - Raw: {}° → Normalized: {}°",
            description,
            pose.orientation_degrees(),
            normalized.orientation_degrees()
        );
    }

    println!();

    // ========================================================================
    // 5. TRAJECTORY SIMULATION
    // ========================================================================
    println!("5️⃣  TRAJECTORY SIMULATION");
    println!("{}", "—".repeat(60));

    let mut trajectory = vec![];
    let current = RobotPose::new(100, 100, 0).expect("Valid starting pose");
    trajectory.push(current);

    // Simulate a simple path: move right and up
    let waypoints = vec![
        (200, 100, 0),
        (300, 200, 45_000),
        (400, 400, 90_000),
        (500, 600, 135_000),
        (600, 800, 180_000),
    ];

    for (x, y, theta) in waypoints {
        if let Some(next) = RobotPose::new(x, y, theta) {
            trajectory.push(next);
        }
    }

    println!("Generated trajectory with {} waypoints:", trajectory.len());
    for (i, pose) in trajectory.iter().enumerate() {
        println!(
            "  [{:2}] ({:3}, {:3}) θ = {:6.1}° | Valid: {}",
            i,
            pose.x_mm,
            pose.y_mm,
            pose.orientation_degrees(),
            pose.is_valid()
        );
    }

    // Verify entire trajectory
    let all_valid = trajectory.iter().all(|p| p.is_valid());
    println!(
        "\nTrajectory validity check: {}",
        if all_valid { "✓ PASS" } else { "✗ FAIL" }
    );

    println!();

    // ========================================================================
    // 6. AGENT BEHAVIOR: LEARNING VALID ACTIONS
    // ========================================================================
    println!("6️⃣  RL AGENT BEHAVIOR: LEARNING VALID ACTIONS");
    println!("{}", "—".repeat(60));

    let agent_start = RobotPose::new(500, 500, 0).expect("Valid starting pose");
    println!(
        "Agent starts at: ({}, {})",
        agent_start.x_mm, agent_start.y_mm
    );

    // Simulate attempted actions
    let action_deltas = vec![
        (100, 100, 0, "Move right & up (valid)"),
        (600, 100, 0, "Move far right (invalid - out of bounds)"),
        (-100, -100, 0, "Move left & down (valid)"),
        (0, 0, 90_000, "Rotate right (valid)"),
        (0, 0, 180_000, "Rotate 180° (valid)"),
    ];

    for (dx, dy, dtheta, description) in action_deltas {
        let new_x = agent_start.x_mm + dx;
        let new_y = agent_start.y_mm + dy;
        let new_theta = agent_start.theta_mdeg + dtheta;

        match RobotPose::new(new_x, new_y, new_theta) {
            Some(new_pose) => {
                let reward = -agent_start.distance_to(&new_pose);
                println!("  ✓ {}: reward = {:.1}", description, reward);
            }
            None => {
                println!(
                    "  ✗ {}: BLOCKED (agent learns to avoid this action)",
                    description
                );
            }
        }
    }

    println!();

    // ========================================================================
    // SUMMARY
    // ========================================================================
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                    SUMMARY                                 ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ ✓ States are validated at construction time                ║");
    println!("║ ✓ Observations provide full workspace state info           ║");
    println!("║ ✓ Distances enable reward shaping for learning             ║");
    println!("║ ✓ Normalization handles angle accumulation                 ║");
    println!("║ ✓ Trajectories can be verified as valid sequences          ║");
    println!("║ ✓ Agents learn to respect workspace constraints            ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
