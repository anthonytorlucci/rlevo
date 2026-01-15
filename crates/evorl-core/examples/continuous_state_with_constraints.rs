//! # RobotPose: Continuous State with Workspace Constraints
//!
//! This example demonstrates how to implement a constrained state representation for reinforcement learning
//! environments. It showcases several key design patterns and best practices for the Burn-EvoRL library.
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

use evorl_core::state::State;

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

// impl State for RobotPose {
//     /// Validates that the robot pose satisfies workspace constraints.
//     ///
//     /// The robot must remain within:
//     /// - X bounds: [0, 1000] mm
//     /// - Y bounds: [0, 1000] mm
//     /// - Orientation: [-180°, 180°] or [-180,000, 180,000] millidegrees
//     fn is_valid(&self) -> bool {
//         // Workspace bounds: 0-1000mm
//         self.x_mm >= 0
//             && self.x_mm <= 1000
//             && self.y_mm >= 0
//             && self.y_mm <= 1000
//             && // Orientation: -180 to 180 degrees (-180,000 to 180,000 millidegrees)
//             self.theta_mdeg >= -180_000
//             && self.theta_mdeg <= 180_000
//     }

//     /// Returns the number of scalar elements in the pose representation.
//     ///
//     /// A robot pose has 3 elements: x, y, and theta.
//     fn numel(&self) -> usize {
//         3
//     }

//     /// Returns the logical shape of this state's tensor representation.
//     ///
//     /// The pose is represented as a flat 1D vector of 3 elements.
//     fn shape(&self) -> Vec<usize> {
//         vec![3] // Single flat vector [x, y, theta]
//     }
// }

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

        // if pose.is_valid() {
        //     Some(pose)
        // } else {
        //     None
        // }
        todo!()
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
    println!("║   RobotPose State Constraint Example for Burn-EvoRL        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. CONSTRUCTING VALID ROBOT POSES
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 1. Constructing Valid Robot Poses                           │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Home position (center of workspace)
    // if let Some(home) = RobotPose::new(500, 500, 0) {
    //     println!("✓ Home Pose (center, facing east):");
    //     println!("  Position: ({}, {}) mm", home.x_mm, home.y_mm);
    //     println!("  Orientation: {}°", home.orientation_degrees());
    //     println!(
    //         "  Valid: {} | Elements: {} | Shape: {:?}",
    //         home.is_valid(),
    //         home.numel(),
    //         home.shape()
    //     );
    //     println!();
    // }

    // Corner position (facing northeast)
    // if let Some(corner) = RobotPose::new(1000, 1000, 45_000) {
    //     println!("✓ Corner Pose (max position, facing northeast):");
    //     println!("  Position: ({}, {}) mm", corner.x_mm, corner.y_mm);
    //     println!("  Orientation: {}°", corner.orientation_degrees());
    //     println!("  Valid: {}", corner.is_valid());
    //     println!();
    // }

    // Negative angle (facing southwest)
    // if let Some(southwest) = RobotPose::new(0, 0, -90_000) {
    //     println!("✓ Southwest Pose (origin, facing south):");
    //     println!("  Position: ({}, {}) mm", southwest.x_mm, southwest.y_mm);
    //     println!("  Orientation: {}°", southwest.orientation_degrees());
    //     println!("  Valid: {}", southwest.is_valid());
    //     println!();
    // }

    // ========================================================================
    // 2. ATTEMPTING INVALID POSES
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 2. Attempting Invalid Poses (Constraint Violations)         │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    let invalid_cases = vec![
        ("X exceeds max", 1500, 500, 0),
        ("X below minimum", -100, 500, 0),
        ("Y exceeds max", 500, 2000, 0),
        ("Y below minimum", 500, -50, 0),
        ("Orientation > 180°", 500, 500, 270_000),
        ("Orientation < -180°", 500, 500, -200_000),
    ];

    // for (reason, x, y, theta) in invalid_cases {
    //     match RobotPose::new(x, y, theta) {
    //         Some(_) => println!("  ✓ {} → Created successfully", reason),
    //         None => println!("  ✗ {} → Rejected (constraint violation)", reason),
    //     }
    // }
    // println!();

    // ========================================================================
    // 3. DISTANCE CALCULATIONS
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 3. Distance Calculations (Euclidean)                        │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    let pose_a = RobotPose::new(100, 100, 0).unwrap();
    let pose_b = RobotPose::new(400, 400, 45_000).unwrap();
    let pose_c = RobotPose::new(100, 400, 90_000).unwrap();

    let dist_ab = pose_a.distance_to(&pose_b);
    let dist_ac = pose_a.distance_to(&pose_c);
    let dist_bc = pose_b.distance_to(&pose_c);

    println!(
        "Pose A: x={}, y={}, θ={}°",
        pose_a.x_mm,
        pose_a.y_mm,
        pose_a.orientation_degrees()
    );
    println!(
        "Pose B: x={}, y={}, θ={}°",
        pose_b.x_mm,
        pose_b.y_mm,
        pose_b.orientation_degrees()
    );
    println!(
        "Pose C: x={}, y={}, θ={}°",
        pose_c.x_mm,
        pose_c.y_mm,
        pose_c.orientation_degrees()
    );
    println!();
    println!("  Distance A→B: {:.2} mm", dist_ab);
    println!("  Distance A→C: {:.2} mm", dist_ac);
    println!("  Distance B→C: {:.2} mm", dist_bc);
    println!();

    // ========================================================================
    // 4. ORIENTATION NORMALIZATION
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 4. Orientation Normalization                                │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Create poses with angles outside the valid range
    let poses_to_normalize = vec![
        RobotPose::new_unchecked(500, 500, 360_000), // 360° → should normalize to 0°
        RobotPose::new_unchecked(500, 500, -270_000), // -270° → should normalize to 90°
        RobotPose::new_unchecked(500, 500, 540_000), // 540° → should normalize to 180°
    ];

    // for pose in poses_to_normalize {
    //     let normalized = pose.normalize_orientation();
    //     println!(
    //         "  {:.0}° → {:.0}° | Valid before: {}, after: {}",
    //         pose.orientation_degrees(),
    //         normalized.orientation_degrees(),
    //         pose.is_valid(),
    //         normalized.is_valid()
    //     );
    // }
    // println!();

    // ========================================================================
    // 5. ROBOT TRAJECTORY / STATE SEQUENCE
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 5. Robot Trajectory (State Transitions)                     │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    let mut trajectory = Vec::new();

    // Simulate a robot moving diagonally while rotating
    println!("Simulating robot motion from (0,0) to (1000,1000):");
    println!("  Step | X (mm) | Y (mm) | Orientation | Valid?");
    println!("  -----|--------|--------|-------------|-------");

    for step in 0..=10 {
        let progress = step as i32 * 100;
        let angle = (step as i32 - 5) * 18_000; // Rotate from -90° to +90°

        if let Some(pose) = RobotPose::new(progress, progress, angle) {
            trajectory.push(pose);
            println!(
                "  {:4} | {:6} | {:6} | {:11.1}° | YES",
                step,
                pose.x_mm,
                pose.y_mm,
                pose.orientation_degrees()
            );
        }
    }
    println!(
        "\nTrajectory: {} valid poses out of 11 steps\n",
        trajectory.len()
    );

    // ========================================================================
    // 6. USE CASE: REINFORCEMENT LEARNING VALIDATION
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ 6. RL Use Case: Agent Accepting/Rejecting State Actions    │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Current robot state
    let current_state = RobotPose::new(500, 500, 0).unwrap();
    println!(
        "Current State: x={}, y={}, θ={}°\n",
        current_state.x_mm,
        current_state.y_mm,
        current_state.orientation_degrees()
    );

    // Potential actions (dx, dy, dθ) that the agent might take
    let potential_actions = vec![
        ("Move +100X", 100, 0, 0),
        ("Move +100Y", 0, 100, 0),
        ("Rotate +45°", 0, 0, 45_000),
        ("Move to boundary", 500, 500, 0),
        ("Move out of bounds", 600, 0, 0), // Would go to x=1100 (invalid)
        ("Rotate out of bounds", 0, 0, 200_000), // Would go to θ=200° (invalid)
    ];

    println!("Evaluating potential actions:");
    for (action_name, dx, dy, dtheta) in potential_actions {
        let next_x = current_state.x_mm + dx;
        let next_y = current_state.y_mm + dy;
        let next_theta = current_state.theta_mdeg + dtheta;

        match RobotPose::new(next_x, next_y, next_theta) {
            Some(next_state) => {
                println!(
                    "  ✓ {} → Valid | New state: ({}, {}) @{:.0}°",
                    action_name,
                    next_state.x_mm,
                    next_state.y_mm,
                    next_state.orientation_degrees()
                );
            }
            None => {
                println!(
                    "  ✗ {} → INVALID | Would result in: ({}, {}) @{:.0}°",
                    action_name,
                    next_x,
                    next_y,
                    next_theta as f64 / 1000.0
                );
            }
        }
    }
    println!();

    // ========================================================================
    // 7. SUMMARY
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Summary: RobotPose Constraint Enforcement                   │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("✓ RobotPose properly constrains states to a 1000×1000mm workspace");
    println!("✓ Orientation bounded to [-180°, 180°] range");
    println!("✓ State trait implementation: is_valid(), numel(), shape()");
    println!("✓ Builder pattern: RobotPose::new() with validation");
    println!("✓ Utility methods: distance_to(), normalize_orientation()");
    println!("✓ Ready for RL agent training with guaranteed constraint satisfaction\n");

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
