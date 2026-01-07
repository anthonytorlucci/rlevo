use evorl_core::action::{Action, ContinuousAction};

#[derive(Debug, Clone)]
struct RobotControl {
    joint_angles: [f32; 6],
}

impl Action for RobotControl {
    fn is_valid(&self) -> bool {
        self.joint_angles.iter().all(|&x| x.is_finite())
    }
}

impl ContinuousAction for RobotControl {
    const DIM: usize = 6;

    fn as_slice(&self) -> &[f32] {
        &self.joint_angles
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        let mut clipped = self.clone();
        clipped
            .joint_angles
            .iter_mut()
            .for_each(|x| *x = x.clamp(min, max));
        clipped
    }

    fn from_slice(values: &[f32]) -> Self {
        let mut angles = [0.0; 6];
        angles.copy_from_slice(values);
        Self {
            joint_angles: angles,
        }
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Robot Control Action Example ===\n");

    // Example 1: Creating a RobotControl with specific joint angles
    println!("1. Creating a RobotControl instance:");
    let robot = RobotControl {
        joint_angles: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    };
    println!("   Robot: {:?}", robot);
    println!("   Action dimension: {}\n", RobotControl::DIM);

    // Example 2: Validating the action
    println!("2. Validating robot control actions:");
    println!("   Valid robot is_valid(): {}", robot.is_valid());

    let invalid_robot = RobotControl {
        joint_angles: [0.5, f32::NAN, 1.5, 2.0, 2.5, 3.0],
    };
    println!(
        "   Robot with NaN is_valid(): {}\n",
        invalid_robot.is_valid()
    );

    // Example 3: Getting the action as a slice
    println!("3. Getting joint angles as slice:");
    let angles = robot.as_slice();
    println!("   Joint angles (radians): {:?}", angles);
    println!("   Slice length: {}\n", angles.len());

    // Example 4: Iterating and converting to degrees
    println!("4. Joint angles in degrees:");
    for (index, &angle) in angles.iter().enumerate() {
        println!(
            "   Joint {}: {:.4} rad = {:.2}°",
            index + 1,
            angle,
            angle.to_degrees()
        );
    }
    println!();

    // Example 5: Clipping values to safe ranges
    println!("5. Clipping actions to safe ranges:");
    let unconstrained = RobotControl {
        joint_angles: [0.5, 2.5, 1.5, 4.5, -1.0, 3.0],
    };
    println!("   Unconstrained: {:?}", unconstrained);

    let min_angle = -std::f32::consts::PI;
    let max_angle = std::f32::consts::PI;
    let constrained = unconstrained.clip(min_angle, max_angle);
    println!("   Clipped to [{:.3}, {:.3}]:", min_angle, max_angle);
    println!("   Result: {:?}\n", constrained);

    // Example 6: Creating from a slice
    println!("6. Creating RobotControl from slice:");
    let angles_array = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
    let robot_from_slice = RobotControl::from_slice(&angles_array);
    println!("   Input slice: {:?}", angles_array);
    println!("   Created robot: {:?}\n", robot_from_slice);

    // Example 7: Practical scenario - robot reaching goal
    println!("7. Practical scenario - validating control signal:");
    let control_signal = RobotControl {
        joint_angles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    };

    if control_signal.is_valid() {
        let signal_slice = control_signal.as_slice();
        let max_signal = signal_slice
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_signal = signal_slice.iter().copied().fold(f32::INFINITY, f32::min);
        println!("   Signal is valid: true");
        println!("   Joint angles: {:?}", signal_slice);
        println!("   Range: [{:.4}, {:.4}]\n", min_signal, max_signal);
    }

    // Example 8: Processing multiple robot trajectories
    println!("8. Processing multiple robot states:");
    let robot_states = vec![
        RobotControl {
            joint_angles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
        RobotControl {
            joint_angles: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        },
        RobotControl {
            joint_angles: [-0.5, -0.3, 0.0, 0.3, 0.5, 0.7],
        },
    ];

    for (idx, state) in robot_states.iter().enumerate() {
        let is_valid = state.is_valid();
        let angles = state.as_slice();
        let mean_angle = angles.iter().sum::<f32>() / angles.len() as f32;
        println!(
            "   State {}: Valid={}, Mean angle={:.4} rad",
            idx, is_valid, mean_angle
        );
    }
    println!();

    // Example 9: Comparing soft and strict safety limits
    println!("9. Applying different safety constraints:");
    let raw_control = RobotControl {
        joint_angles: [2.0, 3.0, 1.5, -2.0, 4.0, 0.5],
    };

    println!("   Raw control: {:?}", raw_control);
    let soft_limit = raw_control.clip(-2.0, 2.0);
    println!("   Soft limit [-2.0, 2.0]: {:?}", soft_limit);

    let strict_limit = raw_control.clip(-1.0, 1.0);
    println!("   Strict limit [-1.0, 1.0]: {:?}", strict_limit);
    println!();

    // Example 10: Verifying round-trip conversion
    println!("10. Verifying round-trip slice conversion:");
    let original = RobotControl {
        joint_angles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    };
    let reconstructed = RobotControl::from_slice(original.as_slice());

    println!("    Original:      {:?}", original);
    println!("    Reconstructed: {:?}", reconstructed);
    println!(
        "    Match: {}",
        original.as_slice() == reconstructed.as_slice()
    );

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
