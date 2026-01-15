use evorl_core::action::{Action, ContinuousAction};

#[derive(Debug, Clone)]
struct VehicleControl {
    steering: f32, // -1.0 (left) to 1.0 (right)
    throttle: f32, // 0.0 (idle) to 1.0 (full)
}

// impl Action for VehicleControl {
//     fn is_valid(&self) -> bool {
//         self.steering.is_finite()
//             && self.throttle.is_finite()
//             && self.steering >= -1.0
//             && self.steering <= 1.0
//             && self.throttle >= 0.0
//             && self.throttle <= 1.0
//     }
// }

// impl ContinuousAction for VehicleControl {
//     const DIM: usize = 2;

//     fn as_slice(&self) -> &[f32] {
//         // Use unsafe to reinterpret struct as slice
//         unsafe { std::slice::from_raw_parts(self as *const _ as *const f32, 2) }
//     }

//     fn clip(&self, min: f32, max: f32) -> Self {
//         Self {
//             steering: self.steering.clamp(min, max),
//             throttle: self.throttle.clamp(min, max),
//         }
//     }

//     fn from_slice(values: &[f32]) -> Self {
//         Self {
//             steering: values[0],
//             throttle: values[1],
//         }
//     }
// }

// --------------------------------------------------------------------------
// Example Usage
// --------------------------------------------------------------------------
// 1. Basic Action Creation** (Examples 1-3)
// - Forward motion with straight steering
// - Left/right turns with throttle control
// - Idle states with no input
//
// **2. Conversion Operations** (Examples 4-5)
// - `as_slice()` conversion for neural network input/output
// - `from_slice()` reconstruction from raw values
// - Demonstrates the `DIM = 2` constant
//
// **3. Action Clipping** (Example 6)
// - Shows how to constrain out-of-bounds values to valid ranges
// - Before/after validation comparison
//
// **4. Validation** (Examples 7, 10)
// - Detects NaN (not-a-number) values
// - Checks boundary constraints (steering: [-1.0, 1.0], throttle: [0.0, 1.0])
// - Batch validation with status indicators (✓/✗)
//
// **5. Round-Trip Conversion** (Example 9)
// - Struct → Slice → Struct conversion pipeline
// - Verifies data integrity with floating-point tolerance check
//
// **6. Practical Scenarios** (Examples 8)
// - Lane change sequence showing realistic driving behavior
// - Sequential action progressio
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VehicleControl Action Examples ===\n");

    // Example 1: Create a straightforward forward action
    println!("Example 1: Straightforward Forward Action");
    let forward_action = VehicleControl {
        steering: 0.0, // straight
        throttle: 0.5, // 50% throttle
    };
    // println!("  Action: {:?}", forward_action);
    // println!("  Valid: {}\n", forward_action.is_valid());

    // Example 2: Create a left turn with full throttle
    println!("Example 2: Left Turn with Full Throttle");
    let left_turn = VehicleControl {
        steering: -0.75, // hard left
        throttle: 1.0,   // full throttle
    };
    // println!("  Action: {:?}", left_turn);
    // println!("  Valid: {}\n", left_turn.is_valid());

    // Example 3: Create an idle action (no steering, no throttle)
    println!("Example 3: Idle Action");
    let idle = VehicleControl {
        steering: 0.0,
        throttle: 0.0,
    };
    // println!("  Action: {:?}", idle);
    // println!("  Valid: {}\n", idle.is_valid());

    // Example 4: Demonstrate conversion to slice
    println!("Example 4: Conversion to Slice");
    let action = VehicleControl {
        steering: 0.25,
        throttle: 0.75,
    };
    // let slice = action.as_slice();
    // println!("  Action: {:?}", action);
    // println!("  As slice: {:?}", slice);
    // println!("  Dimension: {}\n", VehicleControl::DIM);

    // Example 5: Demonstrate conversion from slice
    println!("Example 5: Conversion from Slice");
    let values = [0.5, 0.3];
    // let reconstructed = VehicleControl::from_slice(&values);
    // println!("  Input slice: {:?}", values);
    // println!("  Reconstructed action: {:?}", reconstructed);
    // println!("  Valid: {}\n", reconstructed.is_valid());

    // Example 6: Demonstrate clipping to valid range
    println!("Example 6: Clipping Actions");
    let overconstrained = VehicleControl {
        steering: 1.5, // exceeds max
        throttle: 1.2, // exceeds max
    };
    // println!("  Original action: {:?}", overconstrained);
    // println!("  Valid: {}", overconstrained.is_valid());

    // let clipped = overconstrained.clip(-1.0, 1.0);
    // println!("  Clipped action: {:?}", clipped);
    // println!("  Valid: {}\n", clipped.is_valid());

    // Example 7: Demonstrate invalid actions
    println!("Example 7: Invalid Actions");
    let invalid_steering = VehicleControl {
        steering: f32::NAN,
        throttle: 0.5,
    };
    println!("  Action with NaN: {:?}", invalid_steering);
    // println!("  Valid: {}", invalid_steering.is_valid());

    let invalid_range = VehicleControl {
        steering: 0.5,
        throttle: -0.5, // below minimum
    };
    println!("  Action with negative throttle: {:?}", invalid_range);
    // println!("  Valid: {}\n", invalid_range.is_valid());

    // Example 8: Practical scenario - lane change sequence
    println!("Example 8: Practical Scenario - Lane Change Sequence");
    let lane_change_sequence = vec![
        VehicleControl {
            steering: 0.0,
            throttle: 0.6,
        }, // cruising
        VehicleControl {
            steering: 0.3,
            throttle: 0.6,
        }, // start turn
        VehicleControl {
            steering: 0.8,
            throttle: 0.6,
        }, // sharp turn
        VehicleControl {
            steering: 0.0,
            throttle: 0.6,
        }, // straighten out
    ];

    // for (idx, action) in lane_change_sequence.iter().enumerate() {
    //     println!(
    //         "  Step {}: {:?} | Valid: {}",
    //         idx + 1,
    //         action,
    //         action.is_valid()
    //     );
    // }
    // println!();

    // Example 9: Demonstrate round-trip conversion
    println!("Example 9: Round-Trip Conversion (Struct -> Slice -> Struct)");
    let original = VehicleControl {
        steering: 0.33,
        throttle: 0.67,
    };
    println!("  Original: {:?}", original);

    // let as_slice = original.as_slice();
    // println!("  As slice: {:?}", as_slice);

    // let reconstructed = VehicleControl::from_slice(as_slice);
    // println!("  Reconstructed: {:?}", reconstructed);
    // println!(
    //     "  Match: {}\n",
    //     (original.steering - reconstructed.steering).abs() < 1e-5
    //         && (original.throttle - reconstructed.throttle).abs() < 1e-5
    // );

    // Example 10: Batch validation
    println!("Example 10: Batch Validation");
    let actions = vec![
        VehicleControl {
            steering: 0.0,
            throttle: 0.5,
        },
        VehicleControl {
            steering: 1.0,
            throttle: 1.0,
        },
        VehicleControl {
            steering: -1.0,
            throttle: 0.0,
        },
        VehicleControl {
            steering: 0.5,
            throttle: 2.0,
        }, // invalid
        VehicleControl {
            steering: f32::INFINITY,
            throttle: 0.5,
        }, // invalid
    ];

    // let valid_count = actions.iter().filter(|a| a.is_valid()).count();
    // println!("  Total actions: {}", actions.len());
    // println!("  Valid actions: {}", valid_count);
    // println!("  Invalid actions: {}", actions.len() - valid_count);

    // for (idx, action) in actions.iter().enumerate() {
    //     let status = if action.is_valid() { "✓" } else { "✗" };
    //     println!("    {} Action {}: {:?}", status, idx + 1, action);
    // }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
