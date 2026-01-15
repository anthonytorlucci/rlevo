use evorl_core::action::{Action, MultiDiscreteAction};

#[derive(Debug, Clone)]
struct CombatAction {
    direction: u8, // 0-3: North, East, South, West
    attack: u8,    // 0-2: Light, Heavy, Special
}

// impl Action for CombatAction {
//     fn is_valid(&self) -> bool {
//         self.direction < 4 && self.attack < 3
//     }
// }

// impl MultiDiscreteAction<2> for CombatAction {
//     fn action_space() -> [usize; 2] {
//         [4, 3] // 4 directions × 3 attacks = 12 total combinations
//     }

//     fn from_indices(indices: [usize; 2]) -> Self {
//         Self {
//             direction: indices[0] as u8,
//             attack: indices[1] as u8,
//         }
//     }

//     fn to_indices(&self) -> [usize; 2] {
//         [self.direction as usize, self.attack as usize]
//     }
// }

impl CombatAction {
    /// Get a human-readable direction name.
    fn direction_name(&self) -> &'static str {
        match self.direction {
            0 => "North",
            1 => "East",
            2 => "South",
            3 => "West",
            _ => "Unknown",
        }
    }

    /// Get a human-readable attack name.
    fn attack_name(&self) -> &'static str {
        match self.attack {
            0 => "Light",
            1 => "Heavy",
            2 => "Special",
            _ => "Unknown",
        }
    }

    /// Get a description of the combat action.
    fn describe(&self) -> String {
        format!(
            "{} attack towards {}",
            self.attack_name(),
            self.direction_name()
        )
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Combat Action Space Demo ===\n");

    // 1. Demonstrate action space dimensions
    println!("1. Action Space Dimensions:");
    // let space = CombatAction::action_space();
    // println!("   Dimensions: {}", CombatAction::DIM);
    // println!("   Space: {:?}", space);
    // let total_actions = space.iter().product::<usize>();
    // println!("   Total possible actions: {}\n", total_actions);

    // 2. Create specific actions from indices
    println!("2. Creating Actions from Indices:");
    // let action1 = CombatAction::from_indices([0, 1]); // North + Heavy
    // println!(
    //     "   Action [0, 1]: {} - Valid: {}",
    //     action1.describe(),
    //     action1.is_valid()
    // );

    // let action2 = CombatAction::from_indices([3, 2]); // West + Special
    // println!(
    //     "   Action [3, 2]: {} - Valid: {}",
    //     action2.describe(),
    //     action2.is_valid()
    // );

    // let action3 = CombatAction::from_indices([2, 0]); // South + Light
    // println!(
    //     "   Action [2, 0]: {} - Valid: {}\n",
    //     action3.describe(),
    //     action3.is_valid()
    // );

    // 3. Demonstrate index conversion (roundtrip)
    println!("3. Index Conversion (Roundtrip):");
    // let original = CombatAction::from_indices([1, 2]);
    // println!("   Original indices: [1, 2]");
    // println!("   Original action: {}", original.describe());
    // let converted = original.to_indices();
    // println!("   Converted back: {:?}", converted);
    // println!("   Match: {}\n", original.to_indices() == [1, 2]);

    // 4. Generate random actions
    println!("4. Random Action Generation:");
    // for i in 0..5 {
    //     let random_action = CombatAction::random();
    //     println!(
    //         "   Random action {}: {} - Valid: {}",
    //         i + 1,
    //         random_action.describe(),
    //         random_action.is_valid()
    //     );
    // }
    // println!();

    // 5. Enumerate all possible actions
    println!("5. Enumerating All Possible Actions:");
    // let all_actions = CombatAction::enumerate();
    // println!("   Total actions enumerated: {}", all_actions.len());
    // for (i, action) in all_actions.iter().enumerate() {
    //     let indices = action.to_indices();
    //     println!(
    //         "   {}: {:?} -> {} (valid: {})",
    //         i,
    //         indices,
    //         action.describe(),
    //         action.is_valid()
    //     );
    // }
    // println!();

    // 6. Combat scenario simulation
    println!("6. Combat Scenario Simulation:");
    // let player_action = CombatAction::from_indices([1, 1]); // East + Heavy
    // let enemy_action = CombatAction::random();

    // println!("   Player action: {}", player_action.describe());
    // println!("   Enemy action: {}", enemy_action.describe());

    // Determine clash outcome based on directions
    // let player_dir = player_action.direction;
    // let enemy_dir = enemy_action.direction;

    // if player_dir == enemy_dir {
    //     println!("   Result: Both attack in same direction - MUTUAL STRIKE!");
    // } else if (player_dir + 2) % 4 == enemy_dir {
    //     println!("   Result: Attacks are opposite - BLOCK!");
    // } else {
    //     println!("   Result: Attacks are perpendicular - GLANCING BLOW!");
    // }
    // println!();

    // 7. Batch processing actions
    println!("7. Batch Processing Actions:");
    let action_indices = vec![[0, 0], [1, 1], [2, 2], [3, 0]];
    // for indices in action_indices {
    //     let action = CombatAction::from_indices(indices);
    //     println!("   {:?} -> {}", indices, action.describe());
    // }
    // println!();

    // 8. Validate action constraints
    println!("8. Action Validation:");
    let valid_action = CombatAction {
        direction: 2,
        attack: 1,
    };
    // println!(
    //     "   Valid action (dir=2, atk=1): {}",
    //     valid_action.is_valid()
    // );

    // let invalid_action = CombatAction {
    //     direction: 5, // Invalid: > 3
    //     attack: 1,
    // };
    // println!(
    //     "   Invalid action (dir=5, atk=1): {}",
    //     invalid_action.is_valid()
    // );

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
