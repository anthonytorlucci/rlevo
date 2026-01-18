use evorl_core::action::{Action, MultiDiscreteAction};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    /// Convert Direction to its index representation.
    fn to_index(self) -> usize {
        match self {
            Direction::North => 0,
            Direction::East => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }

    /// Create a Direction from an index.
    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Direction::North),
            1 => Some(Direction::East),
            2 => Some(Direction::South),
            3 => Some(Direction::West),
            _ => None,
        }
    }

    /// Get a human-readable name for this direction.
    fn name(self) -> &'static str {
        match self {
            Direction::North => "North",
            Direction::East => "East",
            Direction::South => "South",
            Direction::West => "West",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttackStrength {
    Light,
    Heavy,
    Special,
}

impl AttackStrength {
    /// Convert AttackStrength to its index representation.
    fn to_index(self) -> usize {
        match self {
            AttackStrength::Light => 0,
            AttackStrength::Heavy => 1,
            AttackStrength::Special => 2,
        }
    }

    /// Create an AttackStrength from an index.
    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(AttackStrength::Light),
            1 => Some(AttackStrength::Heavy),
            2 => Some(AttackStrength::Special),
            _ => None,
        }
    }

    /// Get a human-readable name for this attack strength.
    fn name(self) -> &'static str {
        match self {
            AttackStrength::Light => "Light",
            AttackStrength::Heavy => "Heavy",
            AttackStrength::Special => "Special",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CombatAction {
    direction: Direction,
    attack: AttackStrength,
}

impl Action<2> for CombatAction {
    fn is_valid(&self) -> bool {
        true // Enums are always valid by construction
    }

    fn shape() -> [usize; 2] {
        [4, 3] // 4 directions × 3 attacks = 12 total combinations
    }
}

impl MultiDiscreteAction<2> for CombatAction {
    fn from_indices(indices: [usize; 2]) -> Self {
        Self {
            direction: Direction::from_index(indices[0]).expect("Invalid direction index"),
            attack: AttackStrength::from_index(indices[1]).expect("Invalid attack index"),
        }
    }

    fn to_indices(&self) -> [usize; 2] {
        [self.direction.to_index(), self.attack.to_index()]
    }
}

impl CombatAction {
    /// Create a new CombatAction with the specified direction and attack.
    fn new(direction: Direction, attack: AttackStrength) -> Self {
        Self { direction, attack }
    }

    /// Get a description of the combat action.
    fn describe(&self) -> String {
        format!(
            "{} attack towards {}",
            self.attack.name(),
            self.direction.name()
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
    let action1 = CombatAction::from_indices([0, 1]); // North + Heavy
    println!(
        "   Action [0, 1]: {} - Valid: {}",
        action1.describe(),
        action1.is_valid()
    );

    let action2 = CombatAction::from_indices([3, 2]); // West + Special
    println!(
        "   Action [3, 2]: {} - Valid: {}",
        action2.describe(),
        action2.is_valid()
    );

    let action3 = CombatAction::from_indices([2, 0]); // South + Light
    println!(
        "   Action [2, 0]: {} - Valid: {}\n",
        action3.describe(),
        action3.is_valid()
    );

    // 3. Demonstrate index conversion (roundtrip)
    println!("3. Index Conversion (Roundtrip):");
    let original = CombatAction::from_indices([1, 2]);
    println!("   Original indices: [1, 2]");
    println!("   Original action: {}", original.describe());
    let converted = original.to_indices();
    println!("   Converted back: {:?}", converted);
    println!("   Match: {}\n", original.to_indices() == [1, 2]);

    // 4. Direct enum-based action creation
    println!("4. Direct Enum-Based Action Creation:");
    let attack1 = CombatAction::new(Direction::North, AttackStrength::Heavy);
    println!(
        "   Created action: {} - Valid: {}",
        attack1.describe(),
        attack1.is_valid()
    );

    let attack2 = CombatAction::new(Direction::West, AttackStrength::Special);
    println!(
        "   Created action: {} - Valid: {}\n",
        attack2.describe(),
        attack2.is_valid()
    );

    // 5. Combat scenario simulation
    println!("5. Combat Scenario Simulation:");
    let player_action = CombatAction::from_indices([1, 1]); // East + Heavy
    let enemy_action = CombatAction::from_indices([2, 0]); // South + Light

    println!("   Player action: {}", player_action.describe());
    println!("   Enemy action: {}", enemy_action.describe());

    // Determine clash outcome based on directions
    let player_dir = player_action.direction.to_index();
    let enemy_dir = enemy_action.direction.to_index();

    if player_dir == enemy_dir {
        println!("   Result: Both attack in same direction - MUTUAL STRIKE!");
    } else if (player_dir + 2) % 4 == enemy_dir {
        println!("   Result: Attacks are opposite - BLOCK!");
    } else {
        println!("   Result: Attacks are perpendicular - GLANCING BLOW!");
    }
    println!();

    // 6. Batch processing actions
    println!("6. Batch Processing Actions:");
    let action_indices = vec![[0, 0], [1, 1], [2, 2], [3, 0]];
    for indices in action_indices {
        let action = CombatAction::from_indices(indices);
        println!("   {:?} -> {}", indices, action.describe());
    }
    println!();

    // 7. Validate action constraints
    println!("7. Action Validation:");
    let valid_action = CombatAction::new(Direction::South, AttackStrength::Heavy);
    println!(
        "   Valid action (South, Heavy): {}",
        valid_action.is_valid()
    );

    let another_action = CombatAction::new(Direction::East, AttackStrength::Light);
    println!(
        "   Valid action (East, Light): {}",
        another_action.is_valid()
    );

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
