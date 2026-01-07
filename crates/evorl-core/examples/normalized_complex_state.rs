use evorl_core::state::{FlattenedState, State, StateError};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlayerState {
    health: u8,  // 0-100
    stamina: u8, // 0-100
    level: u8,   // 1-50
    has_sword: bool,
    has_shield: bool,
}

impl State for PlayerState {
    fn numel(&self) -> usize {
        5
    }
    fn shape(&self) -> Vec<usize> {
        vec![5]
    }
}

impl FlattenedState for PlayerState {
    fn flatten(&self) -> Vec<f32> {
        vec![
            self.health as f32 / 100.0,     // Normalize to [0, 1]
            self.stamina as f32 / 100.0,    // Normalize to [0, 1]
            (self.level - 1) as f32 / 49.0, // Normalize to [0, 1]
            if self.has_sword { 1.0 } else { 0.0 },
            if self.has_shield { 1.0 } else { 0.0 },
        ]
    }

    fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
        if data.len() != 5 {
            return Err(StateError::InvalidSize {
                expected: 5,
                got: data.len(),
            });
        }
        Ok(PlayerState {
            health: (data[0] * 100.0).clamp(0.0, 100.0) as u8,
            stamina: (data[1] * 100.0).clamp(0.0, 100.0) as u8,
            level: ((data[2] * 49.0) + 1.0).clamp(1.0, 50.0) as u8,
            has_sword: data[3] > 0.5,
            has_shield: data[4] > 0.5,
        })
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
// Key Features Demonstrated:**
//
// 1. **Construction** - Creating `PlayerState` instances with various configurations
// 2. **State Trait Methods** - Calling `numel()` and `shape()` to inspect state dimensions
// 3. **Flattening** - Converting `PlayerState` to normalized `f32` values for neural network input
// 4. **Reconstruction** - Converting flattened data back to `PlayerState` objects
// 5. **Round-Trip Verification** - Ensuring data integrity through flatten → reconstruct cycles
// 6. **Error Handling** - Demonstrating validation of invalid input sizes
// 7. **Edge Cases** - Testing minimum and maximum values with proper normalization
//
// ### **Normalization Details:**
//
// The example clearly shows how each field is normalized:
// - **health & stamina**: Scaled from `0-100` to `[0.0, 1.0]`
// - **level**: Scaled from `1-50` to `[0.0, 1.0]`
// - **Equipment flags**: Converted to binary `0.0`/`1.0`
//
// ### **Practical Use Cases:**
//
// The example includes realistic scenarios:
// - Comparing weak vs. strong players
// - Testing boundary conditions (min/max values)
// - Error handling for invalid data
// - Round-trip conversions showing data preservation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Create a PlayerState and demonstrate State trait
    println!("=== PlayerState Example ===\n");

    let player = PlayerState {
        health: 75,
        stamina: 90,
        level: 5,
        has_sword: true,
        has_shield: false,
    };

    println!("Original Player State: {:?}", player);
    println!("Number of elements: {}", player.numel());
    println!("Shape: {:?}\n", player.shape());

    // Example 2: Flatten the state for neural network input
    println!("=== Flattening State ===\n");
    let flattened = player.flatten();
    println!("Flattened representation: {:?}", flattened);
    println!("Flattened length: {}\n", flattened.len());

    // Example 3: Reconstruct from flattened representation
    println!("=== Reconstructing from Flattened ===\n");
    match PlayerState::from_flattened(flattened.clone()) {
        Ok(reconstructed) => {
            println!("Reconstructed Player State: {:?}", reconstructed);
            println!(
                "Original and reconstructed are equal: {}\n",
                player == reconstructed
            );
        }
        Err(e) => println!("Error reconstructing state: {:?}\n", e),
    }

    // Example 4: Demonstrate error handling with invalid data
    println!("=== Error Handling ===\n");
    let invalid_data = vec![0.5, 0.5]; // Too few elements (expected 5)
    match PlayerState::from_flattened(invalid_data) {
        Ok(_) => println!("Unexpectedly succeeded"),
        Err(e) => println!("Expected error caught: {:?}\n", e),
    }

    // Example 5: Compare multiple player states with normalization
    println!("=== Comparing Multiple Player States ===\n");

    let player_weak = PlayerState {
        health: 20,
        stamina: 30,
        level: 1,
        has_sword: false,
        has_shield: false,
    };

    let player_strong = PlayerState {
        health: 100,
        stamina: 100,
        level: 50,
        has_sword: true,
        has_shield: true,
    };

    println!("Weak Player:   {:?}", player_weak);
    println!("Weak Flattened: {:?}\n", player_weak.flatten());

    println!("Strong Player:   {:?}", player_strong);
    println!("Strong Flattened: {:?}\n", player_strong.flatten());

    // Example 6: Round-trip conversion with normalization edge cases
    println!("=== Normalization Edge Cases ===\n");

    // Test with minimum values
    let min_state = PlayerState {
        health: 0,
        stamina: 0,
        level: 1,
        has_sword: false,
        has_shield: false,
    };
    let min_flattened = min_state.flatten();
    println!("Minimum State: {:?}", min_state);
    println!("Minimum Flattened: {:?}", min_flattened);

    match PlayerState::from_flattened(min_flattened) {
        Ok(reconstructed) => {
            println!("Reconstructed Min State: {:?}", reconstructed);
            println!("Round-trip successful: {}\n", min_state == reconstructed);
        }
        Err(e) => println!("Error: {:?}\n", e),
    }

    // Test with maximum values
    let max_state = PlayerState {
        health: 100,
        stamina: 100,
        level: 50,
        has_sword: true,
        has_shield: true,
    };
    let max_flattened = max_state.flatten();
    println!("Maximum State: {:?}", max_state);
    println!("Maximum Flattened: {:?}", max_flattened);

    match PlayerState::from_flattened(max_flattened) {
        Ok(reconstructed) => {
            println!("Reconstructed Max State: {:?}", reconstructed);
            println!("Round-trip successful: {}", max_state == reconstructed);
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
