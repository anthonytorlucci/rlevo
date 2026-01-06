use evorl_core::state::State;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GameState {
    Menu,
    Playing { level: u8 },
    GameOver { score: u32 },
}
///
impl State for GameState {
    fn is_valid(&self) -> bool {
        match self {
            GameState::Playing { level } => *level > 0 && *level <= 10,
            _ => true,
        }
    }
    ///
    fn numel(&self) -> usize {
        // Encode as 3 features: [state_id, level, score]
        3
    }
    ///
    fn shape(&self) -> Vec<usize> {
        vec![3]
    }
}

// Key Demonstrations:**
//
// 1. **State Validation (`is_valid()`)** — Shows how the trait validates domain constraints
//    - Menu and GameOver states are always valid
//    - Playing states validate that `level` is between 1-10
//    - Invalid states are correctly identified
//
// 2. **Number of Elements (`numel()`)** — Demonstrates tensor element counting
//    - All states encode to 3 scalar values: `[state_id, level, score]`
//    - Essential for neural network input layer sizing
//
// 3. **State Shape (`shape()`)** — Illustrates tensor shape representation
//    - All states have shape `[3]` (1D vector)
//    - Logical shape independent of framework-specific tensor formats
//
// 4. **Consistency Check** — Validates that `numel() == shape().iter().product()`
//    - Critical requirement for state encoding
//    - All states satisfy this invariant
//
// 5. **Practical Usage** — Real-world filtering based on validity
//    - Demonstrates filtering 4 states down to 3 valid ones
//    - Shows how invalid states are rejected for training
fn main() {
    // Construct different GameState variants
    let menu_state = GameState::Menu;
    let playing_level_5 = GameState::Playing { level: 5 };
    let playing_invalid = GameState::Playing { level: 0 };
    let game_over_state = GameState::GameOver { score: 2500 };

    println!("=== GameState Example Usage ===\n");

    // Demonstrate is_valid() - validates state constraints
    println!("1. State Validation (is_valid):");
    println!("   Menu is valid:               {}", menu_state.is_valid());
    println!(
        "   Playing(level=5) is valid:   {}",
        playing_level_5.is_valid()
    );
    println!(
        "   Playing(level=0) is valid:   {} (invalid: level must be 1-10)",
        playing_invalid.is_valid()
    );
    println!(
        "   GameOver(score=2500) is valid: {}",
        game_over_state.is_valid()
    );
    println!();

    // Demonstrate numel() - total number of scalar elements in state
    println!("2. Number of Elements (numel):");
    println!("   Menu:                {}", menu_state.numel());
    println!("   Playing(level=5):    {}", playing_level_5.numel());
    println!("   GameOver(score=2500):{}", game_over_state.numel());
    println!("   (All states encode as 3 features: [state_id, level, score])");
    println!();

    // Demonstrate shape() - logical shape of state tensor representation
    println!("3. State Shape (shape):");
    println!("   Menu:                {:?}", menu_state.shape());
    println!("   Playing(level=5):    {:?}", playing_level_5.shape());
    println!("   GameOver(score=2500):{:?}", game_over_state.shape());
    println!("   (All states have shape [3]: a 1D vector of 3 elements)");
    println!();

    // Verify consistency: numel() should equal product of shape()
    println!("4. Consistency Check (numel == shape product):");
    for (name, state) in [
        ("Menu", menu_state.clone()),
        ("Playing(level=5)", playing_level_5.clone()),
        ("GameOver(score=2500)", game_over_state.clone()),
    ] {
        let numel = state.numel();
        let shape_product: usize = state.shape().iter().product();
        let is_consistent = numel == shape_product;
        println!(
            "   {:<25} numel={}, shape_product={}, valid={}",
            name, numel, shape_product, is_consistent
        );
    }
    println!();

    // Demonstrate practical use: determining if a state can be processed
    println!("5. Practical Usage Example:");
    let states = vec![
        menu_state,
        playing_level_5,
        playing_invalid,
        game_over_state,
    ];
    let valid_states: Vec<_> = states.into_iter().filter(|s| s.is_valid()).collect();
    println!("   Total states: 4");
    println!("   Valid states: {}", valid_states.len());
    println!("   (Only valid states can be used for transitions/training)");
}
