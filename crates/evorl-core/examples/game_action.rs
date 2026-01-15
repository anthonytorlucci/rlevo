use evorl_core::action::{Action, DiscreteAction};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameAction {
    Left,
    Right,
    Jump,
}

// impl Action for GameAction {
//     fn is_valid(&self) -> bool {
//         true
//     }
// }

// impl DiscreteAction for GameAction {
//     const ACTION_COUNT: usize = 3;

//     fn from_index(index: usize) -> Self {
//         match index {
//             0 => GameAction::Left,
//             1 => GameAction::Right,
//             2 => GameAction::Jump,
//             _ => panic!("Invalid index"),
//         }
//     }

//     fn to_index(&self) -> usize {
//         *self as usize
//     }
// }

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Creating actions directly
    println!("=== Direct Action Creation ===");
    let move_left = GameAction::Left;
    let move_right = GameAction::Right;
    let jump = GameAction::Jump;

    println!(
        "Actions created: {:?}, {:?}, {:?}",
        move_left, move_right, jump
    );

    // Example 2: Validating actions using the Action trait
    println!("\n=== Action Validation ===");
    // println!("Is Left valid? {}", move_left.is_valid());
    // println!("Is Right valid? {}", move_right.is_valid());
    // println!("Is Jump valid? {}", jump.is_valid());

    // Example 3: Converting to/from indices using DiscreteAction trait
    println!("\n=== Index Conversion ===");
    let action = GameAction::Jump;
    // let index = action.to_index();
    // println!("Jump action converted to index: {}", index);

    // let reconstructed = GameAction::from_index(index);
    // println!(
    //     "Index {} converted back to action: {:?}",
    //     index, reconstructed
    // );

    // Example 4: Iterating through all possible actions
    println!("\n=== All Possible Actions ===");
    // for i in 0..GameAction::ACTION_COUNT {
    //     let action = GameAction::from_index(i);
    //     println!("Index {}: {:?} (valid: {})", i, action, action.is_valid());
    // }

    // Example 5: Demonstrating cloning and equality
    println!("\n=== Cloning and Equality ===");
    let action1 = GameAction::Left;
    let action2 = action1.clone();
    let action3 = GameAction::Left;

    println!("action1 == action2: {}", action1 == action2);
    println!("action1 == action3: {}", action1 == action3);
    println!(
        "action1 == GameAction::Right: {}",
        action1 == GameAction::Right
    );

    // Example 6: Simulating a game input sequence
    println!("\n=== Game Input Sequence ===");
    let inputs = vec![
        GameAction::Right,
        GameAction::Right,
        GameAction::Jump,
        GameAction::Left,
        GameAction::Jump,
    ];

    // for (frame, action) in inputs.iter().enumerate() {
    //     if action.is_valid() {
    //         println!("Frame {}: Executing {:?}", frame, action);
    //     }
    // }

    // Example 7:
    // let all_actions = GameAction::enumerate();
    // assert_eq!(all_actions.len(), GameAction::ACTION_COUNT);

    // Example 8: Random
    // let game_action = GameAction::random();
    // assert!(game_action.is_valid());

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
