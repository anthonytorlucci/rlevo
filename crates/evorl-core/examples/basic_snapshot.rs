use evorl_core::state::State;

#[derive(Debug, Clone, PartialEq)]
enum GameState {
    Menu,
    Playing { level: u8 },
    GameOver { score: u32 },
}

// impl State for GameState {
//     fn is_valid(&self) -> bool {
//         match self {
//             GameState::Playing { level } => *level > 0 && *level <= 10,
//             _ => true,
//         }
//     }

//     fn numel(&self) -> usize {
//         // Encode as 3 features: [state_id, level, score]
//         3
//     }

//     fn shape(&self) -> Vec<usize> {
//         vec![3]
//     }
// }

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let menu_state = GameState::Menu;
    let _playing_level_5 = GameState::Playing { level: 5 };
    let _playing_invalid = GameState::Playing { level: 0 };
    let _game_over_state = GameState::GameOver { score: 2500 };

    // // todo! initialize snapshot variable
    // let snapshot: Box<dyn Snapshot<StateType = GameState, RewardType = f32>> =
    //     Box::new(SnapshotBase::<GameState, f32>::new(menu_state, 1.0, false));

    // println!("State: {:?}", snapshot.state()); // State: Menu
    // println!("Reward: {}", snapshot.reward()); // Reward: 1
    // println!("Done: {}", snapshot.is_done()); // Done: false

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
