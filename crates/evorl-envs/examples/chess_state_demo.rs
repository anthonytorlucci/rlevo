//! Chess State Demonstration
//!
//! This example demonstrates the AlphaZero chess state representation,
//! showing how to create, validate, and convert chess positions to tensors
//! for deep reinforcement learning.

use burn::backend::NdArray;
use evorl_core::state::{State, StateTensorConvertible};
use evorl_envs::games::chess::state::{ChessState, Color};

type Backend = NdArray;

fn main() {
    println!("=== ChessState AlphaZero Representation Demo ===\n");

    // 1. Create the standard starting position
    println!("1. Creating starting position...");
    let state = ChessState::new();
    println!("   ✓ State created");
    println!("   - Side to move: {:?}", state.to_move());
    println!("   - Castling rights: {:?}", state.castling_rights());
    println!("   - Full move number: {}", state.fullmove_number());
    println!("   - Half-move clock: {}\n", state.halfmove_clock());

    // 2. Validate the position
    println!("2. Validating position...");
    if state.is_valid() {
        println!("   ✓ Position is valid");
    } else {
        println!("   ✗ Position is invalid!");
    }
    println!();

    // 3. Inspect state dimensions
    println!("3. State dimensions:");
    println!("   - Shape: {:?}", state.shape());
    println!("   - Total elements: {}", state.numel());
    println!("   - Layout: [height, width, planes] = [8, 8, 119]");
    println!();

    // 4. Convert to tensor (AlphaZero observation space)
    println!("4. Converting to tensor...");
    let device = Default::default();
    let tensor = state.to_tensor::<Backend>(&device);
    let shape = tensor.shape();
    println!("   ✓ Tensor created");
    println!("   - Tensor shape: {:?}", shape.dims);
    println!("   - Tensor dtype: f32");
    println!("   - Memory size: ~{} KB", (state.numel() * 4) / 1024);
    println!();

    // 5. Explain the tensor structure
    println!("5. Tensor structure (119 planes):");
    println!("   Historical State (112 planes):");
    println!("   ├─ Time step 0 (current): 14 planes");
    println!("   │  ├─ Own pieces: 6 planes (P,N,B,R,Q,K)");
    println!("   │  ├─ Opponent pieces: 6 planes (P,N,B,R,Q,K)");
    println!("   │  └─ Repetition: 2 planes (once, twice)");
    println!("   ├─ Time step 1 (1 move ago): 14 planes");
    println!("   ├─ ... (6 more time steps)");
    println!("   └─ Time step 7 (7 moves ago): 14 planes");
    println!();
    println!("   Constant Metadata (7 planes):");
    println!("   ├─ Castling rights: 4 planes (own KS/QS, opp KS/QS)");
    println!("   ├─ Side to move: 1 plane (1.0=White, 0.0=Black)");
    println!("   ├─ Move count: 1 plane (normalized)");
    println!("   └─ No-progress count: 1 plane (50-move rule)");
    println!();

    // 6. Demonstrate tensor data inspection
    println!("6. Inspecting tensor data...");
    let tensor_data = tensor.clone().flatten::<1>(0, 2).into_data();
    let values = tensor_data.to_vec::<f32>().expect("Failed to extract data");

    // Check some specific planes
    let plane_size = 64;

    // Plane 0: White pawns (starting position should have 8 pawns on rank 2)
    let white_pawn_plane = &values[0..plane_size];
    let white_pawn_count: usize = white_pawn_plane.iter().filter(|&&v| v > 0.5).count();
    println!("   - White pawns on plane 0: {} pieces", white_pawn_count);

    // Plane 6: Black pawns (should have 8 pawns on rank 7)
    let black_pawn_plane = &values[6 * plane_size..7 * plane_size];
    let black_pawn_count: usize = black_pawn_plane.iter().filter(|&&v| v > 0.5).count();
    println!("   - Black pawns on plane 6: {} pieces", black_pawn_count);

    // Plane 116: Side to move
    let side_plane = &values[116 * plane_size..117 * plane_size];
    let side_value = side_plane[0];
    println!("   - Side to move (plane 116): {:.1} (White)", side_value);
    println!();

    // 7. Demonstrate round-trip conversion
    println!("7. Testing round-trip conversion...");
    match ChessState::from_tensor(&tensor) {
        Ok(reconstructed) => {
            println!("   ✓ State reconstructed from tensor");
            println!("   - Reconstructed side: {:?}", reconstructed.to_move());
            println!(
                "   - Validation: {}",
                if reconstructed.is_valid() {
                    "✓ Valid"
                } else {
                    "✗ Invalid"
                }
            );

            // Check if key properties match
            let matches = reconstructed.to_move() == state.to_move()
                && reconstructed.castling_rights() == state.castling_rights();
            println!(
                "   - Properties match: {}",
                if matches {
                    "✓ Yes"
                } else {
                    "⚠ Partial (expected)"
                }
            );
        }
        Err(e) => {
            println!("   ✗ Reconstruction failed: {}", e);
        }
    }
    println!();

    // 8. Explain perspective normalization
    println!("8. Perspective normalization:");
    println!("   The board is ALWAYS oriented from the current player's perspective:");
    println!("   - When White to move: rank 1 at bottom, rank 8 at top");
    println!("   - When Black to move: board flipped vertically");
    println!("   This allows the network to learn ONE strategy, not two!");
    println!();

    // 9. Key features summary
    println!("9. Key Features:");
    println!("   ✓ Markovian: 8 historical positions capture temporal patterns");
    println!("   ✓ Raw representation: No hand-crafted features");
    println!("   ✓ Translation invariant: CNN can detect patterns anywhere");
    println!("   ✓ Player-agnostic: Single strategy for both colors");
    println!("   ✓ Efficient: Bitboard operations are O(1)");
    println!("   ✓ GPU-ready: Direct tensor conversion for neural networks");
    println!();

    // 10. Use case examples
    println!("10. Typical Use Cases:");
    println!("   - Deep Q-Networks (DQN): state → Q-values for each action");
    println!("   - Policy Gradients (PPO): state → (action probabilities, value)");
    println!("   - AlphaZero MCTS: state → (policy, value) for tree search");
    println!("   - Evolutionary Strategies: state → action scores");
    println!();

    println!("=== Demo Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_runs() {
        // Ensure the demo code is valid
        let state = ChessState::new();
        assert!(state.is_valid());

        let device = Default::default();
        let tensor = state.to_tensor::<Backend>(&device);
        assert_eq!(tensor.shape().dims, [8, 8, 119]);

        let reconstructed = ChessState::from_tensor(&tensor).unwrap();
        assert!(reconstructed.is_valid());
    }
}
