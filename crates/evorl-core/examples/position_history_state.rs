//! # PositionHistory State Example
//!
//! ## Real-World Applications
//!
//! This pattern is used in:
//! - **Atari games**: Track frame history for optical flow understanding
//! - **Robotic control**: Maintain position/velocity history for smooth trajectories
//! - **Stock trading**: Keep historical prices for trend analysis
//! - **Game AI**: Store previous states to detect patterns or compute deltas
//!
//! ## Performance Considerations
//!
//! - The sliding window uses `Vec::remove(0)` and `push()`, which is O(n) for remove.
//! - For high-frequency updates, consider using a circular buffer for O(1) operations.
//! - Flattening allocates a new vector; for tight loops, consider caching or using references.

use evorl_core::dynamics::{History, HistoryRepresentation};
use evorl_core::state::{State, StateError};

/// Represents a sliding window of agent position history.
///
/// This struct maintains the last N x-coordinates observed by an agent, enabling
/// temporal understanding of movement patterns. It's particularly useful for:
/// - Tracking velocity by comparing consecutive positions
/// - Providing temporal context to neural networks
/// - Detecting movement patterns over a rolling window
///

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        PositionHistory State Traits Example                ║");
    println!("║   Demonstrating State, FlattenedState, TemporalState       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // SECTION 1: Construction and Initialization
    // ========================================================================
    println!("┌─ SECTION 1: Construction and Initialization ─────────────────┐");
    // let window_size = 5;
    // let history = PositionHistory::new(window_size);
    // println!("Created PositionHistory with window_size = {}", window_size);
    // println!("  Content: {:?}", history);
    // println!("  Status: Initialized with all positions = 0\n");

    // ========================================================================
    // SECTION 2: State Trait - Metadata and Introspection
    // ========================================================================
    println!("┌─ SECTION 2: State Trait - Metadata & Introspection ─────────┐");
    println!("The State trait provides structural metadata about observations:");
    // println!("  numel()  (number of elements):    {}", history.numel());
    // println!("  shape()  (tensor shape):          {:?}", history.shape());
    // println!("  Verify:  Product of shape = {}", {
    //     let shape = history.shape();
    //     let product: usize = shape.iter().product();
    //     println!("           {:?}.product() = {}", shape, product);
    //     product == history.numel()
    // });
    // println!("  ✓ Shape and numel() are consistent\n");

    // ========================================================================
    // SECTION 3: FlattenedState Trait - Serialization & Conversion
    // ========================================================================
    println!("┌─ SECTION 3: FlattenedState Trait - Serialization ──────────┐");
    println!("FlattenedState enables conversion between internal and numeric formats:");

    // ========================================================================
    // SECTION 4: TemporalState Trait - Temporal Updates
    // ========================================================================
    println!("┌─ SECTION 4: TemporalState Trait - Temporal Updates ────────┐");
    println!("TemporalState manages time-series observations and sliding windows:");
    // println!(
    //     "  sequence_length():  {} (history window size)",
    //     history.sequence_length()
    // );
    // println!();

    // let positions = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    // println!(
    //     "Simulating agent movement over {} timesteps:",
    //     positions.len()
    // );
    // println!("  Observations: {:?}", positions);
    // println!();

    // let mut current_state = history.clone();

    // for (step, &new_position) in positions.iter().enumerate() {
    //     println!(
    //         "  Timestep {}: New observation = {:.0}",
    //         step + 1,
    //         new_position
    //     );

    //     match current_state.push_pop(&[new_position]) {
    //         Ok(new_state) => {
    //             current_state = new_state;
    //             println!("    Updated history: {:?}", current_state.positions);
    //             println!(
    //                 "    Velocity (approx): {:.1}",
    //                 if step > 0 {
    //                     new_position - positions[step - 1]
    //                 } else {
    //                     0.0
    //                 }
    //             );
    //         }
    //         Err(e) => eprintln!("    ERROR: {}", e),
    //     }
    // }
    // println!("  ✓ Temporal window updated {} times\n", positions.len());

    // ========================================================================
    // SECTION 5: Error Handling and Validation
    // ========================================================================
    println!("┌─ SECTION 5: Error Handling & Validation ──────────────────┐");
    println!("Demonstrating robust error handling for invalid inputs:\n");

    // Test 1: Invalid observation size
    println!("Test 1: Invalid observation size");
    let invalid_multi = vec![1.0, 2.0];
    println!("  Input: {:?} (expected 1 element)", invalid_multi);
    // match current_state.push_pop(&invalid_multi) {
    //     Err(StateError::InvalidSize { expected, got }) => {
    //         println!("  ✓ Caught InvalidSize error");
    //         println!("    Expected: {}, Got: {}", expected, got);
    //     }
    //     _ => println!("  ✗ Unexpected result"),
    // }
    // println!();

    // Test 2: Single valid observation (sanity check)
    println!("Test 2: Valid single observation");
    let valid_single = vec![15.0];
    println!("  Input: {:?} (expected 1 element)", valid_single);
    // match current_state.push_pop(&valid_single) {
    //     Ok(new_state) => {
    //         println!("  ✓ Update successful");
    //         println!("    New state: {:?}", new_state.positions);
    //         current_state = new_state;
    //     }
    //     Err(e) => println!("  ✗ Error: {}", e),
    // }
    // println!();

    // ========================================================================
    // SECTION 6: Serialization Roundtrip (Persistence Pattern)
    // ========================================================================
    println!("┌─ SECTION 6: Serialization Roundtrip ──────────────────────┐");
    println!("This pattern is essential for persistence and checkpointing:\n");

    // let original = current_state.clone();
    // println!("1. Original state:    {:?}", original.positions);

    // let serialized = original.flatten();
    // println!("2. Serialized (f32):  {:?}", serialized);
    // println!("   (Could be written to file, sent over network, etc.)\n");

    // let deserialized = PositionHistory::from_flattened(serialized).expect("Failed to deserialize");
    // println!("3. Deserialized:      {:?}", deserialized.positions);

    // let matches = original == deserialized;
    // println!("4. States match:      {}", matches);
    // if matches {
    //     println!("   ✓ Perfect roundtrip - suitable for checkpointing");
    // }
    // println!();

    // ========================================================================
    // SECTION 7: Real-World Use Case - Neural Network Preprocessing
    // ========================================================================
    println!("┌─ SECTION 7: Real-World Pattern - NN Preprocessing ────────┐");
    println!("Typical workflow for feeding state to a neural network:\n");

    // let raw_state = &current_state;
    // println!("Step 1: Extract internal state");
    // println!("  State object: {:?}", raw_state);
    // println!();

    // let flattened_state = raw_state.flatten();
    // println!("Step 2: Flatten to vector");
    // println!("  Flattened: {:?}", flattened_state);
    // println!();

    // println!("Step 3: Normalize to [-1, 1] range");
    // let max_position = 25.0; // Estimate based on typical positions
    // let normalized = flattened_state
    //     .iter()
    //     .map(|&p| (p / max_position) * 2.0 - 1.0)
    //     .collect::<Vec<_>>();
    // println!("  Normalized: {:?}", normalized);
    // println!();

    // println!("Step 4: Get shape for tensor creation");
    // let shape = raw_state.shape();
    // println!("  Shape: {:?}", shape);
    // println!("  Ready to create Burn tensor: Tensor::<B, 1, F>::from_data(data, device)");
    // println!();

    // ========================================================================
    // SECTION 8: Design Patterns Summary
    // ========================================================================
    println!("┌─ SECTION 8: Design Patterns & Best Practices ────────────┐");
    println!();
    println!("PATTERNS DEMONSTRATED:");
    println!("  1. Trait-Based Abstraction");
    println!("     ├─ Separates concerns across State, FlattenedState, TemporalState");
    println!("     ├─ Enables multiple implementations without polymorphic overhead");
    println!("     └─ Improves testability and maintainability\n");

    println!("  2. Builder/Constructor Pattern");
    println!("     ├─ PositionHistory::new() provides clear initialization");
    println!("     └─ Prevents invalid partial states\n");

    println!("  3. Type-Safe Conversions");
    println!("     ├─ from_flattened() returns Result<> for error handling");
    println!("     ├─ Prevents runtime crashes from malformed data");
    println!("     └─ Type system catches errors at compile time\n");

    println!("  4. Sliding Window / Ring Buffer");
    println!("     ├─ Maintains fixed-size temporal context");
    println!("     ├─ push_pop() efficiently rotates the buffer");
    println!("     └─ Critical for tasks requiring velocity, trend detection\n");

    println!("  5. Serialization Pattern");
    println!("     ├─ flatten/from_flattened enables persistence");
    println!("     ├─ Essential for checkpointing and model recovery");
    println!("     └─ Supports distributed training\n");

    println!("BEST PRACTICES:");
    println!("  ✓ Always implement error handling (Result types)");
    println!("  ✓ Validate input sizes and ranges");
    println!("  ✓ Use type system to prevent invalid states");
    println!("  ✓ Document trait methods thoroughly");
    println!("  ✓ Test serialization roundtrips");
    println!("  ✓ Consider performance implications (e.g., Vec::remove is O(n))");
    println!("  ✓ Maintain consistency between shape() and numel()");
    println!();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
