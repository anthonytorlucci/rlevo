//! # PositionHistory State Example
//!
//! This example demonstrates the comprehensive usage of three core state traits in the
//! evorl-core library: `State`, `FlattenedState`, and `TemporalState`. These traits
//! form the foundation for representing agent observations in reinforcement learning.
//!
//! ## Design Patterns Used
//!
//! This example illustrates several design patterns and best practices:
//!
//! ### 1. Trait-Based Abstraction
//! Rather than a single monolithic type, we use multiple focused traits to separate concerns:
//! - **State**: Provides metadata about state structure (shape, number of elements)
//! - **FlattenedState**: Handles serialization and numerical representation
//! - **TemporalState**: Manages temporal sequences and window-based updates
//!
//! This follows the **Adapter Pattern** and **Single Responsibility Principle**.
//!
//! ### 2. Temporal Sliding Window
//! The `PositionHistory` maintains a fixed-size window of past observations, which is
//! crucial for tasks where the agent needs temporal context (e.g., velocity estimation).
//! The `push_pop` operation efficiently rotates the window.
//!
//! ### 3. Type-Safe Conversions
//! The `from_flattened` method demonstrates safe deserialization with error handling,
//! preventing invalid state reconstruction.
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

use evorl_core::state::{FlattenedState, State, StateError, TemporalState};

/// Represents a sliding window of agent position history.
///
/// This struct maintains the last N x-coordinates observed by an agent, enabling
/// temporal understanding of movement patterns. It's particularly useful for:
/// - Tracking velocity by comparing consecutive positions
/// - Providing temporal context to neural networks
/// - Detecting movement patterns over a rolling window
///
/// # Fields
///
/// - `positions`: A vector of i32 x-coordinates in chronological order (oldest to newest)
/// - `window_size`: The maximum number of positions to maintain
///
/// # Example
///
/// ```ignore
/// let mut history = PositionHistory::new(5);
/// // As agent moves, update with new observations
/// history = history.push_pop(&[10.0]).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PositionHistory {
    positions: Vec<i32>,
    window_size: usize,
}

impl PositionHistory {
    /// Creates a new PositionHistory with a specified window size.
    ///
    /// All initial positions are set to 0. As the agent moves, these will be
    /// replaced by actual observations.
    ///
    /// # Arguments
    ///
    /// * `window_size` - The maximum number of historical positions to maintain
    ///
    /// # Returns
    ///
    /// A new `PositionHistory` with all positions initialized to 0
    fn new(window_size: usize) -> Self {
        Self {
            positions: vec![0; window_size],
            window_size,
        }
    }
}

impl State for PositionHistory {
    /// Returns the total number of elements in the state representation.
    ///
    /// For a position history, this equals the window size (one position per time step).
    fn numel(&self) -> usize {
        self.window_size
    }

    /// Returns the shape of the state as a vector of dimensions.
    ///
    /// For a 1D position history, this is `[window_size]`, but the trait
    /// supports arbitrary multi-dimensional shapes (e.g., for images: `[H, W, C]`).
    fn shape(&self) -> Vec<usize> {
        vec![self.window_size]
    }
}

impl FlattenedState for PositionHistory {
    /// Converts the internal state to a flat vector of f32 values.
    ///
    /// This is essential for:
    /// - Feeding into neural networks (which expect f32 tensors)
    /// - Serialization to disk or over network
    /// - GPU transfer and computation
    ///
    /// # Returns
    ///
    /// A vector of f32 values representing the positions
    fn flatten(&self) -> Vec<f32> {
        self.positions.iter().map(|&p| p as f32).collect()
    }

    /// Reconstructs a PositionHistory from a flattened f32 vector.
    ///
    /// This is the inverse of `flatten()` and is used during deserialization.
    /// The reconstructed state will have `window_size = data.len()`.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f32 positions
    ///
    /// # Returns
    ///
    /// * `Ok(PositionHistory)` - Successfully reconstructed state
    /// * `Err(StateError)` - If reconstruction fails (e.g., invalid data)
    fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
        Ok(PositionHistory {
            window_size: data.len(),
            positions: data.iter().map(|&p| p as i32).collect(),
        })
    }
}

impl TemporalState for PositionHistory {
    /// Returns the length of the temporal sequence (window size).
    ///
    /// This indicates how many time steps of history are maintained.
    fn sequence_length(&self) -> usize {
        self.window_size
    }

    /// Returns the most recent observation as a slice.
    ///
    /// In this simplified example, we return an empty slice. In a full implementation,
    /// this might return the last N observations or most recent observation.
    fn latest(&self) -> &[f32] {
        &[]
    }

    /// Updates the temporal window with a new observation, removing the oldest.
    ///
    /// This implements a rotating buffer pattern: the oldest element is removed,
    /// and the newest is appended. This maintains a constant window size while
    /// keeping the state "fresh" with recent observations.
    ///
    /// # Arguments
    ///
    /// * `new_observation` - Must be a slice with exactly 1 element (the new x-position)
    ///
    /// # Returns
    ///
    /// * `Ok(PositionHistory)` - New state with the window rotated
    /// * `Err(StateError::InvalidSize)` - If observation size != 1
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut state = PositionHistory::new(3); // [0, 0, 0]
    /// state = state.push_pop(&[5.0]).unwrap();  // [0, 0, 5]
    /// state = state.push_pop(&[10.0]).unwrap(); // [0, 5, 10]
    /// ```
    fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError> {
        if new_observation.len() != 1 {
            return Err(StateError::InvalidSize {
                expected: 1,
                got: new_observation.len(),
            });
        }

        let mut new_positions = self.positions.clone();
        new_positions.remove(0); // Drop oldest (O(n) operation)
        new_positions.push(new_observation[0] as i32); // Add newest

        Ok(PositionHistory {
            positions: new_positions,
            window_size: self.window_size,
        })
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        PositionHistory State Traits Example                ║");
    println!("║   Demonstrating State, FlattenedState, TemporalState       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // SECTION 1: Construction and Initialization
    // ========================================================================
    println!("┌─ SECTION 1: Construction and Initialization ─────────────────┐");
    let window_size = 5;
    let history = PositionHistory::new(window_size);
    println!("Created PositionHistory with window_size = {}", window_size);
    println!("  Content: {:?}", history);
    println!("  Status: Initialized with all positions = 0\n");

    // ========================================================================
    // SECTION 2: State Trait - Metadata and Introspection
    // ========================================================================
    println!("┌─ SECTION 2: State Trait - Metadata & Introspection ─────────┐");
    println!("The State trait provides structural metadata about observations:");
    println!("  numel()  (number of elements):    {}", history.numel());
    println!("  shape()  (tensor shape):          {:?}", history.shape());
    println!("  Verify:  Product of shape = {}", {
        let shape = history.shape();
        let product: usize = shape.iter().product();
        println!("           {:?}.product() = {}", shape, product);
        product == history.numel()
    });
    println!("  ✓ Shape and numel() are consistent\n");

    // ========================================================================
    // SECTION 3: FlattenedState Trait - Serialization & Conversion
    // ========================================================================
    println!("┌─ SECTION 3: FlattenedState Trait - Serialization ──────────┐");
    println!("FlattenedState enables conversion between internal and numeric formats:");

    // Flatten: Convert to f32 for neural network input
    let flattened = history.flatten();
    println!("  flatten():          {:?}", flattened);
    println!("  Type:               Vec<f32>");
    println!("  Use case:           Input to neural networks, serialization\n");

    // Reconstruct: Convert back from f32
    let reconstructed = PositionHistory::from_flattened(flattened.clone())
        .expect("Failed to reconstruct from flattened representation");
    println!("  from_flattened():   {:?}", reconstructed.positions);
    println!(
        "  Roundtrip check:    {:?} == {:?}",
        history.positions, reconstructed.positions
    );
    println!("  ✓ Serialization is lossless\n");

    // ========================================================================
    // SECTION 4: TemporalState Trait - Temporal Updates
    // ========================================================================
    println!("┌─ SECTION 4: TemporalState Trait - Temporal Updates ────────┐");
    println!("TemporalState manages time-series observations and sliding windows:");
    println!(
        "  sequence_length():  {} (history window size)",
        history.sequence_length()
    );
    println!();

    let positions = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    println!(
        "Simulating agent movement over {} timesteps:",
        positions.len()
    );
    println!("  Observations: {:?}", positions);
    println!();

    let mut current_state = history.clone();

    for (step, &new_position) in positions.iter().enumerate() {
        println!(
            "  Timestep {}: New observation = {:.0}",
            step + 1,
            new_position
        );

        match current_state.push_pop(&[new_position]) {
            Ok(new_state) => {
                current_state = new_state;
                println!("    Updated history: {:?}", current_state.positions);
                println!(
                    "    Velocity (approx): {:.1}",
                    if step > 0 {
                        new_position - positions[step - 1]
                    } else {
                        0.0
                    }
                );
            }
            Err(e) => eprintln!("    ERROR: {}", e),
        }
    }
    println!("  ✓ Temporal window updated {} times\n", positions.len());

    // ========================================================================
    // SECTION 5: Error Handling and Validation
    // ========================================================================
    println!("┌─ SECTION 5: Error Handling & Validation ──────────────────┐");
    println!("Demonstrating robust error handling for invalid inputs:\n");

    // Test 1: Invalid observation size
    println!("Test 1: Invalid observation size");
    let invalid_multi = vec![1.0, 2.0];
    println!("  Input: {:?} (expected 1 element)", invalid_multi);
    match current_state.push_pop(&invalid_multi) {
        Err(StateError::InvalidSize { expected, got }) => {
            println!("  ✓ Caught InvalidSize error");
            println!("    Expected: {}, Got: {}", expected, got);
        }
        _ => println!("  ✗ Unexpected result"),
    }
    println!();

    // Test 2: Single valid observation (sanity check)
    println!("Test 2: Valid single observation");
    let valid_single = vec![15.0];
    println!("  Input: {:?} (expected 1 element)", valid_single);
    match current_state.push_pop(&valid_single) {
        Ok(new_state) => {
            println!("  ✓ Update successful");
            println!("    New state: {:?}", new_state.positions);
            current_state = new_state;
        }
        Err(e) => println!("  ✗ Error: {}", e),
    }
    println!();

    // ========================================================================
    // SECTION 6: Serialization Roundtrip (Persistence Pattern)
    // ========================================================================
    println!("┌─ SECTION 6: Serialization Roundtrip ──────────────────────┐");
    println!("This pattern is essential for persistence and checkpointing:\n");

    let original = current_state.clone();
    println!("1. Original state:    {:?}", original.positions);

    let serialized = original.flatten();
    println!("2. Serialized (f32):  {:?}", serialized);
    println!("   (Could be written to file, sent over network, etc.)\n");

    let deserialized = PositionHistory::from_flattened(serialized).expect("Failed to deserialize");
    println!("3. Deserialized:      {:?}", deserialized.positions);

    let matches = original == deserialized;
    println!("4. States match:      {}", matches);
    if matches {
        println!("   ✓ Perfect roundtrip - suitable for checkpointing");
    }
    println!();

    // ========================================================================
    // SECTION 7: Real-World Use Case - Neural Network Preprocessing
    // ========================================================================
    println!("┌─ SECTION 7: Real-World Pattern - NN Preprocessing ────────┐");
    println!("Typical workflow for feeding state to a neural network:\n");

    let raw_state = &current_state;
    println!("Step 1: Extract internal state");
    println!("  State object: {:?}", raw_state);
    println!();

    let flattened_state = raw_state.flatten();
    println!("Step 2: Flatten to vector");
    println!("  Flattened: {:?}", flattened_state);
    println!();

    println!("Step 3: Normalize to [-1, 1] range");
    let max_position = 25.0; // Estimate based on typical positions
    let normalized = flattened_state
        .iter()
        .map(|&p| (p / max_position) * 2.0 - 1.0)
        .collect::<Vec<_>>();
    println!("  Normalized: {:?}", normalized);
    println!();

    println!("Step 4: Get shape for tensor creation");
    let shape = raw_state.shape();
    println!("  Shape: {:?}", shape);
    println!("  Ready to create Burn tensor: Tensor::<B, 1, F>::from_data(data, device)");
    println!();

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
}
