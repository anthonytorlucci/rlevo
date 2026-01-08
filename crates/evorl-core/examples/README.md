# Burn-EvoRL Examples

This directory contains example implementations demonstrating key concepts and design patterns for the Burn-EvoRL library. Each example serves as a template for building evolutionary reinforcement learning solutions.

## Directory Structure

```
examples/
├── README.md (this file)
├── continuous_state_with_constraints.rs
└── ...
```

## Examples

### 1. Continuous State with Constraints (`continuous_state_with_constraints.rs`)

A comprehensive example demonstrating how to design a **constrained continuous state representation** for RL environments.

#### What It Shows

The example implements `RobotPose`, a 2D robot state with workspace constraints:
- **Position**: Bounded to [0, 1000] mm in both X and Y
- **Orientation**: Bounded to [-180°, 180°]
- **Implementation**: Uses design patterns for type safety, efficiency, and framework integration

#### Key Concepts Illustrated

| Concept | Demonstrated By |
|---------|-----------------|
| **Builder Pattern** | `RobotPose::new()` with validation |
| **Newtype Pattern** | Domain-specific `RobotPose` struct |
| **Copy Semantics** | Lightweight value type (12 bytes) |
| **Trait-Based Abstraction** | `State` trait implementation |
| **Constraint Enforcement** | `is_valid()` method with bounds checks |
| **Utility Methods** | `distance_to()`, `normalize_orientation()` |

#### Running the Example

```bash
cd burn-evorl
cargo run --example continuous_state_with_constraints
```

#### Expected Output

The example demonstrates six key scenarios:

1. **Constructing Valid Poses** - Creating states that satisfy constraints
2. **Rejecting Invalid Poses** - Showing how constraint violations are caught
3. **Distance Calculations** - Computing Euclidean metrics between poses
4. **Orientation Normalization** - Wrapping angles to valid range
5. **Trajectory Simulation** - Modeling a complete robot path
6. **RL Agent Decision Making** - Simulating action validation in learning

Sample output:
```
╔════════════════════════════════════════════════════════════╗
║   RobotPose State Constraint Example for Burn-EvoRL        ║
╚════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────┐
│ 1. Constructing Valid Robot Poses                           │
└──────────────────────────────────────────────────────────────┘

✓ Home Pose (center, facing east):
  Position: (500, 500) mm
  Orientation: 0°
  Valid: true | Elements: 3 | Shape: [3]
```

#### Design Documentation

Detailed design rationale, pattern explanations, and extension guidance are provided in:
- **[CONTINUOUS_STATE_DESIGN.md](./CONTINUOUS_STATE_DESIGN.md)** - Comprehensive design documentation
- **[Inline Documentation](./continuous_state_with_constraints.rs)** - Module and function documentation

#### Use This As A Template For

1. **Bounded Continuous Spaces** - Any environment with position/orientation constraints
2. **Game States** - Chess, checkers, or other discrete-space games with validation rules
3. **Physics Simulations** - Robotic control, dynamics modeling, trajectory planning
4. **Sensor Networks** - Modeling agent positions and orientations in space
5. **Navigation Systems** - Path planning with workspace boundaries

#### Key Takeaways

✓ **Type Safety**: Invalid states cannot be constructed (caught at compile-time via `Option`)
✓ **Efficiency**: `Copy` semantics + O(1) validation enable high-frequency simulation
✓ **Integration**: `State` trait implementation enables automatic NN/serialization support
✓ **Clarity**: Domain knowledge is embedded in the type system
✓ **Extensibility**: Add new constraints and methods without breaking existing code

---

## Design Patterns Reference

This section maps Burn-EvoRL examples to established design patterns:

### Behavioral Patterns

| Pattern | Example | Purpose |
|---------|---------|---------|
| **Strategy** | Auth provider abstraction | Pluggable behavior selection |
| **Command** | Job queue | Deferred/queued execution |
| **Observer** | Event bus | Decoupled notifications |
| **Visitor** | AST traversal | Transform hierarchical data |

### Creational Patterns

| Pattern | Example | Purpose |
|---------|---------|---------|
| **Builder** | `RobotPose::new()` | Complex type construction with validation |
| **Factory** | State construction | Polymorphic instance creation |

### Structural Patterns

| Pattern | Example | Purpose |
|---------|---------|---------|
| **Newtype** | `RobotPose` wrapper | Domain-specific type abstraction |
| **Composite** | Trait-based state | Hierarchical component composition |

---

## Running All Examples

```bash
# List available examples
cargo build --examples

# Run a specific example
cargo run --example continuous_state_with_constraints

# Run with release optimizations
cargo run --release --example continuous_state_with_constraints

# Check for compilation without running
cargo check --example continuous_state_with_constraints
```

## Writing Your Own Examples

### Template: Implementing a Constrained State

Use `RobotPose` as a template for your domain:

```rust
use evorl_core::state::State;

/// Your domain-specific state type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MyState {
    // Your fields here
}

impl State for MyState {
    fn is_valid(&self) -> bool {
        // Define your domain constraints
        true
    }

    fn numel(&self) -> usize {
        // Total number of scalar elements
        0
    }

    fn shape(&self) -> Vec<usize> {
        // Logical tensor dimensions
        vec![]
    }
}

impl MyState {
    /// Smart constructor with validation
    pub fn new(/* args */) -> Option<Self> {
        let state = MyState { /* ... */ };
        if state.is_valid() {
            Some(state)
        } else {
            None
        }
    }

    /// Domain-specific utility methods
    pub fn some_calculation(&self) -> f64 {
        // Implement domain logic
        0.0
    }
}

fn main() {
    // 1. Construct valid states
    if let Some(state) = MyState::new(/* args */) {
        println!("Valid state: {:?}", state);
    }

    // 2. Demonstrate invalid state rejection
    if MyState::new(/* invalid args */).is_none() {
        println!("Invalid state rejected ✓");
    }

    // 3. Show utility methods in action
    let calculation = state.some_calculation();
    println!("Calculation result: {}", calculation);
}
```

### Checklist for New Examples

- [ ] **Documentation**: Module-level `//!` doc comments explaining the example
- [ ] **Structure**: Clear sections demonstrating different concepts
- [ ] **Error Handling**: Show both valid and invalid cases
- [ ] **Metrics**: Display key measurements (distance, validity, shape)
- [ ] **Output**: Formatted output that's easy to read and understand
- [ ] **Design Patterns**: Reference the patterns being demonstrated
- [ ] **Guidelines**: Link to relevant Rust guidelines (API, style, pragmatic)

---

## Related Documentation

### Within This Crate

- **[continuous_state_with_constraints.rs](./continuous_state_with_constraints.rs)** - Main example with inline docs
- **[CONTINUOUS_STATE_DESIGN.md](./CONTINUOUS_STATE_DESIGN.md)** - Detailed design rationale
- **[../src/state.rs](../src/state.rs)** - `State` trait definition and documentation

### External References

#### Design Patterns
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/)
- [Pragmatic Rust Guidelines](https://microsoft.github.io/rust-guidelines/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

#### Core Concepts
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust Reference](https://doc.rust-lang.org/reference/)
- [Rust Nomicon](https://doc.rust-lang.org/nomicon/)

#### Project-Specific
- [Burn Deep Learning Framework](https://github.com/tracel-ai/burn)
- [Burn-EvoRL Repository](https://github.com/yourusername/burn-evorl)

---

## Tips for Success

### When Building RL Environments

1. **Constrain Early**: Use the builder pattern to validate state invariants at construction
2. **Type Safety First**: Leverage Rust's type system to prevent invalid states
3. **Performance Matters**: Use `Copy` for small value types; profile hot paths
4. **Test Boundaries**: Ensure edge cases (min/max bounds) are handled correctly
5. **Document Intent**: Make constraints explicit in code and comments

### When Implementing States

1. **Keep It Lightweight**: Minimize state size for efficient copying and caching
2. **Implement Common Traits**: Debug, Clone, Copy, Eq, Hash for framework compatibility
3. **Use Integer Math**: Avoid floating-point arithmetic in state representation
4. **Normalize Outputs**: Ensure state values fit expected ranges (e.g., [-1, 1] for NNs)
5. **Encapsulate Logic**: Put domain-specific methods on the state type

### When Writing Examples

1. **Start Simple**: Begin with basic valid/invalid cases
2. **Progress Gradually**: Build complexity through numbered sections
3. **Show Output**: Include formatted output that demonstrates success
4. **Link References**: Point to design documentation and guidelines
5. **Use Comments**: Explain what each section demonstrates

---

## Troubleshooting

### Compilation Errors

**Error**: `cannot move value, does not implement `Copy` trait`
- **Solution**: If your state type uses `Copy` semantics, ensure all fields implement `Copy`
- Example: Replace `String` with `&'static str` or `Box<str>` with fixed-size array

**Error**: `trait bounds not satisfied`
- **Solution**: Ensure your type implements all required traits (Debug, Clone, PartialEq, Eq, Hash)
- Add derives: `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]`

### Runtime Issues

**Problem**: Agent keeps producing invalid states
- **Diagnosis**: Check `is_valid()` implementation for correctness
- **Solution**: Add debug assertions and print state values

**Problem**: Neural network input size mismatch
- **Diagnosis**: `numel()` may not match actual field count
- **Solution**: Verify `numel()` equals sum of all state elements

---

## Contributing New Examples

To contribute a new example to Burn-EvoRL:

1. **Create Example File**: `examples/my_example.rs`
2. **Add Documentation**: Module-level `//!` docs + inline comments
3. **Implement Main**: Demonstrate key concepts clearly
4. **Create Design Doc**: `examples/MY_EXAMPLE_DESIGN.md` (optional but recommended)
5. **Test Thoroughly**: Run and verify output
6. **Submit PR**: Include references to guidelines and patterns

---

## Quick Start

New to Burn-EvoRL? Start here:

1. **Read This README** - Understand what examples are available ✓
2. **Run the Example** - Execute `cargo run --example continuous_state_with_constraints`
3. **Read the Code** - Study the inline documentation
4. **Read the Design Doc** - Understand the rationale (CONTINUOUS_STATE_DESIGN.md)
5. **Modify It** - Change constraints, add methods, experiment
6. **Build Your Own** - Use it as a template for your environment

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial examples with `continuous_state_with_constraints` |

---

## License

These examples are part of Burn-EvoRL and follow the same license as the parent project.

---

## Support

For questions or issues:

- **Examples Discussion**: Open an issue in the repository
- **Design Questions**: Check CONTINUOUS_STATE_DESIGN.md
- **API Questions**: Refer to inline documentation in source files
- **Community**: Join the Burn community on Discord/GitHub

---

**Last Updated**: 2024
**Maintained By**: Burn-EvoRL Contributors
