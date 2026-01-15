# Deep Q-Networks




### Implementing the DqnModel trait

There are two networks in DQN:

1. **The Active/Policy Network**: Uses an `AutodiffBackend` (for training).
2. **The Target Network**: Uses the `InnerBackend` (no gradients, for stability).

```rust
#[derive(Module, Debug)]
pub struct MyDqn<B: Backend> {
    linear: nn::Linear<B>,
    // ... other layers
}

impl<B: AutodiffBackend> DqnModel<B> for MyDqn<B> {
    fn soft_update(
        active: &Self, 
        target: Self::InnerModule, 
        tau: f64
    ) -> Self::InnerModule {
        // Implementation logic typically involves using a ParamMapper 
        // to iterate through 'active' weights and interpolate them into 'target'
        unimplemented!("Interpolate weights here")
    }
}
```
