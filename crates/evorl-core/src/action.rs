use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::Rng;
use std::error::Error;
use std::fmt::Debug; // Need to import Rng trait for gen_range

/// Core trait for all action types. Minimal and framework-agnostic.
pub trait Action: Debug + Clone + Sized {
    /// Validate that this action is legal in its action space
    fn is_valid(&self) -> bool;
}

/// Trait for discrete actions with finite, indexable values
pub trait DiscreteAction: Action {
    /// Number of possible actions
    const ACTION_COUNT: usize;

    /// Convert from a flat index
    fn from_index(index: usize) -> Self;

    /// Convert to a flat index
    fn to_index(&self) -> usize;

    /// Sample a random valid action (no instance needed)
    fn random() -> Self
    where
        Self: Sized,
    {
        use rand::Rng;
        let mut rng = rand::rng();
        let index = rng.gen_range(0..Self::ACTION_COUNT);
        Self::from_index(index)
    }

    /// Enumerate all possible actions
    fn enumerate() -> Vec<Self>
    where
        Self: Sized,
    {
        (0..Self::ACTION_COUNT).map(Self::from_index).collect()
    }
}

/// Trait for multi-dimensional discrete actions
pub trait MultiDiscreteAction<const D: usize>: Action {
    /// Number of dimensions in the action space
    const DIM: usize = D;

    /// Size of each dimension
    fn action_space() -> [usize; D];

    /// Convert from multi-dimensional indices
    fn from_indices(indices: [usize; D]) -> Self;

    /// Convert to multi-dimensional indices
    fn to_indices(&self) -> [usize; D];

    /// Sample a random valid action
    fn random() -> Self
    where
        Self: Sized,
    {
        use rand::Rng;
        let mut rng = rand::rng();
        let space = Self::action_space();
        let indices = space.map(|dim| rng.gen_range(0..dim));
        Self::from_indices(indices)
    }

    /// Enumerate all possible actions (use with caution - can be huge!)
    fn enumerate() -> Vec<Self>
    where
        Self: Sized,
    {
        let space = Self::action_space();
        let total: usize = space.iter().product();
        let mut actions = Vec::with_capacity(total);

        fn generate<const D: usize, T: MultiDiscreteAction<D>>(
            space: &[usize; D],
            current: &mut [usize; D],
            dim: usize,
            actions: &mut Vec<T>,
        ) {
            if dim == D {
                actions.push(T::from_indices(*current));
                return;
            }
            for i in 0..space[dim] {
                current[dim] = i;
                generate(space, current, dim + 1, actions);
            }
        }

        let mut current = [0; D];
        generate(&space, &mut current, 0, &mut actions);
        actions
    }
}

/// Trait for continuous actions
pub trait ContinuousAction: Action {
    /// Dimensionality of the action vector
    const DIM: usize;

    /// Get the action as a slice of f32 values
    fn as_slice(&self) -> &[f32];

    /// Clip the action to valid bounds
    fn clip(&self, min: f32, max: f32) -> Self;

    /// Sample a random action from a normal distribution
    fn random() -> Self
    where
        Self: Sized,
    {
        use rand::Rng;
        let mut rng = rand::rng();
        // Default implementation - override for custom behavior
        let values: Vec<f32> = (0..Self::DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self::from_slice(&values)
    }

    /// Create from a slice (required for random() default)
    fn from_slice(values: &[f32]) -> Self;
}

/// Error type for invalid actions
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidActionError {
    pub message: String,
}

impl std::fmt::Display for InvalidActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid action: {}", self.message)
    }
}

impl Error for InvalidActionError {}

/// Framework-Specific Conversion Trait
/// Separate trait for converting actions to tensors
/// Keeps core action traits framework-agnostic
pub trait ActionTensorConvertible<const R: usize> {
    const R1: usize = R + 1; // dimensions including batch_size
    fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R>;
}
