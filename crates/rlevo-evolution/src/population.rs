//! Population containers.
//!
//! [`Population<B, K>`] is a thin wrapper around a device tensor plus the
//! shape metadata strategies need. For real-valued kinds it holds a
//! `Tensor<B, 2>`; binary and integer kinds use `Tensor<B, 2, Int>`.
//!
//! The wrapper exists so operators and strategies have a single shape
//! contract to validate against (they check `pop_size` and `genome_dim`
//! rather than repeatedly interrogating `tensor.shape().dims`).

use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Int, Tensor};

use crate::genome::{Binary, Integer, Real};

/// Population stored on a Burn backend device.
///
/// The concrete tensor type depends on the genome kind `K`. Most
/// consumers interact with [`Population<B, Real>`] via [`tensor`](Population::tensor),
/// but strategies parameterized on the kind can keep the `K` generic and
/// reach for the right tensor flavor through the inherent impls below.
///
/// Invariant: for every `Population<B, K>` produced by the public
/// constructors, exactly one of `tensor_real` / `tensor_int` is `Some`,
/// determined by `K`. `Real` populates `tensor_real`; `Binary`,
/// `Integer`, and `Permutation` populate `tensor_int`. The inherent
/// `tensor(&self)` accessors `.expect()` on the matching field because
/// the constructor contract pins the invariant — a mismatch would be a
/// bug in this module.
#[derive(Debug, Clone)]
pub struct Population<B: Backend, K> {
    pop_size: usize,
    genome_dim: usize,
    _kind: PhantomData<K>,
    tensor_real: Option<Tensor<B, 2>>,
    tensor_int: Option<Tensor<B, 2, Int>>,
}

impl<B: Backend, K> Population<B, K> {
    /// Returns the number of individuals (rows) in the population.
    #[must_use]
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    /// Returns the genome dimensionality (columns).
    #[must_use]
    pub fn genome_dim(&self) -> usize {
        self.genome_dim
    }
}

impl<B: Backend> Population<B, Real> {
    /// Constructs a real-valued population from a `Tensor<B, 2>`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not rank 2.
    #[must_use]
    pub fn new_real(tensor: Tensor<B, 2>) -> Self {
        let dims = tensor.shape().dims;
        assert_eq!(dims.len(), 2, "population tensor must be rank 2");
        Self {
            pop_size: dims[0],
            genome_dim: dims[1],
            _kind: PhantomData,
            tensor_real: Some(tensor),
            tensor_int: None,
        }
    }

    /// Borrows the backing real-valued tensor.
    ///
    /// # Panics
    ///
    /// Never panics for a correctly constructed `Population<B, Real>`.
    #[must_use]
    pub fn tensor(&self) -> &Tensor<B, 2> {
        self.tensor_real
            .as_ref()
            .expect("real population always has a tensor_real")
    }

    /// Consumes the wrapper and returns the owned tensor.
    ///
    /// # Panics
    ///
    /// Never panics for a correctly constructed `Population<B, Real>`.
    #[must_use]
    pub fn into_tensor(self) -> Tensor<B, 2> {
        self.tensor_real
            .expect("real population always has a tensor_real")
    }
}

impl<B: Backend> Population<B, Binary> {
    /// Constructs a binary population from a `Tensor<B, 2, Int>`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not rank 2.
    #[must_use]
    pub fn new_binary(tensor: Tensor<B, 2, Int>) -> Self {
        let dims = tensor.shape().dims;
        assert_eq!(dims.len(), 2, "population tensor must be rank 2");
        Self {
            pop_size: dims[0],
            genome_dim: dims[1],
            _kind: PhantomData,
            tensor_real: None,
            tensor_int: Some(tensor),
        }
    }

    /// Borrows the backing integer tensor holding 0/1 values.
    ///
    /// # Panics
    ///
    /// Never panics for a correctly constructed `Population<B, Binary>`.
    #[must_use]
    pub fn tensor(&self) -> &Tensor<B, 2, Int> {
        self.tensor_int
            .as_ref()
            .expect("binary population always has a tensor_int")
    }
}

impl<B: Backend> Population<B, Integer> {
    /// Constructs an integer population from a `Tensor<B, 2, Int>`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not rank 2.
    #[must_use]
    pub fn new_integer(tensor: Tensor<B, 2, Int>) -> Self {
        let dims = tensor.shape().dims;
        assert_eq!(dims.len(), 2, "population tensor must be rank 2");
        Self {
            pop_size: dims[0],
            genome_dim: dims[1],
            _kind: PhantomData,
            tensor_real: None,
            tensor_int: Some(tensor),
        }
    }

    /// Borrows the backing integer tensor.
    ///
    /// # Panics
    ///
    /// Never panics for a correctly constructed `Population<B, Integer>`.
    #[must_use]
    pub fn tensor(&self) -> &Tensor<B, 2, Int> {
        self.tensor_int
            .as_ref()
            .expect("integer population always has a tensor_int")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    type TestBackend = NdArray;

    #[test]
    fn real_population_reports_shape() {
        let device = Default::default();
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);
        let pop = Population::<TestBackend, Real>::new_real(tensor);
        assert_eq!(pop.pop_size(), 2);
        assert_eq!(pop.genome_dim(), 2);
        assert_eq!(pop.tensor().shape().dims, vec![2, 2]);
    }

    #[test]
    fn binary_population_uses_int_tensor() {
        let device = Default::default();
        let data = TensorData::new(vec![0i64, 1, 1, 0, 1, 0], [2, 3]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &device);
        let pop = Population::<TestBackend, Binary>::new_binary(tensor);
        assert_eq!(pop.pop_size(), 2);
        assert_eq!(pop.genome_dim(), 3);
    }
}
