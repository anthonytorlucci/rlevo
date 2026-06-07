//! Population containers.
//!
//! [`Population<B, K>`] is a thin wrapper around a device tensor plus the
//! shape metadata strategies need. For real-valued kinds it holds a
//! `Tensor<B, 2>`; binary and integer kinds use `Tensor<B, 2, Int>`.
//!
//! The wrapper exists so operators and strategies have a single shape
//! contract to validate against (they check `pop_size` and `genome_dim`
//! rather than repeatedly interrogating `tensor.dims()`).
//!
//! # Constructing a population
//!
//! Each genome kind has a dedicated constructor that takes the
//! already-allocated tensor:
//!
//! ```no_run
//! use burn::backend::Flex;
//! use burn::tensor::{Tensor, TensorData};
//! use rlevo_evolution::genome::Real;
//! use rlevo_evolution::population::Population;
//!
//! let device = Default::default();
//! // 4 individuals, each with a 3-gene real-valued genome.
//! let data = TensorData::new(vec![0.1f32, 0.2, 0.3,
//!                                 0.4, 0.5, 0.6,
//!                                 0.7, 0.8, 0.9,
//!                                 1.0, 1.1, 1.2], [4, 3]);
//! let tensor = Tensor::<Flex, 2>::from_data(data, &device);
//! let pop = Population::<Flex, Real>::new_real(tensor);
//! assert_eq!(pop.pop_size(), 4);
//! assert_eq!(pop.genome_dim(), 3);
//! ```

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
    ///
    /// This value equals `tensor.dims()[0]` for any population produced by
    /// the public constructors.
    #[must_use]
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    /// Returns the genome dimensionality (number of genes, i.e. columns).
    ///
    /// This value equals `tensor.dims()[1]` for any population produced by
    /// the public constructors.
    #[must_use]
    pub fn genome_dim(&self) -> usize {
        self.genome_dim
    }
}

impl<B: Backend> Population<B, Real> {
    /// Constructs a real-valued population from a `Tensor<B, 2>`.
    ///
    /// Shape is read from `tensor.dims()` at construction time; subsequent
    /// calls to [`pop_size`](Population::pop_size) and
    /// [`genome_dim`](Population::genome_dim) reflect those dimensions.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn::backend::Flex;
    /// use burn::tensor::{Tensor, TensorData};
    /// use rlevo_evolution::genome::Real;
    /// use rlevo_evolution::population::Population;
    ///
    /// let device = Default::default();
    /// let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]);
    /// let pop = Population::<Flex, Real>::new_real(
    ///     Tensor::from_data(data, &device),
    /// );
    /// assert_eq!(pop.pop_size(), 2);
    /// assert_eq!(pop.genome_dim(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `tensor` is not rank 2.
    #[must_use]
    pub fn new_real(tensor: Tensor<B, 2>) -> Self {
        let dims = tensor.dims();
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
    /// The returned tensor has shape `[pop_size, genome_dim]`. Use this
    /// to pass the population to fitness functions or operator kernels
    /// without giving up ownership.
    ///
    /// # Panics
    ///
    /// Never panics in practice: a real-valued population always holds a
    /// real tensor by construction.
    #[must_use]
    pub fn tensor(&self) -> &Tensor<B, 2> {
        self.tensor_real
            .as_ref()
            .expect("real population always has a tensor_real")
    }

    /// Consumes the wrapper and returns the owned tensor.
    ///
    /// Prefer this over [`tensor`](Population::tensor) when handing the
    /// population off to a strategy or operator that needs ownership (e.g.
    /// to avoid a clone on the hot path).
    ///
    /// # Panics
    ///
    /// Never panics in practice: a real-valued population always holds a
    /// real tensor by construction.
    #[must_use]
    pub fn into_tensor(self) -> Tensor<B, 2> {
        self.tensor_real
            .expect("real population always has a tensor_real")
    }
}

impl<B: Backend> Population<B, Binary> {
    /// Constructs a binary population from a `Tensor<B, 2, Int>`.
    ///
    /// Each element is expected to be `0` or `1`; the constructor does not
    /// validate element values. Shape is read from `tensor.dims()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn::backend::Flex;
    /// use burn::tensor::{Int, Tensor, TensorData};
    /// use rlevo_evolution::genome::Binary;
    /// use rlevo_evolution::population::Population;
    ///
    /// let device = Default::default();
    /// // 3 individuals, each with a 4-bit binary genome.
    /// let data = TensorData::new(vec![0i64, 1, 0, 1,
    ///                                 1, 0, 1, 0,
    ///                                 0, 0, 1, 1], [3, 4]);
    /// let pop = Population::<Flex, Binary>::new_binary(
    ///     Tensor::from_data(data, &device),
    /// );
    /// assert_eq!(pop.pop_size(), 3);
    /// assert_eq!(pop.genome_dim(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `tensor` is not rank 2.
    #[must_use]
    pub fn new_binary(tensor: Tensor<B, 2, Int>) -> Self {
        let dims = tensor.dims();
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
    /// The returned tensor has shape `[pop_size, genome_dim]` and element
    /// type `Int`. Callers performing crossover or mutation should work
    /// directly with this tensor.
    ///
    /// # Panics
    ///
    /// Never panics in practice: a binary population always holds an integer
    /// tensor by construction.
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
    /// Elements represent non-negative integer indices (e.g. node indices in
    /// CGP, symbol indices in integer-coded GA). The constructor does not
    /// validate element bounds. Shape is read from `tensor.dims()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn::backend::Flex;
    /// use burn::tensor::{Int, Tensor, TensorData};
    /// use rlevo_evolution::genome::Integer;
    /// use rlevo_evolution::population::Population;
    ///
    /// let device = Default::default();
    /// // 2 individuals, each with a 5-gene integer-valued genome.
    /// let data = TensorData::new(vec![0i64, 3, 1, 4, 2,
    ///                                 2, 0, 4, 1, 3], [2, 5]);
    /// let pop = Population::<Flex, Integer>::new_integer(
    ///     Tensor::from_data(data, &device),
    /// );
    /// assert_eq!(pop.pop_size(), 2);
    /// assert_eq!(pop.genome_dim(), 5);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `tensor` is not rank 2.
    #[must_use]
    pub fn new_integer(tensor: Tensor<B, 2, Int>) -> Self {
        let dims = tensor.dims();
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
    /// The returned tensor has shape `[pop_size, genome_dim]` and element
    /// type `Int`. Element values are non-negative indices whose domain is
    /// determined by the problem (e.g. `0..n_nodes` for CGP).
    ///
    /// # Panics
    ///
    /// Never panics in practice: an integer population always holds an integer
    /// tensor by construction.
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
    use burn::backend::Flex;
    use burn::tensor::TensorData;
    type TestBackend = Flex;

    #[test]
    fn real_population_reports_shape() {
        let device = Default::default();
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);
        let pop = Population::<TestBackend, Real>::new_real(tensor);
        assert_eq!(pop.pop_size(), 2);
        assert_eq!(pop.genome_dim(), 2);
        assert_eq!(pop.tensor().dims(), [2, 2]);
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
