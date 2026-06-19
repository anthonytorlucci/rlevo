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

use burn::tensor::{backend::Backend, Int, Tensor};

use crate::genome::{Binary, Integer, Real, TensorGenome};

/// Population stored on a Burn backend device.
///
/// The concrete tensor type depends on the genome kind `K`, chosen at compile
/// time through [`TensorGenome::Tensor`]: `Real` is backed by `Tensor<B, 2>`,
/// `Binary` and `Integer` by `Tensor<B, 2, Int>`. Because the storage type is a
/// function of `K`, there is a single tensor field and no run-time tag — a
/// population can never hold the wrong tensor flavour for its kind, so the
/// [`tensor`](Population::tensor) accessor is total (it cannot fail).
///
/// The `K: TensorGenome` bound is what keeps this honest: kinds without a
/// rectangular tensor form (e.g. [`Tree`](crate::genome::Tree)) do not implement
/// `TensorGenome`, so `Population<B, Tree>` does not type-check.
#[derive(Debug, Clone)]
pub struct Population<B: Backend, K: TensorGenome> {
    pop_size: usize,
    genome_dim: usize,
    tensor: K::Tensor<B>,
}

impl<B: Backend, K: TensorGenome> Population<B, K> {
    /// Returns the number of individuals (rows) in the population.
    ///
    /// This value equals `tensor.dims()[0]`.
    #[must_use]
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    /// Returns the genome dimensionality (number of genes, i.e. columns).
    ///
    /// This value equals `tensor.dims()[1]`.
    #[must_use]
    pub fn genome_dim(&self) -> usize {
        self.genome_dim
    }

    /// Borrows the backing tensor for this population's kind.
    ///
    /// The concrete type is [`K::Tensor<B>`](TensorGenome::Tensor) — a
    /// `Tensor<B, 2>` for `Real`, a `Tensor<B, 2, Int>` for `Binary`/`Integer`
    /// — with shape `[pop_size, genome_dim]`. Use it to pass the population to
    /// fitness functions or operator kernels without giving up ownership.
    #[must_use]
    pub fn tensor(&self) -> &K::Tensor<B> {
        &self.tensor
    }

    /// Consumes the wrapper and returns the owned tensor.
    ///
    /// Prefer this over [`tensor`](Population::tensor) when handing the
    /// population off to a strategy or operator that needs ownership (e.g. to
    /// avoid a clone on the hot path).
    #[must_use]
    pub fn into_tensor(self) -> K::Tensor<B> {
        self.tensor
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
    #[must_use]
    pub fn new_real(tensor: Tensor<B, 2>) -> Self {
        let dims = tensor.dims();
        Self {
            pop_size: dims[0],
            genome_dim: dims[1],
            tensor,
        }
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
    #[must_use]
    pub fn new_binary(tensor: Tensor<B, 2, Int>) -> Self {
        let dims = tensor.dims();
        Self {
            pop_size: dims[0],
            genome_dim: dims[1],
            tensor,
        }
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
    #[must_use]
    pub fn new_integer(tensor: Tensor<B, 2, Int>) -> Self {
        let dims = tensor.dims();
        Self {
            pop_size: dims[0],
            genome_dim: dims[1],
            tensor,
        }
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
