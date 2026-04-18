//! Auxiliary-phase replay buffer for PPG.
//!
//! Accumulates the last `n_iteration` policy-phase rollouts so the auxiliary
//! phase can train over the combined pool. Each slice stores:
//!
//! - `obs` — the rollout's observations (CPU-resident, cloned at push time)
//! - `returns` — GAE target returns `advantages + values`, used as the
//!   regression target for both the main value net and the auxiliary value
//!   head
//!
//! The pre-aux-phase policy logits (`π_old` for the distillation KL) are
//! **not** stored here; they are computed once at the start of the
//! auxiliary phase against the policy that has just finished `n_iteration`
//! policy-phase updates, matching CleanRL's `ppg_procgen.py`. Storing them
//! per-step at rollout collection time would anchor the distillation to a
//! stale policy and undo the policy-phase's learning.
//!
//! Sampling is i.i.d. random across the concatenated pool (the "simpler
//! random-obs sampling" choice from the PPG spec §8.1). The more elaborate
//! contiguous-per-rollout sampling used in CleanRL's `ppg_procgen.py` is a
//! follow-up once benchmarks demand it.

use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use std::marker::PhantomData;

use evorl_core::base::{Observation, TensorConvertible};

/// One rollout's worth of auxiliary training data.
#[derive(Debug)]
pub struct AuxRolloutSlice<O> {
    /// Observations in collection order.
    pub obs: Vec<O>,
    /// GAE target returns, aligned with `obs`.
    pub returns: Vec<f32>,
}

/// Accumulates [`AuxRolloutSlice`]s until the auxiliary phase is ready to run.
#[derive(Debug)]
pub struct AuxRolloutBuffer<B: Backend, O> {
    slices: Vec<AuxRolloutSlice<O>>,
    _marker: PhantomData<B>,
}

impl<B: Backend, O> Default for AuxRolloutBuffer<B, O> {
    fn default() -> Self {
        Self {
            slices: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<B: Backend, O: Clone> AuxRolloutBuffer<B, O> {
    /// Creates an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one rollout's worth of auxiliary data.
    ///
    /// # Panics
    /// - If `returns.len() != obs.len()`.
    pub fn push_slice(&mut self, obs: Vec<O>, returns: Vec<f32>) {
        assert_eq!(
            returns.len(),
            obs.len(),
            "aux buffer: returns len {} ≠ obs len {}",
            returns.len(),
            obs.len()
        );
        self.slices.push(AuxRolloutSlice { obs, returns });
    }

    /// `true` when the buffer holds at least `n_iteration` rollouts.
    #[must_use]
    pub fn is_ready(&self, n_iteration: usize) -> bool {
        self.slices.len() >= n_iteration
    }

    /// Number of rollouts currently held.
    #[must_use]
    pub fn num_slices(&self) -> usize {
        self.slices.len()
    }

    /// Total number of steps across all rollouts.
    #[must_use]
    pub fn len_steps(&self) -> usize {
        self.slices.iter().map(|s| s.obs.len()).sum()
    }

    /// Drops all accumulated rollouts.
    pub fn clear(&mut self) {
        self.slices.clear();
    }

    /// Returns the `(slice, inner)` address for a global step index.
    fn locate(&self, mut global: usize) -> (usize, usize) {
        for (i, s) in self.slices.iter().enumerate() {
            if global < s.obs.len() {
                return (i, global);
            }
            global -= s.obs.len();
        }
        panic!("aux buffer index out of range");
    }

    /// Flat CPU-side access to one observation by global step index.
    /// Used by the agent to batch-compute `π_old` logits once per aux phase.
    pub fn obs_at(&self, global: usize) -> &O {
        let (si, ii) = self.locate(global);
        &self.slices[si].obs[ii]
    }

    /// Materialises a minibatch of `indices` into device tensors.
    ///
    /// Returns `(obs_tensor: [k, ..obs_shape], returns_tensor: [k])`.
    pub fn gather_minibatch<const DO: usize, const DB: usize>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, DB>, Tensor<B, 1>)
    where
        O: Observation<DO> + TensorConvertible<DO, B>,
    {
        let n = indices.len();
        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();

        let mut obs_flat: Vec<f32> = Vec::with_capacity(n * numel_per_obs);
        let mut returns: Vec<f32> = Vec::with_capacity(n);

        for &global in indices {
            let (si, ii) = self.locate(global);
            let slice = &self.slices[si];
            let t: Tensor<B, DO> = slice.obs[ii].to_tensor(device);
            let data = t.into_data().convert::<f32>();
            obs_flat.extend_from_slice(data.as_slice::<f32>().expect("f32 obs"));
            returns.push(slice.returns[ii]);
        }

        let mut batched_shape: Vec<usize> = Vec::with_capacity(DB);
        batched_shape.push(n);
        batched_shape.extend_from_slice(&obs_shape);
        let obs_tensor: Tensor<B, DB> =
            Tensor::from_data(TensorData::new(obs_flat, batched_shape), device);
        let returns_tensor: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(returns, vec![n]), device);
        (obs_tensor, returns_tensor)
    }

    /// Shuffled global step indices for one auxiliary epoch.
    pub fn indices_shuffled<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let total = self.len_steps();
        let mut idx: Vec<usize> = (0..total).collect();
        idx.shuffle(rng);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::base::TensorConversionError;
    type Be = burn::backend::NdArray;

    #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
    struct TestObs([f32; 2]);

    impl Observation<1> for TestObs {
        fn shape() -> [usize; 1] {
            [2]
        }
    }

    impl<Bk: burn::tensor::backend::Backend> TensorConvertible<1, Bk> for TestObs {
        fn to_tensor(&self, device: &Bk::Device) -> Tensor<Bk, 1> {
            Tensor::from_floats(self.0, device)
        }
        fn from_tensor(_t: Tensor<Bk, 1>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by this test")
        }
    }

    fn push(buf: &mut AuxRolloutBuffer<Be, TestObs>, n: usize) {
        let obs: Vec<TestObs> = (0..n)
            .map(|i| TestObs([i as f32, i as f32 + 0.5]))
            .collect();
        let returns: Vec<f32> = (0..n).map(|i| i as f32).collect();
        buf.push_slice(obs, returns);
    }

    #[test]
    fn aux_buffer_accumulates_and_drains() {
        let mut buf: AuxRolloutBuffer<Be, TestObs> = AuxRolloutBuffer::new();
        assert!(!buf.is_ready(3));
        push(&mut buf, 4);
        push(&mut buf, 4);
        push(&mut buf, 4);
        assert!(buf.is_ready(3));
        assert_eq!(buf.len_steps(), 12);
        assert_eq!(buf.num_slices(), 3);
        buf.clear();
        assert_eq!(buf.len_steps(), 0);
        assert!(!buf.is_ready(3));
    }

    #[test]
    fn gather_minibatch_shapes() {
        let device: <Be as burn::tensor::backend::Backend>::Device = Default::default();
        let mut buf: AuxRolloutBuffer<Be, TestObs> = AuxRolloutBuffer::new();
        push(&mut buf, 3);
        push(&mut buf, 3);
        let (o, r) = buf.gather_minibatch::<1, 2>(&[0, 2, 3, 5], &device);
        assert_eq!(o.dims(), [4, 2]);
        assert_eq!(r.dims(), [4]);
    }

    #[test]
    fn gather_minibatch_preserves_row_values() {
        let device: <Be as burn::tensor::backend::Backend>::Device = Default::default();
        let mut buf: AuxRolloutBuffer<Be, TestObs> = AuxRolloutBuffer::new();
        push(&mut buf, 3);
        let (_, r) = buf.gather_minibatch::<1, 2>(&[0, 1, 2], &device);
        let d = r.into_data().convert::<f32>();
        let slice = d.as_slice::<f32>().unwrap().to_vec();
        assert_eq!(slice, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn obs_at_spans_slices() {
        let mut buf: AuxRolloutBuffer<Be, TestObs> = AuxRolloutBuffer::new();
        push(&mut buf, 2); // indices 0, 1
        push(&mut buf, 3); // indices 2, 3, 4
        assert_eq!(buf.obs_at(0).0, [0.0, 0.5]);
        assert_eq!(buf.obs_at(2).0, [0.0, 0.5]);
        assert_eq!(buf.obs_at(4).0, [2.0, 2.5]);
    }
}
