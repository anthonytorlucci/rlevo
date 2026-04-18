//! Categorical projection operator (Algorithm 1, Bellemare et al. 2017).
//!
//! Given the target network's probability mass at each atom for the bootstrap
//! action, together with the observed reward and terminal mask, the
//! categorical projection returns the *target distribution* on the same fixed
//! atom support that the policy network is trying to match.
//!
//! # Algorithm
//!
//! For each sample in the batch, each atom `z_i` of the next-state
//! distribution is shifted by the Bellman backup:
//!
//! ```text
//! Tz_i = clamp(r + γ · (1 − done) · z_i, v_min, v_max)
//! ```
//!
//! `Tz_i` generally falls between two atoms of the *fixed* support. The
//! projection redistributes each `next_probs_i` between the two neighbouring
//! atoms, weighted by their relative distance, producing the projected
//! probability mass `target_probs`.
//!
//! The implementation is fully vectorised; all batched bin updates use a
//! single call to [`Tensor::scatter`] per neighbour.

use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor};

/// Projects the target distribution onto the fixed categorical support.
///
/// # Parameters
/// - `next_probs`: target-network probability mass for the bootstrap action,
///   shape `(batch, num_atoms)`. Must sum to ≈1 along the atom axis.
/// - `rewards`: per-sample reward `r`, shape `(batch,)`.
/// - `dones`: per-sample terminal mask in `{0.0, 1.0}`, shape `(batch,)`.
/// - `support`: atom values `z_0 … z_{N-1}`, shape `(num_atoms,)`, assumed
///   uniformly spaced between `v_min` and `v_max`.
/// - `gamma`: discount factor.
/// - `v_min`, `v_max`: support bounds used for clamping.
/// - `num_atoms`: `N`. Kept as an explicit argument because it feeds the
///   `delta_z` computation without an extra GPU round-trip.
///
/// # Returns
/// A `(batch, num_atoms)` tensor of projected probabilities. Rows sum to ≈1.
#[allow(clippy::too_many_arguments)]
pub fn project_distribution<B: Backend>(
    next_probs: Tensor<B, 2>,
    rewards: Tensor<B, 1>,
    dones: Tensor<B, 1>,
    support: Tensor<B, 1>,
    gamma: f32,
    v_min: f32,
    v_max: f32,
    num_atoms: usize,
) -> Tensor<B, 2> {
    let [batch_size, n_atoms] = next_probs.dims();
    debug_assert_eq!(
        n_atoms, num_atoms,
        "next_probs atom axis must match num_atoms",
    );
    assert!(num_atoms >= 2, "C51 requires at least two atoms");
    let device = next_probs.device();
    let delta_z = (v_max - v_min) / (num_atoms as f32 - 1.0);

    // Broadcast each argument to (B, N).
    let rewards_bn: Tensor<B, 2> = rewards.unsqueeze_dim::<2>(1); // (B, 1)
    let dones_bn: Tensor<B, 2> = dones.unsqueeze_dim::<2>(1); // (B, 1)
    let support_bn: Tensor<B, 2> = support.unsqueeze_dim::<2>(0); // (1, N)

    // Bellman shift: Tz = clamp(r + γ · (1 − done) · z, v_min, v_max).
    let keep = dones_bn.neg().add_scalar(1.0); // 1 − done
    let tz = rewards_bn + keep * support_bn * gamma;
    let tz = tz.clamp(v_min, v_max);

    // Continuous atom coordinate b ∈ [0, N-1], and its floor/ceil as Int.
    let b = (tz.sub_scalar(v_min)).div_scalar(delta_z);
    let l_idx: Tensor<B, 2, Int> = b.clone().floor().int();
    let u_idx: Tensor<B, 2, Int> = b.clone().ceil().int();

    // When b lands exactly on an atom, l == u and both distance weights are
    // zero. Adding a `(l == u)` indicator onto the "lower" weight preserves
    // the full probability mass on the hit atom — mirrors the CleanRL fix.
    let l_eq_u_mask: Tensor<B, 2> = l_idx.clone().equal(u_idx.clone()).float();

    let u_f = u_idx.clone().float();
    let l_f = l_idx.clone().float();

    let weight_lower = next_probs.clone() * (u_f - b.clone() + l_eq_u_mask);
    let weight_upper = next_probs * (b - l_f);

    let zeros: Tensor<B, 2> = Tensor::zeros([batch_size, num_atoms], &device);
    let m = zeros.scatter(1, l_idx, weight_lower, IndexingUpdateOp::Add);
    m.scatter(1, u_idx, weight_upper, IndexingUpdateOp::Add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn make_support(v_min: f32, v_max: f32, n: usize, device: &<B as Backend>::Device) -> Tensor<B, 1> {
        let delta = (v_max - v_min) / (n as f32 - 1.0);
        let data: Vec<f32> = (0..n).map(|i| v_min + (i as f32) * delta).collect();
        Tensor::from_data(TensorData::new(data, vec![n]), device)
    }

    fn into_vec(tensor: Tensor<B, 2>) -> Vec<f32> {
        tensor.into_data().convert::<f32>().into_vec::<f32>().expect("f32 vec")
    }

    #[test]
    fn projection_identity_when_support_aligned() {
        // With reward = 0, γ = 1, done = 0 and a support that passes through
        // every atom, each atom maps to itself — the projection is the
        // identity on the input probabilities.
        let device: <B as Backend>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.3, 0.2], vec![1, 3]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let dones = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let support = make_support(-1.0, 1.0, 3, &device);

        let out = project_distribution(next_probs, rewards, dones, support, 1.0, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] - 0.3).abs() < 1e-6);
        assert!((v[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn projection_terminal_reward_half_splits_between_atoms_one_and_two() {
        // Hand-computed Bellemare-style reference on a 3-atom support:
        //   support = [-1, 0, 1], reward = 0.5, done = 1 → Tz ≡ 0.5 ∀ z_i
        //   b = 1.5 → mass evenly split between atoms 1 and 2, independent of
        //   next_probs.
        let device: <B as Backend>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.1_f32, 0.4, 0.5], vec![1, 3]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![0.5_f32], vec![1]), &device);
        let dones = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32], vec![1]), &device);
        let support = make_support(-1.0, 1.0, 3, &device);

        let out = project_distribution(next_probs, rewards, dones, support, 0.99, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!(v[0].abs() < 1e-6, "atom 0 should be empty, got {}", v[0]);
        assert!((v[1] - 0.5).abs() < 1e-6, "atom 1 should hold half the mass, got {}", v[1]);
        assert!((v[2] - 0.5).abs() < 1e-6, "atom 2 should hold half the mass, got {}", v[2]);
    }

    #[test]
    fn projection_clamps_above_support() {
        // reward ≫ v_max ⇒ Tz clamps at v_max for every atom ⇒ all mass at
        // the top atom.
        let device: <B as Backend>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32 / 3.0; 3], vec![1, 3]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![100.0_f32], vec![1]), &device);
        let dones = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let support = make_support(-1.0, 1.0, 3, &device);

        let out = project_distribution(next_probs, rewards, dones, support, 1.0, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!(v[0].abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn projection_preserves_total_mass() {
        // Arbitrary batch, random-ish probabilities normalised to 1: rows of
        // the projection should still sum to ≈1.
        let device: <B as Backend>::Device = Default::default();
        let batch = 4;
        let n = 5;
        let next_probs_raw: Vec<f32> = (0..batch * n).map(|i| 1.0 + (i as f32) * 0.1).collect();
        let mut next_probs_norm = Vec::with_capacity(batch * n);
        for row in next_probs_raw.chunks(n) {
            let s: f32 = row.iter().sum();
            next_probs_norm.extend(row.iter().map(|x| x / s));
        }
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(next_probs_norm, vec![batch, n]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.3_f32, -0.5, 1.2, 0.0], vec![batch]),
            &device,
        );
        let dones = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 0.0, 0.0], vec![batch]),
            &device,
        );
        let support = make_support(-2.0, 2.0, n, &device);

        let out = project_distribution(next_probs, rewards, dones, support, 0.9, -2.0, 2.0, n);
        let v = into_vec(out);
        for row in v.chunks(n) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum should be 1, got {s}: {row:?}");
        }
    }
}
