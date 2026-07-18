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
//! Tz_i = clamp(r + γ · (1 − terminated) · z_i, v_min, v_max)
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

/// Spacing `Δz` between adjacent atoms of a uniform categorical support.
///
/// `Δz = (v_max − v_min) / (num_atoms − 1)` — the single source of truth for
/// the atom scale. Both the *construction* of the support tensor
/// (`z_i = v_min + i · Δz`) and the *index* computation inside
/// [`project_distribution`] must use this one function; two independent
/// spellings of the same constant can disagree by an ULP and push a bin index
/// off the end of the support.
///
/// Kept as a free function taking scalars (rather than a method on a config)
/// so the projection operator stays usable by any distributional agent —
/// C51 today, a future Rainbow tomorrow — without a config dependency.
///
/// # Arguments
/// - `v_min`, `v_max`: support bounds `z_0` and `z_{N-1}`.
/// - `num_atoms`: number of atoms `N`.
///
/// # Returns
/// The uniform atom spacing. A support needs **at least two** atoms for a
/// spacing to be defined; for `num_atoms < 2` this returns [`f32::NAN`] rather
/// than panicking or dividing by zero, so the degeneracy propagates visibly
/// instead of silently producing `±inf` coordinates. Callers that require a
/// well-formed support validate `num_atoms >= 2` up front
/// (`C51TrainingConfig::validate`, and the assertion in
/// [`project_distribution`]).
///
/// # Examples
/// ```
/// use rlevo_reinforcement_learning::algorithms::c51::projection::atom_spacing;
///
/// assert!((atom_spacing(-10.0, 10.0, 51) - 0.4).abs() < 1e-6);
/// assert!(atom_spacing(0.0, 1.0, 1).is_nan());
/// ```
#[must_use]
pub fn atom_spacing(v_min: f32, v_max: f32, num_atoms: usize) -> f32 {
    if num_atoms < 2 {
        return f32::NAN;
    }
    (v_max - v_min) / (num_atoms as f32 - 1.0)
}

/// Projects the target distribution onto the fixed categorical support.
///
/// # Parameters
/// - `next_probs`: target-network probability mass for the bootstrap action,
///   shape `(batch, num_atoms)`. Must sum to ≈1 along the atom axis.
/// - `rewards`: per-sample reward `r`, shape `(batch,)`.
/// - `terminated`: per-sample terminal mask in `{0.0, 1.0}`, shape `(batch,)`.
/// - `support`: atom values `z_0 … z_{N-1}`, shape `(num_atoms,)`, assumed
///   uniformly spaced between `v_min` and `v_max`.
/// - `gamma`: discount factor.
/// - `v_min`, `v_max`: support bounds used for clamping.
/// - `num_atoms`: `N`. Kept as an explicit argument because it feeds the
///   `delta_z` computation without an extra GPU round-trip.
///
/// # Returns
/// A `(batch, num_atoms)` tensor of projected probabilities. Rows sum to ≈1.
///
/// # Panics
/// - If `num_atoms < 2`. A categorical support needs at least two atoms for the
///   `N-1` spacing denominator to be meaningful.
/// - If the atom spacing `Δz = (v_max − v_min) / (N − 1)` is not finite and
///   strictly positive — i.e. the support is degenerate (`v_max == v_min`) or
///   inverted (`v_max < v_min`), or a bound is non-finite. This is *not*
///   implied by `num_atoms >= 2`: a well-sized `num_atoms = 51` support with
///   `v_min == v_max` gives `Δz = 0`, so `b = (Tz − v_min) / Δz` is `NaN`.
///   `f32::clamp` propagates `NaN` rather than rescuing it, and `NaN as i32`
///   saturates to `0`, which would collapse every atom index to zero and pile
///   the whole distribution onto atom 0 — a silently corrupted target rather
///   than an observable failure. The assertion converts that into a panic.
#[allow(clippy::too_many_arguments)]
pub fn project_distribution<B: Backend>(
    next_probs: Tensor<B, 2>,
    rewards: Tensor<B, 1>,
    terminated: Tensor<B, 1>,
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
    let delta_z = atom_spacing(v_min, v_max, num_atoms);
    // A zero or non-finite spacing makes `b` NaN below; `clamp` propagates NaN
    // and `NaN as i32` saturates to 0, so the projection would return a
    // silently corrupted distribution with all mass on atom 0. Fail loudly.
    assert!(
        delta_z.is_finite() && delta_z > 0.0,
        "C51 support must satisfy v_max > v_min with finite bounds \
         (got v_min={v_min}, v_max={v_max}, num_atoms={num_atoms}, delta_z={delta_z})",
    );

    // Broadcast each argument to (B, N).
    let rewards_bn: Tensor<B, 2> = rewards.unsqueeze_dim::<2>(1); // (B, 1)
    let terminated_bn: Tensor<B, 2> = terminated.unsqueeze_dim::<2>(1); // (B, 1)
    let support_bn: Tensor<B, 2> = support.unsqueeze_dim::<2>(0); // (1, N)

    // Bellman shift: Tz = clamp(r + γ · (1 − terminated) · z, v_min, v_max).
    let keep = terminated_bn.neg().add_scalar(1.0); // 1 − terminated
    let tz = rewards_bn + keep * support_bn * gamma;
    let tz = tz.clamp(v_min, v_max);

    // Continuous atom coordinate b ∈ [0, N-1], and its floor/ceil as Int.
    //
    // `tz` is already clamped to [v_min, v_max], so `b ∈ [0, N-1]` holds
    // exactly in real arithmetic — Bellemare et al. 2017 Algorithm 1 states it
    // as an inline assertion. It does *not* hold in IEEE-754: for many valid
    // supports (e.g. v_min = -10, v_max = 0.1, N = 8) the f32 division rounds a
    // few ULPs above N-1, so `ceil` yields index N and the scatter below
    // indexes off the end of the support. Re-clamping `b` — not the derived
    // indices — restores the invariant and keeps `u_f - b == 0` exact at the
    // boundary, so the `l == u` mask below still carries the full mass. Same
    // guard as CleanRL's `c51.py`. No-op for any support that does not round
    // out of range.
    let b = (tz.sub_scalar(v_min)).div_scalar(delta_z);
    let b = b.clamp(0.0, num_atoms as f32 - 1.0);
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
    use burn::backend::Flex;
    use burn::tensor::TensorData;

    type B = Flex;

    fn make_support(
        v_min: f32,
        v_max: f32,
        n: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let delta = atom_spacing(v_min, v_max, n);
        let data: Vec<f32> = (0..n).map(|i| v_min + (i as f32) * delta).collect();
        Tensor::from_data(TensorData::new(data, vec![n]), device)
    }

    fn into_vec(tensor: Tensor<B, 2>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 vec")
    }

    #[test]
    fn projection_identity_when_support_aligned() {
        // With reward = 0, γ = 1, terminated = 0 and a support that passes through
        // every atom, each atom maps to itself — the projection is the
        // identity on the input probabilities.
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.3, 0.2], vec![1, 3]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let support = make_support(-1.0, 1.0, 3, &device);

        let out = project_distribution(next_probs, rewards, terminated, support, 1.0, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] - 0.3).abs() < 1e-6);
        assert!((v[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn projection_terminal_reward_half_splits_between_atoms_one_and_two() {
        // Hand-computed Bellemare-style reference on a 3-atom support:
        //   support = [-1, 0, 1], reward = 0.5, terminated = 1 → Tz ≡ 0.5 ∀ z_i
        //   b = 1.5 → mass evenly split between atoms 1 and 2, independent of
        //   next_probs.
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.1_f32, 0.4, 0.5], vec![1, 3]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![0.5_f32], vec![1]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32], vec![1]), &device);
        let support = make_support(-1.0, 1.0, 3, &device);

        let out =
            project_distribution(next_probs, rewards, terminated, support, 0.99, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!(v[0].abs() < 1e-6, "atom 0 should be empty, got {}", v[0]);
        assert!(
            (v[1] - 0.5).abs() < 1e-6,
            "atom 1 should hold half the mass, got {}",
            v[1]
        );
        assert!(
            (v[2] - 0.5).abs() < 1e-6,
            "atom 2 should hold half the mass, got {}",
            v[2]
        );
    }

    #[test]
    fn projection_clamps_above_support() {
        // reward ≫ v_max ⇒ Tz clamps at v_max for every atom ⇒ all mass at
        // the top atom.
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs =
            Tensor::<B, 2>::from_data(TensorData::new(vec![1.0_f32 / 3.0; 3], vec![1, 3]), &device);
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![100.0_f32], vec![1]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let support = make_support(-1.0, 1.0, 3, &device);

        let out = project_distribution(next_probs, rewards, terminated, support, 1.0, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!(v[0].abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn projection_preserves_total_mass() {
        // Arbitrary batch, random-ish probabilities normalised to 1: rows of
        // the projection should still sum to ≈1.
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let batch = 4;
        let n = 5;
        let next_probs_raw: Vec<f32> = (0..batch * n).map(|i| 1.0 + (i as f32) * 0.1).collect();
        let mut next_probs_norm = Vec::with_capacity(batch * n);
        for row in next_probs_raw.chunks(n) {
            let s: f32 = row.iter().sum();
            next_probs_norm.extend(row.iter().map(|x| x / s));
        }
        let next_probs =
            Tensor::<B, 2>::from_data(TensorData::new(next_probs_norm, vec![batch, n]), &device);
        let rewards = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.3_f32, -0.5, 1.2, 0.0], vec![batch]),
            &device,
        );
        let terminated = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 0.0, 0.0], vec![batch]),
            &device,
        );
        let support = make_support(-2.0, 2.0, n, &device);

        let out = project_distribution(next_probs, rewards, terminated, support, 0.9, -2.0, 2.0, n);
        let v = into_vec(out);
        for row in v.chunks(n) {
            let s: f32 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row sum should be 1, got {s}: {row:?}"
            );
        }
    }

    /// Drives one clamped-to-`v_max` projection and returns the projected row.
    ///
    /// A reward far above `v_max` with `terminated = 1` forces `Tz ≡ v_max` for
    /// every atom, which is exactly the `b == N-1` boundary where the f32
    /// rounding defect bites.
    fn project_saturated_row(v_min: f32, v_max: f32, n: usize, reward: f32) -> Vec<f32> {
        assert!(
            reward >= v_max,
            "reward {reward} must saturate the support to force Tz = v_max",
        );
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32 / (n as f32); n], vec![1, n]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![reward], vec![1]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32], vec![1]), &device);
        let support = make_support(v_min, v_max, n, &device);

        let out = project_distribution(
            next_probs, rewards, terminated, support, 0.99, v_min, v_max, n,
        );
        into_vec(out)
    }

    #[test]
    fn projection_handles_f32_rounding_at_top_atom() {
        // Regression, issue #180. On this *valid* support the continuous atom
        // coordinate rounds to b = 7.000000477 > N-1 = 7, so `ceil(b) = 8`
        // scattered off the end of a size-8 axis and panicked. Note the
        // default support (-10, 10, 51) cannot catch this: it lands on 50.0
        // exactly, which is why the bug survived.
        let v = project_saturated_row(-10.0, 0.1, 8, 5.0);

        let s: f32 = v.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "row sum should be 1, got {s}: {v:?}"
        );
        assert!(
            (v[7] - 1.0).abs() < 1e-5,
            "clamped Tz = v_max should put all mass on the top atom, got {}",
            v[7]
        );
        for (i, p) in v.iter().enumerate().take(7) {
            assert!(*p < 1e-6, "atom {i} should be empty, got {p}");
        }
    }

    #[test]
    fn projection_handles_f32_rounding_on_symmetric_support() {
        // Second known overflow from the #180 sweep: (-13, 13, 12) yields
        // b = 11.000000954 against a max valid index of 11.
        let v = project_saturated_row(-13.0, 13.0, 12, 100.0);

        let s: f32 = v.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "row sum should be 1, got {s}: {v:?}"
        );
        assert!(
            (v[11] - 1.0).abs() < 1e-5,
            "clamped Tz = v_max should put all mass on the top atom, got {}",
            v[11]
        );
    }

    #[test]
    fn projection_preserves_mass_across_overflowing_supports() {
        // A spread of (v_min, v_max, N) triples from the #180 sweep, every one
        // of which passes `C51TrainingConfig::validate` and every one of which
        // panicked before the clamp. Mass must land wholly on the top atom.
        let supports = [
            (-10.0_f32, 0.1_f32, 8_usize),
            (-13.0, 13.0, 12),
            (-10.0, 0.1, 15),
            (-0.1, 0.1, 12),
            (-0.1, 0.2, 20),
        ];

        for (v_min, v_max, n) in supports {
            let v = project_saturated_row(v_min, v_max, n, v_max + 1000.0);
            let s: f32 = v.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row sum should be 1 for ({v_min}, {v_max}, {n}), got {s}"
            );
            assert!(
                (v[n - 1] - 1.0).abs() < 1e-5,
                "top atom should hold all mass for ({v_min}, {v_max}, {n}), got {}",
                v[n - 1]
            );
        }
    }

    /// Drives one projection on an arbitrary (possibly degenerate) support.
    ///
    /// Deliberately does *not* pre-validate `v_min`/`v_max`: the point is to
    /// reach [`project_distribution`]'s own guard. Note `make_support` calls
    /// `atom_spacing` too, so for a degenerate support the support tensor
    /// itself is degenerate — the tests below assert on the panic *message* to
    /// confirm the guard fired rather than some incidental failure.
    fn project_on_support(v_min: f32, v_max: f32, n: usize) -> Vec<f32> {
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32 / (n as f32); n], vec![1, n]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let support = make_support(v_min, v_max, n, &device);

        let out = project_distribution(
            next_probs, rewards, terminated, support, 0.99, v_min, v_max, n,
        );
        into_vec(out)
    }

    #[test]
    #[should_panic(expected = "C51 support must satisfy v_max > v_min")]
    fn projection_rejects_degenerate_support_v_min_equals_v_max() {
        // Δz = 0 ⇒ b = NaN ⇒ every index collapses to 0 and the whole
        // distribution silently lands on atom 0. Must panic instead.
        let _ = project_on_support(5.0, 5.0, 51);
    }

    #[test]
    #[should_panic(expected = "C51 support must satisfy v_max > v_min")]
    fn projection_rejects_inverted_support_v_min_greater_than_v_max() {
        // Δz < 0 ⇒ the atom coordinate runs backwards; the support is not a
        // valid ordered categorical support at all.
        let _ = project_on_support(1.0, -1.0, 51);
    }

    #[test]
    fn atom_spacing_matches_uniform_support_and_is_nan_when_degenerate() {
        assert!((atom_spacing(-10.0, 10.0, 51) - 0.4).abs() < 1e-6);
        assert!((atom_spacing(-1.0, 1.0, 3) - 1.0).abs() < 1e-6);
        assert!(
            atom_spacing(0.0, 1.0, 1).is_nan(),
            "spacing is undefined for a single atom"
        );
        assert!(
            atom_spacing(0.0, 1.0, 0).is_nan(),
            "spacing is undefined for an empty support"
        );
    }
}
