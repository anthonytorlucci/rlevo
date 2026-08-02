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
//! ```math
//! Tz_i = \text{clamp}(r + \gamma \cdot (1 - \text{terminated}) \cdot z_i,\; v_{min},\; v_{max})
//! ```
//!
//! `Tz_i` generally falls between two atoms of the *fixed* support. The
//! projection redistributes each `next_probs_i` between the two neighbouring
//! atoms, weighted by their relative distance, producing the projected
//! probability mass `target_probs`.
//!
//! The implementation is fully vectorised; all batched bin updates use a
//! single call to [`Tensor::scatter`] per neighbour.
//!
//! # Non-finite inputs (issue #1044)
//!
//! This is the only place in the workspace where a float tensor is converted to
//! `Int` and fed to a `scatter`, so it is the only place where a non-finite
//! *value* can become an out-of-range *index*. Two properties are held here,
//! and they are a pair — neither is safe without the other:
//!
//! 1. A `NaN` reward stays observable. Every clamp goes through
//!    [`clamp_preserving_nan`] rather than [`Tensor::clamp`], because `clamp`'s
//!    `NaN` behaviour differs between the host and GPU backends: Flex
//!    propagates, wgpu/Metal rescues to the lower bound. Under the plain clamp
//!    a `NaN` reward yielded an all-`NaN` row on Flex but a *well-formed* row
//!    summing to exactly `1.0` on Metal — one that no finiteness guard
//!    downstream could detect.
//! 2. The derived atom indices are clamped to `[0, N-1]`. Preserving the `NaN`
//!    means `floor(b).int()` now sees one, and that conversion is itself
//!    backend-dependent (the host saturates to `0`; Metal yields `i32::MIN`).
//!    Burn's wgpu `scatter` performs **no** bounds check at any layer, so an
//!    out-of-range index there writes into an unrelated tensor's memory instead
//!    of panicking.
//!
//! The result is a row that is non-finite on every backend — loud enough for
//! the caller's `FiniteLossGuard` to skip the step — and an index that is
//! in range on every backend. Assert on **finiteness** of the projected row,
//! never on its sum: the pre-fix Metal row summed to exactly `1.0`.
//!
//! [`clamp_preserving_nan`]: crate::algorithms::shared::clamp_preserving_nan

use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor};

use crate::algorithms::shared::clamp_preserving_nan;

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
// Structural size of the distributional support (atom / quantile count). The
// configs cap these in the low hundreds, so the value is exact in f32; a
// non-finite or zero spacing is rejected by assertion before use.
#[allow(clippy::cast_precision_loss)]
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
/// A `(batch, num_atoms)` tensor of projected probabilities. Rows sum to ≈1 for
/// finite inputs. A row whose reward was `NaN` is **non-finite** on every
/// backend — see the module docs; do not postcondition on the row sum, which is
/// exactly what the pre-#1044 GPU path satisfied while being wrong.
///
/// # Panics
/// - If `num_atoms < 2`. A categorical support needs at least two atoms for the
///   `N-1` spacing denominator to be meaningful.
/// - If the atom spacing `Δz = (v_max − v_min) / (N − 1)` is not finite and
///   strictly positive — i.e. the support is degenerate (`v_max == v_min`) or
///   inverted (`v_max < v_min`), or a bound is non-finite. This is *not*
///   implied by `num_atoms >= 2`: a well-sized `num_atoms = 51` support with
///   `v_min == v_max` gives `Δz = 0`, so `b = (Tz − v_min) / Δz` is `NaN` for
///   **every** element of **every** batch, for the whole run.
///
///   The assertion is a config-degeneracy backstop, not a memory-safety guard.
///   The operator is `NaN`-safe by construction — [`clamp_preserving_nan`]
///   keeps the `NaN` observable and the clamp on the derived `Int` indices
///   keeps the scatter in range on both backends — so a degenerate support
///   could no longer corrupt anything; it would merely emit an all-`NaN` target
///   on every step. That is a run which trains on nothing and reports it only
///   once, through the caller's one-shot `FiniteLossGuard` warning. `Δz` is a
///   *configuration* constant rather than per-batch data, so the right response
///   is to reject it at the call site with a message naming the offending
///   bounds, not to let it propagate.
///
/// [`clamp_preserving_nan`]: crate::algorithms::shared::clamp_preserving_nan
#[allow(clippy::too_many_arguments)]
// Structural size of the distributional support (atom / quantile count). The
// configs cap these in the low hundreds, so the value is exact in f32 and
// cannot wrap an `i64`; a non-finite or zero spacing is rejected by assertion
// before use.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
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
    // A zero or non-finite spacing makes `b` NaN below — for every element of
    // every batch, for the whole run, because `delta_z` is config and not data.
    // The operator below is NaN-safe (it neither hides the NaN nor derives an
    // out-of-range index from it), so this is not a safety guard: it is the
    // difference between naming the mis-specified support here and emitting an
    // all-NaN target forever while the loss guard warns once. Fail loudly.
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
    // `clamp_preserving_nan`, not `clamp`: on wgpu/Metal the plain clamp
    // rescues a NaN reward to `v_min`, producing a projected row that sums to
    // exactly 1.0 and asserts certainty of the worst return, with no NaN left
    // for any downstream guard to catch (issue #1044). ±inf is untouched by the
    // wrapper and still pins to v_min/v_max, which is the intended semantics.
    let tz = clamp_preserving_nan(tz, v_min, v_max);

    // Continuous atom coordinate b ∈ [0, N-1], and its floor/ceil as Int.
    //
    // `tz` is already clamped to [v_min, v_max], so `b ∈ [0, N-1]` holds
    // exactly in real arithmetic — Bellemare et al. 2017 Algorithm 1 states it
    // as an inline assertion. It does *not* hold in IEEE-754: for many valid
    // supports (e.g. v_min = -10, v_max = 0.1, N = 8) the f32 division rounds a
    // few ULPs above N-1, so `ceil` yields index N and the scatter below
    // indexes off the end of the support. Clamping `b` restores the invariant
    // and keeps `u_f - b == 0` exact at the boundary, so the `l == u` mask
    // below still carries the full mass. Same guard as CleanRL's `c51.py`.
    // No-op for any support that does not round out of range.
    //
    // Applying `clamp_preserving_nan` at BOTH clamps is deliberate. Applying it
    // at only one is *accidentally* sufficient on the two measured backends —
    // on Metal the `tz` clamp eats the NaN so this one never sees it; on Flex
    // this one propagates — but that argument turns on WHICH clamp eats the
    // NaN first, which is precisely the backend-specific fact this code exists
    // to stop depending on. Each site is locally correct instead.
    //
    // The `b` clamp is therefore NOT the out-of-range guard for the non-finite
    // path: it deliberately preserves NaN, so a NaN reaches `floor().int()`
    // below. That conversion is itself backend-dependent (the host saturates
    // NaN to 0; wgpu/Metal yields i32::MIN), and Burn's wgpu `scatter` is
    // launched unchecked with `bounds_checks: false` — an out-of-range index
    // does not panic there, it writes into whatever tensor shares the memory
    // pool at that offset. The clamp on the derived Int indices is what makes
    // that unreachable, and it is load-bearing: keeping the NaN-preserving
    // clamps without it would be strictly worse than the pre-#1044 code.
    // Measured, by deleting just the index clamp and running
    // `tests/c51_projection_backend_parity.rs` on an Apple M2 Pro / Metal 4:
    // the NaN row came back all ZEROS on wgpu (the NaN weights were written
    // somewhere outside the target tensor) and nothing panicked, while Flex
    // returned `[NaN, 0, …]`. Land the two changes together or neither.
    //
    // The index clamp is a no-op on the finite path. Once `b ∈ [0, N-1]`, both
    // `0` and `N-1` are exactly representable in f32 for N ≤ 2^24 (the configs
    // cap N in the low hundreds), so `floor`/`ceil` land inside [0, N-1] and
    // the clamp is the identity — in particular at `b == N-1`, where the
    // `u_f - b == 0` exactness the paragraph above protects must survive.
    let b = (tz.sub_scalar(v_min)).div_scalar(delta_z);
    let b = clamp_preserving_nan(b, 0.0, num_atoms as f32 - 1.0);
    let max_index = num_atoms as i64 - 1;
    let l_idx: Tensor<B, 2, Int> = b.clone().floor().int().clamp(0, max_index);
    let u_idx: Tensor<B, 2, Int> = b.clone().ceil().int().clamp(0, max_index);

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

    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn make_support(
        v_min: f32,
        v_max: f32,
        n: usize,
        device: <B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let delta = atom_spacing(v_min, v_max, n);
        let data: Vec<f32> = (0..n).map(|i| v_min + (i as f32) * delta).collect();
        Tensor::from_data(TensorData::new(data, vec![n]), &device)
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
        let support = make_support(-1.0, 1.0, 3, device);

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
        let support = make_support(-1.0, 1.0, 3, device);

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
        let support = make_support(-1.0, 1.0, 3, device);

        let out = project_distribution(next_probs, rewards, terminated, support, 1.0, -1.0, 1.0, 3);
        let v = into_vec(out);
        assert!(v[0].abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
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
        let support = make_support(-2.0, 2.0, n, device);

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
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
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
        let support = make_support(v_min, v_max, n, device);

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

    // -------- non-finite rewards (issue #1044) --------

    /// Projects one row per entry of `rewards` on the `(v_min, v_max, n)`
    /// support, with a uniform `next_probs` and `terminated = 0`.
    ///
    /// Returns the flattened `(rewards.len(), n)` output. `terminated = 0` is
    /// the live path: `keep = 1`, so the reward is added to a *finite* shifted
    /// support rather than to zero, which is where a real `NaN` reward enters.
    // Test fixture data: the element count is bounded by a small constant
    // declared at each call site, far below f32's 2^24 exact-integer limit, so
    // every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn project_rows_for_rewards(v_min: f32, v_max: f32, n: usize, rewards: &[f32]) -> Vec<f32> {
        let batch = rewards.len();
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32 / (n as f32); batch * n], vec![batch, n]),
            &device,
        );
        let rewards_t =
            Tensor::<B, 1>::from_data(TensorData::new(rewards.to_vec(), vec![batch]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32; batch], vec![batch]), &device);
        let support = make_support(v_min, v_max, n, device);

        let out = project_distribution(
            next_probs, rewards_t, terminated, support, 0.99, v_min, v_max, n,
        );
        into_vec(out)
    }

    #[test]
    fn projection_nan_reward_yields_a_non_finite_row() {
        // Regression, issue #1044. The postcondition is FINITENESS, never the
        // row sum: before the fix this row summed to exactly 1.0 on wgpu/Metal
        // — a well-formed probability vector claiming certainty of the worst
        // return — because Metal's `clamp` lowers to `fmin(fmax(..))` and
        // rescues NaN to `v_min`. A row-sum assertion would have passed on the
        // corrupted output and caught nothing.
        let v = project_rows_for_rewards(-10.0, 10.0, 8, &[f32::NAN]);

        assert!(
            v.iter().any(|p| !p.is_finite()),
            "a NaN reward must leave the projected row non-finite so the \
             caller's FiniteLossGuard can skip the step, got {v:?}"
        );
    }

    #[test]
    fn projection_nan_reward_does_not_corrupt_a_neighbouring_row() {
        // The NaN row must stay non-finite AND the healthy row in the same
        // batch must be untouched. This is the assertion that would fail if the
        // NaN-derived index escaped `[0, N-1]`: Burn's wgpu scatter carries no
        // bounds check at any layer, so an out-of-range index writes into
        // whatever tensor shares the memory pool at that offset rather than
        // panicking. On the host the same index would panic instead — either
        // way, an unclamped index is observable here.
        let n = 8;
        let v = project_rows_for_rewards(-10.0, 10.0, n, &[f32::NAN, 0.5]);
        let (nan_row, good_row) = v.split_at(n);

        assert!(
            nan_row.iter().any(|p| !p.is_finite()),
            "the NaN row must be non-finite, got {nan_row:?}"
        );
        assert!(
            good_row.iter().all(|p| p.is_finite()),
            "a finite-reward row must not be poisoned by a NaN row beside it, \
             got {good_row:?}"
        );
        let s: f32 = good_row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "the healthy row must still be a probability distribution, \
             got sum {s}: {good_row:?}"
        );
    }

    #[test]
    fn projection_finite_reward_row_is_unchanged_and_still_sums_to_one() {
        // The NaN-preserving clamps and the index clamp are no-ops on the
        // finite path; this pins that they did not perturb ordinary inputs.
        let n = 8;
        let v = project_rows_for_rewards(-10.0, 10.0, n, &[0.0, 0.5, -2.5, 3.75]);

        for (i, row) in v.chunks(n).enumerate() {
            assert!(
                row.iter().all(|p| p.is_finite()),
                "row {i} must be finite for a finite reward, got {row:?}"
            );
            let s: f32 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row {i} must still sum to 1, got {s}: {row:?}"
            );
        }
    }

    #[test]
    fn projection_positive_infinite_reward_saturates_the_top_atom() {
        // Pins the `is_nan`-not-`is_finite` decision in `clamp_preserving_nan`.
        // `clamp` handles ±inf correctly and IDENTICALLY on both backends, and
        // an infinite reward genuinely means "a return of at least v_max" — so
        // it must still clamp onto the top atom rather than being masked to
        // NaN. Masking on `is_finite` would regress this to an all-NaN row.
        let n = 8;
        let v = project_rows_for_rewards(-10.0, 10.0, n, &[f32::INFINITY]);

        assert!(
            v.iter().all(|p| p.is_finite()),
            "+inf must clamp to v_max, not become NaN, got {v:?}"
        );
        assert!(
            (v[n - 1] - 1.0).abs() < 1e-5,
            "+inf must put all mass on the top atom, got {}: {v:?}",
            v[n - 1]
        );
    }

    #[test]
    fn projection_negative_infinite_reward_saturates_the_bottom_atom() {
        let n = 8;
        let v = project_rows_for_rewards(-10.0, 10.0, n, &[f32::NEG_INFINITY]);

        assert!(
            v.iter().all(|p| p.is_finite()),
            "-inf must clamp to v_min, not become NaN, got {v:?}"
        );
        assert!(
            (v[0] - 1.0).abs() < 1e-5,
            "-inf must put all mass on the bottom atom, got {}: {v:?}",
            v[0]
        );
    }

    /// Drives one projection on an arbitrary (possibly degenerate) support.
    ///
    /// Deliberately does *not* pre-validate `v_min`/`v_max`: the point is to
    /// reach [`project_distribution`]'s own guard. Note `make_support` calls
    /// `atom_spacing` too, so for a degenerate support the support tensor
    /// itself is degenerate — the tests below assert on the panic *message* to
    /// confirm the guard fired rather than some incidental failure.
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn project_on_support(v_min: f32, v_max: f32, n: usize) -> Vec<f32> {
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let next_probs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32 / (n as f32); n], vec![1, n]),
            &device,
        );
        let rewards = Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let terminated =
            Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32], vec![1]), &device);
        let support = make_support(v_min, v_max, n, device);

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
