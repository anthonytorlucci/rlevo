//! Cross-backend parity for the C51 categorical projection's non-finite path:
//! on Metal, a `NaN` reward used to produce a well-formed, wrongly-confident
//! probability row instead of the loud `NaN` the host backend produced on the
//! same input.
//!
//! # This test does NOT run in CI
//!
//! Every workflow in `.github/workflows/` runs on `ubuntu-latest` with no GPU,
//! and `cubecl-wgpu` **aborts** (not a catchable error) on a host with no
//! adapter: it panics on a worker thread and the calling thread then panics on
//! the severed channel. There is no clean in-process probe across the cubecl
//! boundary, so the whole test is `#[ignore]`d and is only reached when a human
//! asks for it on a GPU host. Regression protection for this class of defect is
//! therefore a **manual GPU run**, not automation. Nothing here is covered by
//! the pipeline; treat the in-source Flex unit tests in `projection.rs` as the
//! automated half and this as the half that only fires when someone remembers.
//!
//! # What it pins
//!
//! `Tensor::clamp`'s `NaN` behaviour is backend-divergent and undocumented:
//!
//! - `burn-flex` lowers it to `f32::clamp`, which **propagates** `NaN`.
//! - `burn-cubecl`/wgpu lowers it to the WGSL `clamp` builtin, which naga emits
//!   as Metal's `fmin(fmax(…))` — a composition that **rescues** `NaN` to the
//!   lower bound.
//!
//! Measured on an Apple M2 Pro (macOS 26.5.2, Metal 4; burn 0.21, cubecl 0.10,
//! wgpu 29): `clamp(NaN, -10, 10)` is `NaN` on Flex and `-10.0` on Metal.
//! Before the fix that made `project_distribution` return `[NaN, 0, …]` on Flex
//! — loud, and caught by the caller's `FiniteLossGuard` — but `[1.0, 0, …]` on
//! Metal, a well-formed probability vector **summing to exactly 1.0** that
//! claims certainty of the worst return and that no finiteness guard can see.
//!
//! Hence the assertions below are on the **finiteness pattern**, never on the
//! row sum: a row-sum postcondition is exactly what the corrupted Metal output
//! satisfied.
//!
//! Note also that the second half of the fix — clamping the derived `Int` atom
//! indices — is only observable here. Preserving the `NaN` through the clamps
//! means `floor(b).int()` sees one, and that conversion yields `0` on the host
//! but `i32::MIN` on Metal; Burn's wgpu `scatter` carries no bounds check at
//! any layer (`launch_unchecked`, `bounds_checks: false`), so an out-of-range
//! index writes into an unrelated tensor's memory rather than panicking. The
//! host path cannot exhibit that at all.
//!
//! # Single test, serial execution
//!
//! One `#[test]` function per file, mirroring
//! `crates/rlevo-evolution/tests/backend_parity.rs`: both backends carry
//! process-global device/RNG state, so nothing else in this binary should run
//! concurrently with the two runs being compared.

use burn::backend::{Flex, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use rlevo_reinforcement_learning::algorithms::c51::projection::{
    atom_spacing, project_distribution,
};

/// Support bounds and atom count every case below shares.
const V_MIN: f32 = -10.0;
const V_MAX: f32 = 10.0;
const N_ATOMS: usize = 8;

/// Projects one row per entry of `rewards` on the `(V_MIN, V_MAX, N_ATOMS)`
/// support with a uniform `next_probs` and `terminated = 0`, and reads the
/// `(rewards.len(), N_ATOMS)` result back to the host.
///
/// `terminated = 0` is the live path: `keep = 1`, so the reward is added to a
/// finite shifted support rather than to zero — where a real `NaN` reward
/// enters the backup.
// Fixture sizes are small compile-time constants, far below f32's 2^24
// exact-integer limit, so every generated value is represented exactly.
#[allow(clippy::cast_precision_loss)]
fn project_rows<B: Backend>(rewards: &[f32], device: &B::Device) -> Vec<f32> {
    let batch = rewards.len();
    let delta = atom_spacing(V_MIN, V_MAX, N_ATOMS);
    let support_data: Vec<f32> = (0..N_ATOMS).map(|i| V_MIN + (i as f32) * delta).collect();

    let next_probs = Tensor::<B, 2>::from_data(
        TensorData::new(
            vec![1.0_f32 / (N_ATOMS as f32); batch * N_ATOMS],
            vec![batch, N_ATOMS],
        ),
        device,
    );
    let rewards_t =
        Tensor::<B, 1>::from_data(TensorData::new(rewards.to_vec(), vec![batch]), device);
    let terminated =
        Tensor::<B, 1>::from_data(TensorData::new(vec![0.0_f32; batch], vec![batch]), device);
    let support = Tensor::<B, 1>::from_data(TensorData::new(support_data, vec![N_ATOMS]), device);

    project_distribution(
        next_probs, rewards_t, terminated, support, 0.99, V_MIN, V_MAX, N_ATOMS,
    )
    .into_data()
    .convert::<f32>()
    .into_vec::<f32>()
    .expect("f32 host read of the projected distribution")
}

#[test]
#[ignore = "requires a wgpu/Metal or wgpu/Vulkan adapter; every CI workflow runs on ubuntu-latest \
            with no GPU and cubecl-wgpu aborts on device init — run on a GPU host with \
            `cargo xtask test-gpu c51_projection_backend_parity`"]
fn wgpu_matches_flex_on_non_finite_rewards() {
    // Row 0: NaN — must be non-finite on BOTH backends after the fix.
    // Row 1: +inf — must clamp to v_max (top atom) on both, WITHOUT becoming
    //        NaN; this is the `is_nan`-not-`is_finite` decision.
    // Row 2: -inf — must clamp to v_min (bottom atom) on both.
    // Row 3: a finite reward beside them — must be untouched, which is also the
    //        witness that no NaN-derived index escaped `[0, N-1]` and scribbled
    //        over neighbouring memory in the unchecked wgpu scatter.
    let rewards = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];

    // Flex first: it has no device init that can abort, so a failure here is
    // unambiguously about the operator rather than the adapter.
    let flex_device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
    let flex = project_rows::<Flex>(&rewards, &flex_device);

    let wgpu_device: burn::backend::wgpu::WgpuDevice = Default::default();
    let wgpu = project_rows::<Wgpu>(&rewards, &wgpu_device);

    assert_eq!(
        flex.len(),
        wgpu.len(),
        "both backends must project to the same shape"
    );

    // 1. The finiteness pattern must agree element for element. This is the
    //    whole point: pre-fix, Flex row 0 was all-NaN and Metal row 0 was a
    //    clean one-hot.
    for (i, (&f, &w)) in flex.iter().zip(&wgpu).enumerate() {
        assert_eq!(
            f.is_finite(),
            w.is_finite(),
            "element {i} must have the same finiteness on both backends: \
             flex={f}, wgpu={w} (flex={flex:?}, wgpu={wgpu:?})"
        );
    }

    // 2. NaN reward → non-finite row, on both.
    for (name, row) in [("flex", &flex[..N_ATOMS]), ("wgpu", &wgpu[..N_ATOMS])] {
        assert!(
            row.iter().any(|p| !p.is_finite()),
            "{name}: a NaN reward must leave the projected row non-finite so the \
             caller's FiniteLossGuard can skip the step, got {row:?}"
        );
    }

    // 3. ±inf still saturate the support and stay finite, on both.
    for (name, all) in [("flex", &flex), ("wgpu", &wgpu)] {
        let pos_inf_row = &all[N_ATOMS..2 * N_ATOMS];
        assert!(
            pos_inf_row.iter().all(|p| p.is_finite()),
            "{name}: +inf must clamp to v_max, not become NaN, got {pos_inf_row:?}"
        );
        assert!(
            (pos_inf_row[N_ATOMS - 1] - 1.0).abs() < 1e-5,
            "{name}: +inf must put all mass on the top atom, got {pos_inf_row:?}"
        );

        let neg_inf_row = &all[2 * N_ATOMS..3 * N_ATOMS];
        assert!(
            neg_inf_row.iter().all(|p| p.is_finite()),
            "{name}: -inf must clamp to v_min, not become NaN, got {neg_inf_row:?}"
        );
        assert!(
            (neg_inf_row[0] - 1.0).abs() < 1e-5,
            "{name}: -inf must put all mass on the bottom atom, got {neg_inf_row:?}"
        );

        // 4. The healthy row beside the poisoned ones is untouched and still a
        //    probability distribution.
        let finite_row = &all[3 * N_ATOMS..4 * N_ATOMS];
        assert!(
            finite_row.iter().all(|p| p.is_finite()),
            "{name}: a finite-reward row must not be poisoned by its neighbours, \
             got {finite_row:?}"
        );
        let sum: f32 = finite_row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "{name}: the healthy row must still sum to 1, got {sum}: {finite_row:?}"
        );
    }
}
