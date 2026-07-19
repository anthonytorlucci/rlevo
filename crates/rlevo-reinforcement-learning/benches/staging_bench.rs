//! A/B micro-bench of the minibatch observation-staging seam: the device
//! round trip (`obs.to_tensor(device)` then immediately `.into_data()`,
//! upload-then-download-unchanged) vs. `TensorConvertible::write_host_row`
//! (pure host staging, one batched upload). Answers #187 / #362's open
//! measurement question (issue #365 step 1): both strategies are implemented
//! **inline, in this file**, so a single `cargo bench` run yields the delta.
//!
//! # A note on this checkout's actual state
//!
//! Commit `797d18c` ("Stage minibatch observations host-side") is *not* an
//! ancestor of this worktree's HEAD — `git merge-base --is-ancestor 797d18c
//! HEAD` returns false, and `797d18c`'s parent is this same HEAD commit. It
//! sits on a sibling branch reachable in the shared object store, not in this
//! branch's history. Concretely: `dqn_agent.rs::learn_step` in *this*
//! checkout still round-trips (`t.obs.to_tensor(&self.device)` then
//! `.into_data().convert::<f32>()`, lines ~417-422) — the fix has not been
//! wired into any agent here. What *does* already exist here is the
//! `TensorConvertible::write_host_row` primitive itself (`rlevo-core`), which
//! is enough to build this A/B independently of whether any agent calls it
//! yet. Treat this bench as answering "is the not-yet-applied fix worth
//! applying", not "did the applied fix help".
//!
//! # Design
//!
//! - `stage_roundtrip` — the round-trip path, reproduced verbatim from the
//!   minibatch loop live in this checkout's `dqn_agent.rs::learn_step`: per
//!   row, `to_tensor(device)`, then `.into_data().convert::<f32>()`, then
//!   `extend_from_slice` into the flat host buffer; one batched
//!   `Tensor::from_data` at the end.
//! - `stage_host_row` — the candidate replacement: per row,
//!   `TensorConvertible::write_host_row(&mut flat)` (no device touched), then
//!   the same batched `Tensor::from_data`.
//!
//! Both are generic over the row type `T: TensorConvertible<R, B>` and the
//! backend `B`, so one implementation covers `CartPoleObservation` (`R = 1`,
//! 4 floats/row) and `PixelObservation` (`R = 3`, 1200 floats/row) and both
//! `Flex` and `Wgpu`, per the #365 sweep requirement (sync count is invariant
//! to row size; bytes moved are not).
//!
//! # Methodology — forcing completion inside the timed region
//!
//! wgpu is asynchronous: `Tensor::from_data` and friends *submit* work and
//! return immediately. A timer that stops there measures submission, not
//! execution — and would make `stage_roundtrip` look artificially fast,
//! because its per-row `into_data()` calls are precisely what forces
//! synchronization (confirmed by reading `Tensor::into_data` →
//! `try_into_data` → `crate::try_read_sync(self.into_data_async())`, which
//! blocks the calling thread until the device queue drains up to that
//! tensor). `stage_host_row` has no such call anywhere in its loop.
//!
//! So **both** timed closures below end with an explicit
//! `black_box(tensor.into_data())` on the *final batched* tensor — applied
//! identically to both strategies — guaranteeing every timed iteration
//! includes full device completion, not just command submission. This is the
//! only correctness-relevant addition to the "obvious" A/B: without it the
//! result direction inverts.
//!
//! # Correctness gate
//!
//! [`assert_bit_identical`] runs once per (observation, backend) group,
//! outside the timed region, before any `bench_with_input` call. If the two
//! strategies ever disagree, the timing comparison is meaningless.
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo-reinforcement-learning --bench staging_bench
//! ```
//!
//! # Reporting
//!
//! Results are backend- and hardware-scoped by construction: every criterion
//! group name carries the backend label (`flex` / `wgpu`), and `wgpu` on this
//! development machine is Metal on an Apple M2 Pro (verified independently —
//! see the step-0 smoke test in the #365 session notes). A `wgpu` number here
//! does not transfer to CUDA or any other machine.

#[path = "support/bench_backend.rs"]
mod bench_backend;

use std::hint::black_box;

use burn::backend::{Flex, Wgpu};
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::base::TensorConvertible;
use rlevo_core::state::Observable;
use rlevo_environments::classic::cartpole::CartPoleObservation;
use rlevo_environments::pixel_grid::{CELL_COUNT, PixelGridState, PixelObservation};

use bench_backend::BenchBackend;

/// Device type alias so generic signatures below don't repeat the projection.
type DeviceOf<B> = <B as BackendTypes>::Device;

/// Batch sizes swept per (observation, backend) combination — at least
/// 32/64/256 per the #365 request; 64 matches the existing `dqn_bench.rs`
/// `dqn_learn_step_batch64`.
const BATCH_SIZES: [usize; 3] = [32, 64, 256];

// ---------------------------------------------------------------------------
// The two staging strategies under test
// ---------------------------------------------------------------------------

/// The round-trip path: per row, upload then immediately read back, purely
/// to obtain a host `f32` slice to append. Reproduced verbatim from the
/// minibatch loop currently live in this checkout's
/// `dqn_agent.rs::learn_step` (`obs_tensor.into_data().convert::<f32>()` +
/// `extend_from_slice(..as_slice::<f32>()..)`).
fn stage_roundtrip<const R: usize, const BR: usize, T, B>(
    items: &[T],
    device: &DeviceOf<B>,
) -> Tensor<B, BR>
where
    T: TensorConvertible<R, B>,
    B: Backend,
{
    debug_assert_eq!(BR, R + 1, "batched rank BR must equal row rank R + 1");
    let row_shape = T::row_shape();
    let row_len: usize = row_shape.iter().product();
    let mut flat: Vec<f32> = Vec::with_capacity(items.len() * row_len);
    for item in items {
        let row_tensor: Tensor<B, R> = item.to_tensor(device);
        let row_data = row_tensor.into_data().convert::<f32>();
        flat.extend_from_slice(row_data.as_slice::<f32>().expect("float data"));
    }
    let mut shape = [0usize; BR];
    shape[0] = items.len();
    shape[1..].copy_from_slice(&row_shape);
    Tensor::from_data(TensorData::new(flat, shape), device)
}

/// The candidate replacement: write straight into the host buffer, one
/// batched upload, zero per-row device round trips. Uses the
/// `TensorConvertible::write_host_row` primitive that already exists in
/// `rlevo-core` but is not yet wired into any agent's minibatch loop in this
/// checkout (see the module-level note on `797d18c`).
fn stage_host_row<const R: usize, const BR: usize, T, B>(
    items: &[T],
    device: &DeviceOf<B>,
) -> Tensor<B, BR>
where
    T: TensorConvertible<R, B>,
    B: Backend,
{
    debug_assert_eq!(BR, R + 1, "batched rank BR must equal row rank R + 1");
    let row_shape = T::row_shape();
    let row_len: usize = row_shape.iter().product();
    let mut flat: Vec<f32> = Vec::with_capacity(items.len() * row_len);
    for item in items {
        item.write_host_row(&mut flat);
    }
    let mut shape = [0usize; BR];
    shape[0] = items.len();
    shape[1..].copy_from_slice(&row_shape);
    Tensor::from_data(TensorData::new(flat, shape), device)
}

/// Runs both strategies on the same batch and asserts the resulting tensors
/// are bit-identical. Call **once**, outside any timed region.
fn assert_bit_identical<const R: usize, const BR: usize, T, B>(items: &[T], device: &DeviceOf<B>)
where
    T: TensorConvertible<R, B>,
    B: Backend,
{
    let rt: Tensor<B, BR> = stage_roundtrip::<R, BR, T, B>(items, device);
    let hr: Tensor<B, BR> = stage_host_row::<R, BR, T, B>(items, device);
    let rt_v: Vec<f32> = rt.into_data().into_vec().expect("host read (roundtrip)");
    let hr_v: Vec<f32> = hr.into_data().into_vec().expect("host read (host_row)");
    assert_eq!(
        rt_v, hr_v,
        "stage_roundtrip and stage_host_row diverged -- staging is not bit-identical; \
         the timing comparison below would be meaningless"
    );
}

// ---------------------------------------------------------------------------
// Synthetic batch data
// ---------------------------------------------------------------------------

fn cartpole_batch(n: usize, rng: &mut StdRng) -> Vec<CartPoleObservation> {
    (0..n)
        .map(|_| CartPoleObservation {
            cart_pos: rng.random_range(-2.4_f32..2.4_f32),
            cart_vel: rng.random_range(-3.0_f32..3.0_f32),
            pole_angle: rng.random_range(-0.2_f32..0.2_f32),
            pole_ang_vel: rng.random_range(-3.0_f32..3.0_f32),
        })
        .collect()
}

fn pixel_batch(n: usize, rng: &mut StdRng) -> Vec<PixelObservation> {
    (0..n)
        .map(|_| {
            let agent = rng.random_range(0..CELL_COUNT as u32);
            let goal = rng.random_range(0..CELL_COUNT as u32);
            PixelGridState::new(agent, goal).project()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bench driver
// ---------------------------------------------------------------------------

/// Runs the `roundtrip` vs. `host_row` A/B for one (observation type,
/// backend) combination across [`BATCH_SIZES`], inside criterion group
/// `group_name`.
fn bench_stage<const R: usize, const BR: usize, T, B>(
    c: &mut Criterion,
    group_name: &str,
    make_batch: impl Fn(usize, &mut StdRng) -> Vec<T>,
) where
    T: TensorConvertible<R, B>,
    B: BenchBackend,
{
    let device = B::device();
    let mut rng = StdRng::seed_from_u64(365);

    // Correctness gate: once, outside the timed region (see module doc).
    let check_batch = make_batch(*BATCH_SIZES.last().expect("non-empty"), &mut rng);
    assert_bit_identical::<R, BR, T, B>(&check_batch, &device);

    let mut group = c.benchmark_group(group_name);
    // 20 samples (vs. criterion's default 100): `stage_roundtrip` on wgpu at
    // batch 256 costs ~1.5 ms/row * 256 rows ~= 384 ms/iteration purely from
    // per-row synchronization, so 100 iterations is ~38 s for that one cell
    // alone. Cut 5x here to keep the full 24-benchmark sweep tractable while
    // still reporting a real confidence interval (criterion's minimum is 10).
    group.sample_size(20);
    for &batch_size in &BATCH_SIZES {
        let items = make_batch(batch_size, &mut rng);
        group.bench_with_input(
            BenchmarkId::new("roundtrip", batch_size),
            &items,
            |b, items| {
                b.iter(|| {
                    let t = stage_roundtrip::<R, BR, T, B>(black_box(items), &device);
                    // Force completion inside the timed region -- see module doc.
                    black_box(t.into_data());
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("host_row", batch_size),
            &items,
            |b, items| {
                b.iter(|| {
                    let t = stage_host_row::<R, BR, T, B>(black_box(items), &device);
                    black_box(t.into_data());
                });
            },
        );
    }
    group.finish();
}

fn bench_staging(c: &mut Criterion) {
    bench_stage::<1, 2, CartPoleObservation, Flex>(c, "stage_cartpole_flex", cartpole_batch);
    bench_stage::<1, 2, CartPoleObservation, Wgpu>(c, "stage_cartpole_wgpu", cartpole_batch);
    bench_stage::<3, 4, PixelObservation, Flex>(c, "stage_pixel_flex", pixel_batch);
    bench_stage::<3, 4, PixelObservation, Wgpu>(c, "stage_pixel_wgpu", pixel_batch);
}

criterion_group!(benches, bench_staging);
criterion_main!(benches);
