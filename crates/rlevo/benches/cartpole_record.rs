//! Throughput benchmark for the recording tier: a uniformly-random
//! [`CartPole`] rollout driven through a [`RecordingTap`] into an
//! in-memory sink.
//!
//! This is the perf vehicle the (now-deleted) `record_cartpole` example
//! used to be — a random policy never *learns*, so it makes a poor
//! example but a clean, deterministic benchmark of the per-step record
//! encoding path. PPO's learning curve lives in the
//! `report_ppo_cartpole_with_client` example instead.
//!
//! [`InMemoryRecordSink`] keeps the measurement off the filesystem so the
//! numbers reflect frame-encoding cost, not disk I/O jitter.
//!
//! # Run with
//!
//! ```bash
//! cargo bench -p rlevo --features viz-record --bench cartpole_record
//! ```

use std::hint::black_box;
use std::sync::Arc;

use parking_lot::Mutex;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_benchmarks::record::{InMemoryRecordSink, RecordSink, RecordingTap};

use rlevo_core::action::DiscreteAction;
use rlevo_core::environment::{Environment, Snapshot};

use rlevo_environments::classic::{CartPole, CartPoleAction, CartPoleConfig};

const SEED: u64 = 2026;

/// Drives a fresh random `CartPole` rollout of `steps` steps through a
/// `RecordingTap` backed by an in-memory sink, resetting on episode end.
fn record_random_rollout(steps: usize) {
    let sink: Arc<Mutex<dyn RecordSink>> = Arc::new(Mutex::new(InMemoryRecordSink::new()));
    let env = CartPole::with_config(CartPoleConfig {
        seed: SEED,
        ..CartPoleConfig::default()
    });
    let mut tap: RecordingTap<CartPole, 1, 1, 1> = RecordingTap::new(env, sink);
    let mut rng = StdRng::seed_from_u64(SEED);

    tap.reset().expect("reset");
    for _ in 0..steps {
        let idx = rng.random_range(0..CartPoleAction::ACTION_COUNT);
        let snapshot = tap.step(CartPoleAction::from_index(idx)).expect("step");
        if snapshot.is_done() {
            tap.reset().expect("reset");
        }
    }
}

fn bench_record_rollout(c: &mut Criterion) {
    let mut group = c.benchmark_group("cartpole_record_rollout");
    for &steps in &[1_000_usize, 4_000, 16_000] {
        group.throughput(criterion::Throughput::Elements(steps as u64));
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &steps| {
            b.iter(|| record_random_rollout(black_box(steps)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_record_rollout);
criterion_main!(benches);
