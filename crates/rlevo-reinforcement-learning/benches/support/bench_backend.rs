//! Backend-parametric scaffolding shared by benches that need to sweep the
//! same workload across more than one Burn backend.
//!
//! Every `[[bench]]` is its own compilation unit, so this module is pulled in
//! per-file via `#[path = "support/bench_backend.rs"] mod bench_backend;`; it
//! sits in a subdirectory so Cargo's bench autodiscovery does not treat it as
//! a standalone target (mirrors `crates/rlevo/benches/support/`).
//!
//! [`BenchBackend`] exists so a bench body is written once, generic over
//! `B: BenchBackend`, and instantiated per backend at the call site — adding a
//! backend later (CUDA, ROCm, ...) is one new `impl` block here, not a
//! rewrite of every bench that sweeps backends. See `staging_bench.rs` for
//! the reference usage.

#![allow(dead_code)]

use burn::backend::{Flex, Wgpu};
use burn::tensor::backend::{Backend, BackendTypes};

/// A `Backend` benches can select by a stable, printable label.
///
/// This is deliberately *not* a blanket impl over `Backend` — only backends a
/// bench author has actually validated on the target hardware (see the wgpu
/// smoke test in the #365 step-1 session) get an `impl` here, so a typo'd or
/// unavailable backend fails at the `impl` site, not at bench-report time.
pub trait BenchBackend: Backend {
    /// Short, lowercase, filesystem/CLI-safe label used in criterion group
    /// names and in this bench's own report table (e.g. `"flex"`, `"wgpu"`).
    /// Report text must pair this label with the concrete hardware/adapter it
    /// ran on — the label alone does not carry that (a `"wgpu"` number on
    /// this machine is Metal on an Apple M2 Pro; it says nothing about CUDA).
    const NAME: &'static str;

    /// Constructs the device this backend benches against.
    ///
    /// `Default` is sufficient for every backend implemented today: `Flex`
    /// has exactly one (CPU) device, and wgpu's `WgpuDevice::DefaultDevice`
    /// selects the highest-priority GPU adapter present. On this machine
    /// (Apple M2 Pro) that resolves to Metal — confirmed independently by
    /// forcing `cubecl::wgpu::init_setup::<Metal>` in the step-0 smoke test,
    /// which reported `backend: Metal` in the resulting `WgpuSetup`.
    fn device() -> <Self as BackendTypes>::Device {
        Default::default()
    }
}

impl BenchBackend for Flex {
    const NAME: &'static str = "flex";
}

impl BenchBackend for Wgpu {
    const NAME: &'static str = "wgpu";
}

// Adding CUDA later is exactly this shape, gated behind the `cuda` feature:
//
// #[cfg(feature = "cuda")]
// impl BenchBackend for burn::backend::Cuda {
//     const NAME: &'static str = "cuda";
// }
//
// No existing bench body changes — callers add one more monomorphized call
// (`bench_stage::<_, _, _, Cuda>(...)`) alongside the Flex/Wgpu ones.
