//! Neuroevolution strategies — evolving the parameters of a Burn `Module`.
//!
//! Phase 3d1 ships **weight-only** neuroevolution: a fixed-topology network is
//! declared once and a phase-1 [`Strategy`](crate::strategy::Strategy)
//! (GA / ES / DE / …) evolves its flattened weights. The
//! [`WeightOnly`](weight_only::WeightOnly) wrapper composes any such strategy
//! with any [`Module`](burn::module::Module); reshaping between the flat
//! genome and the network happens at the fitness boundary
//! ([`ModuleEvalFn`](crate::module_eval_fn::ModuleEvalFn)), never inside the
//! strategy.
//!
//! Topology evolution (NEAT) and indirect encodings are future phases (3d2);
//! a batched/vectorized forward pass is deferred to issue #41.

pub mod weight_only;

pub use weight_only::WeightOnly;
