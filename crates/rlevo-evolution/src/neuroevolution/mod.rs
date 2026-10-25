//! Topology-evolving neuroevolution (NEAT) — the host-side graph data model.
//!
//! This module holds the building blocks shared by NEAT: the graph genome
//! ([`topology`]), the per-run innovation registry ([`innovation`]), speciation
//! ([`species`]), and the phenotype builders ([`phenotype`]). The
//! [`NeatStrategy`](crate::algorithms::neuroevolution::neat::NeatStrategy)
//! that drives them lives in [`crate::algorithms::neuroevolution::neat`].
//!
//! These types are *graphs*, not tensors: unlike the
//! [`Strategy`](crate::strategy::Strategy) genomes elsewhere in the crate,
//! [`TopologyGenome`] is plain host-side data (and is [`Clone`]). A genome is
//! scored either one at a time, through the interpreted [`InterpretedPhenotype`]
//! reference path, or a whole population at once, through the device-batched
//! [`DensePaddedEvaluator`]; the two agree to float epsilon.
//!
//! # Orientation
//!
//! NEAT is **maximization** (higher fitness is better) — matching the crate-wide
//! maximise convention. [`Species`] best/stagnation tracking and fitness
//! sharing treat higher as better; a cost objective is reconciled into
//! canonical space by the loop/adapter chokepoint, not by hand here, and
//! [`speciate`] shifts a generation with negative fitness before sharing.

pub mod innovation;
pub mod phenotype;
pub mod species;
pub mod topology;

pub use innovation::{InnovationRegistry, NodeSplit};
pub use phenotype::{DensePaddedEvaluator, InterpretedPhenotype};
pub use species::{
    Species, SpeciesId, allocate_offspring, compatibility_distance, remove_stagnant, speciate,
};
pub use topology::{
    ActivationFn, ConnectionGene, InnovationId, NodeGene, NodeId, NodeKind, TopologyGenome,
};
