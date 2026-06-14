//! Topology-evolving neuroevolution (NEAT) — the host-side graph data model.
//!
//! This module holds the building blocks shared by the interpreted NEAT path
//! (issue #34): the graph genome ([`topology`]), the per-run innovation registry
//! ([`innovation`]), speciation ([`species`]), and the interpreted phenotype
//! ([`phenotype`]). The [`NeatStrategy`] custom harness that drives them lives in
//! [`crate::algorithms::neuroevolution::neat`].
//!
//! These types are *graphs*, not tensors: unlike the
//! [`Strategy`](crate::strategy::Strategy) genomes elsewhere in the crate,
//! [`TopologyGenome`] is plain host-side data (and is [`Clone`]). The tensorized
//! / GPU-batched evaluation path is out of scope and tracked in #41.
//!
//! # Orientation
//!
//! NEAT is **maximization** (higher fitness is better) — the opposite of the
//! crate-wide minimize convention. [`Species`] best/stagnation tracking, fitness
//! sharing, and the [`GraphFitnessFn`](crate::algorithms::neuroevolution::neat::GraphFitnessFn)
//! seam all treat higher as better; a cost objective supplies `−cost`.

pub mod innovation;
pub mod phenotype;
pub mod species;
pub mod topology;

pub use innovation::{InnovationRegistry, NodeSplit};
pub use phenotype::{InterpretedBuilder, InterpretedPhenotype, Phenotype, PhenotypeBuilder};
pub use species::{Species, SpeciesId, allocate_offspring, compatibility_distance, remove_stagnant, speciate};
pub use topology::{
    ActivationFn, ConnectionGene, InnovationId, NodeGene, NodeId, NodeKind, TopologyGenome,
};
