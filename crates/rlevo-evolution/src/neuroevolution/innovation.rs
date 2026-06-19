//! Innovation-number bookkeeping — the per-run registry that assigns the
//! historical markers NEAT crossover aligns on.
//!
//! A single [`InnovationRegistry`] is created per run and shared across the NEAT
//! harness via `Arc`. It is the **only** allocator of [`InnovationId`]s and
//! hidden [`NodeId`]s, so two identical structural mutations occurring in
//! different genomes *within the same run* receive the same ids — without which
//! innovation-aligned crossover silently misclassifies genes.
//!
//! # Determinism
//!
//! Mutation is applied sequentially host-side inside `tell` under the seeded
//! `seed_stream` RNG, so the `Mutex` never contends and id-issue order is
//! seed-fixed. The caches additionally make the *result* of a repeated
//! structural mutation order-independent: only first-issue assignment depends on
//! order, and that order is seeded.
//!
//! Independent runs (parallel trials, different seeds) each create their own
//! registry, so their innovation spaces are isolated — which is correct, because
//! cross-run gene alignment is meaningless.

use std::collections::HashMap;

use parking_lot::Mutex;

use super::topology::{InnovationId, NodeId};

/// The result of an add-node mutation that splits a connection.
///
/// Stable for a given split innovation within a run: the same split always
/// yields the same new node and the same two connection innovations, so the
/// inserted node aligns across genomes during crossover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeSplit {
    /// Id of the hidden node inserted into the split connection.
    pub new_node: NodeId,
    /// Innovation of the `source -> new_node` edge (canonical weight `1.0`).
    pub in_innov: InnovationId,
    /// Innovation of the `new_node -> target` edge (inherits the old weight).
    pub out_innov: InnovationId,
}

/// Per-run innovation bookkeeping, shared via `Arc` across the NEAT harness.
///
/// Thread-safe via interior mutability behind a [`parking_lot::Mutex`]
/// (ADR-0010). See the [module docs](self) for the determinism argument.
#[derive(Debug)]
pub struct InnovationRegistry {
    inner: Mutex<RegistryInner>,
}

#[derive(Debug)]
struct RegistryInner {
    next_innovation: InnovationId,
    next_node: NodeId,
    /// `(source, target) -> innovation`, so the same add-connection re-uses its
    /// id across genomes and generations.
    conn_cache: HashMap<(NodeId, NodeId), InnovationId>,
    /// split-connection innovation `-> NodeSplit`, so the same split is stable
    /// across the run even after the surrounding topology changes.
    node_cache: HashMap<InnovationId, NodeSplit>,
}

impl InnovationRegistry {
    /// Create a registry whose counters start *after* the minimal seed topology.
    ///
    /// `initial_node_count` is the number of input + output nodes (their ids are
    /// `0..initial_node_count`), and `initial_innovation_count` is the number of
    /// fully-connected seed connection genes (their innovations are
    /// `0..initial_innovation_count`). The first hidden node and the first new
    /// connection are therefore allocated *after* the seed, matching
    /// [`TopologyGenome::minimal`](super::topology::TopologyGenome::minimal).
    #[must_use]
    pub fn new(initial_node_count: usize, initial_innovation_count: usize) -> Self {
        Self {
            inner: Mutex::new(RegistryInner {
                next_innovation: initial_innovation_count as InnovationId,
                next_node: initial_node_count as NodeId,
                conn_cache: HashMap::new(),
                node_cache: HashMap::new(),
            }),
        }
    }

    /// Innovation id for an add-connection between `source` and `target`,
    /// allocating a fresh one on first sight and re-using it thereafter.
    ///
    /// The returned id must be wired into the new [`ConnectionGene`]; discarding
    /// it leaks an allocation, hence `#[must_use]`.
    ///
    /// [`ConnectionGene`]: super::topology::ConnectionGene
    #[must_use]
    pub fn register_connection(&self, source: NodeId, target: NodeId) -> InnovationId {
        let mut inner = self.inner.lock();
        if let Some(&id) = inner.conn_cache.get(&(source, target)) {
            return id;
        }
        let id = inner.next_innovation;
        inner.next_innovation += 1;
        inner.conn_cache.insert((source, target), id);
        id
    }

    /// Node + two innovations for splitting connection `split`, allocated once
    /// and cached so the same split is stable across the run.
    ///
    /// Keying on the **split innovation** (not on the `(source, target)` pair)
    /// makes a node inserted into "the same place" align even after the
    /// surrounding topology diverges.
    ///
    /// The returned [`NodeSplit`] carries the node and two innovation ids the
    /// caller must build the split connections from, hence `#[must_use]`.
    #[must_use]
    pub fn register_node_split(&self, split: InnovationId) -> NodeSplit {
        let mut inner = self.inner.lock();
        if let Some(&existing) = inner.node_cache.get(&split) {
            return existing;
        }
        let new_node = inner.next_node;
        inner.next_node += 1;
        let in_innov = inner.next_innovation;
        let out_innov = inner.next_innovation + 1;
        inner.next_innovation += 2;
        let result = NodeSplit {
            new_node,
            in_innov,
            out_innov,
        };
        inner.node_cache.insert(split, result);
        result
    }

    /// Next innovation id that would be allocated (the count of innovations
    /// issued so far). Used for checkpointing surfaces and invariant assertions.
    #[must_use]
    pub fn next_innovation(&self) -> InnovationId {
        self.inner.lock().next_innovation
    }

    /// Next node id that would be allocated (the count of nodes issued so far,
    /// including the seed inputs/outputs).
    #[must_use]
    pub fn next_node_id(&self) -> NodeId {
        self.inner.lock().next_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_starts_after_seed() {
        let registry = InnovationRegistry::new(3, 2);
        assert_eq!(registry.next_node_id(), 3, "node ids start after I+O seed nodes");
        assert_eq!(
            registry.next_innovation(),
            2,
            "innovations start after the I*O seed connections"
        );
    }

    #[test]
    fn test_register_connection_caches_id() {
        let registry = InnovationRegistry::new(3, 2);
        let a = registry.register_connection(0, 2);
        let b = registry.register_connection(1, 2);
        // Distinct pairs get distinct, monotone ids.
        assert_eq!(a, 2);
        assert_eq!(b, 3);
        // The same pair re-uses the cached id (crossover alignment).
        assert_eq!(registry.register_connection(0, 2), a);
        assert_eq!(registry.next_innovation(), 4);
    }

    #[test]
    fn test_register_node_split_caches_and_allocates_one_node_two_innovs() {
        let registry = InnovationRegistry::new(3, 2);
        let s = registry.register_node_split(0);
        assert_eq!(s.new_node, 3, "first hidden node id is after the seed nodes");
        assert_eq!(s.in_innov, 2);
        assert_eq!(s.out_innov, 3);
        assert_eq!(registry.next_node_id(), 4);
        assert_eq!(registry.next_innovation(), 4);
        // Splitting the SAME connection again returns the cached split.
        assert_eq!(registry.register_node_split(0), s);
        // Splitting a DIFFERENT connection allocates a fresh node + 2 innovs.
        let t = registry.register_node_split(1);
        assert_eq!(t.new_node, 4);
        assert_eq!(t.in_innov, 4);
        assert_eq!(t.out_innov, 5);
    }

    #[test]
    fn test_independent_registries_replay_identical_ids() {
        // Registry-level determinism: replaying the same allocation script on
        // two fresh registries yields identical innovation AND node id sequences.
        fn run() -> (Vec<InnovationId>, Vec<NodeId>) {
            let reg = InnovationRegistry::new(3, 2);
            let mut innovs = Vec::new();
            let mut nodes = Vec::new();
            innovs.push(reg.register_connection(0, 2));
            let s = reg.register_node_split(0);
            nodes.push(s.new_node);
            innovs.push(s.in_innov);
            innovs.push(s.out_innov);
            innovs.push(reg.register_connection(1, s.new_node));
            (innovs, nodes)
        }
        let (i1, n1) = run();
        let (i2, n2) = run();
        assert_eq!(i1, i2, "innovation sequence is reproducible across runs");
        assert_eq!(n1, n2, "node id sequence is reproducible across runs");
    }

    #[test]
    fn test_shared_registry_aligns_same_split_across_genomes() {
        // The cross-genome alignment guarantee: two genomes splitting the same
        // connection (same innovation) via the SAME shared registry get the same
        // node id and the same in/out innovations.
        let registry = InnovationRegistry::new(3, 2);
        let split_in_genome_a = registry.register_node_split(0);
        let split_in_genome_b = registry.register_node_split(0);
        assert_eq!(split_in_genome_a, split_in_genome_b);
    }
}
