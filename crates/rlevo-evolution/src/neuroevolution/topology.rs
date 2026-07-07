//! NEAT topology genome — a direct graph encoding of node and connection genes.
//!
//! A [`TopologyGenome`] is the **genotype**: two gene lists (nodes and
//! connections) that NEAT mutates by adding neurons and edges and recombines via
//! historical **innovation numbers**. The network actually evaluated — the
//! *phenotype* — is built from this genome by
//! [`crate::neuroevolution::phenotype`] and walks the enabled connections in
//! topological order.
//!
//! # Invariants
//!
//! - `connections` is kept **sorted by `innovation`** (see
//!   [`TopologyGenome::insert_connection_sorted`]). Mutations append the
//!   largest-so-far innovation, so insertion keeps it sorted cheaply, and
//!   crossover / compatibility distance become `O(n)` merges.
//! - The directed graph over **all** structural edges (enabled *or* disabled) is
//!   acyclic — the feedforward invariant. [`TopologyGenome::would_create_cycle`]
//!   checks against all edges so the DAG stays stable under enable/disable
//!   toggles.
//!
//! Unlike the tensor-backed genomes elsewhere in the crate,
//! [`TopologyGenome`] **is** [`Clone`]: it is plain host-side data with no
//! Burn-tensor storage aliasing.

use std::collections::HashSet;

use rand::Rng;
use rand_distr::{Distribution as _, Normal};

use super::innovation::InnovationRegistry;

/// Stable identifier for a node gene. Monotone within a run; allocated only by
/// the [`InnovationRegistry`] (for hidden nodes) or fixed by the minimal
/// topology (for inputs/outputs).
///
/// An **opaque newtype** over `u64` (not a bare alias): a `NodeId` cannot be
/// interchanged with an [`InnovationId`] or a raw integer, so the mutation and
/// crossover logic can never confuse the two id spaces. It has no invariant —
/// every `u64` is a legal id — so [`new`](NodeId::new) is infallible. Construct
/// with `new`, read with [`get`](NodeId::get); the crate-internal
/// [`succ`](NodeId::succ) is the only arithmetic, used solely by the
/// [`InnovationRegistry`] counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);

impl NodeId {
    /// Wrap a raw id. Infallible — a node id has no invariant.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The underlying id.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The next id in sequence. Crate-internal because only the
    /// [`InnovationRegistry`] allocates fresh node ids.
    #[must_use]
    pub(crate) const fn succ(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Historical marker for a connection gene — the *innovation number* that lets
/// crossover align structurally-different genomes. Globally monotone within a
/// run, but sparse within any one genome.
///
/// An **opaque newtype** over `u64` (not a bare alias); see [`NodeId`] for the
/// rationale. Its derived [`Ord`] is what keeps `connections` innovation-sorted
/// and drives the `O(n)` crossover / compatibility-distance merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnovationId(u64);

impl InnovationId {
    /// Wrap a raw innovation number. Infallible — it has no invariant.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The underlying innovation number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The next innovation in sequence. Crate-internal because only the
    /// [`InnovationRegistry`] allocates fresh innovations.
    #[must_use]
    pub(crate) const fn succ(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Steepening gain of the canonical NEAT logistic sigmoid (Stanley &
/// Miikkulainen 2002 use `4.9`). Shared by the host-side
/// [`ActivationFn::apply`] and the tensor forward pass in
/// [`crate::neuroevolution::phenotype`] so the two never disagree.
pub(crate) const SIGMOID_GAIN: f32 = 4.9;

/// Role of a node within the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// Sensor node: holds an input value verbatim (no bias, no activation).
    Input,
    /// Output node: its activation is read as a network output.
    Output,
    /// Hidden node introduced by an add-node mutation.
    Hidden,
    /// Always-on bias node. Reserved for completeness; v1's minimal topology
    /// carries bias on the node gene ([`NodeGene::bias`]) instead, so this
    /// variant is unused by [`TopologyGenome::minimal`].
    Bias,
}

/// Activation applied at a node.
///
/// The canonical NEAT starting set. Marked `#[non_exhaustive]` so CPPN
/// activations (`sin`/`gauss`/`abs`) needed by a future `HyperNEAT` builder are a
/// non-breaking addition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivationFn {
    /// Steepened logistic sigmoid (gain `4.9`) — the canonical NEAT activation.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Rectified linear unit.
    Relu,
    /// Identity.
    Linear,
}

impl ActivationFn {
    /// Apply the activation to a scalar, host-side.
    ///
    /// The tensor forward pass in [`crate::neuroevolution::phenotype`] mirrors
    /// these exact formulas (including the [`SIGMOID_GAIN`] steepening) so a
    /// hand-computed truth table matches the interpreted phenotype.
    #[must_use]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-SIGMOID_GAIN * x).exp()),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Relu => x.max(0.0),
            ActivationFn::Linear => x,
        }
    }
}

/// A single node gene: identity, role, activation, and per-node bias.
#[derive(Clone, Debug)]
pub struct NodeGene {
    /// Stable node id (see [`NodeId`]).
    pub id: NodeId,
    /// Node role.
    pub kind: NodeKind,
    /// Activation applied to this node's pre-activation sum.
    pub activation: ActivationFn,
    /// Additive bias folded into the node's pre-activation sum. Mutated by the
    /// weight-perturbation operator (a bias is, functionally, a weight).
    pub bias: f32,
}

/// A single connection gene: a weighted directed edge tagged with its
/// historical innovation number.
#[derive(Clone, Debug)]
pub struct ConnectionGene {
    /// Historical marker aligning this edge across genomes (see [`InnovationId`]).
    pub innovation: InnovationId,
    /// Source node id.
    pub source: NodeId,
    /// Target node id.
    pub target: NodeId,
    /// Edge weight.
    pub weight: f32,
    /// Whether the edge carries signal in the phenotype. Disabled genes are
    /// skipped in the forward pass but still counted in compatibility distance
    /// and crossover alignment.
    pub enabled: bool,
}

/// A network genotype: a node-gene list plus an innovation-sorted
/// connection-gene list.
///
/// See the [module docs](self) for the two structural invariants.
///
/// Fields are `pub(crate)` rather than public: the innovation-sort invariant on
/// `connections` is maintained collectively by the NEAT mutation, crossover,
/// and speciation operators (`neat.rs`, `species.rs`, `phenotype.rs`), which
/// edit the vectors in place. Exposing them publicly would let external code
/// build an unsorted genome by struct literal; instead, construct one with
/// [`TopologyGenome::new`] / [`TopologyGenome::minimal`] (which establish the
/// invariant) and extend it with [`insert_connection_sorted`](TopologyGenome::insert_connection_sorted).
#[derive(Clone, Debug)]
pub struct TopologyGenome {
    /// Node genes (inputs, outputs, and any hidden nodes).
    pub(crate) nodes: Vec<NodeGene>,
    /// Connection genes, kept sorted by [`ConnectionGene::innovation`].
    pub(crate) connections: Vec<ConnectionGene>,
}

impl TopologyGenome {
    /// Build a genome from explicit gene lists, sorting connections by
    /// innovation to establish the sorted invariant.
    ///
    /// Hand-built test genomes should prefer this over a struct literal so the
    /// sorted invariant holds regardless of input order.
    #[must_use]
    pub fn new(nodes: Vec<NodeGene>, mut connections: Vec<ConnectionGene>) -> Self {
        connections.sort_by_key(|c| c.innovation);
        Self { nodes, connections }
    }

    /// Build the minimal seed topology: `num_inputs` input nodes fully connected
    /// to `num_outputs` output nodes, with no hidden nodes (NEAT's
    /// minimal-topology principle).
    ///
    /// Node ids are fixed by convention — inputs `0..num_inputs`, outputs
    /// `num_inputs..num_inputs + num_outputs` — and initial connection
    /// innovations are `input_index * num_outputs + output_index`, i.e. the
    /// range `0..num_inputs * num_outputs`. A matching registry is created with
    /// `InnovationRegistry::new(num_inputs + num_outputs, num_inputs *
    /// num_outputs)` so its counters start *after* this seed; the registry is
    /// passed only to assert that agreement (it is not used to allocate the seed
    /// ids, which would double-count them).
    ///
    /// Calling this once per individual with the *same* registry yields aligned
    /// initial genomes (identical ids, per-individual random weights).
    ///
    /// # Panics
    ///
    /// Panics if `weight_init_std` is negative (degenerate normal), or (in debug
    /// builds) if `registry`'s counters disagree with the seed sizes.
    #[must_use]
    pub fn minimal(
        num_inputs: usize,
        num_outputs: usize,
        registry: &InnovationRegistry,
        rng: &mut dyn Rng,
        weight_init_std: f32,
    ) -> Self {
        debug_assert!(
            registry.next_node_id().get() >= (num_inputs + num_outputs) as u64
                && registry.next_innovation().get() >= (num_inputs * num_outputs) as u64,
            "registry counters must start after the minimal seed (H6)"
        );
        let normal = Normal::new(0.0_f32, weight_init_std)
            .expect("weight_init_std must be finite and non-negative");

        let mut nodes: Vec<NodeGene> = Vec::with_capacity(num_inputs + num_outputs);
        for i in 0..num_inputs {
            nodes.push(NodeGene {
                id: NodeId::new(i as u64),
                kind: NodeKind::Input,
                activation: ActivationFn::Linear,
                bias: 0.0,
            });
        }
        for o in 0..num_outputs {
            nodes.push(NodeGene {
                id: NodeId::new((num_inputs + o) as u64),
                kind: NodeKind::Output,
                activation: ActivationFn::Sigmoid,
                bias: normal.sample(rng),
            });
        }

        let mut connections: Vec<ConnectionGene> = Vec::with_capacity(num_inputs * num_outputs);
        for i in 0..num_inputs {
            for o in 0..num_outputs {
                connections.push(ConnectionGene {
                    innovation: InnovationId::new((i * num_outputs + o) as u64),
                    source: NodeId::new(i as u64),
                    target: NodeId::new((num_inputs + o) as u64),
                    weight: normal.sample(rng),
                    enabled: true,
                });
            }
        }
        // Already innovation-sorted by construction.
        Self { nodes, connections }
    }

    /// Insert a connection gene, preserving the innovation-sorted invariant.
    ///
    /// The caller must guarantee `gene.innovation` is not already present.
    pub fn insert_connection_sorted(&mut self, gene: ConnectionGene) {
        debug_assert!(
            self.connections
                .iter()
                .all(|c| c.innovation != gene.innovation),
            "insert_connection_sorted requires a fresh innovation id"
        );
        let pos = self
            .connections
            .partition_point(|c| c.innovation < gene.innovation);
        self.connections.insert(pos, gene);
    }

    /// Look up a node gene by id.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&NodeGene> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Whether a directed edge `source -> target` already exists (enabled or
    /// disabled). Used by add-connection to avoid duplicate edges.
    #[must_use]
    pub fn is_connected(&self, source: NodeId, target: NodeId) -> bool {
        self.connections
            .iter()
            .any(|c| c.source == source && c.target == target)
    }

    /// Whether adding edge `source -> target` would create a cycle, considering
    /// **all** structural edges (enabled or disabled).
    ///
    /// A cycle forms exactly when `target` can already reach `source`; checking
    /// over all edges (not just enabled ones) keeps the feedforward DAG stable
    /// under enable/disable toggles.
    #[must_use]
    pub fn would_create_cycle(&self, source: NodeId, target: NodeId) -> bool {
        if source == target {
            return true;
        }
        let mut stack: Vec<NodeId> = vec![target];
        let mut visited: HashSet<NodeId> = HashSet::new();
        while let Some(node) = stack.pop() {
            if node == source {
                return true;
            }
            if !visited.insert(node) {
                continue;
            }
            for c in &self.connections {
                if c.source == node {
                    stack.push(c.target);
                }
            }
        }
        false
    }

    /// Whether `connections` is **strictly** increasing by innovation — i.e.
    /// sorted with no duplicate innovation ids, the full structural invariant.
    #[must_use]
    pub fn is_innovation_sorted(&self) -> bool {
        self.connections
            .windows(2)
            .all(|w| w[0].innovation < w[1].innovation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_id_newtypes_round_trip_and_succ() {
        // The opaque-id surface: `new`/`get` round-trip and `succ` steps by one.
        // `NodeId` and `InnovationId` are distinct types — a program that mixed
        // them would not compile, which is the whole point of the newtype.
        assert_eq!(NodeId::new(7).get(), 7);
        assert_eq!(NodeId::new(7).succ(), NodeId::new(8));
        assert_eq!(InnovationId::new(0).get(), 0);
        assert_eq!(InnovationId::new(0).succ().succ(), InnovationId::new(2));
        // Ordering (needed by the innovation-sorted invariant and BTreeMap keys).
        assert!(InnovationId::new(1) < InnovationId::new(2));
    }

    #[test]
    fn test_activation_fn_apply_known_values() {
        // Linear is identity; Relu clamps negatives; Tanh is odd; Sigmoid is
        // steepened logistic centered at 0.5.
        approx::assert_relative_eq!(ActivationFn::Linear.apply(0.7), 0.7, epsilon = 1e-6);
        approx::assert_relative_eq!(ActivationFn::Relu.apply(-2.0), 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(ActivationFn::Relu.apply(3.5), 3.5, epsilon = 1e-6);
        approx::assert_relative_eq!(ActivationFn::Tanh.apply(0.0), 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(ActivationFn::Sigmoid.apply(0.0), 0.5, epsilon = 1e-6);
        // Steepened sigmoid saturates fast: sigmoid(4.9 * 1) ~ 0.9926.
        approx::assert_relative_eq!(
            ActivationFn::Sigmoid.apply(1.0),
            1.0 / (1.0 + (-SIGMOID_GAIN).exp()),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_minimal_topology_ids_and_innovations() {
        let registry = InnovationRegistry::new(3, 2); // 2 inputs + 1 output, 2 connections
        let mut rng = StdRng::seed_from_u64(1);
        let g = TopologyGenome::minimal(2, 1, &registry, &mut rng, 1.0);

        // 2 inputs (0, 1) + 1 output (2).
        assert_eq!(g.nodes.len(), 3, "minimal seed has I + O nodes");
        assert_eq!(g.node(NodeId::new(0)).unwrap().kind, NodeKind::Input);
        assert_eq!(g.node(NodeId::new(1)).unwrap().kind, NodeKind::Input);
        assert_eq!(g.node(NodeId::new(2)).unwrap().kind, NodeKind::Output);

        // Fully connected inputs -> output with innovations 0 and 1.
        assert_eq!(g.connections.len(), 2, "I * O connections");
        let innovs: Vec<u64> = g.connections.iter().map(|c| c.innovation.get()).collect();
        assert_eq!(innovs, vec![0, 1], "initial innovations are 0..I*O");
        assert!(g.is_innovation_sorted());
    }

    #[test]
    fn test_insert_connection_sorted_keeps_order() {
        let registry = InnovationRegistry::new(3, 2);
        let mut rng = StdRng::seed_from_u64(1);
        let mut g = TopologyGenome::minimal(2, 1, &registry, &mut rng, 1.0);
        // Insert a smaller-than-max innovation out of order; must land in place.
        g.insert_connection_sorted(ConnectionGene {
            innovation: InnovationId::new(5),
            source: NodeId::new(0),
            target: NodeId::new(2),
            weight: 0.1,
            enabled: true,
        });
        g.insert_connection_sorted(ConnectionGene {
            innovation: InnovationId::new(3),
            source: NodeId::new(1),
            target: NodeId::new(2),
            weight: 0.2,
            enabled: true,
        });
        assert!(
            g.is_innovation_sorted(),
            "sorted invariant preserved on insert"
        );
        let innovs: Vec<u64> = g.connections.iter().map(|c| c.innovation.get()).collect();
        assert_eq!(innovs, vec![0, 1, 3, 5]);
    }

    #[test]
    fn test_would_create_cycle_rejects_back_edge() {
        // Build 0 -> 2 -> 3 (feedforward). Adding 3 -> 0 would close a cycle.
        let nodes = vec![
            NodeGene {
                id: NodeId::new(0),
                kind: NodeKind::Input,
                activation: ActivationFn::Linear,
                bias: 0.0,
            },
            NodeGene {
                id: NodeId::new(2),
                kind: NodeKind::Hidden,
                activation: ActivationFn::Relu,
                bias: 0.0,
            },
            NodeGene {
                id: NodeId::new(3),
                kind: NodeKind::Output,
                activation: ActivationFn::Sigmoid,
                bias: 0.0,
            },
        ];
        let conns = vec![
            ConnectionGene {
                innovation: InnovationId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: 1.0,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(1),
                source: NodeId::new(2),
                target: NodeId::new(3),
                weight: 1.0,
                enabled: true,
            },
        ];
        let g = TopologyGenome::new(nodes, conns);
        assert!(
            g.would_create_cycle(NodeId::new(3), NodeId::new(0)),
            "3 -> 0 closes a cycle through 0 -> 2 -> 3"
        );
        assert!(
            g.would_create_cycle(NodeId::new(3), NodeId::new(2)),
            "3 -> 2 closes a cycle through 2 -> 3"
        );
        assert!(
            !g.would_create_cycle(NodeId::new(0), NodeId::new(3)),
            "0 -> 3 is a forward edge"
        );
        assert!(
            g.would_create_cycle(NodeId::new(0), NodeId::new(0)),
            "self-loop is a cycle"
        );
    }

    #[test]
    fn test_would_create_cycle_counts_disabled_edges() {
        // Disabled 2 -> 3 still constrains acyclicity (H2).
        let nodes = vec![
            NodeGene {
                id: NodeId::new(2),
                kind: NodeKind::Hidden,
                activation: ActivationFn::Relu,
                bias: 0.0,
            },
            NodeGene {
                id: NodeId::new(3),
                kind: NodeKind::Hidden,
                activation: ActivationFn::Relu,
                bias: 0.0,
            },
        ];
        let conns = vec![ConnectionGene {
            innovation: InnovationId::new(0),
            source: NodeId::new(2),
            target: NodeId::new(3),
            weight: 1.0,
            enabled: false,
        }];
        let g = TopologyGenome::new(nodes, conns);
        assert!(
            g.would_create_cycle(NodeId::new(3), NodeId::new(2)),
            "disabled edges are counted so the DAG survives re-enable"
        );
    }
}
