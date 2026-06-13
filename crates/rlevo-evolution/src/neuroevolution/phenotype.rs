//! Phenotype construction — turning a [`TopologyGenome`] into a callable network.
//!
//! [`PhenotypeBuilder`] abstracts *how* a genome becomes a network so a future
//! CPPN/`HyperNEAT` builder is a new impl, not a trait change (spec §3.H). The v1
//! reference builder, [`InterpretedBuilder`], produces an
//! [`InterpretedPhenotype`]: a host-side **feedforward** evaluator built from
//! bare `Tensor<B, 2>` arithmetic over the genome's enabled connections in
//! topological order — **no Burn `Module`**, no autodiff, no `Recorder` (spec
//! §3.C). NEAT phenotypes need only a forward pass.
//!
//! The batched/GPU population evaluator (`BatchPhenotypeEvaluator`) is out of
//! scope and tracked in #41; these traits are the interpreted-path interface
//! only.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, activation};

use super::topology::{ActivationFn, NodeId, SIGMOID_GAIN, TopologyGenome};

/// Builds a callable [`Phenotype`] from a [`TopologyGenome`].
///
/// Implementors decide the network representation; the v1 reference is
/// [`InterpretedBuilder`]. A substrate-based builder (`HyperNEAT`) would take its
/// substrate layout in its constructor and implement this same trait.
pub trait PhenotypeBuilder<B: Backend> {
    /// Compile `genome` into a callable [`Phenotype`], allocating any
    /// device-side resources on `device`.
    fn build(
        &self,
        genome: &TopologyGenome,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Box<dyn Phenotype<B>>;
}

/// A callable network: a forward pass from a batch of inputs to a batch of
/// outputs.
pub trait Phenotype<B: Backend>: Send + Sync {
    /// Run a forward pass, mapping `[batch, num_inputs]` to
    /// `[batch, num_outputs]`. The compute device is taken from `input`.
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// The v1 reference builder — produces an [`InterpretedPhenotype`].
#[derive(Debug, Clone, Copy, Default)]
pub struct InterpretedBuilder;

impl<B: Backend> PhenotypeBuilder<B> for InterpretedBuilder {
    fn build(
        &self,
        genome: &TopologyGenome,
        _device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Box<dyn Phenotype<B>> {
        Box::new(InterpretedPhenotype::<B>::new(genome))
    }
}

/// Pre-computed evaluation plan for one non-input node.
#[derive(Debug, Clone)]
struct NodeEval {
    id: NodeId,
    /// Enabled incoming edges as `(source_node, weight)` pairs.
    incoming: Vec<(NodeId, f32)>,
    bias: f32,
    activation: ActivationFn,
}

/// Host-side feedforward reference phenotype (spec §3.C).
///
/// Stores only host-side evaluation metadata (topological order, per-node
/// incoming edges, biases, activations) — **no** Burn tensors — so it is
/// trivially `Send + Sync` and cheap to construct. The forward device is taken
/// from the input tensor.
#[derive(Debug, Clone)]
pub struct InterpretedPhenotype<B: Backend> {
    /// Input node ids in input-column order.
    input_ids: Vec<NodeId>,
    /// Output node ids in output-column order.
    output_ids: Vec<NodeId>,
    /// Non-input nodes in topological order, with their evaluation plan.
    eval_order: Vec<NodeEval>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> InterpretedPhenotype<B> {
    /// Compile a genome into an evaluation plan.
    ///
    /// Nodes are ordered topologically over **all** structural edges; because
    /// every genome maintains the feedforward DAG invariant, this order is valid
    /// for the enabled subgraph the forward pass actually follows.
    #[must_use]
    pub fn new(genome: &TopologyGenome) -> Self {
        let mut input_ids: Vec<NodeId> = genome
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, super::topology::NodeKind::Input))
            .map(|n| n.id)
            .collect();
        input_ids.sort_unstable();

        let mut output_ids: Vec<NodeId> = genome
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, super::topology::NodeKind::Output))
            .map(|n| n.id)
            .collect();
        output_ids.sort_unstable();

        let input_set: HashSet<NodeId> = input_ids.iter().copied().collect();
        let order = topological_order(genome);

        let mut eval_order: Vec<NodeEval> = Vec::with_capacity(order.len());
        for nid in order {
            if input_set.contains(&nid) {
                continue;
            }
            let Some(node) = genome.node(nid) else {
                continue;
            };
            let incoming: Vec<(NodeId, f32)> = genome
                .connections
                .iter()
                .filter(|c| c.target == nid && c.enabled)
                .map(|c| (c.source, c.weight))
                .collect();
            eval_order.push(NodeEval {
                id: nid,
                incoming,
                bias: node.bias,
                activation: node.activation,
            });
        }

        Self {
            input_ids,
            output_ids,
            eval_order,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Phenotype<B> for InterpretedPhenotype<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, _num_inputs] = input.dims();
        let device = input.device();

        let mut values: HashMap<NodeId, Tensor<B, 2>> = HashMap::new();
        for (col, &iid) in self.input_ids.iter().enumerate() {
            // Column `col` of the input → input node's value, shape [batch, 1].
            let column = input.clone().slice([0..batch, col..col + 1]);
            values.insert(iid, column);
        }

        for node in &self.eval_order {
            let mut acc = Tensor::<B, 2>::zeros([batch, 1], &device);
            for (src, weight) in &node.incoming {
                if let Some(src_value) = values.get(src) {
                    acc = acc + src_value.clone().mul_scalar(*weight);
                }
            }
            acc = acc.add_scalar(node.bias);
            values.insert(node.id, apply_activation::<B>(node.activation, acc));
        }

        let columns: Vec<Tensor<B, 2>> = self
            .output_ids
            .iter()
            .map(|oid| {
                values
                    .get(oid)
                    .cloned()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([batch, 1], &device))
            })
            .collect();
        Tensor::cat(columns, 1)
    }
}

/// Tensor-side activation, mirroring [`ActivationFn::apply`] exactly (including
/// the [`SIGMOID_GAIN`] steepening) so a hand-computed truth table matches.
fn apply_activation<B: Backend>(act: ActivationFn, x: Tensor<B, 2>) -> Tensor<B, 2> {
    match act {
        ActivationFn::Sigmoid => activation::sigmoid(x.mul_scalar(SIGMOID_GAIN)),
        ActivationFn::Tanh => activation::tanh(x),
        ActivationFn::Relu => activation::relu(x),
        ActivationFn::Linear => x,
    }
}

/// Kahn topological sort over **all** structural edges, breaking ties by
/// ascending node id for reproducibility. Returns every node id; any node left
/// unordered by a (non-invariant) cycle is appended at the end so the forward
/// pass never indexes a missing node.
fn topological_order(genome: &TopologyGenome) -> Vec<NodeId> {
    let mut in_degree: BTreeMap<NodeId, usize> =
        genome.nodes.iter().map(|n| (n.id, 0usize)).collect();
    let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for c in &genome.connections {
        if let Some(d) = in_degree.get_mut(&c.target) {
            *d += 1;
        }
        adj.entry(c.source).or_default().push(c.target);
    }

    // BTreeMap iteration is sorted, so the seed queue is in ascending id order.
    let mut queue: VecDeque<NodeId> = in_degree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut order: Vec<NodeId> = Vec::with_capacity(genome.nodes.len());
    while let Some(n) = queue.pop_front() {
        order.push(n);
        if let Some(succ) = adj.get(&n) {
            let mut targets = succ.clone();
            targets.sort_unstable();
            for t in targets {
                if let Some(d) = in_degree.get_mut(&t) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(t);
                    }
                }
            }
        }
    }

    if order.len() < genome.nodes.len() {
        for n in &genome.nodes {
            if !order.contains(&n.id) {
                order.push(n.id);
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuroevolution::topology::{ConnectionGene, NodeGene, NodeKind};
    use burn::backend::Flex;

    type TestBackend = Flex;

    /// AC4: a hand-built feedforward genome reproduces a known truth table.
    ///
    /// Network: inputs 0, 1 → hidden 2 (Relu) → output 3 (Linear). With
    /// `h = relu(1·in0 + 1·in1)` and `out = 2·h + 0.5`, the four binary input
    /// rows give outputs `0.5, 2.5, 2.5, 4.5`.
    #[test]
    fn test_interpreted_phenotype_reproduces_truth_table() {
        let device = Default::default();
        let nodes = vec![
            NodeGene { id: 0, kind: NodeKind::Input, activation: ActivationFn::Linear, bias: 0.0 },
            NodeGene { id: 1, kind: NodeKind::Input, activation: ActivationFn::Linear, bias: 0.0 },
            NodeGene { id: 2, kind: NodeKind::Hidden, activation: ActivationFn::Relu, bias: 0.0 },
            NodeGene { id: 3, kind: NodeKind::Output, activation: ActivationFn::Linear, bias: 0.5 },
        ];
        let conns = vec![
            ConnectionGene { innovation: 0, source: 0, target: 2, weight: 1.0, enabled: true },
            ConnectionGene { innovation: 1, source: 1, target: 2, weight: 1.0, enabled: true },
            ConnectionGene { innovation: 2, source: 2, target: 3, weight: 2.0, enabled: true },
        ];
        let genome = TopologyGenome::new(nodes, conns);

        let builder = InterpretedBuilder;
        let pheno = PhenotypeBuilder::<TestBackend>::build(&builder, &genome, &device);

        let input = Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(
                vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [4, 2],
            ),
            &device,
        );
        let out = pheno.forward(input).into_data().into_vec::<f32>().unwrap();
        let expected = [0.5f32, 2.5, 2.5, 4.5];
        for (got, want) in out.iter().zip(expected.iter()) {
            approx::assert_relative_eq!(*got, *want, epsilon = 1e-5);
        }
    }

    /// A disabled connection is skipped in the forward pass.
    #[test]
    fn test_interpreted_phenotype_skips_disabled_edges() {
        let device = Default::default();
        let nodes = vec![
            NodeGene { id: 0, kind: NodeKind::Input, activation: ActivationFn::Linear, bias: 0.0 },
            NodeGene { id: 1, kind: NodeKind::Output, activation: ActivationFn::Linear, bias: 1.0 },
        ];
        // Single edge 0 -> 1 is DISABLED, so output = bias = 1.0 regardless.
        let conns = vec![ConnectionGene {
            innovation: 0,
            source: 0,
            target: 1,
            weight: 99.0,
            enabled: false,
        }];
        let genome = TopologyGenome::new(nodes, conns);
        let pheno = InterpretedPhenotype::<TestBackend>::new(&genome);
        let input = Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(vec![5.0f32, 7.0], [2, 1]),
            &device,
        );
        let out = pheno.forward(input).into_data().into_vec::<f32>().unwrap();
        approx::assert_relative_eq!(out[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[1], 1.0, epsilon = 1e-6);
    }
}
