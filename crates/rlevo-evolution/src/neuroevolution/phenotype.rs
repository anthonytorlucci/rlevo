//! Phenotype construction — turning a [`TopologyGenome`] into a callable network.
//!
//! [`PhenotypeBuilder`] abstracts *how* a genome becomes a network so a future
//! CPPN/`HyperNEAT` builder is a new impl, not a trait change. The reference
//! builder, [`InterpretedBuilder`], produces an [`InterpretedPhenotype`]: a
//! host-side **feedforward** evaluator built from bare `Tensor<B, 2>` arithmetic
//! over the genome's enabled connections in topological order — **no Burn
//! `Module`**, no autodiff, no `Recorder`. NEAT phenotypes need only a forward
//! pass, so skipping Burn's `Module` (whose `#[derive]` needs a static field
//! structure a data-defined topology cannot supply) costs nothing.
//!
//! The interpreted seam ([`PhenotypeBuilder`]/[`Phenotype`]) evaluates one genome
//! at a time. Its population-batched companion, [`BatchPhenotypeEvaluator`], runs
//! the *whole* population in one device-resident forward pass; the dense-padded
//! [`DensePaddedEvaluator`] is the stock-Burn implementation.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor, TensorData, activation};

use super::topology::{ActivationFn, NodeId, NodeKind, SIGMOID_GAIN, TopologyGenome};

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

/// Host-side feedforward reference phenotype.
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

// ===========================================================================
// Population-batched evaluation
// ===========================================================================

/// Population-level batched forward pass — the device-resident companion to the
/// per-genome [`PhenotypeBuilder`]/[`Phenotype`] interpreted seam.
///
/// Where [`Phenotype::forward`] evaluates one genome, an implementor of this
/// trait evaluates an entire population on a shared observation batch in a single
/// pass. The [`DensePaddedEvaluator`] v1 impl does so with stock Burn tensor ops
/// (no custom kernel); the interpreted path remains the correctness oracle (a
/// numerical-parity test pins the two within float epsilon).
pub trait BatchPhenotypeEvaluator<B: Backend>: Send + Sync {
    /// Evaluate every genome on the shared observation batch `obs`.
    ///
    /// `obs` has shape `[batch, obs_dim]`, where `obs_dim` must equal the
    /// population's (constant) input-node count. The result has shape
    /// `[pop, batch, action_dim]` — the population dimension is kept explicit so
    /// a caller can reduce fitness per genome without index arithmetic. Column
    /// order along `obs_dim`/`action_dim` matches the interpreted phenotype's
    /// (input/output nodes in ascending-id order).
    fn evaluate_population(
        &self,
        genomes: &[TopologyGenome],
        obs: Tensor<B, 2>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 3>;
}

/// Dense-padded [`BatchPhenotypeEvaluator`].
///
/// Each call compiles the `P` genomes into padded `(P, N, N)` weight, `(P, N)`
/// bias, and per-activation mask tensors over a node budget `N = max_nodes`, then
/// runs a synchronous-update forward pass. Because v1 NEAT is feedforward, the
/// pass is **exact**: after `d` synchronous updates every node at topological
/// depth `d` has settled, so iterating the population's **deepest enabled path**
/// (≤ `N − 1`) resolves every genome. The dense path uses that tight depth bound
/// rather than the static `N − 1` worst case — NEAT topologies are typically
/// sparse and shallow, and the per-step cost is a dense `N×N` matmul, so
/// over-iterating dominates the runtime at scale. The whole pass is stock Burn
/// (batched `matmul`, broadcast add, `mask_where`, the four elementwise
/// activations); the absent-edge mask folds into the zero weight.
///
/// Memory is dominated by the `weights` tensor: `(256, 50) → 2.56 MB`,
/// `(256, 200) → 41 MB` (f32). The [`max_nodes_cap`](Self::max_nodes_cap) guards
/// it when topologies grow.
#[derive(Debug, Clone, Copy)]
pub struct DensePaddedEvaluator {
    /// Hard ceiling on `N = max_nodes`. A population whose largest genome exceeds
    /// it panics rather than silently allocating an oversized weight tensor.
    pub max_nodes_cap: usize,
}

impl DensePaddedEvaluator {
    /// Build an evaluator with the given node-budget ceiling.
    #[must_use]
    pub fn new(max_nodes_cap: usize) -> Self {
        Self { max_nodes_cap }
    }
}

impl Default for DensePaddedEvaluator {
    /// A `512`-node ceiling — far past where the dense path stays affordable
    /// (41 MB at `N = 200`), so it never binds in practice.
    fn default() -> Self {
        Self { max_nodes_cap: 512 }
    }
}

impl<B: Backend> BatchPhenotypeEvaluator<B> for DensePaddedEvaluator {
    fn evaluate_population(
        &self,
        genomes: &[TopologyGenome],
        obs: Tensor<B, 2>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 3> {
        let pop = genomes.len();
        let [batch, obs_dim] = obs.dims();
        if pop == 0 {
            return Tensor::<B, 3>::zeros([0, batch, 0], device);
        }
        let max_nodes = genomes.iter().map(|g| g.nodes.len()).max().unwrap_or(0);
        assert!(
            max_nodes <= self.max_nodes_cap,
            "largest genome has {max_nodes} nodes, exceeding max_nodes_cap {}",
            self.max_nodes_cap
        );

        let PaddedPopulation {
            weights,
            bias,
            act_masks,
            input_slots,
            num_inputs,
            num_outputs,
            n,
            iterations,
        } = PaddedPopulation::<B>::compile(genomes, device);
        assert_eq!(
            obs_dim, num_inputs,
            "obs feature dim {obs_dim} must equal the population's input-node count {num_inputs}"
        );

        // Seed input rows with the observation (others 0): obsᵀ stacked over the
        // padding rows, broadcast across the population. Held fixed every step.
        let obs_t = obs.swap_dims(0, 1); // (num_inputs, batch)
        let seed_2d = if n > num_inputs {
            let pad = Tensor::<B, 2>::zeros([n - num_inputs, batch], device);
            Tensor::cat(vec![obs_t, pad], 0) // (N, batch)
        } else {
            obs_t
        };
        let seeded = seed_2d.unsqueeze_dim::<3>(0).repeat_dim(0, pop); // (P, N, batch)

        // Broadcast the per-node bias/masks across the batch dimension.
        let bias = bias.unsqueeze_dim::<3>(2); // (P, N, 1) — broadcasts on add
        let input_slots = input_slots.unsqueeze_dim::<3>(2).repeat_dim(2, batch);
        let [mask_sigmoid, mask_tanh, mask_relu, mask_linear] = act_masks;
        let mask_sigmoid = mask_sigmoid.unsqueeze_dim::<3>(2).repeat_dim(2, batch);
        let mask_tanh = mask_tanh.unsqueeze_dim::<3>(2).repeat_dim(2, batch);
        let mask_relu = mask_relu.unsqueeze_dim::<3>(2).repeat_dim(2, batch);
        let mask_linear = mask_linear.unsqueeze_dim::<3>(2).repeat_dim(2, batch);

        let mut values = seeded.clone(); // (P, N, batch)
        for _ in 0..iterations {
            let acc = weights.clone().matmul(values.clone()) + bias.clone(); // (P, N, batch)
            // Heterogeneous per-node activation: one masked pass per variant,
            // reusing the interpreted formulas (incl. SIGMOID_GAIN) verbatim.
            let mut out = Tensor::<B, 3>::zeros([pop, n, batch], device);
            out = out.mask_where(
                mask_sigmoid.clone(),
                activation::sigmoid(acc.clone().mul_scalar(SIGMOID_GAIN)),
            );
            out = out.mask_where(mask_tanh.clone(), activation::tanh(acc.clone()));
            out = out.mask_where(mask_relu.clone(), activation::relu(acc.clone()));
            out = out.mask_where(mask_linear.clone(), acc);
            // Re-seat the inputs so they keep carrying the observation.
            values = out.mask_where(input_slots.clone(), seeded.clone());
        }

        // Output rows are contiguous at `num_inputs..num_inputs + num_outputs`.
        let result = values.slice([0..pop, num_inputs..num_inputs + num_outputs, 0..batch]);
        result.swap_dims(1, 2) // (P, num_outputs, batch) -> (P, batch, action_dim)
    }
}

/// Per-generation host compile of `P` genomes into padded dense tensors.
///
/// Within each genome, node ids are assigned local rows in the order
/// `[inputs by id][outputs by id][others by id]`. Because NEAT fixes the
/// input/output node set across the whole population, input rows are always
/// `0..num_inputs` and output rows always `num_inputs..num_inputs + num_outputs`
/// — uniform across `P`, so seeding and output-slicing need no per-genome index
/// arithmetic. Padding rows (id-free, beyond a genome's node count) carry no
/// activation mask and no edges, so they stay zero and never feed a real node.
struct PaddedPopulation<B: Backend> {
    /// `(P, N, N)` — `weights[p, i, j]` is the weight of enabled edge `j → i`.
    weights: Tensor<B, 3>,
    /// `(P, N)` per-node bias (0 for input and padding rows).
    bias: Tensor<B, 2>,
    /// One `(P, N)` bool mask per [`ActivationFn`] variant, in the order
    /// `[Sigmoid, Tanh, Relu, Linear]`.
    act_masks: [Tensor<B, 2, Bool>; 4],
    /// `(P, N)` bool mask, true at input rows (held fixed each iteration).
    input_slots: Tensor<B, 2, Bool>,
    num_inputs: usize,
    num_outputs: usize,
    n: usize,
    /// Synchronous-update iterations needed to settle every genome: the maximum
    /// enabled-subgraph longest path (in edges) across the population, floored at
    /// `1` so bias-only nodes get their activation applied. This is the **tight,
    /// exact** bound — `N − 1` is the safe static worst case, but NEAT topologies
    /// are typically far shallower, and over-iterating is the dominant cost at
    /// scale (a dense `N×N` matmul per step), so the depth bound is what makes
    /// the dense path competitive on wide-but-shallow populations.
    iterations: usize,
}

impl<B: Backend> PaddedPopulation<B> {
    fn compile(
        genomes: &[TopologyGenome],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self {
        let pop = genomes.len();
        let num_inputs = count_kind(&genomes[0], NodeKind::Input);
        let num_outputs = count_kind(&genomes[0], NodeKind::Output);
        let n = genomes.iter().map(|g| g.nodes.len()).max().unwrap_or(0);
        // Tight exact iteration bound: the deepest enabled path in the whole
        // population (≤ N − 1), floored at 1 so a bias-only node still activates.
        let iterations = genomes
            .iter()
            .map(longest_path_edges)
            .max()
            .unwrap_or(0)
            .max(1);

        let mut weights = vec![0.0f32; pop * n * n];
        let mut bias = vec![0.0f32; pop * n];
        let mut masks: [Vec<f32>; 4] = [
            vec![0.0f32; pop * n],
            vec![0.0f32; pop * n],
            vec![0.0f32; pop * n],
            vec![0.0f32; pop * n],
        ];
        let mut input_slots = vec![0.0f32; pop * n];

        for (p, genome) in genomes.iter().enumerate() {
            debug_assert_eq!(
                count_kind(genome, NodeKind::Input),
                num_inputs,
                "every genome must share the population input-node count"
            );
            debug_assert_eq!(
                count_kind(genome, NodeKind::Output),
                num_outputs,
                "every genome must share the population output-node count"
            );
            let local = local_rows(genome);
            for node in &genome.nodes {
                let base = p * n + local[&node.id];
                if matches!(node.kind, NodeKind::Input) {
                    input_slots[base] = 1.0;
                } else {
                    bias[base] = node.bias;
                    masks[act_index(node.activation)][base] = 1.0;
                }
            }
            let wbase = p * n * n;
            for conn in &genome.connections {
                if !conn.enabled {
                    continue;
                }
                let i = local[&conn.target];
                let j = local[&conn.source];
                weights[wbase + i * n + j] = conn.weight;
            }
        }

        let weights = Tensor::<B, 3>::from_data(TensorData::new(weights, [pop, n, n]), device);
        let bias = Tensor::<B, 2>::from_data(TensorData::new(bias, [pop, n]), device);
        let act_masks = masks
            .map(|m| Tensor::<B, 2>::from_data(TensorData::new(m, [pop, n]), device).greater_elem(0.5));
        let input_slots =
            Tensor::<B, 2>::from_data(TensorData::new(input_slots, [pop, n]), device).greater_elem(0.5);

        Self {
            weights,
            bias,
            act_masks,
            input_slots,
            num_inputs,
            num_outputs,
            n,
            iterations,
        }
    }
}

/// Count nodes of a given kind in a genome.
fn count_kind(genome: &TopologyGenome, kind: NodeKind) -> usize {
    genome.nodes.iter().filter(|n| n.kind == kind).count()
}

/// Longest path, in **enabled** edges, through the genome's feedforward DAG.
///
/// This is the number of synchronous updates the dense forward pass needs to
/// settle every node (signal advances one edge per step). Disabled edges carry a
/// zero weight in the dense matrix, so they never propagate and are excluded
/// here, giving a tighter bound than counting all structural edges. Relaxing in
/// the all-edges topological order is valid because the enabled subgraph is a
/// subset of that DAG.
fn longest_path_edges(genome: &TopologyGenome) -> usize {
    let mut out_edges: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for conn in &genome.connections {
        if conn.enabled {
            out_edges.entry(conn.source).or_default().push(conn.target);
        }
    }
    let mut depth: HashMap<NodeId, usize> = genome.nodes.iter().map(|n| (n.id, 0usize)).collect();
    for node in topological_order(genome) {
        let here = depth.get(&node).copied().unwrap_or(0);
        if let Some(targets) = out_edges.get(&node) {
            for &t in targets {
                let slot = depth.entry(t).or_insert(0);
                *slot = (*slot).max(here + 1);
            }
        }
    }
    depth.into_values().max().unwrap_or(0)
}

/// Dense-table index for an activation variant (the `act_masks` array order).
fn act_index(act: ActivationFn) -> usize {
    match act {
        ActivationFn::Sigmoid => 0,
        ActivationFn::Tanh => 1,
        ActivationFn::Relu => 2,
        ActivationFn::Linear => 3,
    }
}

/// Map every node id to its local row, ordered `[inputs][outputs][others]`, each
/// group ascending by id (so input/output rows align with the interpreted
/// phenotype's column order).
fn local_rows(genome: &TopologyGenome) -> HashMap<NodeId, usize> {
    let mut inputs: Vec<NodeId> = filter_ids(genome, |k| matches!(k, NodeKind::Input));
    let mut outputs: Vec<NodeId> = filter_ids(genome, |k| matches!(k, NodeKind::Output));
    let mut others: Vec<NodeId> =
        filter_ids(genome, |k| !matches!(k, NodeKind::Input | NodeKind::Output));
    inputs.sort_unstable();
    outputs.sort_unstable();
    others.sort_unstable();

    let mut map: HashMap<NodeId, usize> = HashMap::with_capacity(genome.nodes.len());
    for (row, id) in inputs.into_iter().chain(outputs).chain(others).enumerate() {
        map.insert(id, row);
    }
    map
}

/// Collect the ids of nodes whose kind satisfies `pred`.
fn filter_ids(genome: &TopologyGenome, pred: impl Fn(NodeKind) -> bool) -> Vec<NodeId> {
    genome
        .nodes
        .iter()
        .filter(|n| pred(n.kind))
        .map(|n| n.id)
        .collect()
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
    use crate::neuroevolution::topology::{ConnectionGene, NodeGene};
    use burn::backend::Flex;

    type TestBackend = Flex;

    /// A hand-built feedforward genome reproduces a known truth table.
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
