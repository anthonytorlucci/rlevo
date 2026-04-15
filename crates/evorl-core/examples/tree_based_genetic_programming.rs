use evorl_core::evolution::{
    Crossover, FitnessEvaluator, Individual, Mutation, Replacement, Selection, Termination,
};
use rand::RngExt;
use std::rc::Rc;

// Tree node for GP
#[derive(Clone, Debug)]
pub enum GpNode {
    Function {
        name: String,
        arity: usize,
        children: Vec<Rc<GpNode>>,
    },
    Terminal {
        value: f64,
    },
    Variable {
        name: String,
    },
}

// GP Individual with tree representation
#[derive(Clone, Debug)]
pub struct GpIndividual {
    root: Rc<GpNode>,
    fitness: f64,
}

impl Individual for GpIndividual {
    type Gene = Rc<GpNode>;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn genes(&self) -> &[Self::Gene] {
        std::slice::from_ref(&self.root)
    }

    fn genes_mut(&mut self) -> &mut [Self::Gene] {
        std::slice::from_mut(&mut self.root)
    }
}

impl GpIndividual {
    pub fn new(root: GpNode) -> Self {
        Self {
            root: Rc::new(root),
            fitness: 0.0,
        }
    }

    pub fn depth(&self) -> usize {
        self.node_depth(&self.root)
    }

    fn node_depth(&self, node: &GpNode) -> usize {
        match node {
            GpNode::Terminal { .. } | GpNode::Variable { .. } => 1,
            GpNode::Function { children, .. } => {
                1 + children
                    .iter()
                    .map(|child| self.node_depth(child))
                    .max()
                    .unwrap_or(0)
            }
        }
    }

    pub fn size(&self) -> usize {
        self.node_size(&self.root)
    }

    fn node_size(&self, node: &GpNode) -> usize {
        match node {
            GpNode::Terminal { .. } | GpNode::Variable { .. } => 1,
            GpNode::Function { children, .. } => {
                1 + children
                    .iter()
                    .map(|child| self.node_size(child))
                    .sum::<usize>()
            }
        }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.eval_node(&self.root, x)
    }

    fn eval_node(&self, node: &GpNode, x: f64) -> f64 {
        match node {
            GpNode::Terminal { value } => *value,
            GpNode::Variable { .. } => x,
            GpNode::Function { name, children, .. } => {
                let child_values: Vec<f64> =
                    children.iter().map(|c| self.eval_node(c, x)).collect();
                match name.as_str() {
                    "+" => child_values.iter().sum(),
                    "*" => child_values.iter().product(),
                    "-" if child_values.len() == 2 => child_values[0] - child_values[1],
                    "/" if child_values.len() == 2 => {
                        if child_values[1].abs() < 1e-10 {
                            1.0
                        } else {
                            child_values[0] / child_values[1]
                        }
                    }
                    "sin" => child_values.get(0).map(|v| v.sin()).unwrap_or(0.0),
                    "cos" => child_values.get(0).map(|v| v.cos()).unwrap_or(0.0),
                    _ => 0.0,
                }
            }
        }
    }
}

// Subtree crossover for GP
pub struct SubtreeCrossover {
    rate: f64,
}

impl SubtreeCrossover {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }

    fn select_random_node(
        &self,
        node: &Rc<GpNode>,
        target_index: usize,
        current_index: &mut usize,
    ) -> Option<Rc<GpNode>> {
        if *current_index == target_index {
            return Some(Rc::clone(node));
        }
        *current_index += 1;

        match node.as_ref() {
            GpNode::Function { children, .. } => {
                for child in children {
                    if let Some(found) = self.select_random_node(child, target_index, current_index)
                    {
                        return Some(found);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn replace_node(
        &self,
        node: &Rc<GpNode>,
        target_index: usize,
        replacement: &Rc<GpNode>,
        current_index: &mut usize,
    ) -> Rc<GpNode> {
        if *current_index == target_index {
            *current_index += 1;
            return Rc::clone(replacement);
        }
        *current_index += 1;

        match node.as_ref() {
            GpNode::Function {
                name,
                arity,
                children,
            } => {
                let new_children: Vec<_> = children
                    .iter()
                    .map(|child| self.replace_node(child, target_index, replacement, current_index))
                    .collect();

                Rc::new(GpNode::Function {
                    name: name.clone(),
                    arity: *arity,
                    children: new_children,
                })
            }
            terminal => Rc::clone(&Rc::new(terminal.clone())),
        }
    }
}

impl Crossover<GpIndividual> for SubtreeCrossover {
    fn crossover(
        &self,
        parent1: &GpIndividual,
        parent2: &GpIndividual,
    ) -> (GpIndividual, GpIndividual) {
        let mut rng = rand::rng();

        if rng.random::<f64>() > self.rate {
            return (parent1.clone(), parent2.clone());
        }

        let size1 = parent1.size();
        let size2 = parent2.size();

        let point1 = if size1 > 0 {
            rng.random_range(0..size1)
        } else {
            0
        };
        let point2 = if size2 > 0 {
            rng.random_range(0..size2)
        } else {
            0
        };

        let mut index = 0;
        let subtree1 = self
            .select_random_node(&parent1.root, point1, &mut index)
            .unwrap_or_else(|| Rc::clone(&parent1.root));

        index = 0;
        let subtree2 = self
            .select_random_node(&parent2.root, point2, &mut index)
            .unwrap_or_else(|| Rc::clone(&parent2.root));

        index = 0;
        let child1_root = self.replace_node(&parent1.root, point1, &subtree2, &mut index);

        index = 0;
        let child2_root = self.replace_node(&parent2.root, point2, &subtree1, &mut index);

        (
            GpIndividual::new((*child1_root).clone()),
            GpIndividual::new((*child2_root).clone()),
        )
    }

    fn crossover_rate(&self) -> f64 {
        self.rate
    }
}

// Subtree mutation for GP
pub struct SubtreeMutation {
    rate: f64,
}

impl SubtreeMutation {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }

    fn generate_random_tree(depth: usize) -> Rc<GpNode> {
        let mut rng = rand::rng();
        if depth == 0 || rng.random::<f64>() < 0.3 {
            if rng.random::<bool>() {
                Rc::new(GpNode::Variable {
                    name: "x".to_string(),
                })
            } else {
                Rc::new(GpNode::Terminal {
                    value: rng.random_range(-5.0..5.0),
                })
            }
        } else {
            let functions = vec!["+", "-", "*"];
            let name = functions[rng.random_range(0..functions.len())].to_string();
            let children = vec![
                Self::generate_random_tree(depth - 1),
                Self::generate_random_tree(depth - 1),
            ];
            Rc::new(GpNode::Function {
                name,
                arity: 2,
                children,
            })
        }
    }

    fn replace_node_mutation(
        &self,
        node: &Rc<GpNode>,
        target_index: usize,
        replacement: &Rc<GpNode>,
        current_index: &mut usize,
    ) -> Rc<GpNode> {
        if *current_index == target_index {
            *current_index += 1;
            return Rc::clone(replacement);
        }
        *current_index += 1;

        match node.as_ref() {
            GpNode::Function {
                name,
                arity,
                children,
            } => {
                let new_children: Vec<_> = children
                    .iter()
                    .map(|child| {
                        self.replace_node_mutation(child, target_index, replacement, current_index)
                    })
                    .collect();

                Rc::new(GpNode::Function {
                    name: name.clone(),
                    arity: *arity,
                    children: new_children,
                })
            }
            terminal => Rc::clone(&Rc::new(terminal.clone())),
        }
    }
}

impl Mutation<GpIndividual> for SubtreeMutation {
    fn mutate(&self, individual: &mut GpIndividual) {
        let mut rng = rand::rng();
        if rng.random::<f64>() >= self.rate {
            return;
        }

        let size = individual.size();
        if size == 0 {
            return;
        }

        let target_index = rng.random_range(0..size);
        let new_subtree = Self::generate_random_tree(3);
        let mut current_index = 0;

        let new_root = self.replace_node_mutation(
            &individual.root,
            target_index,
            &new_subtree,
            &mut current_index,
        );
        individual.root = new_root;
    }

    fn mutation_rate(&self) -> f64 {
        self.rate
    }
}

// Symbolic regression fitness evaluator
pub struct SymbolicRegressionEvaluator {
    test_points: Vec<(f64, f64)>,
}

impl SymbolicRegressionEvaluator {
    pub fn new() -> Self {
        let mut test_points = Vec::new();
        for i in 0..10 {
            let x = i as f64;
            let y = x * x + 2.0 * x + 1.0;
            test_points.push((x, y));
        }
        Self { test_points }
    }
}

impl FitnessEvaluator<GpIndividual> for SymbolicRegressionEvaluator {
    fn evaluate(&self, individual: &GpIndividual) -> f64 {
        let mut error = 0.0;
        for (x, target_y) in &self.test_points {
            let predicted_y = individual.evaluate(*x);
            let diff = predicted_y - target_y;
            error += diff * diff;
        }
        -error / self.test_points.len() as f64
    }
}

// Tournament selection
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

impl Selection<GpIndividual> for TournamentSelection {
    fn select<'a>(&self, population: &'a [GpIndividual], count: usize) -> Vec<&'a GpIndividual> {
        let mut rng = rand::rng();
        let mut selected = Vec::new();

        for _ in 0..count {
            let mut best = &population[rng.random_range(0..population.len())];
            for _ in 1..self.tournament_size {
                let candidate = &population[rng.random_range(0..population.len())];
                if candidate.fitness() > best.fitness() {
                    best = candidate;
                }
            }
            selected.push(best);
        }

        selected
    }
}

// Generational replacement with elitism
pub struct GenerationalReplacement {
    elite_count: usize,
}

impl GenerationalReplacement {
    pub fn new(elite_count: usize) -> Self {
        Self { elite_count }
    }
}

impl Replacement<GpIndividual> for GenerationalReplacement {
    fn replace(
        &self,
        mut current: Vec<GpIndividual>,
        mut offspring: Vec<GpIndividual>,
    ) -> Vec<GpIndividual> {
        current.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        offspring.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_gen: Vec<GpIndividual> =
            current.iter().take(self.elite_count).cloned().collect();
        next_gen.extend(offspring.into_iter().take(current.len() - self.elite_count));

        next_gen
    }
}

// Termination criteria
pub struct GenerationTermination {
    max_generations: usize,
}

impl GenerationTermination {
    pub fn new(max_generations: usize) -> Self {
        Self { max_generations }
    }
}

impl Termination<GpIndividual> for GenerationTermination {
    fn should_terminate(&self, generation: usize, _population: &[GpIndividual]) -> bool {
        generation >= self.max_generations
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tree Based Genetic Programming Demo ===\n");
    println!("Target function: y = x² + 2x + 1\n");

    // Initialize parameters
    let population_size = 100;
    let max_generations = 50;
    let crossover_rate = 0.8;
    let mutation_rate = 0.2;

    // Create initial population
    let mut population = Vec::new();
    for _ in 0..population_size {
        let root = GpNode::Function {
            name: "+".to_string(),
            arity: 2,
            children: vec![
                Rc::new(GpNode::Variable {
                    name: "x".to_string(),
                }),
                Rc::new(GpNode::Terminal { value: 1.0 }),
            ],
        };
        population.push(GpIndividual::new(root));
    }

    // Create evolutionary components
    let fitness_eval = SymbolicRegressionEvaluator::new();
    let selection = TournamentSelection::new(3);
    let crossover = SubtreeCrossover::new(crossover_rate);
    let mutation = SubtreeMutation::new(mutation_rate);
    let replacement = GenerationalReplacement::new(10);
    let termination = GenerationTermination::new(max_generations);

    // Run evolutionary loop
    for generation in 0..max_generations {
        // Evaluate fitness
        fitness_eval.evaluate_population(&mut population);

        // Check termination
        if termination.should_terminate(generation, &population) {
            println!("Termination criteria met at generation {}\n", generation);
            break;
        }

        // Calculate statistics
        let best = population
            .iter()
            .max_by(|a, b| {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let avg_fitness =
            population.iter().map(|i| i.fitness()).sum::<f64>() / population.len() as f64;
        let avg_depth =
            population.iter().map(|i| i.depth() as f64).sum::<f64>() / population.len() as f64;

        println!(
            "Gen {:3}: Best Fitness: {:8.4}, Avg Fitness: {:8.4}, Avg Depth: {:5.2}",
            generation,
            best.fitness(),
            avg_fitness,
            avg_depth
        );

        // Selection
        let parents = selection.select(&population, population_size);

        // Reproduction (crossover)
        let mut offspring = Vec::new();
        for i in (0..parents.len()).step_by(2) {
            if i + 1 < parents.len() {
                let (child1, child2) = crossover.crossover(parents[i], parents[i + 1]);
                offspring.push(child1);
                offspring.push(child2);
            }
        }

        // Ensure correct population size
        while offspring.len() < population_size {
            offspring.push(population[0].clone());
        }
        offspring.truncate(population_size);

        // Apply mutation
        for child in &mut offspring {
            mutation.mutate(child);
        }

        // Replacement
        population = replacement.replace(population, offspring);
    }

    // Final evaluation
    fitness_eval.evaluate_population(&mut population);
    let best = population
        .iter()
        .max_by(|a, b| {
            a.fitness()
                .partial_cmp(&b.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    println!("\n=== Final Results ===");
    println!("Best Fitness: {:.6}", best.fitness());
    println!("Tree Size: {}", best.size());
    println!("Tree Depth: {}", best.depth());
    println!("\nTesting on range [0, 9]:");
    for x in 0..10 {
        let x_f = x as f64;
        let target = x_f * x_f + 2.0 * x_f + 1.0;
        let predicted = best.evaluate(x_f);
        let error = (predicted - target).abs();
        println!(
            "x={:5.1}: target={:8.3}, predicted={:8.3}, error={:8.5}",
            x_f, target, predicted, error
        );
    }

    Ok(())
}
