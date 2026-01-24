use evorl_core::evolution::{
    Crossover, FitnessEvaluator, Individual, Mutation, Replacement, Selection,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};

// Real-valued individual with strategy parameters for ES
#[derive(Clone, Debug)]
pub struct RealIndividual {
    parameters: Vec<f64>,
    strategy_params: Vec<f64>, // Self-adaptive mutation strengths
    fitness: f64,
}

impl Individual for RealIndividual {
    type Gene = f64;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn genes(&self) -> &[Self::Gene] {
        &self.parameters
    }

    fn genes_mut(&mut self) -> &mut [Self::Gene] {
        &mut self.parameters
    }
}

impl RealIndividual {
    pub fn new(dim: usize, bounds: (f64, f64)) -> Self {
        let mut rng = rand::rng();
        Self {
            parameters: (0..dim)
                .map(|_| rng.random_range(bounds.0..bounds.1))
                .collect(),
            strategy_params: vec![0.1; dim], // Initial mutation strength
            fitness: 0.0,
        }
    }

    pub fn strategy_params(&self) -> &[f64] {
        &self.strategy_params
    }

    pub fn strategy_params_mut(&mut self) -> &mut [f64] {
        &mut self.strategy_params
    }
}

// Self-adaptive Gaussian mutation for ES
pub struct SelfAdaptiveMutation {
    tau: f64,       // Learning rate for strategy parameters
    tau_prime: f64, // Global learning rate
}

impl SelfAdaptiveMutation {
    pub fn new(dimension: usize) -> Self {
        let n = dimension as f64;
        Self {
            tau: 1.0 / (2.0 * n).sqrt(),
            tau_prime: 1.0 / (2.0 * n.sqrt()).sqrt(),
        }
    }
}

impl Mutation<RealIndividual> for SelfAdaptiveMutation {
    fn mutate(&self, individual: &mut RealIndividual) {
        let mut rng = rand::rng();
        let standard_normal = Normal::new(0.0, 1.0).unwrap();

        // Global mutation component
        let global_factor = (self.tau_prime * standard_normal.sample(&mut rng)).exp();

        // Mutate strategy parameters and apply to object parameters
        for i in 0..individual.parameters.len() {
            // Mutate strategy parameter (self-adaptation)
            let local_factor = (self.tau * standard_normal.sample(&mut rng)).exp();
            individual.strategy_params_mut()[i] *= global_factor * local_factor;

            // Ensure minimum mutation strength
            individual.strategy_params_mut()[i] = individual.strategy_params_mut()[i].max(1e-10);

            // Mutate object parameter using adapted strategy
            let mutation = Normal::new(0.0, individual.strategy_params()[i]).unwrap();
            individual.genes_mut()[i] += mutation.sample(&mut rng);
        }
    }

    fn mutation_rate(&self) -> f64 {
        1.0 // ES always mutates
    }
}

// Intermediate recombination for ES
pub struct IntermediateRecombination;

impl Crossover<RealIndividual> for IntermediateRecombination {
    fn crossover(
        &self,
        parent1: &RealIndividual,
        parent2: &RealIndividual,
    ) -> (RealIndividual, RealIndividual) {
        let mut child = parent1.clone();

        // Average parameters
        for i in 0..child.parameters.len() {
            child.genes_mut()[i] = (parent1.genes()[i] + parent2.genes()[i]) / 2.0;
            child.strategy_params_mut()[i] =
                (parent1.strategy_params()[i] + parent2.strategy_params()[i]) / 2.0;
        }

        (child.clone(), child)
    }

    fn crossover_rate(&self) -> f64 {
        1.0 // ES typically always recombines
    }
}

// (μ+λ) replacement strategy
pub struct PlusReplacement {
    mu: usize,
}

impl<I: Individual> Replacement<I> for PlusReplacement {
    fn replace(&self, mut current: Vec<I>, offspring: Vec<I>) -> Vec<I> {
        current.extend(offspring);
        current.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        current.truncate(self.mu);
        current
    }
}

// (μ,λ) replacement strategy
pub struct CommaReplacement {
    mu: usize,
}

impl<I: Individual> Replacement<I> for CommaReplacement {
    fn replace(&self, _current: Vec<I>, mut offspring: Vec<I>) -> Vec<I> {
        offspring.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        offspring.truncate(self.mu);
        offspring
    }
}

// Sphere function fitness evaluator (minimize sum of squares)
pub struct SphereFunction;

impl FitnessEvaluator<RealIndividual> for SphereFunction {
    fn evaluate(&self, individual: &RealIndividual) -> f64 {
        // Sum of squares - lower is better, negate for maximization
        let sum_sq: f64 = individual.genes().iter().map(|x| x * x).sum();
        -sum_sq
    }
}

// Simple tournament selection
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

impl Selection<RealIndividual> for TournamentSelection {
    fn select<'a>(
        &self,
        population: &'a [RealIndividual],
        count: usize,
    ) -> Vec<&'a RealIndividual> {
        let mut rng = rand::rng();
        let mut selected = Vec::new();

        for _ in 0..count {
            let mut best_idx = rng.random_range(0..population.len());
            let mut best_fitness = population[best_idx].fitness();

            for _ in 1..self.tournament_size {
                let idx = rng.random_range(0..population.len());
                if population[idx].fitness() > best_fitness {
                    best_idx = idx;
                    best_fitness = population[idx].fitness();
                }
            }

            selected.push(&population[best_idx]);
        }

        selected
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Real-Valued Evolution Strategy Demo ===\n");

    // Configuration
    let dimension = 5;
    let mu = 15; // Population size
    let lambda = 30; // Offspring size
    let generations = 50;
    let bounds = (-5.0, 5.0);

    // Initialize components
    let fitness_eval = SphereFunction;
    let selection = TournamentSelection::new(3);
    let mutation = SelfAdaptiveMutation::new(dimension);
    let crossover = IntermediateRecombination;
    let replacement = CommaReplacement { mu };

    // Initialize population
    let mut population: Vec<RealIndividual> = (0..mu)
        .map(|_| RealIndividual::new(dimension, bounds))
        .collect();

    // Evaluate initial population
    fitness_eval.evaluate_population(&mut population);

    println!(
        "Generation 0: Best fitness = {:.6}",
        population
            .iter()
            .map(|ind| ind.fitness())
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Main ES loop
    for generation in 1..=generations {
        // Selection: select parents for reproduction
        let parents = selection.select(&population, lambda);

        // Reproduction: create offspring through crossover and mutation
        let mut offspring: Vec<RealIndividual> = Vec::new();
        let mut rng = rand::rng();

        for i in (0..lambda).step_by(2) {
            let parent1 = parents[rng.random_range(0..parents.len())];
            let parent2 = parents[rng.random_range(0..parents.len())];

            // Crossover
            let (mut child1, mut child2) = crossover.crossover(parent1, parent2);

            // Mutation
            mutation.mutate(&mut child1);
            if i + 1 < lambda {
                mutation.mutate(&mut child2);
            }

            offspring.push(child1);
            if i + 1 < lambda {
                offspring.push(child2);
            }
        }

        // Evaluate offspring
        fitness_eval.evaluate_population(&mut offspring);

        // Replacement: (μ,λ) strategy - only keep best from offspring
        population = replacement.replace(population, offspring);

        // Print progress
        let best_fitness = population
            .iter()
            .map(|ind| ind.fitness())
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "Generation {}: Best fitness = {:.6}, Avg strategy param = {:.6}",
            generation,
            best_fitness,
            population[0].strategy_params().iter().sum::<f64>()
                / population[0].strategy_params().len() as f64
        );
    }

    // Find and display best solution
    let best = population
        .iter()
        .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
        .unwrap();

    println!("\n=== Final Results ===");
    println!("Best fitness: {:.6}", best.fitness());
    println!("Best solution:");
    for (i, param) in best.genes().iter().enumerate() {
        println!("  x[{}] = {:.6}", i, param);
    }

    Ok(())
}
