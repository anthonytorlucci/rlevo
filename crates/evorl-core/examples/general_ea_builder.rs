use evorl_core::evolution::{
    Crossover, EvolutionaryAlgorithm, FitnessEvaluator, Individual, Mutation, Replacement,
    Selection, Termination,
};
use rand::Rng;

// Generic EA that composes different components
pub struct GenericEA<I, S, C, M, R, T>
where
    I: Individual,
    S: Selection<I>,
    C: Crossover<I>,
    M: Mutation<I>,
    R: Replacement<I>,
    T: Termination<I>,
{
    selection: S,
    crossover: C,
    mutation: M,
    replacement: R,
    termination: T,
    _phantom: std::marker::PhantomData<I>,
}

impl<I, S, C, M, R, T> GenericEA<I, S, C, M, R, T>
where
    I: Individual,
    S: Selection<I>,
    C: Crossover<I>,
    M: Mutation<I>,
    R: Replacement<I>,
    T: Termination<I>,
{
    pub fn new(selection: S, crossover: C, mutation: M, replacement: R, termination: T) -> Self {
        Self {
            selection,
            crossover,
            mutation,
            replacement,
            termination,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I, S, C, M, R, T> EvolutionaryAlgorithm<I> for GenericEA<I, S, C, M, R, T>
where
    I: Individual,
    S: Selection<I>,
    C: Crossover<I>,
    M: Mutation<I>,
    R: Replacement<I>,
    T: Termination<I>,
{
    fn initialize_population(&self, _size: usize) -> Vec<I> {
        // This would need to be customized per individual type
        // In practice, you'd pass an initializer trait
        vec![]
    }

    fn evolve(&mut self, population: Vec<I>) -> Vec<I> {
        let pop_size = population.len();
        let mut offspring = Vec::with_capacity(pop_size);

        // Generate offspring
        while offspring.len() < pop_size {
            let parent1 = self.selection.select_one(&population);
            let parent2 = self.selection.select_one(&population);

            let (mut child1, mut child2) = self.crossover.crossover(parent1, parent2);

            self.mutation.mutate(&mut child1);
            self.mutation.mutate(&mut child2);

            offspring.push(child1);
            if offspring.len() < pop_size {
                offspring.push(child2);
            }
        }

        // Replace old population with offspring
        self.replacement.replace(population, offspring)
    }

    fn run(&mut self, population_size: usize, max_generations: usize) -> I {
        let mut population = self.initialize_population(population_size);
        let mut generation = 0;

        while generation < max_generations
            && !self.termination.should_terminate(generation, &population)
        {
            population = self.evolve(population);
            generation += 1;
        }

        population
            .into_iter()
            .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
            .unwrap()
    }
}

// Termination based on max generations or fitness threshold
#[derive(Clone)]
pub struct StandardTermination {
    fitness_threshold: Option<f64>,
}

impl<I: Individual> Termination<I> for StandardTermination {
    fn should_terminate(&self, _generation: usize, population: &[I]) -> bool {
        if let Some(threshold) = self.fitness_threshold {
            population.iter().any(|ind| ind.fitness() >= threshold)
        } else {
            false
        }
    }
}

// --------------------------------------------------------------------------
// Concrete implementations for the example
// --------------------------------------------------------------------------

/// Simple bit-string individual for maximization
#[derive(Clone, Debug)]
struct BitStringIndividual {
    genes: Vec<u8>,
    fitness: f64,
}

impl Individual for BitStringIndividual {
    type Gene = u8;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn genes(&self) -> &[Self::Gene] {
        &self.genes
    }

    fn genes_mut(&mut self) -> &mut [Self::Gene] {
        &mut self.genes
    }
}

/// Tournament selection implementation
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

impl Selection<BitStringIndividual> for TournamentSelection {
    fn select<'a>(
        &self,
        population: &'a [BitStringIndividual],
        count: usize,
    ) -> Vec<&'a BitStringIndividual> {
        let mut rng = rand::thread_rng();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let first_idx = rng.gen_range(0..population.len());
            let mut best_idx = first_idx;
            let mut best_fitness = population[first_idx].fitness();

            for _ in 1..self.tournament_size {
                let idx = rng.gen_range(0..population.len());
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

/// Uniform crossover implementation
pub struct UniformCrossover {
    rate: f64,
}

impl UniformCrossover {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

impl Crossover<BitStringIndividual> for UniformCrossover {
    fn crossover(
        &self,
        parent1: &BitStringIndividual,
        parent2: &BitStringIndividual,
    ) -> (BitStringIndividual, BitStringIndividual) {
        let mut rng = rand::thread_rng();
        let mut child1_genes = parent1.genes.clone();
        let mut child2_genes = parent2.genes.clone();

        for i in 0..child1_genes.len() {
            if rng.r#gen::<f64>() < 0.5 {
                (child1_genes[i], child2_genes[i]) = (child2_genes[i], child1_genes[i]);
            }
        }

        (
            BitStringIndividual {
                genes: child1_genes,
                fitness: 0.0,
            },
            BitStringIndividual {
                genes: child2_genes,
                fitness: 0.0,
            },
        )
    }

    fn crossover_rate(&self) -> f64 {
        self.rate
    }
}

/// Bit-flip mutation implementation
pub struct BitFlipMutation {
    rate: f64,
}

impl BitFlipMutation {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

impl Mutation<BitStringIndividual> for BitFlipMutation {
    fn mutate(&self, individual: &mut BitStringIndividual) {
        let mut rng = rand::thread_rng();
        for gene in individual.genes_mut() {
            if rng.r#gen::<f64>() < self.rate {
                *gene = if *gene == 0 { 1 } else { 0 };
            }
        }
    }

    fn mutation_rate(&self) -> f64 {
        self.rate
    }
}

/// Generational replacement - offspring replace entire population
pub struct GenerationalReplacement;

impl Replacement<BitStringIndividual> for GenerationalReplacement {
    fn replace(
        &self,
        _current: Vec<BitStringIndividual>,
        offspring: Vec<BitStringIndividual>,
    ) -> Vec<BitStringIndividual> {
        offspring
    }
}

/// Fitness evaluator - counts ones in bit string
pub struct OnesCountEvaluator;

impl FitnessEvaluator<BitStringIndividual> for OnesCountEvaluator {
    fn evaluate(&self, individual: &BitStringIndividual) -> f64 {
        individual.genes().iter().map(|&g| g as f64).sum()
    }
}

/// Builder pattern for configuration
pub struct EABuilder<I: Individual> {
    population_size: usize,
    max_generations: usize,
    _phantom: std::marker::PhantomData<I>,
}

impl<I: Individual> EABuilder<I> {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            max_generations: 1000,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    pub fn max_generations(mut self, gens: usize) -> Self {
        self.max_generations = gens;
        self
    }

    pub fn build<S, C, M, R, T>(
        self,
        selection: S,
        crossover: C,
        mutation: M,
        replacement: R,
        termination: T,
    ) -> GenericEA<I, S, C, M, R, T>
    where
        S: Selection<I>,
        C: Crossover<I>,
        M: Mutation<I>,
        R: Replacement<I>,
        T: Termination<I>,
    {
        GenericEA::new(selection, crossover, mutation, replacement, termination)
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Genetic Algorithm Demo - Maximizing Ones in Bit String ===\n");

    let population_size = 50;
    let max_generations = 100;
    let gene_length = 20;
    let target_fitness = gene_length as f64;

    // Configure the EA using the builder pattern
    let ea_builder = EABuilder::<BitStringIndividual>::new()
        .population_size(population_size)
        .max_generations(max_generations);

    // Create concrete components implementing each trait
    let selection = TournamentSelection::new(3);
    let crossover = UniformCrossover::new(1.0);
    let mutation = BitFlipMutation::new(0.01);
    let replacement = GenerationalReplacement;
    let termination = StandardTermination {
        fitness_threshold: Some(target_fitness),
    };

    // Build the EA using the builder pattern
    let mut ea = ea_builder.build(selection, crossover, mutation, replacement, termination);

    // Initialize population with random bit strings
    let mut population: Vec<BitStringIndividual> = (0..population_size)
        .map(|_| {
            let mut rng = rand::thread_rng();
            BitStringIndividual {
                genes: (0..gene_length).map(|_| rng.gen_range(0..=1)).collect(),
                fitness: 0.0,
            }
        })
        .collect();

    // Create fitness evaluator and evaluate initial population
    let evaluator = OnesCountEvaluator;
    evaluator.evaluate_population(&mut population);

    // Display initial population statistics
    let best_fitness = population
        .iter()
        .map(|i| i.fitness())
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_fitness = population.iter().map(|i| i.fitness()).sum::<f64>() / population.len() as f64;

    println!("Initial population (generation 0):");
    println!("  Population size: {}", population_size);
    println!("  Best fitness: {:.1} / {}", best_fitness, gene_length);
    println!("  Average fitness: {:.1}\n", avg_fitness);

    // Run evolution loop
    let mut generation = 0;
    while generation < max_generations {
        // Check termination condition
        if population.iter().any(|ind| ind.fitness() >= target_fitness) {
            break;
        }

        // Evolve the population through one generation
        population = ea.evolve(population);
        evaluator.evaluate_population(&mut population);

        // Periodically report progress
        if generation % 10 == 0 || generation == max_generations - 1 {
            let best = population
                .iter()
                .map(|i| i.fitness())
                .fold(f64::NEG_INFINITY, f64::max);
            let avg = population.iter().map(|i| i.fitness()).sum::<f64>() / population.len() as f64;
            println!(
                "Generation {:3}: best = {:.1} / {}, avg = {:.1}",
                generation, best, gene_length, avg
            );
        }

        generation += 1;
    }

    // Display final results
    let best_individual = population
        .iter()
        .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
        .unwrap();

    println!("\n=== Final Results ===");
    println!("Total generations: {}", generation);
    println!(
        "Best fitness: {:.0} / {}",
        best_individual.fitness(),
        gene_length
    );
    println!(
        "Best genome: {}",
        best_individual
            .genes()
            .iter()
            .map(|g| g.to_string())
            .collect::<String>()
    );

    if best_individual.fitness() == gene_length as f64 {
        println!("✓ Success! Found optimal solution with all ones.");
    } else {
        println!(
            "Terminated after {} generations without reaching optimum.",
            generation
        );
    }

    Ok(())
}
