use evorl_core::evolution::{Crossover, Individual, Mutation, Selection};
use rand::Rng;
use rand::prelude::IndexedRandom;
use rand::seq::SliceRandom;

// Binary individual for classic GA
#[derive(Clone, Debug)]
pub struct BinaryIndividual {
    genes: Vec<bool>,
    fitness: f64,
}

impl Individual for BinaryIndividual {
    type Gene = bool;

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

impl BinaryIndividual {
    pub fn new(length: usize) -> Self {
        let mut rng = rand::rng();
        Self {
            genes: (0..length).map(|_| rng.random_bool(0.5)).collect(),
            fitness: 0.0,
        }
    }
}

// Single-point crossover for binary encoding
pub struct SinglePointCrossover {
    rate: f64,
}

impl Crossover<BinaryIndividual> for SinglePointCrossover {
    fn crossover(
        &self,
        parent1: &BinaryIndividual,
        parent2: &BinaryIndividual,
    ) -> (BinaryIndividual, BinaryIndividual) {
        let mut rng = rand::rng();

        if rng.random::<f64>() > self.rate {
            return (parent1.clone(), parent2.clone());
        }

        let point = rng.random_range(1..parent1.genes().len());
        let mut child1_genes = parent1.genes().to_vec();
        let mut child2_genes = parent2.genes().to_vec();

        child1_genes[point..].copy_from_slice(&parent2.genes()[point..]);
        child2_genes[point..].copy_from_slice(&parent1.genes()[point..]);

        (
            BinaryIndividual {
                genes: child1_genes,
                fitness: 0.0,
            },
            BinaryIndividual {
                genes: child2_genes,
                fitness: 0.0,
            },
        )
    }

    fn crossover_rate(&self) -> f64 {
        self.rate
    }
}

// Bit-flip mutation
pub struct BitFlipMutation {
    rate: f64,
}

impl Mutation<BinaryIndividual> for BitFlipMutation {
    fn mutate(&self, individual: &mut BinaryIndividual) {
        let mut rng = rand::rng();
        for gene in individual.genes_mut() {
            if rng.random::<f64>() < self.rate {
                *gene = !*gene;
            }
        }
    }

    fn mutation_rate(&self) -> f64 {
        self.rate
    }
}

// Tournament selection
pub struct TournamentSelection {
    tournament_size: usize,
}

impl<I: Individual> Selection<I> for TournamentSelection {
    fn select<'a>(&self, population: &'a [I], count: usize) -> Vec<&'a I> {
        let mut rng = rand::rng();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let tournament: Vec<&I> = population
                .choose_multiple(&mut rng, self.tournament_size)
                .collect();

            let winner = tournament
                .iter()
                .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
                .unwrap();

            selected.push(*winner);
        }

        selected
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Binary Encoded Genetic Algorithm Demo ===\n");

    // Problem: OneMax - maximize the number of true bits
    let genome_length = 20;
    let population_size = 50;
    let generations = 100;

    // Initialize GA components
    let crossover = SinglePointCrossover { rate: 0.8 };
    let mutation = BitFlipMutation { rate: 0.01 };
    let selection = TournamentSelection { tournament_size: 3 };

    // Create initial population
    let mut population: Vec<BinaryIndividual> = (0..population_size)
        .map(|_| BinaryIndividual::new(genome_length))
        .collect();

    // Fitness function: count number of true bits
    let evaluate = |individual: &mut BinaryIndividual| {
        let fitness = individual.genes().iter().filter(|&&gene| gene).count() as f64;
        individual.set_fitness(fitness);
    };

    // Evolution loop
    for r#gen in 0..generations {
        // Evaluate fitness
        for individual in &mut population {
            evaluate(individual);
        }

        // Find best individual in generation
        let best = population.iter().max_by(|a, b| {
            a.fitness()
                .partial_cmp(&b.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(best_individual) = best {
            if r#gen % 10 == 0 {
                println!(
                    "Generation {}: Best fitness = {:.1} / {}",
                    r#gen,
                    best_individual.fitness(),
                    genome_length
                );
            }
        }

        // Selection
        let parents = selection.select(&population, population_size);

        // Create new population through crossover and mutation
        let mut new_population = Vec::new();
        for i in (0..parents.len()).step_by(2) {
            let (mut child1, mut child2) =
                crossover.crossover(parents[i], parents[(i + 1) % parents.len()]);

            mutation.mutate(&mut child1);
            mutation.mutate(&mut child2);

            new_population.push(child1);
            if new_population.len() < population_size {
                new_population.push(child2);
            }
        }

        // Trim to exact population size if needed
        new_population.truncate(population_size);
        population = new_population;
    }

    // Final evaluation and report
    for individual in &mut population {
        evaluate(individual);
    }

    if let Some(best) = population.iter().max_by(|a, b| {
        a.fitness()
            .partial_cmp(&b.fitness())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "\nFinal result: Best fitness = {:.1} / {}",
            best.fitness(),
            genome_length
        );
        println!(
            "Genome: {}",
            best.genes()
                .iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect::<String>()
        );
    }

    Ok(())
}
