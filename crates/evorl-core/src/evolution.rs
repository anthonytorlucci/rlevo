use std::fmt::Debug;

// Core trait for representing individuals in the population
pub trait Individual: Clone + Debug {
    type Gene: Clone;

    fn fitness(&self) -> f64;
    fn set_fitness(&mut self, fitness: f64);
    fn genes(&self) -> &[Self::Gene];
    fn genes_mut(&mut self) -> &mut [Self::Gene];
}

// Trait for encoding schemes - converting between problem and genetic representation
pub trait Encoding<Problem, Genome> {
    fn encode(&self, problem: &Problem) -> Genome;
    fn decode(&self, genome: &Genome) -> Problem;
}

// Trait for fitness evaluation
pub trait FitnessEvaluator<I: Individual> {
    fn evaluate(&self, individual: &I) -> f64;
    fn evaluate_population(&self, population: &mut [I]) {
        for individual in population.iter_mut() {
            let fitness = self.evaluate(individual);
            individual.set_fitness(fitness);
        }
    }
}

// Trait for selection strategies
pub trait Selection<I: Individual> {
    fn select<'a>(&self, population: &'a [I], count: usize) -> Vec<&'a I>;
    fn select_one<'a>(&self, population: &'a [I]) -> &'a I {
        &self.select(population, 1)[0]
    }
}

// Trait for crossover operations
pub trait Crossover<I: Individual> {
    fn crossover(&self, parent1: &I, parent2: &I) -> (I, I);
    fn crossover_rate(&self) -> f64;
}

// Trait for mutation operations
pub trait Mutation<I: Individual> {
    fn mutate(&self, individual: &mut I);
    fn mutation_rate(&self) -> f64;
}

// Trait for reproduction - combines crossover and mutation
pub trait Reproduction<I: Individual> {
    fn reproduce(&self, parents: &[&I]) -> Vec<I>;
}

// Trait for population management and replacement strategies
pub trait Replacement<I: Individual> {
    fn replace(&self, current: Vec<I>, offspring: Vec<I>) -> Vec<I>;
}

// Trait for termination criteria
pub trait Termination<I: Individual> {
    fn should_terminate(&self, generation: usize, population: &[I]) -> bool;
}

// Main evolutionary algorithm trait
pub trait EvolutionaryAlgorithm<I: Individual> {
    fn initialize_population(&self, size: usize) -> Vec<I>;
    fn evolve(&mut self, population: Vec<I>) -> Vec<I>;
    fn run(&mut self, population_size: usize, max_generations: usize) -> I;
}

// More sophisticated individual trait with associated types
pub trait AdvancedIndividual: Clone + Debug {
    type Genome: Clone + Debug;
    type Fitness: PartialOrd + Clone + Debug;

    fn genome(&self) -> &Self::Genome;
    fn genome_mut(&mut self) -> &mut Self::Genome;
    fn fitness(&self) -> &Self::Fitness;
    fn set_fitness(&mut self, fitness: Self::Fitness);
}

// Multi-objective fitness
#[derive(Clone, Debug, PartialEq)]
pub struct MultiObjectiveFitness {
    objectives: Vec<f64>,
}

impl PartialOrd for MultiObjectiveFitness {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Pareto dominance comparison
        let mut dominates = false;
        let mut dominated = false;

        for (a, b) in self.objectives.iter().zip(&other.objectives) {
            if a > b {
                dominates = true;
            } else if a < b {
                dominated = true;
            }
        }

        if dominates && !dominated {
            Some(std::cmp::Ordering::Greater)
        } else if dominated && !dominates {
            Some(std::cmp::Ordering::Less)
        } else if !dominates && !dominated {
            Some(std::cmp::Ordering::Equal)
        } else {
            None // Non-comparable (both dominate in different objectives)
        }
    }
}
