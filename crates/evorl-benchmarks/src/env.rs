//! Minimal environment interface consumed by the evaluator.
//!
//! `BenchEnv` is intentionally narrower than `evorl_core::Environment` so
//! the harness does not have to thread const-generic dimensions through its
//! signatures. Adapters that wrap concrete `Environment` impls live in
//! `evorl-envs`.

#[derive(Debug, Clone)]
pub struct BenchStep<Obs> {
    pub observation: Obs,
    pub reward: f64,
    pub done: bool,
}

pub trait BenchEnv {
    type Observation;
    type Action;

    fn reset(&mut self) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> BenchStep<Self::Observation>;
}
