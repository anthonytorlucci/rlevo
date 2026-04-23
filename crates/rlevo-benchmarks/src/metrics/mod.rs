//! Metric primitives and aggregators.

pub mod core;
pub mod ea;
pub mod rl;

#[derive(Debug, Clone)]
pub enum Metric {
    Scalar { name: String, value: f64 },
    Histogram { name: String, values: Vec<f64> },
    Counter { name: String, count: u64 },
}

/// Trait implemented by agents (and internal collectors) that can report
/// method-specific metrics at trial boundaries.
pub trait MetricsProvider {
    fn emit(&self) -> Vec<Metric>;
}
