//! Method-specific metric name constants for RL agents.
//!
//! RL agents populate these via `BenchableAgent::emit_metrics`. The harness
//! does not compute them — they come from whatever state the frozen policy
//! chooses to expose (e.g. the epsilon at training end, the LR schedule value).

pub const POLICY_LOSS: &str = "rl/policy_loss";
pub const VALUE_LOSS: &str = "rl/value_loss";
pub const APPROX_KL: &str = "rl/approx_kl";
pub const ENTROPY: &str = "rl/entropy";
pub const EPSILON: &str = "rl/epsilon";
pub const LEARNING_RATE: &str = "rl/learning_rate";
