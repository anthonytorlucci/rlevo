//! Metric name constants for RL agent state emitted via `emit_metrics`.
//!
//! RL agents populate these via `BenchableAgent::emit_metrics`. The harness
//! does not compute them — they come from whatever state the frozen policy
//! chooses to expose (e.g. the epsilon at training end, the LR schedule value).
//! Use these constants rather than raw string literals so that name changes
//! are caught at compile time.

/// Metric key for the policy gradient loss — emitted by on-policy algorithms (PPO, A2C).
pub const POLICY_LOSS: &str = "rl/policy_loss";
/// Metric key for the value-function loss — emitted by actor-critic algorithms.
pub const VALUE_LOSS: &str = "rl/value_loss";
/// Metric key for the approximate KL divergence between old and new policy — PPO stability signal.
pub const APPROX_KL: &str = "rl/approx_kl";
/// Metric key for the policy entropy — higher values indicate more exploratory behaviour.
pub const ENTROPY: &str = "rl/entropy";
/// Metric key for the ε-greedy exploration rate — emitted by value-based agents (DQN, etc.).
pub const EPSILON: &str = "rl/epsilon";
/// Metric key for the current learning rate — useful when a LR schedule is active.
pub const LEARNING_RATE: &str = "rl/learning_rate";
