//! Model traits implemented by every actor and critic used with
//! [`Td3Agent`](super::td3_agent::Td3Agent).
//!
//! TD3 reuses DDPG's actor / critic contracts verbatim — a deterministic
//! policy mapping observations to continuous actions, and a continuous Q
//! critic scoring `(obs, action)` pairs. Both traits live in
//! [`crate::algorithms::ddpg::ddpg_model`]; this module re-exports them so
//! TD3 implementors can `use rlevo_reinforcement_learning::algorithms::td3::td3_model::*` without
//! leaking the DDPG path into their code.
//!
//! The TD3 agent instantiates the [`ContinuousQ`] trait twice (one per
//! critic) and relies on [`DeterministicPolicy`] exactly once for the single
//! shared actor.

pub use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
