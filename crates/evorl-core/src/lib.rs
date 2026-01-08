//! Core types and traits for evolutionary reinforcement learning.
//!
//! This crate provides the foundational abstractions for building RL agents
//! and environments, designed around traits and enums rather than inheritance.
//!
//! # Overview
//!
//! The core library defines three main concepts:
//! - **Agent**: Learns to maximize rewards through interaction with an environment
//! - **Environment**: Defines the problem domain and state/action spaces
//! - **Model**: Neural network or policy function used by the agent
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌──────────┐
//! │   Agent     │────→│ Environment  │────→│  Model   │
//! │ (Learner)   │     │ (Problem)    │     │ (Policy) │
//! └─────────────┘     └──────────────┘     └──────────┘
//! ```
//!
//! # Example
//!
//! ```no_run
//! use evorl_core::agent::Agent;
//! use evorl_core::environment::Environment;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create environment
//! let mut env = your_env::MyEnvironment::new()?;
//!
//! // Create agent
//! let mut agent = your_agent::MyAgent::new()?;
//!
//! // Training loop
//! for episode in 0..1000 {
//!     let state = env.reset()?;
//!     // ... interact with environment
//! }
//! # Ok(())
//! # }
//! ```

pub mod action;
pub mod agent;
pub mod environment;
pub mod memory;
pub mod metrics;
pub mod model;
pub mod state;
