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

pub mod action;
pub mod agent;
pub mod base;
pub mod environment;
pub mod evolution;
pub mod experience;
pub mod memory;
pub mod metrics;
pub mod reward;
pub mod state;
