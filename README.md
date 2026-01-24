# burn-evorl

> [!CAUTION]
> **Alpha Stage**: This project is in the early stages of development. Currently focusing on project structure and design.

Evolutionary Deep Reinforcement Learning library written entirely in the Rust programming language. It uses [burn](https://github.com/tracel-ai/burn) for tensor operations and neural network architectures for deep learning.

## Project Structure
```
burn-evorl/
|-- Cargo.toml (workspace)
|-- crates/
    |-- evorl-core/
        |-- Cargo.toml
        |-- src/
    |-- evorl-evolution/
        |-- Cargo.toml
        |-- src/
            |-- algorithms/        // ES, GA, etc.
    |-- evorl-rl/
        |-- Cargo.toml
        |-- src/
            |-- algorithms/        // Deep Q-Networks, Double DQN, Dueling DQN, Rainbow, REINFORCE, Trust Region Policy Optimization, Proximal Policy Optimization, Soft Actor-Critic, Advantage Actor-Critic, Deep Deterministic Policy Gradient, AlphaZero and MuZero
                |-- dqn\
            |-- environment/       // Sepcific RL environments, e.g. chess, connect4, cartpole, etc.
          |-- utils.rs             // utility functions specific to reinforcement learning
    |-- evorl-hybrid/
        |-- Cargo.toml
        |-- src/
            |-- lib.rs 
            |-- evorl.rs           // Combined approaches
            |-- strategies.rs      // Population-based training, etc.
    ├── evorl-envs/                // Environment implementations
        ├── Cargo.toml
        └── src/
            ├── lib.rs
            ├── classic/           // Classic control problems
            ├── games/             // Board games
                ├── mod.rs
                ├── chess/
                └── connect4.rs
            └── benchmarks/        // Standard benchmarks
                └── mod.rs
    └── evorl-utils/                   // Utilities crate
        ├── Cargo.toml
        └── src/
            ├── lib.rs
            ├── math.rs                // Mathematical utilities
            ├── logging.rs             // Logging helpers
            ├── validations.rs         // Data validation
            └── serialization.rs         // Data validation      
```

## Crates

### `evorl-core`
This crate provides generic triats and the core components for the evolutionary reinforcement learning library. It forms the base for all other crates.

### `evorl-evolution`
This crate is the evoluionary engine. It manages the population

#### Key Components:
- **Population Manager**: A container for the current generation of agents.
- **Selection Strategies**:
  - _Tournament Selection_: Picking the best from a random subset.
  - _Elite Preservation_: Automatically carrying the top 1% to the next generation.
- **Mutation Operators**:
  - _Gaussian Noise_: Adding small random values to weights.
  - _Crossover_: Swapping "genes" between two successful agents.

### `evorl-rl`
This crate implements the standard components for reinforcement learning with a focus on the use of deep neural networks.

#### Key Components:
**1. The Environment Interface (The World)**
The environment provides the context for the agent. In DRL, this is typically modeled as a Markov Decision Process (MDP).
  - **Standardized API**: Most libraries adopt a "Step" and "Reset" pattern (popularized by OpenAI Gymnasium).
  - **Space Definitions**: Formal definitions for Observation Spaces (what the agent sees) and Action Spaces (what the agent can do), often categorized as Discrete or Continuous.
  - **Vectorization**: A critical component for modern DRL is the ability to run multiple environment instances in parallel to collect data faster.

**2. The Agent & Policy (The Brain)**
The Agent is the wrapper that contains the logic for interacting with the environment, while the Policy is the specific mathematical function (usually a Neural Network) that maps observations to actions.
  - **Actor-Critic Architectures**: Many libraries separate the "Actor" (which chooses actions) from the "Critic" (which estimates the value of states).
  - **Exploration Strategy**: A module that manages the trade-off between trying new things (Exploration) and using known good actions (Exploitation), such as $\epsilon$-greedy or entropy bonuses.

**3. Data Storage: Replay Buffers**
Deep RL algorithms, especially "off-policy" ones like DQN or SAC, are notoriously unstable. Replay buffers help solve this by breaking the correlation between consecutive experiences.
  - **Experience Tuples**: A storage system for $(s, a, r, s', d)$ — state, action, reward, next state, and "done" flag.
  - **Prioritized Sampling**: Advanced libraries implement "Prioritized Experience Replay" (PER), which samples "surprising" transitions more frequently to speed up learning.

**4. Algorithm Implementations (The Logic)**
This is the library's core "zoo" of algorithms. These are usually categorized into:
- **Value-Based**
  - Examples: DQN, Rainbow 
  - Key Characteristics: Learns to estimate the value of actions ($Q$-values)
- **Policy Gradient** 
  - Examples: PPO, REINFORCE 
  - Key Characteristics: Directly optimizes the action probabilities
- **Actor-Critic** 
  - Examples: A2C, SAC, TD3 
  - Key Characteristics: Combines both value estimation and policy optimization

**5. Training Infrasctructure (The Engine)**
This component manages the actual optimization process.
- **Gradient Updates**: Integration with a deep learning backend `burn` to perform backpropagation.
- **Target Networks**: Mechanisms to maintain "frozen" copies of neural networks to stabilize the $Q$-value targets during training.
- **Schedulers**: Logic to decay hyperparameters like the learning rate or exploration rate over time.

**6. Logger and Evaluation (The Dashboard)**
DRL is highly sensitive to hyperparameters, making visualization essential.
- **Metrics Tracking**: Real-time logging of "Mean Reward," "Episode Length," and "Loss."
- **Checkpointing**: Automatically saving model weights at specific intervals.
- **Evaluation Runner**: A separate loop that runs the agent in a deterministic mode (no exploration) to gauge its "true" performance.

### `evorl-hybrid`
This crate combines deep reinforcement learning algorithms with evolutionary optimization algorithms.

#### Key Components:
**1. The Genome (The Genotype)**
The `Genome` is the raw data that describes an agent. In most Rust implementations, this is represented as a "flat" structure that is easy to mutate and cross-breed.
  - **Data Representation**: Usually a `Vec<f64>` or `ndarray::Array1`.
  - **Encoding**: It maps these raw numbers to specific neural network parameters (weights and biases).
  - **Serialization**: Crucial for saving the "best" agents to disk (using `serde`) so they can be reloaded later.

**2. The Agent (The Phenotype)**
While the Genome is the blueprint, the `Agent` is the actual living instance that interacts with the environment. It translates the Genome into behavior.
  - **Policy Mapping**: It contains the logic to feed environmental observations into a neural network and output an action.
  - **Internal State**: Some agents might have "memory" (like LSTM or GRU layers) that needs to be reset between evaluation runs.
  - **Decoupling**: The Agent should not know how it was evolved; it only knows how to act based on its current parameters.

**3. The Fitness Evaluator**
In RL, we don't have "correct" labels; we have rewards. The Fitness Evaluator is the bridge that converts environment rewards into a single scalar value that the evolutionary engine can use to rank individuals.
  - **Simulation Loop**: It handles the boilerplate of resetting the environment, stepping through frames, and accumulating rewards.
  - **Objective Functions**: You might implement different ways to calculate fitness (e.g., average reward over 5 runs vs. the single best run to account for noise).

**4. The Population Orchestrator**
This component manages the "society" of agents. It handles the lifecycle of a generation.
  - **Generation Tracking**: Keeping track of the current generation number and the "Hall of Fame" (the best agents ever found).
  - **Parallel Dispatch**: Because evaluating 100 agents is independent, the orchestrator uses libraries like Rayon to spread agents across all available CPU cores.

#### Summary Table
- **Genome**
  - Responsibility: Stores raw numeric traits
  - Analogous To: DNA / Blueprint
- **Agent**
  - Responsibility: Performs inference/actions
  - Analogous To: The physical body
- **Fitness**
  - Responsibility: Measures performance
  - Analogous To: Natural Selection
- **Orchestrator**
  - Responsibility: Manages the group
  - Analogous To: The Ecosystem
