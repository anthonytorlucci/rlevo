# Acknowledgements

`rlevo` did not emerge from a vacuum. Every abstraction in this library traces
back to ideas, APIs, and community effort that came before it. This page credits
the projects and people this work depends on most directly.

## Open-Source Libraries

**[Burn](https://github.com/tracel-ai/burn)** is the tensor and neural network
framework that underpins everything in `rlevo` that touches a neural network.
Burn's backend-agnostic design — swap `wgpu` for `ndarray` with a type parameter
— is the reason `rlevo` can target both CPU and GPU without conditional
compilation scattered through training code. The `burn` team's commitment to
ergonomic Rust API design set the tone for how `rlevo`'s own traits are shaped.

**[OpenAI Gym](https://github.com/openai/gym)** (now
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium), maintained by the
Farama Foundation) established the `reset` / `step` / `observation, reward, done`
interface that almost every RL library since 2016 has converged on. `rlevo`'s
`Environment` trait is a statically-typed, const-generic Rust translation of that
contract.

**[DEAP](https://github.com/deap/deap)** (Distributed Evolutionary Algorithms in
Python) showed that evolutionary computation deserves a first-class software
abstraction — not just a script — and that the ask/tell pattern scales to complex
operator pipelines. `rlevo`'s `Strategy` trait draws directly from this
philosophy.

**[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)** provided a
clear reference implementation of DQN, PPO, and SAC that `rlevo` used as a
correctness anchor during algorithm development. The clarity of its API design
influenced how `rlevo` structures agent configuration structs.

**[rand](https://github.com/rust-random/rand)** and the Rust random number
ecosystem underpin `rlevo`'s reproducible seeding strategy. The `SeedableRng`
trait made it possible to derive every stochastic draw in an evolutionary run
from a single root seed, which is essential for reproducible research.

**[rayon](https://github.com/rayon-rs/rayon)** makes parallelism in Rust
ergonomic enough that population evaluation — the dominant cost in evolutionary
computation — can be parallelised with a one-line change.

**[serde](https://github.com/serde-rs/serde)** provides the serialisation
layer that makes run records, manifests, and checkpoints portable across
machines and runtimes.

## Research Communities

The evolutionary computation community has been building, benchmarking, and
critiquing population-based optimisation methods for over five decades. The CEC
benchmark suite, the GECCO conference series, and the authors of the textbooks
cited throughout this book — Thomas Back, Zbigniew Michalewicz, Kalyanmoy Deb,
and Nils Reimann among others — created the shared vocabulary that makes it
possible to describe algorithms precisely.

The reinforcement learning community's tradition of open publication — from
Sutton and Barto's freely available second edition to DeepMind and OpenAI
releasing paper code alongside results — made it possible to verify `rlevo`'s
implementations against canonical baselines.

## A Note on Standing on Shoulders

Where this book describes an algorithm, a citation points to the paper or
textbook where it was introduced or given its canonical treatment. We have aimed
to be specific: a vague "as described in the literature" is less useful than
"Watkins & Dayan (1992), Theorem 1." The [Bibliography](bibliography.md) collects
all references in one place.
