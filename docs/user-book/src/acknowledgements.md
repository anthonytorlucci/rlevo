# Acknowledgements

`rlevo` did not emerge from a vacuum. Much of the library has been influenced 
by ideas, APIs, and community efforts that came before it. This page credits
the projects and people this work depends on most directly.

## Open-Source Libraries

**[Burn](https://github.com/tracel-ai/burn)** is the tensor and neural network
framework that underpins everything in `rlevo` that touches a neural network.
Burn's backend-agnostic design — swap `wgpu` for `flex` with a type parameter
— is the reason `rlevo` can target both GPU and CPU without conditional
compilation scattered through training code. The `burn` team's commitment to
ergonomic Rust API design set the tone for how `rlevo`'s own traits are shaped.
Furthermore, the Burn community has also been a driving force for continued 
learning while developing `rlevo` and we (the author) are very grateful.

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
clear reference implementation of DQN, PPO, and SAC. The clarity of its API design
influenced how `rlevo` structures agent configuration structs.

**[CleanRL](https://github.com/vwxyzjn/cleanrl)** demonstrated that single-file,
dependency-minimal RL implementations are a legitimate research artifact — not a
sign of poor engineering, but a deliberate trade-off that makes algorithms
auditable at a glance. `rlevo`'s example programs aim for the same property:
every hyperparameter visible, every data-flow traceable without jumping between
modules.

**[neat-python](https://github.com/CodeReclaimers/neat-python)** is the canonical
Python implementation of Stanley and Miikkulainen's NEAT algorithm (NeuroEvolution
of Augmenting Topologies). Its treatment of speciation, compatibility distance, and
historical markings served as the primary reference when `rlevo` was evaluating
topology-evolving neuroevolution as a future direction.

**[EvoJAX](https://github.com/google/evojax)** (Google Research) showed that
hardware-accelerated neuroevolution — evaluating thousands of neural network
genomes in parallel on a single accelerator — is practical today, not just a
theoretical speedup. Its task / policy / solver decomposition influenced how
`rlevo` separates `Environment`, genome encoding, and `Strategy`.

**[evosax](https://github.com/RobertTLange/evosax)** provides a comprehensive JAX
library of evolution strategies (CMA-ES, OpenES, PGPE, and many others) with a
uniform `ask`/`tell` interface. Its breadth demonstrated that a single strategy
abstraction can accommodate both gradient-free search and learned parameter
adaptation, reinforcing `rlevo`'s commitment to a single `Strategy` trait rather
than a separate hierarchy per algorithm family.

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
cited throughout this book — Thomas Bäck, Zbigniew Michalewicz, Kalyanmoy Deb,
and Dan Simon among others — created the shared vocabulary that makes it
possible to describe algorithms precisely.

The reinforcement learning community's tradition of open publication — from
Sutton and Barto's freely available second edition to DeepMind and OpenAI
releasing paper code alongside results — made it possible to verify `rlevo`'s
implementations against canonical baselines.

## A Note on Standing on Shoulders

Where this book describes an algorithm, a citation points to the paper or
textbook where it was introduced or given its canonical treatment. The 
[Bibliography](bibliography.md) collects all references in one place.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
