# Part I — Foundations: The "How it Works" Bit

Welcome to the engine room. 

Before you start cranking out populations of neural networks or tuning PPO 
hyperparameters, it’s worth taking a second to understand how `rlevo` actually 
thinks about the world. If you've used Python-based RL libs, most of this will 
feel familiar, but we're doing a few things differently here—mostly because we 
have the luxury of Rust’s type system and Burn’s backend flexibility.

<!-- todo! discuss here the const generics and what makes rlevo different -->

At its core, `rlevo` is a marriage between two different philosophies of 
"getting better at something."

### 1. The Gradient Path (Deep RL)
 
When you use algorithms like **PPO**, **SAC**, or **DQN**, you're dealing with 
a single agent (or a small batch) using calculus to nudge network weights in 
the direction of higher rewards. It’s fast and efficient, but as any RL 
practitioner knows, it’s incredibly easy to get stuck in a "local optimum" - a 
little hill that feels like the summit but is actually just a bump in the road.

**In `rlevo`, this looks like:**
- **Policies & Value Functions:** Neural networks built with Burn.
- **Experience Replay:** Buffers that store transitions so we can learn from the past without forgetting it.
- **The Gradient Loop:** Standard backpropagation guided by reward signals.

### 2. The Survival Path (Evolutionary Computing)

Think of evolutionary computing as nature’s way of crowdsourcing the perfect 
solution, where we swap out manual tuning for a survival-of-the-fittest 
sandbox. Instead of putting all our chips on a single, lonely mathematical 
guess and hoping it climbs the right hill, population-based optimization lets 
us throw an entire diverse collective of candidate solutions - our 
"population" - into the wild at once. Over successive generations, these 
candidates compete, swap their best traits through crossover, and occasionally 
mutate in weird, unexpected ways, naturally driving the whole group toward 
global optima that traditional gradient descent might completely miss. It’s 
messy, highly parallelizable, and honestly a bit chaotic, but when you're 
dealing with rugged, non-differentiable search spaces where you don't even know 
what the perfect answer looks like, letting a digital ecosystem evolve the 
solution provides a genuinely unfair advantage — because while your 
gradient-based colleagues are carefully tiptoeing down one hill, your 
population is simultaneously faceplanting on thousands of them, and it turns 
out faceplanting at scale is a surprisingly effective search strategy.

**In `rlevo`, this looks like:**
- **The Population:** A collection of agents, often represented as a large 
  tensor for raw throughput.
- **Fitness:** Instead of step-by-step rewards, we look at the *episodic 
  return* (the total score).
- **Genetic Operators:** Crossover and mutation functions that shuffle weights 
  around without needing a single derivative.

### 3. The Hybrid Space: Where it gets interesting

The "secret sauce" of `rlevo` is that we don't think you should have to choose. 
We’re implementing hybrid strategies where a population evolves to explore the 
map, and gradient descent is used to refine the winners. It’s essentially 
using evolution to find the right mountain and gradients to climb to the very 
top of it.

### 4. The "Rust" Layer (The Safety Net)

You'll notice a lot of talk in this guide about `State<SR>`, `Observation<R>`, 
and `Action<AR>`. 

In most RL libraries, you find out your observation tensor is the wrong size 
when your training run crashes at runtime. We hate that. An another option is 
being forced to use a flattened vector to represent a high-dimensional space. 
All structural information is lost or, at minimum, obfuscated! By using 
**const generics**, we've baked the dimensions of your environment directly 
into the types. If your network expects a 8 x 8 x 111 input but your 
environment provides 64, `rlevo` won't let you compile the code. 

It might feel like fighting the borrow checker or the type system at first, but 
it means that once your code runs, the "plumbing" is guaranteed to be correct.

>[!note] PettingZoo - Chess
> The chess environment in [PettingZoo](https://pettingzoo.farama.org/environments/classic/chess/) has an (8,8,11) observation space.

---

Where an algorithm or derivation deserves more than a summary, a callout box points to the relevant appendix.

**You do not have to read this part before Part II.** If you learn better by doing, start with the tour and come back here when a term needs grounding. The cross-references work in both directions.

## What is covered

| Section | Core idea |
| ------- | --------- |
| [What Is Optimization?](10-optimization.md) | Fitness landscapes, the exploitation–exploration trade-off, and why gradient descent is not always the answer |
| [Evolutionary Computation](20-evolutionary-computation.md) | Populations, selection, variation operators, and the family of algorithms that descend from Holland's genetic algorithm |
| [Reinforcement Learning](30-reinforcement-learning.md) | The agent–environment loop, Markov decision processes, value functions, and the road from Q-learning to deep RL |
| [Why Combine Them?](40-why-combine.md) | Neuroevolution, evolutionary RL, and what happens when you let evolution drive gradient-based agents |

---

*Co-Authored-By: Anthropic Claude Sonnet 4.6, Google Gemini Flash 3.5*\
*Reviewed-By: (Human) Anthony Torlucci*
