# What Is Optimisation?

Every algorithm in `rlevo` — a genetic algorithm searching a fitness landscape,
a DQN agent chasing reward, a hybrid doing both — is, underneath, doing the same
thing: searching for the input that makes some function come out best. Before we
introduce a single trait, it is worth pinning down what that means, because the
vocabulary splits along community lines and the sign conventions are a classic
source of bugs. This chapter fixes both, then names the three questions every
search algorithm must answer — questions the rest of Part I spends its time
answering in detail.

## The one idea

Suppose you are tuning a single knob — the learning rate of a network, say — and
you can score any setting by running a short training job and reading off the
final loss. You want the setting that scores best. That is optimisation in
miniature: a space of inputs, a function that scores them, and a search for the
input with the best score. Formally, we want

```math
\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})
```

where \\(f : \mathcal{X} \to \mathbb{R}\\) is the **objective function** (also
called a **fitness function**, a **cost function**, or a **loss**) and
\\(\mathcal{X}\\) is the **search space**. The single knob above is a
one-dimensional \\(\mathcal{X}\\); a neural network's weights are a
\\(\mathcal{X}\\) with millions of dimensions, but the shape of the problem is
identical. Everything else in optimisation — every algorithm in this book — is a
strategy for searching \\(\mathcal{X}\\) efficiently when you cannot afford to
score every point in it.

### A note on terminology (or: why everyone calls it something different)

The four terms above all name the same object — a scalar-valued function you are
trying to push to an extreme. The terminology tells you which community is
speaking and, crucially, **which direction they are pushing**. *If you have ever
optimised in the wrong direction, you know why this matters.*

- **Loss** / **cost function** is deep learning's dialect. You are minimising
  it, it is differentiable (or you are pretending it is), and your optimiser is
  almost certainly some flavour of SGD. Mean squared error, cross-entropy, Huber
  loss — all losses. The distinction between "loss" and "cost" is mostly vibes:
  loss tends to refer to a per-sample quantity, cost to its aggregate over a
  batch or dataset, though nobody enforces this at the border.

- **Objective function** is the neutral, field-agnostic term. Operations
  research, convex optimisation, and anyone who wants to sound rigorous without
  committing to a community uses this one. It can be minimised *or* maximised,
  which makes it the honest choice when the direction is not obvious from
  context.

- **Fitness function** is evolutionary computing's term, and the direction
  flips — you *maximise* it, because organisms with higher fitness survive.
  `rlevo`'s engine is maximise-native for exactly this reason. But maximise-native
  is also a footgun: hand a maximiser a loss and forget to negate it, and you
  breed a population that is confidently, spectacularly wrong. We close that gap
  with [`ObjectiveSense`](https://docs.rs/rlevo-core). Rather than negating by
  hand, you hand the engine your *natural* objective and declare its direction:

  ```rust
  ObjectiveSense::Minimize  // a cost — lower is better
  ObjectiveSense::Maximize  // a reward — higher is better
  ```

  The harness reconciles the sense at one chokepoint, so a forgotten negation can
  no longer silently invert your whole run. The mechanics live in
  [Fitness Evaluation](evolutionary-computation/23-fitness.md#the-engine-maximises--and-you-declare-your-objectives-sense).

- **Reward** is reinforcement learning's contribution to the pile. Strictly, the
  reward signal \\(r_t\\) and the objective (expected cumulative return
  \\(\mathbb{E}[\sum_t \gamma^t r_t]\\)) are distinct, but colloquially people
  call the whole thing "the reward function." Like fitness, it is maximised — RL
  agents are optimists by design.

The upshot, and a useful reading habit: when you see \\(\arg\min\\) in a paper,
someone is probably doing ML or OR; when you see \\(\arg\max\\), someone is doing
RL or EC. `rlevo` spans all four communities, so it commits to one internal
convention — maximise a canonical fitness — and makes you state the sense once,
explicitly, at the boundary.

## The first question: what does the landscape look like?

A useful mental model is the **fitness landscape** — picture \\(f\\) as a
physical terrain where elevation represents cost: valleys are good solutions,
peaks are bad ones. A perfect optimiser would teleport to the lowest valley
without looking at the terrain. Every real algorithm is a compromise: it gathers
*local* information and uses it to decide where to look next. The shape of the
terrain therefore decides which algorithm has a chance.

The metaphor comes from evolutionary biology — Sewall Wright introduced it to
describe how populations move through genotype space under selection
[[Wright, 1932]](#bibliography). Stuart Kauffman later made *ruggedness* — how
many local optima a landscape has — precise in the NK model
[[Kauffman, 1993]](#bibliography). Both ideas transferred directly into
evolutionary computation, and three landscape properties end up driving almost
every algorithm choice:

| Property | What it means | Implication |
| -------- | ------------- | ----------- |
| **Unimodal** | One global optimum, no local traps | Gradient descent works well |
| **Multimodal** | Many local optima | Local search gets trapped; a *population* helps |
| **Deceptive** | The local gradient points *away* from the global optimum | Gradient methods walk the wrong way; EAs can still escape |

You will meet these three first-hand in Part II. The sphere function minimised
there is unimodal and convex — an easy landscape, chosen to show the mechanics
cleanly. The Rastrigin and Rosenbrock functions (in `rlevo::envs::landscapes`)
are multimodal and non-separable respectively, and that is exactly where
evolutionary methods start to earn their keep. For a fuller tour of landscape
geometry and the benchmark suite `rlevo` ships — including how to *see* these
shapes rendered — see [Fitness Landscapes](../appendix-d-suppl/fitness-landscape.md)
in the Appendices.

## The second question: explore or exploit?

Knowing the terrain is half the battle; the other half is a tension every search
algorithm must resolve at every step:

- **Exploitation** — concentrate effort near solutions already known to be good.
- **Exploration** — probe unknown regions that might be better.

A pure exploiter converges fast and gets trapped in the first valley it finds. A
pure explorer wanders forever and never converges. The right balance depends on
the landscape, the budget (how many evaluations you can afford), and how much you
are willing to risk missing the global optimum. As Sutton and Barto put it, "the
agent has to exploit what it has already experienced in order to obtain reward,
but it also has to explore in order to make better action selections in the
future" — and "neither exploration nor exploitation can be pursued exclusively
without failing at the task" [[Sutton and Barto, 2018]](#bibliography). The
balance is not fixed, either: the more stable your problem, the more you can lean
on what you already know; the more it shifts, the more exploration earns its
keep.

This same trade-off wears different costumes across the field — the
**exploration–exploitation dilemma** in reinforcement learning, the
**diversity–selection-pressure** trade-off in evolutionary computation (John
Holland called striking it well "the optimal allocation of trials"
[[Holland, 1975]](#bibliography)), and the **bias–variance trade-off** in
statistics. Recognising them as one problem is worth the effort: a tactic that
buys exploration in one setting — injecting mutation, widening a search
distribution, adding an exploration bonus — usually has a direct analogue in the
others.

<!-- Author note: a fuller exploration/exploitation treatment (Simon 2013 pp.28-29;
Sutton & Barto 2018 p.3) is a candidate supplementary appendix; trimmed here to
keep the chapter at overview tier. -->

## The third question: do you even have a gradient?

Gradient-based methods — stochastic gradient descent, Adam, L-BFGS — dominate
modern machine learning because neural-network loss surfaces, though
high-dimensional, are empirically well-behaved to train on. They are the right
default *when they apply*. They stop applying when:

1. **The objective is not differentiable.** Reward in a game, a simulation
   result, or a human preference score has no gradient to follow.
2. **The landscape is highly multimodal.** A gradient tells you which way is
   downhill *locally* and nothing about whether a better valley exists elsewhere.
3. **The search space is discrete or combinatorial.** Gene sequences, network
   architectures, and routing tables are not real-valued, so there is nothing to
   differentiate.
4. **Evaluation is noisy.** A single observation of \\(f(\mathbf{x})\\) can
   mislead; you need a strategy robust to stochastic scores.

Evolutionary computation was built for exactly these regimes — it asks only that
you can *score* a candidate, never that you can differentiate the scoring. That
is why `rlevo` carries a population engine alongside its gradient one, and why
the [next chapter](20-evolutionary-computation.md) is where the population path
begins.

## Putting it together: there is no universal best algorithm

These three questions are not idle taxonomy. They are how you choose, because of
a hard theoretical result: Wolpert and Macready proved that **no search algorithm
beats every other algorithm when performance is averaged uniformly over all
possible objective functions** [[Wolpert and Macready, 1997]](#bibliography). If
an algorithm excels on one class of problems, there is necessarily another class
on which it does equivalently worse. There is no free lunch.

The practical reading is liberating rather than gloomy: choosing an algorithm
*is* making an assumption about your problem's structure, so the work is to match
the assumption to the landscape. Each family in `rlevo` bets on a different
structure:

| If your problem is… | A fitting bet | Why |
| ------------------- | ------------- | --- |
| Smooth, differentiable, unimodal-ish | Gradient RL (DQN, PPO) | A deep network can approximate the value/policy surface |
| Rugged, multimodal, derivative-free | A genetic algorithm | Good solutions share reusable building blocks |
| Continuous, ill-conditioned, low-to-medium dimensional | CMA-ES | The landscape is locally well-approximated by a Gaussian whose shape you can learn |
| Discrete or program-structured | GP / CGP / NEAT | The genotype decodes to a valid structure without repair |

Part II makes the bet explicit for every algorithm it demonstrates, so you are
never trusting a method blind — you are choosing one whose assumptions you have
checked against the terrain.

> **Deeper reading.** For a rigorous treatment of optimisation landscapes and
> algorithm analysis, see Nocedal and Wright, *Numerical Optimization* (2nd ed.,
> Springer, 2006) [[Nocedal and Wright, 2006]](#bibliography) for the gradient
> side and Eiben and Smith, *Introduction to Evolutionary Computing* (Springer,
> 2015) for the population side. The NK model is developed in Kauffman, *The
> Origins of Order* (Oxford, 1993) [[Kauffman, 1993]](#bibliography). The No Free
> Lunch theorem is Wolpert and Macready (1997), *IEEE Transactions on
> Evolutionary Computation* [[Wolpert and Macready, 1997]](#bibliography); Simon
> (2013) [[Simon, 2013]](#bibliography) discusses it with worked examples in
> Appendix B.1, p. 614.

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
