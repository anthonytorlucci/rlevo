# Fitness landscape

<!-- see also https://en.wikipedia.org/wiki/Fitness_landscape -->

In the context of evolutionary optimization (EO), a **fitness landscape** is a conceptual and mathematical framework used to visualize and analyze the relationship between potential solutions in a search space and their corresponding "fitness" (how well they perform).

When we talk about **benchmark functions**, we are looking at standardized, mathematically defined fitness landscapes designed to test specific characteristics of an optimization algorithm.

Here is how you can describe a fitness landscape across four key dimensions:

---

### 1. The Mathematical Mapping (The "Map")
At its core, a fitness landscape is a function $f(x)$, where:
*   **The Domain ($x$):** Represents the search space (e.g., all possible weights in a neural network or all possible dimensions of a robot's movement).
*   **The Codomain ($f(x)$):** Represents the fitness value (a scalar number indicating success).

In this context, a "landscape" is an abstraction: we treat every possible input as a point on a map. If you imagine a 2D search space (two variables), it becomes a 3D surface where height represents fitness. The goal of any evolutionary algorithm (EA) is to navigate this terrain to find the highest peak (global optimum).

### 2. Topographical Characteristics (The "Terrain")
When analyzing benchmark functions, we describe the landscape by its "geography." This determines how difficult it is for an algorithm to succeed:

*   **Modality:** Is the landscape **Unimodal** (one single peak) or **Multimodal** (many peaks and valleys)? Benchmark functions like *Rastrigin* are highly multimodal, testing an algorithm's ability to avoid getting stuck in "local optima" (false peaks).
*   **Ruggedness:** A "rugged" landscape has frequent, sharp changes in fitness over small distances. This mimics problems with high noise or complex interactions between variables.
*   **Reachability/Connectivity:** Can the algorithm move from one peak to another? Some landscapes have "ridges" or "plateaus" that make it hard for an algorithm to jump across gaps.

### 3. The Concept of "Epistasis" (The Interaction)
In fitness landscapes, **epistasis** refers to how much the variables interact with each other.
*   On a **separable landscape**, changing variable $A$ has no effect on the impact of variable $B$. These are easy for EAs because they can optimize one dimension at a time.
*   On a **non-separable (decoupled) landscape**, variables are deeply intertwined. A change in $x_1$ might drastically change the "height" of the fitness only if $x_2$ is also a certain value. This creates complex, diagonal "valleys" (like those found in the *Rosenbrock* function).

### 4. Why Benchmark Functions Exist
We use specific benchmark functions because they isolate these geometric traits into "test tracks." By observing how an algorithm performs on different landscapes, we can diagnose its strengths and weaknesses:

| Benchmark Function | Landscape Characteristics | What it tests in an algorithm |
| :--- | :--- | :--- |
| **Sphere / Quadratic** | Perfectly smooth, unimodal. | Basic convergence speed and ability to "climb" a hill. |
| **Rosenbrock** | A long, narrow, curved valley. | Ability to navigate "flat" regions where gradients are very small. |
| **Rastrigin** | Highly multimodal (a grid of many peaks). | Ability to escape local optima and explore globally. |
| **Ackley** | A nearly flat surface with a very steep hole in the center. | Ability to find a needle in a haystack (global search capability). |
| **Schaffer (RK109)** | Highly non-separable/complex dimensions. | Handling of complex interactions between variables. |
<!-- todo! list all available landscapes in rlevo-environments -->

### Summary Analogy
Think of a **fitness landscape** as a mountain range and the **evolutionary algorithm** as a hiker lost in the fog. 
*   The **hiker** (the algorithm) can only see the ground immediately around their feet. 
*   If the **mountain range** is one big smooth hill (Unimodal), any step upward eventually leads to the top. 
*   If the mountain range is a jagged maze of thousands of small peaks (Multimodal/Rugged), the hiker might reach a small hill and think they are at the summit, only to find a much taller mountain hidden behind the next ridge.

**Benchmark functions** are the standardized "maps" we give to different hikers to see which one has the best navigation tools.

---

*Co-Authored-By: Anthropic Claude Sonnet 4.6*\
*Reviewed-By: (Human) Anthony Torlucci*
