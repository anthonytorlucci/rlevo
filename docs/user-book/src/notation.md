# Notation

| Symbol | Meaning |
| ------ | ------- |
| \\(\mathcal{S}\\) | State space |
| \\(\mathcal{A}\\) | Action space |
| \\(s_t, a_t, r_t\\) | State, action, reward at timestep \\(t\\) |
| \\(\pi(a \mid s)\\) | Policy: probability of action \\(a\\) in state \\(s\\) |
| \\(V^\pi(s)\\) | State-value function under policy \\(\pi\\) |
| \\(Q^\pi(s, a)\\) | Action-value function under policy \\(\pi\\) |
| \\(\gamma\\) | Discount factor \\(\in [0, 1)\\) |
| \\(G_t\\) | Discounted return from timestep \\(t\\) |
| \\(\theta\\) | Neural network parameters |
| \\(\mathbf{x}\\) | Genome / candidate solution vector |
| \\(f(\mathbf{x})\\) | Objective / fitness function |
| \\(\mathcal{N}(\mu, \sigma^2)\\) | Normal distribution with mean \\(\mu\\), variance \\(\sigma^2\\) |
| \\(\lambda\\) | Population size (number of offspring) |
| \\(\mu\\) | Number of parents selected |
| \\(\sigma\\) | Mutation step size (evolution strategies) |
