# Appendix B — Reinforcement Learning Algorithms

This appendix gives derivations, pseudocode, and implementation notes for the RL
algorithms in `rlevo::rl`. See
[Part I — Reinforcement Learning](../part-1-foundations/30-reinforcement-learning.md)
for the conceptual overview.

> This appendix is a stub. Algorithm pages will be added incrementally.

## Contents (planned)

- DQN — full pseudocode, experience replay details, target network update schedule
- Double DQN — decoupled action selection and evaluation
- Dueling DQN — value / advantage decomposition
- PPO — clipped surrogate objective, advantage estimation (GAE) with
  partial-episode bootstrapping on truncation, deliberately diverging from
  CleanRL's default
  ([ADR 0048](https://github.com/anthonytorlucci/rlevo/blob/main/docs/adr/0048-partial-episode-bootstrapping-in-gae.md))
- SAC — entropy-regularised objective, twin critics, automatic temperature tuning

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
