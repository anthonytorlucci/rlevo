# Part III — Advanced Evolutionary Methods

Part I established the single-population evolutionary loop: one population, scored
by a fixed objective, driven by the `Strategy` ask/tell contract and the
`EvolutionaryHarness`. Part II put that loop to work. This part goes beyond it —
to settings where the structure of the search itself changes.

The first such setting is **coevolution**, where fitness is no longer measured
against a fixed yardstick but *between* populations: each population's score
depends on the others. That single change — coupling the objective to a moving
target — opens the door to decomposing high-dimensional problems, to adversarial
arms races, and to a class of failure modes (cycling, forgetting) that the
single-population loop never encounters.

These methods build directly on Part I. If the `Strategy` trait, the ask/tell
loop, or the harness are not yet familiar, read
[Evolutionary Computation](../part-1-foundations/20-evolutionary-computation.md)
first — the coevolution machinery is layered on top of exactly those pieces.

| Chapter | Core idea |
| ------- | --------- |
| [Coevolution](10-coevolution.md) | Cooperative decomposition, competitive arms races, coupled fitness, and the hall-of-fame fix for cycling |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
