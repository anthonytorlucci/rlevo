---
project: rlevo
status: active
type: decision
date: 2026-10-24
tags: [adr, decision, architecture, rlevo-core, rlevo-harness, rlevo-reinforcement-learning, agent, trial, public-surface, breaking-change]
---

# ADR 0095: The agent seam is one `Agent` trait with defaulted learning hooks

## Status

**Accepted (2026-10-24).**

Fills the `rlevo_core::agent` module, empty and reserved since ADR
[0003](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) kept
its `Agent` stub as a roadmap marker. Supersedes a clause of each of six
ADRs; the rest of each stands.

- ADR [0004](0004-move-bench-traits-into-rlevo-core.md): `BenchableAgent` in
  `rlevo_core::fitness`. It is deleted; `Agent` in `rlevo_core::agent`
  replaces it (Decision 1).
- ADR [0050](0050-replay-strategy-seam.md): the stored `Transition<O, P>` is
  the replay module's own, with a `terminated: bool`. It is
  `rlevo_core::agent::Transition`, which carries the `EpisodeStatus`
  (Decision 2).
- ADR [0051](0051-replay-kind-dispatch-and-validated-importance-exponent.md)
  Decision 1: "`ReplayStrategy<T>` itself remains public and
  implementable". It is crate-private, with the rest of the replay module
  but `PrioritizedReplaySettings` (Decision 6).
- ADR [0057](0057-target-soft-update-path-is-fallible.md): each off-policy
  agent's error enum gains a `Polyak` variant and `learn_step` returns it.
  The enums are deleted and `learn_step` returns `PolyakError` itself; ADR
  0057's choice of `Result` over a panic stands (Decision 4).
- ADRs [0070](0070-avg-score-transits-non-finite-scores-the-hardened-mean-is-additive.md)
  and [0071](0071-best-score-latches-plus-infinity-the-finite-best-is-additive-and-counted.md):
  `AgentStats`. It is deleted with the training loops that filled it
  (Decision 5).

Amends ADR [0076](0076-trial-seam-splits-machinery-from-trial-shape.md)
Decision 1: the `EpisodicTrial` that `run_suite` builds drives an `Agent`,
and trains it before evaluating when asked (Decision 3). The `Trial` seam
itself is unchanged.

## Context

`BenchableAgent<Obs, Act>` had two items, `act` and a defaulted
`emit_metrics`. `EpisodicTrial` called `act`, stepped the environment, and
added the reward to its own total; the agent never saw the reward, the next
observation, or how the episode ended. So `Evaluator::run_suite`, the
documented entry point, could measure a frozen policy and could not train a
learner. A learning submission had to write a `Trial` of its own.

The RL crate had the learning half, as nine public `train` functions (DQN,
C51, QR-DQN, DDPG, TD3, SAC, PPG, and PPO's discrete and continuous loops),
each a copy of one loop: act, step, store the transition with its
bootstrap mask, run the family's cadence methods, record the episode into
an `AgentStats`. The cadence methods those loops called (`remember`,
`on_env_step`, `should_train`, `decay_exploration`, `record_step`,
`finalize_rollout`, `update`, ...) were public too, so each agent had two
public ways to be driven, and every loop had to pass the episode's status to
its agent the same way. A wrapper over the trial was not available: the RL
crate cannot depend on the harness.

The crate's public surface also carried its building blocks: the replay
buffers and their trait, the loss kernels, the exploration schedules, the
target-update helpers, PPO's rollout buffer, the network `Slot`. Nothing
outside the workspace named one, and no probe or example built an algorithm
from them. Inside it, eight files did: five of the crate's benches, one of
its tests, two of the umbrella's tests.

## Decision

1. **`Agent<Obs, Act>` in `rlevo_core::agent`** replaces `BenchableAgent`,
   with no alias. `act(&mut self, obs, rng) -> Act` is required, with
   `BenchableAgent::act`'s signature and contract: every random draw comes
   from the driver's `rng`. Four hooks default to doing nothing, so a
   frozen policy is one method long:
   `on_transition(&mut self, Transition<&Obs, &Act>, rng)`, after every
   step; `on_episode_end(&mut self, EpisodeStatus)`, when an episode ends;
   `set_training(&mut self, bool)`, before a phase; and `emit_metrics`,
   pulled once per trial. There is no `learn` method. The hooks return
   nothing.

2. **One `Transition<O, A>`** in the same module, with plain public fields
   `obs`, `action`, `reward: f32`, `next_obs` and `status: EpisodeStatus`,
   and no `#[non_exhaustive]`, as `Snapshot`. A driver shows the borrowed
   form; the replay stores `Transition<O, usize>` or
   `Transition<O, Vec<f32>>`, so it keeps the status where it kept a
   `terminated` flag. `remember` takes the status, and each agent masks the
   bootstrap on `status.is_terminated()` where it builds the target.

3. **`EpisodicTrial` trains, then evaluates.** `EvaluatorConfig` gains
   `train_steps` (default 0). The trial calls `set_training(true)` and runs
   that many environment steps, then `set_training(false)` and its
   `num_episodes` evaluation episodes, on one RNG. In both phases every
   step's transition carries the snapshot's status unchanged, and an ended
   episode gets `on_episode_end` with that status. An episode the trial
   cuts at `max_steps` ends with `on_episode_end(Truncated)` after a last
   transition whose status is still `Running`, since the environment did not
   end it. The report holds the evaluation episodes only. The training phase
   is `rlevo_harness::evaluator::train(env, agent, steps, max_steps, rng)`,
   public for a caller that keeps the agent; it resets only to open an
   episode its budget can step, so with `max_steps` at 0 it opens none, and
   returns the episodes it completed.

4. **One `impl Agent` per RL agent.** The value-based three explore
   ε-greedily while training and act greedily otherwise; the actor-critic
   three act with exploration noise while training and on the policy
   otherwise; PPO, for each of its two shipped heads, and PPG sample while
   training and act on the policy's mean or most probable action otherwise.
   PPO and PPG carry no action type, so their impls are per head, and the
   continuous one is for rank-1 actions, the shape a Gaussian head emits.
   While training, `on_transition` runs what the family's loop did; while
   evaluating it does nothing. PPO and PPG also implement `on_episode_end`
   and close a step in `set_training`: a step the driver ends without the
   environment ending it, at a `max_steps` cut or a change of phase, was
   recorded `Running`, and becomes a truncation that bootstraps from its own
   next observation (ADR 0048). Otherwise GAE would run on from it into the
   first step of the next episode, after the driver's reset. The methods the impl replaces are private:
   each agent's sampling `act`, the value-based `act_greedy`, and the
   cadence methods. `remember` and `learn_step` stay public on the six
   off-policy agents as the externally paced path, for a caller that
   pushes transitions and takes gradient steps on a schedule of its own;
   `remember` now counts the environment step itself, so a caller cannot
   forget to and leave learning gated off. `learn_step` returns
   `Result<Option<LearnOutcome>, PolyakError>`; the hooks, which have no
   error channel, turn that error into a panic, which a trial folds into an
   errored report.

5. **The nine `train` functions are deleted**, with `AgentStats`,
   `PerformanceRecord`, the eight `*Metrics` structs, each agent's `stats`
   and `record_episode`, the eight `*AgentError` enums, and the
   `log_every` progress throttle. PPO and PPG log one `tracing` event per
   update, with the update's statistics under the metric registry's names;
   the off-policy agents log no progress event.

6. **The RL crate's parts are crate-private**: the replay buffers, their
   trait and errors, the loss kernels, the exploration schedules, the
   target-update helpers and `PolyakTau`, the rollout buffers, `Slot`, and
   `MAX_BUFFER_CAPACITY`. `replay` exports `PrioritizedReplaySettings`, the
   config opt-in, and `target` re-exports `PolyakError`. A part can be made
   public later, when an author needs it, without breaking anyone; making
   one private after a release could not be done that way.

7. **No capability trait ships yet.** `OffPolicyLearner`, `ActorParams` and
   `Checkpoint` were planned beside `Agent`; nothing in the workspace is
   generic over any of them, so under ADR
   [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md) each
   waits for its first consumer.

## Consequences

- `run_suite` trains: a learner implements `Agent` and is trained and
  measured through the documented entry point, and a random baseline under
  the same configuration spends the same training steps learning nothing.
  A learner never writes a `Trial`.
- One loop drives every agent, so the status rule is written once, in the
  harness, and the bootstrap mask once per agent, in `remember`'s caller.
- Evaluation is unchanged. With `train_steps` at 0 the trial runs as
  before, and an agent with default hooks behaves as a `BenchableAgent` did.
- PPO and PPG update on whole rollouts only. The deleted loops updated on a
  partial rollout when their budget ended mid-rollout; through the hooks
  the agent cannot know the budget ended, so a budget that is not a
  multiple of `num_steps` leaves its last partial rollout unused.
- A direct call to a hook on a PPO or PPG agent names its action type
  (`Agent::<Obs, Act>::set_training(&mut agent, false)`); a driver infers it
  from the environment.
- A crate-private item is checked by lints that skip exported items, and
  code that only the tests reached shows as dead. The few test-only methods
  are now `#[cfg(test)]`; what not even the tests reached is deleted.

## Alternatives considered

- **A deprecated `BenchableAgent` alias for one release.** One more public
  item, for a trait with no implementor outside this workspace's own tests,
  examples and probe templates.
- **Keeping the `train` functions beside the trial.** Two loops per agent,
  ten places to keep the status rule right, and every cadence method public.
- **A `LearnError` in place of the eight error enums.** Their environment
  variants lost their only producer with the loops, and `PolyakError` is
  the one error left; a wrapper around it adds a name and no information.
- **The capability traits now**, so a new algorithm crate could implement
  them from its first version. They would be public items with an
  implementor and no consumer.
- **Keeping the parts public for algorithm authors.** It needs an example or
  probe that builds an algorithm from them, and a consumer for
  `ReplayStrategy`, which no agent accepts from outside; and it fixes 57
  items' signatures at the next release.

## Measured

On an Apple M2 Pro, 12 CPU cores, 32 GiB, macOS, rustc 1.99.0, Flex.

- **Training is unchanged, bit for bit.** Before the change, each agent was
  trained through its `train` function: DQN, C51 and QR-DQN on a
  `TimeLimit`-capped CartPole, DDPG, TD3 and SAC on `LinearEnv`, PPO on both
  and PPG on CartPole, from one seed, with budgets that are a whole number
  of PPO's and PPG's rollouts. After it, each was trained through the
  generic `train`. The hash of every observation, reward and status the
  environment returned, the episode lengths, and the actions of a 200-step
  evaluation are identical for all nine runs, and so are the trained
  parameters for the eight whose network a caller can read; SAC exposes
  none, and its evaluation stands in for them.
- **Evaluation is unchanged.** The two BYOE-1 probe environments return what
  they returned before: −213.564 and −45.564 for the thermostat, −1.467 and
  −2.893 for the tank farm.
- **Public surface** (`tools/public-surface.sh`). `rlevo-core` gains `Agent`
  and `Transition` and loses `BenchableAgent`. `rlevo-harness` gains `train`
  and `EvaluatorConfig::train_steps`. The umbrella's prelude gains `Agent`
  and `Transition`. `rlevo-reinforcement-learning` goes from 1,059 listing
  lines to 585: 81 top-level items leave (26 structs, 12 enums, 33 free
  functions, 6 constants, 2 traits, 2 type aliases), and `target::PolyakError`
  is the one new path, for an existing type.

## References

- ADR [0029](0029-host-rng-seeding-convention.md): the host-seeded RNG an
  agent's draws come from.
- Pardo et al., "Time Limits in Reinforcement Learning", ICML 2018: why the
  status, not a `done` flag, reaches the learner.
