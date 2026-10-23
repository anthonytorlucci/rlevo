---
project: rlevo
status: active
type: decision
date: 2026-10-23
tags: [adr, decision, architecture, rlevo-reinforcement-learning, rl, model-traits, default-networks, public-surface, breaking-change]
---

# ADR 0094: One trait for every Q-network, and default networks

## Status

**Accepted (2026-10-23).**

Supersedes one clause. The rest of the ADR stands.

- ADR [0024](0024-rlevo-test-support-dev-crate.md), "What deliberately
  stays in each test file": the per-algorithm networks stay test-local.
  The value-side networks those test files defined are now the RL crate's
  own (Decision 2), and the files build them from it. ADR 0024's reason
  against moving them into `rlevo-test-support` still holds, and they do
  not go there.

## Context

`rlevo-reinforcement-learning` asks a user for a network through a model
trait per algorithm family. After ADR 0093 each is one method:

| Trait | `forward` | Shipped implementor |
|---|---|---|
| `DqnModel<BOR>` | `Tensor<BOR>` to `Tensor<2>`, Q-values | none |
| `C51Model<BS>` | `Tensor<BS>` to `Tensor<3>`, atom logits | none |
| `QrDqnModel<BOR>` | `Tensor<BOR>` to `Tensor<3>`, quantile values | none |
| `PpoValue<BOR>` | `Tensor<BOR>` to `Tensor<1>`, `V(s)` | none |
| `DeterministicPolicy<BOR, BAR>` | `Tensor<BOR>` to `Tensor<BAR>`, an action | none |
| `ContinuousQ<BOR, BAR>` | observation and action to `Tensor<1>`, `Q(s, a)` | none |
| `PpoPolicy<BOR>` | | three heads |
| `SquashedGaussianPolicy<BOR, BAR>` | | one head |

Two things follow from the table.

**`C51Model` and `QrDqnModel` are one trait twice.** Same input, same
output shape, one method. The difference is how the agent reads the
output: C51 takes a softmax over the last axis, QR-DQN reads quantile
values. A network that serves one serves the other, and a test fixture
already implemented both with one body.

**The policy side ships networks; the value side does not.** A first DQN,
PPO or DDPG run therefore starts with writing a network. The workspace
shows what that costs. A search for every impl of the eight model traits
(`rg` over `impl … for`, 57 lines) finds 53 hand-written networks besides
the four shipped heads. They sit in benches, integration tests, two
examples and the crate's own test fixtures. Most are the same few shapes:
two hidden layers of one width (`ReLU`, or `tanh` for a value network),
then a linear head. Fixed at two layers, a default would reproduce 24 of
them; with a list of widths, 37. Anything that starts a user's first run,
an example or a project template, needs a default to point at, or it
starts with a hand-written network too.

## Decision

### 1. `DqnModel` takes the output rank; `C51Model` and `QrDqnModel` go

```rust
pub trait DqnModel<const BOR: usize, const OUT: usize = 2>: Module {
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<OUT>;
}
```

A Q-network is `DqnModel<BOR>`, as before; the default makes every DQN
bound and impl compile unchanged. A C51 or QR-DQN network is
`DqnModel<BOR, 3>`, `(batch, num_actions, num_atoms)`. The trait's docs say
how each agent reads the rank-3 output, which is what the two deleted
traits' docs said. The modules `c51::c51_model` and `qrdqn::qrdqn_model`
are deleted. No alias: a trait alias is unstable, a sub-trait with a
blanket impl is a new public item, and ADRs 0083 and 0085 kept none.

### 2. The crate ships a default network for each value-side trait

| Network | Implements | Config |
|---|---|---|
| `MlpQNetwork` | `DqnModel<BOR>` | `obs_dim`, `hidden`, `num_actions` |
| `MlpDistributionalQ` | `DqnModel<BOR, 3>` | `obs_dim`, `hidden`, `num_actions`, `num_atoms` |
| `MlpValue` | `PpoValue<BOR>` | `obs_dim`, `hidden` |
| `MlpActor` | `DeterministicPolicy<BOR, 2>` | `obs_dim`, `hidden`, `action_dim`, `action_scale` |
| `MlpCritic` | `ContinuousQ<BOR, 2>` | `obs_dim`, `action_dim`, `hidden` |

Each lives beside the trait it implements, as the heads do, and is built
the way the heads are: a config with public fields, `try_init(&device)`,
which runs `Validate` first, and no infallible `init`, so a zero width
cannot build a zero-width layer. The hidden layers use `ReLU`, and `tanh`
in `MlpValue`, as CleanRL does and as every copy in the workspace did.
`MlpDistributionalQ` has no activation at its head, so it serves C51's
logits and QR-DQN's quantiles alike. `MlpActor` emits `tanh(..) *
action_scale`, a range symmetric about zero; no copy used a bias, and an
actor for another range is the author's, or a scale of 1 with the agent's
per-component clip, as four copies did. `MlpCritic` concatenates the
observation and the action on the feature axis, observation first, as all
eleven critics in the workspace did.

### 3. `hidden` is a list of widths

The heads take one `hidden` width for two layers. The networks take
`hidden: Vec<usize>`, the widths first to last; empty is a linear model.
A list reproduces the one-layer actors and critics of the DDPG, TD3 and
SAC integration tests, a three-layer bench network and six no-hidden-layer
fixtures exactly, and it is the shape a researcher knows from other
libraries. The heads are not changed here.

### 4. Every network works at any observation rank

Each impl is generic over `BOR` and flattens the observation to
`(batch, obs_dim)` before its first layer. At `BOR = 2` that is the
identity. One network serves vector and grid observations, where the
benches kept one network for each. A rank below 2 has no feature axis and
is a compile error where the network is used.

### 5. The copies go; the special-purpose networks stay

39 of the 53 hand-written networks are replaced: by a default with the
copy's own layer widths, or, for two bench SAC actors that were copies of
the shipped `SquashedGaussianPolicyHead`, by that head. The 14 that stay
do something no default should: two convolutional networks; fixtures with
hand-set weights that back known answers; critics with a `ReLU` on the Q
output; SAC's one-layer test actor, where the shipped head has two; and
value fixtures whose tests read their fields.

### 6. An outside-in probe builds a derived network

Burn's `Module` and `Config` derives expand to `burn::` paths, so a crate
that reaches Burn as `rlevo::burn` keeps `use rlevo::burn;` in the module
that derives, as the umbrella's crate docs say. A doctest of `rlevo`
cannot check that: rustdoc compiles it with `rlevo`'s own `burn` in scope.
The `consumer-umbrella` probe, which builds against the workspace as
packaged from outside the checkout, now derives a `Module` and a `Config`
that way and builds a default network beside it. Once code outside the
workspace builds on a default, nothing there derives a network of its own,
so the check comes with the defaults.

## Consequences

### Positive

- A first DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3 or SAC run writes no
  network, and an example or a template has a default to build on.
- One trait where there were three for the DQN family, and one network
  for C51 and QR-DQN.
- 39 networks and the benches' `support/` directory are gone, and no
  seeded result moved (Measured).
- The crate's own tests drive the defaults through all six off-policy
  `train` loops.

### Negative / costs

- **Breaking:** `C51Model`, `QrDqnModel` and their two module paths are
  removed. An implementor writes `impl DqnModel<2, 3> for MyNet` where it
  wrote `impl C51Model<2> for MyNet`; the method body does not change.
- `DqnModel<2, 3>` says less at a glance than `C51Model<2>`; the trait's
  docs carry the difference.
- The surface grows (Measured): the step adds more public items than it
  removes, all of them the networks and their configs.
- The heads and the networks spell `hidden` differently (one width, a
  list of widths) until the heads are revisited.

### Neutral

- Decisions 1 and 2 of ADR 0093 are unchanged: a network is built on the
  plain device, and the agent switches autodiff on.
- The probe's package counts do not change; the derive macros were
  already in its build.

## Alternatives considered

- **Keep `C51Model` and `QrDqnModel`, or merge them into a new
  `DistributionalQModel`.** Two traits or three where one carries the
  difference as a const parameter. Rejected: the audit's test (one trait
  before one network) is that a concept has one name.
- **Fix the depth at two layers, as the heads do.** Reproduces 24 copies
  instead of 37, and the 13 others (the DDPG, TD3 and SAC integration
  networks among them) would have to change their numbers, and their
  seeded convergence thresholds be measured again.
- **One `networks` module.** One import path for a template, at the cost
  of a module, and a reader who finds a trait would not find its default
  beside it. Rejected.
- **A scale and a bias on `MlpActor`, or the bounds of the action type.**
  No copy used either; per-component bounds need a vector, and the agent
  already clips per component.
- **A sixth probe for the derive.** A CI job for one file, where an
  existing probe already depends on `rlevo` alone and only builds.

## Measured

Apple M2 Pro, 12 CPU cores, 32 GiB, macOS, rustc 1.99.0, Flex.

- **The defaults reproduce the copies.** Before deleting them, an
  uncommitted test built each family of copy and its replacement from the
  same seed, and compared every parameter and one forward pass on a fixed
  batch: 19 cases, from a 9-parameter fixture to a 528,898-parameter
  three-layer network and the 82,947-parameter rank-4 car-racing actor, all
  bit-identical. Burn 0.22 initialises a parameter on first read, so each
  network was read right after it was built, as an agent does when it is
  built.
- **The surface delta** (`tools/public-surface.sh
  rlevo-reinforcement-learning`, base against branch): 1,029 lines to
  1,059. Out, 6 lines: the traits `C51Model` and `QrDqnModel`, their
  `forward`, and their two modules. In, 36 lines: the five networks and
  their five configs, each config's `try_init`, and the configs' 16 public
  fields. `DqnModel`'s line does not change; the listing does not show a
  const parameter.
- **The derive check fails without the import.** With the probe's imports
  spelled `rlevo::burn::nn::…` and `rlevo::burn::prelude::*` and no
  `use rlevo::burn;`, the probe fails at the derive: "cannot find module
  or crate `burn` in this scope".

## References

- ADR [0024](0024-rlevo-test-support-dev-crate.md), the clause superseded
  above.
- ADR [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md),
  the test a public item answers to.
- ADR [0093](0093-the-backend-is-a-runtime-device.md), which left each
  model trait one method.
- CleanRL's DQN, C51, PPO, DDPG, TD3 and SAC single-file implementations,
  whose network shapes the defaults follow.
