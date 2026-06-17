---
project: rlevo
status: active
type: decision
date: 2026-06-04
tags:
  - adr
  - decision
  - architecture
  - record-schema
  - metrics
  - benchmarks
  - reproducibility
  - wire-format
  - rlevo
---

# ADR 0014: Record schema v6 — single-agent RL richness, typed run provenance, eval/train tagging

## Status

Active. Adopted 2026-06-04. Bumps the on-disk record format from `FORMAT_VERSION = 5`
to `6`. Extends — does not supersede — [0013-metrics-only-live-tui](0013-metrics-only-live-tui.md): the `EpisodeRecord`
seam, the per-family `FamilyPayload`, and the production-crate isolation rules are all
preserved. Driven by the gap analysis in `research/2026-06-04-benchmark-client-metrics.md`.

## Context

The record surface (`crates/rlevo-benchmarks/src/record/`) at `FORMAT_VERSION = 5`
captures enough to *replay* an episode (frames, per-family render payload) and a thin
slice of *training signal* (six RL metric names, four EA metric names, a free-form
`hyperparameters: BTreeMap<String,String>` on the manifest). The 2026-06-04 survey of
RLlib, CleanRL, Stable-Baselines3, Tianshou, Gymnasium, and the reproducibility
literature (Henderson et al. 2018; Agarwal et al. 2021 / rliable) found three concrete
classes of gap that prevent a recorded run from answering *"is it learning, and is this
run reproducible?"*:

1. **Metric coverage is too thin to read a training curve.** Missing the single most
   important value-net diagnostic (`explained_variance`), per-iteration episode-return
   statistics emitted *during* training (we compute them only at trial completion), a
   named lifetime env-step counter (the x-axis of every learning curve), and the
   algorithm-specific losses for algorithms we already ship (DQN `q_values`/`td_loss`;
   SAC `alpha`/`actor_loss`/`qf{1,2}_loss`).

2. **Run provenance is untyped and incomplete.** Algorithm identity, software-stack
   versions, git commit, hardware/backend, seed count, and the success threshold either
   live as opaque strings in `hyperparameters` or are not persisted at all. A loader
   cannot reliably pick which loss panels to render, nor reconstruct the run.

3. **No evaluation-vs-training distinction.** Deterministic eval rollouts and exploration
   rollouts are indistinguishable in the record; RLlib and Tianshou separate them at the
   schema level.

The research also identified three **future** paradigms (multi-agent RL, variable-topology
neuroevolution, Bayesian-network learning) whose data shapes do not fit v5. This ADR
**deliberately excludes** all three from v6 (see Decision part 6) and instead records the
additive invariants that keep their eventual bumps cheap. Two facts make that exclusion
free of cost:

- `MIN_SUPPORTED_VERSION == FORMAT_VERSION` — there is no pre-release backward
  compatibility, so each bump simply regenerates the test corpora; deferring a field
  costs nothing later.
- bincode `standard()` is **positional** (no field names on the wire). Adding a struct
  field or reordering one is a breaking layout change regardless. The schema already
  documents "append enum variants at the end so existing tags keep decoding"; v6 holds to
  that, and speculative empty fields (the v5 `parents_of_best` / `inner_rl_returns`
  "emitted empty until a producer lands" pattern) are an anti-pattern we do not repeat.

## Decision

**Bump the on-disk record format to `FORMAT_VERSION = 6`. v6 lands single-agent RL metric
richness, typed run-provenance fields on `RunManifest`, an `EpisodeKind` train/eval tag on
`EpisodeRecordHeader`, and an episode wall-clock seam — and nothing else. Multi-agent,
neuroevolution-topology, and Bayesian-network record shapes are out of scope and deferred
to their own future bumps. All v6 changes are additive at the type level and obey the
design invariants in part 7.**

### Concrete parts

1. **Expand the canonical metric registry (no wire-format impact).**
   `CANONICAL_METRICS` (`crates/rlevo-benchmarks/src/metrics_registry.rs`) is a *filter*,
   not a wire type — adding names does not change `MetricSample`'s layout and is
   version-neutral. It ships in the same PR for coherence. Add, as flat `snake_case`
   field names (the `tracing`-field convention; no `/` namespacing, which `tracing`
   fields cannot carry):
   - **Value/policy diagnostics:** `explained_variance`, `old_approx_kl`.
   - **Per-iteration training stats:** `episode_return_mean`, `episode_return_std`,
     `episode_return_min`, `episode_return_max`, `episode_length_mean`,
     `env_steps_sampled` (lifetime counter), `steps_per_sec`, `learning_rate`.
   - **DQN family:** `td_loss`, `q_values`.
   - **SAC family:** `qf1_loss`, `qf2_loss`, `actor_loss`, `alpha`, `alpha_loss`.
   - **Schedules:** `clip_range`, `n_updates`.

2. **Emit per-iteration training stats during training, not only at trial end.**
   A producer-side change: the RL training loops emit the `episode_return_*` /
   `env_steps_sampled` / `steps_per_sec` samples through the existing `tracing` →
   `RecordingLayer` → `MetricSample` path at training cadence. No schema change beyond
   part 1; `core_metrics()` at trial completion is retained as the cross-seed summary.

3. **Episode wall-clock via a terminal metric, not a new record type.**
   The Gymnasium `(r, l, t)` triple is mostly already recoverable — return = Σ frame
   rewards, length = frame count — so only the wall-clock `t` is genuinely new. Rather
   than add an episode-summary chunk, the recording surface emits a terminal
   `episode_wall_clock_secs` (and, for convenience and eval episodes, `episode_return` /
   `episode_length`) `MetricSample` at `on_episode_end`. No new `RecordChunk` variant.

4. **Typed run provenance on `RunManifest` (wire bump).** Add, all `Option`/`#[serde(default)]`
   so the TOML stays tolerant, alongside the retained `hyperparameters` map (which keeps
   the untyped long tail):
   - `algorithm: Option<String>` — the load-bearing field; lets the report tier choose
     loss panels without grepping `hyperparameters`.
   - `rlevo_version`, `rustc_version`, `burn_version`, `platform` — software stack.
   - `git_commit: Option<String>`, `git_dirty: Option<bool>` — source provenance.
   - `device: Option<String>` — CPU/GPU backend descriptor.
   - `num_seeds: Option<u32>` — for cross-seed (IQM/CI) aggregation at the report tier.
   - `success_threshold: Option<f64>` — the threshold that produced `success_rate`,
     currently stranded in `EvaluatorConfig`.

   **Provenance gathering:** version/git/rustc embedded at **compile time** via a
   `build.rs` (vergen-style: `CARGO_PKG_VERSION` for `rlevo_version`, git rev + dirty
   flag, `rustc -V`, `burn` version from the resolved lockfile); `platform`/`device`
   resolved at **runtime** (`std::env::consts::OS`/`ARCH`, Burn backend type). Reversal:
   if the `build.rs` proves heavy or flaky in CI, fall back to a runtime `git` shell call
   guarded behind the `record` feature.

5. **`EpisodeKind` train/eval tag on `EpisodeRecordHeader` (wire bump).**
   Add `kind: EpisodeKind { Training, Evaluation }` (a `#[non_exhaustive]` enum). Eval
   rollouts are recorded as **separate episode files** tagged `Evaluation`; the metric
   samples captured into that file inherit the tag by file membership — no per-sample
   discriminator needed. `Training` is the default for existing producers.

6. **Learner-checkpoint seam — deep-RL only (manifest field + sink method).**
   Deep-RL produces a trained artifact the record cannot currently point to: the **learner**
   (policy / value networks). Burn owns model serialisation via its `Recorder` trait
   (`NamedMpkFileRecorder`, `BinFileRecorder`, …,
   https://burn.dev/books/burn/saving-and-loading.html), writing its own files. The record
   therefore **references** those files; it never embeds weights. This is the deep-RL analog
   of EA's `best_genome_digest` (a pointer, not the genome). Add:
   - `checkpoints: Vec<CheckpointRef>` on `RunManifest`, `#[serde(default)]` (empty for EA and
     un-wired RL — the field is genuinely free on the tolerant TOML/JSON path).
   - `CheckpointRef { step, kind: CheckpointKind {Periodic,Best,Final}, format:
     CheckpointFormat {NamedMpk,NamedMpkGz,Bin,Json,Other}, path (relative to run dir),
     metric: Option<f64>, digest: Option<[u8;16]> }`.
   - `register_checkpoint(&mut self, CheckpointRef)` on `RecordSink` (default no-op, like
     `on_population_sample`); the writer accumulates and merges into the manifest at
     `on_run_end`.

   Unlike the deferred paradigms in part 7, this seam has a concrete near-term producer
   (DQN/PPO training, Milestone 3/4), so the seam lands in v6; the **producer wiring** (the
   actual `Recorder` call + `register_checkpoint`), the recorder-format choice, retention
   policy, and resume-from-checkpoint are deferred (Tier D). Pure evolutionary-optimisation
   runs are unaffected.

7. **Out of scope for v6 — deferred to future bumps.**
   - **Multi-agent** (`FrameRecord.agents` map, `alive_agents`, header `agent_roster`,
     per-policy metric discriminators) — defer until a multi-agent producer exists.
   - **Variable-topology neuroevolution** (`PopulationSample.family_payload` enum for
     NEAT/ES/QD/Coevolution, `std_fitness`, a `GenealogyLog`) — defer to the Milestone 2
     neuroevolution work (neuroevolution-neat, advanced-hybrid-specialized-ea),
     where the strategy-side data shape will actually be known.
   - **Bayesian-network learning** (`EnvFamily::BayesNet`, `FamilyPayload::BnGraph`,
     `RecordChunk::BnIteration`, a distinct `BnLearningRecord`) — parking-lot; defer.

   Each is introducible **additively** later (new `Option` field, new `#[non_exhaustive]`
   enum variant, new `RecordChunk` variant — exactly how `RecordChunk::Population` arrived
   in v3) without disturbing v6. We do not add their empty shells now. (The learner-checkpoint
   seam in part 6 is the one exception that *does* land now, because it has a concrete
   near-term producer.)

8. **Design invariants (binding on v6 and every later bump).**
   - **Additive only.** Never replace `action: Vec<u8>` / `reward: f32` or the
     fixed-shape `PopulationSample` scalars; new structure is wrapped in `Option<…>` so
     the single-agent / fixed-genome fast path stays zero-cost. Append enum variants at
     the end; never reorder (bincode is positional).
   - **`BTreeMap`, not `HashMap`**, for any new map field (deterministic bincode).
   - **Typed fields over the `hyperparameters` string map** for anything the report tier
     branches on.
   - **Digests + external stores** for large/variable artifacts; sample on
     change / interval / run-end, never store full structure every step.
   - **Mirror obligation.** `FORMAT_VERSION`, `MIN_SUPPORTED_VERSION`, and every new wire
     type must be mirrored in `crates/rlevo-benchmarks-report-client/src/wire.rs`;
     `crates/rlevo-benchmarks/tests/wire_format_compat.rs` enforces the version equality
     at compile time. Both consts move to `6` together; v5 test corpora are regenerated.

### Reversal criteria

- If the typed manifest provenance proves to churn (e.g. `burn_version` extraction is
  brittle across Burn releases), demote the brittle subset back into the
  `hyperparameters` map and keep only `algorithm` + `rlevo_version` + git typed.
- If recording eval episodes as separate tagged files turns out to fragment a run's
  metric stream awkwardly for the report tier, promote `EpisodeKind` to a per-`MetricSample`
  discriminator instead (the rejected alternative below) — still additive.
- If the expanded allowlist becomes a maintenance bottleneck (every new algorithm edits a
  central list), replace the allowlist with a namespacing convention (`metric.*` prefix)
  parsed by the layer — a larger change deferred until the list actually hurts.

## Consequences

**Positive**

- A recorded run now supports a real return-vs-step learning curve, value-net health
  (`explained_variance`), and the per-algorithm losses for every RL algorithm we ship.
- A run is reproducible from its manifest alone (algorithm, versions, git, seed count),
  closing the Henderson/rliable provenance gap without a database.
- Eval and training rollouts are distinguishable, enabling honest eval-only reporting.
- The future paradigms are unblocked but uncommitted: v7+ can add them additively, and v6
  carries no speculative dead fields.
- Deep-RL trained models are addressable from the run: a `CheckpointRef` lets a post-run
  consumer locate (and Burn-reload) the learner without the record owning weight
  serialisation.

**Negative / accepted costs**

- **v5 files do not decode under v6.** Accepted under the standing
  `MIN_SUPPORTED_VERSION == FORMAT_VERSION` no-backcompat policy; corpora regenerate.
- **A `build.rs` enters `rlevo-benchmarks`** (behind/for the `record` path) to embed git
  and toolchain provenance — a small build-time surface with a runtime-shell reversal.
- **Producer wiring work.** The new per-iteration metrics and the episode wall-clock are
  only as good as the training loops that emit them; the schema permits but does not
  populate them. Tracked separately from this ADR.

**Neutral**

- Expanding `CANONICAL_METRICS` is version-neutral; it rides this PR only for coherence.
- `FamilyPayload`, `FrameRecord`'s render fields, and the writer state machine are
  untouched apart from the additive header/manifest fields.

## Alternatives considered

**Split the bump — metrics-allowlist now, manifest restructure later.** Rejected. The
allowlist expansion is version-neutral and the manifest/header changes are small and
cohesive; splitting them doubles the corpus-regeneration and mirror-sync churn for no
benefit, since both land in the same record surface.

**Introduce the MARL / NEAT / BN seams now as empty `Option` fields.** Rejected. It
repeats the v5 `parents_of_best` / `inner_rl_returns` anti-pattern (wire fields that
decode to empty because no producer populates them), and the additive-only invariant means
deferring them costs nothing — they slot in as new optional fields / enum variants when a
producer exists and the real data shape is known.

**Tag eval/train per `MetricSample` rather than per episode file.** Rejected for v6 as
heavier than needed — separate tagged episode files give the distinction for free via file
membership. Retained in the reversal criteria if the report tier needs finer granularity.

**Add an `EpisodeSummaryRecord` chunk for the `(r, l, t)` triple.** Rejected — `r` and `l`
are already recoverable from frames, so a whole new `RecordChunk` variant for one genuinely
new scalar (`t`) is overkill; a terminal `MetricSample` carries it with no schema surface.

## References

- `research/2026-06-04-benchmark-client-metrics.md` — the gap analysis and per-domain
  surveys this ADR acts on; §6 "Now" list maps to Decision parts 1–5, §6 "future seams"
  to part 6.
- [0013-metrics-only-live-tui](0013-metrics-only-live-tui.md) — preserved `EpisodeRecord` seam, `FamilyPayload`, and
  isolation rules that v6 extends rather than supersedes.
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — `SeedStream` (replay determinism),
  `Metric`/`MetricsProvider` underpinning the metric stream.
- [0003-collapse-rl-modules-into-rlevo-reinforcement-learning](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) — `AgentStats`/
  `PerformanceRecord`, the RL-side metric sources feeding the new per-iteration stats.
- reference_flex_gemm_nondeterminism — why a single `seed` is insufficient provenance;
  motivates documenting seed coverage and `num_seeds`.
- `crates/rlevo-benchmarks/src/record/{schema,manifest,writer,tracing_layer}.rs`,
  `crates/rlevo-benchmarks/src/metrics_registry.rs` — the surfaces v6 edits.
- `crates/rlevo-benchmarks-report-client/src/wire.rs`,
  `crates/rlevo-benchmarks/tests/wire_format_compat.rs` — the mirror + enforcement that
  must move to `6` in lockstep.
- neuroevolution-neat, advanced-hybrid-specialized-ea — Milestone 2 work that will
  own the deferred neuroevolution population-payload bump.
