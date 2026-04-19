# Benchmarks

Criterion-based micro and macro benchmarks provide standardized, reproducible
data points for the hot paths in each crate. All benches use the `NdArray`
backend (or `Autodiff<NdArray>`) with fixed RNG seeds so runs are comparable
across machines and commits. Save a baseline with
`cargo bench -- --save-baseline <name>` and compare with `--baseline <name>`
to check for regressions.

## evorl-evolution

Run with `cargo bench -p evorl-evolution`. Reference point for the custom
CubeCL kernel work in `ops/kernels/` — kernels must strictly beat the
pure-tensor baseline at `pop_size ≥ 256`.

- **`tournament_select`** — `ops::selection::tournament_select` at
  `pop_size ∈ {64, 256, 1024}`, `dim = 10`, tournament size 2. Seeded
  `StdRng`, shared pre-built population tensor.
- **`de_one_generation`** — one full `ask → evaluate → tell` cycle of
  `DifferentialEvolution` (`Rand1Bin`) at `pop_size ∈ {64, 256, 1024}`,
  `dim = 10`, `ZeroFitness`. Uses `iter_batched` with a per-iteration fresh
  harness that is warmed for one generation so init costs are excluded from
  the measurement window; `sample_size = 10`.

## evorl-rl

Run with `cargo bench -p evorl-rl --bench <name>`. All benches are micro
unless noted. Macro CartPole reward-curve reproductions (e.g. DQN) are gated
behind the `macro` env var and compared against
`tests/baselines/<algo>_<env>.csv`.

### `dqn_bench`
- **`dqn_act_single_obs`** — single-observation ε-greedy `act` on a
  4→64→64→2 MLP over `CartPoleObservation`.
- **`dqn_learn_step_batch64`** — one `learn_step` on a buffer pre-populated
  by 2 000 real CartPole transitions, batch size 64, replay capacity 10 000.

### `c51_bench`
- **`project_distribution/atoms={21,51,101}/batch={32,128}`** — C51
  categorical Bellman projection (`algorithms::c51::projection`) with
  uniform `next_probs`, `support ∈ [-10, 10]`, `γ = 0.99`, periodic
  terminals. Sized to catch any regression that turns the linear scatter-add
  into a super-linear broadcast.

### `qrdqn_bench`
- **`quantile_huber_loss/quantiles={51,101,200}/batch={32,128}`** —
  `quantile_huber_loss` with deterministic saw-tooth `pred`/`target` and
  midpoint `taus`, κ = 1.0. Sized to catch `(B, N, N) → (B, B, N, N)`
  broadcast regressions.

### `ppo_bench`
- **`ppo_compute_gae/{128,512,2048}`** — pure-Rust GAE over a synthetic
  trajectory, `γ = 0.99`, `λ = 0.95`, no terminals/truncations.
- **`ppo_advantage_norm/{64,256,1024}`** — `normalize_advantages` over a
  1-D tensor of monotone values.
- **`ppo_clipped_surrogate/{64,256,1024}`** — PPO clipped surrogate
  objective at `ε = 0.2` with constant `new_lp`/`old_lp`/`advs`.

### `ppg_bench`
(PPG inherits PPO's hot paths; only PPG-specific kernels live here.)
- **`ppg_policy_kl_categorical`** — distillation KL at
  `(batch, num_actions) ∈ {(64,2),(256,2),(1024,2),(256,18)}`.
- **`ppg_aux_value_loss_mse/{256,1024,4096}`** — aux-phase value-target MSE
  (same kernel as PPO's unclipped value loss) at aux-phase-shaped sizes.

### `ddpg_bench`
- **`ddpg_act_single_obs_eval`** — deterministic actor forward (no
  exploration noise) on a 3→256→256→1 `tanh`-scaled MLP over
  `PendulumObservation`.
- **`ddpg_learn_step_batch256`** — one `learn_step` (critic update +
  delayed actor/Polyak step at `policy_frequency = 2`) on a Pendulum-primed
  replay buffer (2 000 transitions), batch size 256, capacity 10 000.

### `td3_bench`
- **`td3_act_single_obs_eval`** — deterministic actor forward, same MLP
  shape as DDPG.
- **`td3_learn_step_batch256`** — one `learn_step` with twin-critic update
  and delayed actor/Polyak step (`policy_frequency = 2`) on a
  Pendulum-primed buffer, batch size 256.

### `sac_bench`
- **`sac_act_single_obs_eval`** — deterministic (mean) actor forward on the
  stochastic actor, no sampling.
- **`sac_learn_step_batch256`** — one `learn_step` (twin-critic + entropy
  temperature + delayed actor/Polyak) on a Pendulum-primed buffer, batch
  size 256, `policy_frequency = 2`.
