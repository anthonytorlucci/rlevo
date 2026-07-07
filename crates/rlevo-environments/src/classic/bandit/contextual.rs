//! Contextual k-armed bandit — tabular discrete-context formulation.
//!
//! At each step the environment reveals a discrete context `c ∈ {0, …, C-1}`
//! (drawn uniformly from the seeded RNG) and the agent picks an arm
//! `a ∈ {0, …, K-1}`. The reward is sampled from `N(q*(c, a), 1)` where the
//! per-context, per-arm means `q*(c, a)` are drawn once from `N(0, 1)` at
//! construction (and re-drawn from the same seed on
//! [`Environment::reset`]).
//!
//! This is the simplest contextual-bandit testbed — a `C × K` table of
//! Gaussian means — which exercises algorithms that must learn a separate
//! policy per context (e.g. tabular contextual ε-greedy, contextual
//! Thompson sampling). For continuous-feature contextual bandits (LinUCB et
//! al.) a separate environment with a vector-valued context is appropriate;
//! it is intentionally out of scope for this module.
//!
//! # Example
//!
//! ```rust
//! use rlevo_core::environment::{ConstructableEnv, Environment, Snapshot};
//! use rlevo_environments::classic::{ContextualBandit, ContextualBanditConfig, KArmedBanditAction};
//!
//! let mut env = ContextualBandit::<4, 10>::with_config(ContextualBanditConfig::default())
//!     .expect("valid config");
//! let _ = <ContextualBandit<4, 10> as Environment<1, 1, 1>>::reset(&mut env)
//!     .expect("reset succeeds");
//! let action = KArmedBanditAction::<10>::new(2).expect("arm in range");
//! let snap = <ContextualBandit<4, 10> as Environment<1, 1, 1>>::step(&mut env, action)
//!     .expect("step");
//! assert!(!snap.is_done());
//! ```

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rlevo_core::base::{
    Action, Observation, Reward, State, TensorConversionError, TensorConvertible,
};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use super::k_armed::KArmedBanditAction;

// ---------------------------------------------------------------------------
// State / Observation
// ---------------------------------------------------------------------------

/// Contextual-bandit state: the index of the currently-presented context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ContextualBanditState<const C: usize> {
    context: usize,
}

impl<const C: usize> ContextualBanditState<C> {
    /// The currently-presented context index in `0..C`.
    #[must_use]
    pub fn context(&self) -> usize {
        self.context
    }
}

impl<const C: usize> Display for ContextualBanditState<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContextualBanditState<{C}>(context={})", self.context)
    }
}

/// Observation for the contextual bandit — the index of the context revealed
/// to the agent before it acts.
///
/// The raw struct holds the context index as a `usize`. When converted via
/// [`TensorConvertible::to_tensor`], the index is encoded as a one-hot vector
/// of length `C` for use with neural-network policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ContextualBanditObservation<const C: usize> {
    /// Index of the revealed context (`0..C`).
    pub context: usize,
}

impl<const C: usize> Observation<1> for ContextualBanditObservation<C> {
    fn shape() -> [usize; 1] {
        [C]
    }
}

impl<const C: usize> State<1> for ContextualBanditState<C> {
    type Observation = ContextualBanditObservation<C>;

    fn shape() -> [usize; 1] {
        [C]
    }

    fn observe(&self) -> Self::Observation {
        ContextualBanditObservation {
            context: self.context,
        }
    }

    fn is_valid(&self) -> bool {
        self.context < C
    }

    fn numel(&self) -> usize {
        C
    }
}

impl<const C: usize, B: Backend> TensorConvertible<1, B> for ContextualBanditObservation<C> {
    /// Row shape of the one-hot context encoding: `[C]`.
    fn row_shape() -> [usize; 1] {
        [C]
    }

    /// One-hot encoding of the current context, length `C`.
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        let mut one_hot = [0.0_f32; C];
        one_hot[self.context] = 1.0;
        buf.extend_from_slice(&one_hot);
    }

    /// Reconstructs an observation from a one-hot tensor by argmax.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[C]`.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [C] {
            return Err(TensorConversionError {
                message: format!("expected shape [{C}], got {dims:?}"),
            });
        }
        let data = tensor.into_data();
        let values: Vec<f32> = data.to_vec().map_err(|e| TensorConversionError {
            message: format!("failed to extract tensor data: {e:?}"),
        })?;
        let (argmax, _) = values.iter().enumerate().fold(
            (0_usize, f32::NEG_INFINITY),
            |(i_best, v_best), (i, &v)| {
                if v > v_best { (i, v) } else { (i_best, v_best) }
            },
        );
        Ok(Self { context: argmax })
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`ContextualBandit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualBanditConfig {
    /// Maximum number of steps before the episode terminates.
    pub max_steps: usize,
    /// RNG seed driving both the per-`(context, arm)` mean draw and the
    /// per-step context sampling. Default: `42`.
    pub seed: u64,
}

impl Default for ContextualBanditConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            seed: 42,
        }
    }
}

impl Validate for ContextualBanditConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "ContextualBanditConfig";
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

/// Parses the same `"N"` / `"max_steps=N,seed=S"` formats as
/// [`super::k_armed::KArmedBanditConfig`].
///
/// # Errors
///
/// Returns a `String` error if the input is neither a bare integer nor a
/// comma-separated list of `key=value` pairs with keys `max_steps` and
/// `seed`, or if any numeric value fails to parse.
impl FromStr for ContextualBanditConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();

        if let Ok(max_steps) = trimmed.parse::<usize>() {
            return Ok(Self {
                max_steps,
                ..Self::default()
            });
        }

        let mut cfg = Self::default();
        let mut saw_key = false;
        for pair in trimmed.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let Some(eq_pos) = pair.find('=') else {
                return Err(format!(
                    "Invalid ContextualBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
                ));
            };
            let key = pair[..eq_pos].trim();
            let value_str = pair[eq_pos + 1..].trim();
            match key {
                "max_steps" => {
                    cfg.max_steps = value_str
                        .parse::<usize>()
                        .map_err(|e| format!("Failed to parse max_steps value: {e}"))?;
                }
                "seed" => {
                    cfg.seed = value_str
                        .parse::<u64>()
                        .map_err(|e| format!("Failed to parse seed value: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "Unknown ContextualBanditConfig key {other:?} (expected max_steps or seed)"
                    ));
                }
            }
            saw_key = true;
        }

        if saw_key {
            Ok(cfg)
        } else {
            Err(format!(
                "Invalid ContextualBanditConfig format. Expected either a number or 'key=value' pairs, got: {s}"
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Tabular contextual bandit with `C` discrete contexts and `K` arms.
///
/// On each step the environment draws the current context uniformly from
/// `{0, …, C-1}` and the agent selects an arm `{0, …, K-1}`. Reward is
/// sampled from `N(q*(context, arm), 1)` using a `C × K` table of means
/// fixed at construction and reset by [`Environment::reset`].
///
/// The snapshot returned by [`Environment::step`] carries the *next* context
/// in its observation (the one the agent will act on in the following step),
/// not the context that produced the current reward. This matches the standard
/// contextual-bandit loop where the context is revealed before each decision.
#[derive(Debug)]
pub struct ContextualBandit<const C: usize, const K: usize> {
    state: ContextualBanditState<C>,
    steps: usize,
    done: bool,
    config: ContextualBanditConfig,
    rng: StdRng,
    arm_means: [[f32; K]; C],
}

impl<const C: usize, const K: usize> Display for ContextualBandit<C, K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ContextualBandit<{C},{K}>(step={}/{}, context={}, done={})",
            self.steps, self.config.max_steps, self.state.context, self.done
        )
    }
}

impl<const C: usize, const K: usize> ContextualBandit<C, K> {
    /// Construct with a specific seed (other config fields default).
    pub fn with_seed(seed: u64) -> Self {
        let config = ContextualBanditConfig {
            seed,
            ..ContextualBanditConfig::default()
        };
        Self::with_config(config).expect("default-derived config must validate")
    }

    /// Construct with an explicit config.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]
    /// (`max_steps == 0`).
    pub fn with_config(config: ContextualBanditConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let arm_means = sample_arm_means::<C, K>(&mut rng);
        let context = rng.random_range(0..C);
        Ok(Self {
            state: ContextualBanditState { context },
            steps: 0,
            done: false,
            config,
            rng,
            arm_means,
        })
    }

    /// Read-only view of the per-context, per-arm means.
    #[must_use]
    pub fn arm_means(&self) -> &[[f32; K]; C] {
        &self.arm_means
    }

    /// The context currently revealed to the agent (`0..C`).
    #[must_use]
    pub fn current_context(&self) -> usize {
        self.state.context
    }

    fn sample_reward(&mut self, context: usize, arm: usize) -> f32 {
        let mean = self.arm_means[context][arm];
        Normal::new(mean, 1.0)
            .expect("N(mean, 1) is always valid")
            .sample(&mut self.rng)
    }
}

fn sample_arm_means<const C: usize, const K: usize>(rng: &mut StdRng) -> [[f32; K]; C] {
    let normal = Normal::new(0.0_f32, 1.0).expect("N(0, 1) is always valid");
    let mut means = [[0.0_f32; K]; C];
    for context_row in &mut means {
        for mean in context_row.iter_mut() {
            *mean = normal.sample(rng);
        }
    }
    means
}

impl<const C: usize, const K: usize> ConstructableEnv for ContextualBandit<C, K> {
    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(ContextualBanditConfig::default()).expect("default config must validate")
    }
}

impl<const C: usize, const K: usize> Environment<1, 1, 1> for ContextualBandit<C, K> {
    type StateType = ContextualBanditState<C>;
    type ObservationType = ContextualBanditObservation<C>;
    type ActionType = KArmedBanditAction<K>;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, ContextualBanditObservation<C>, ScalarReward>;

    /// Reset episode state and reveal a fresh first context.
    ///
    /// The fixed per-context arm means (the problem) are sampled once at
    /// construction and **preserved** across resets; the persistent RNG is
    /// **not** re-seeded, so the revealed context and the reward realisations
    /// vary independently each episode (host-RNG seeding convention,
    /// `docs/rules.md` §8). For a reproducible problem, construct with a fixed
    /// seed.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state.context = self.rng.random_range(0..C);
        self.steps = 0;
        self.done = false;
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward::zero(),
        ))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "arm index {} out of range [0, {K})",
                action.arm(),
            )));
        }
        let context = self.state.context;
        let reward = ScalarReward(self.sample_reward(context, action.arm()));
        self.steps += 1;
        // Reveal the next context after computing reward for the current one.
        self.state.context = self.rng.random_range(0..C);
        let obs = self.state.observe();
        let snap = if self.steps >= self.config.max_steps {
            self.done = true;
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl<const K: usize, const C: usize> crate::render::AsciiRenderable for ContextualBandit<K, C> {
    fn render_ascii(&self) -> String {
        let ctx = self.state.context;
        let (best_arm, best_mean) = super::k_armed::argmax(&self.arm_means[ctx]);
        format!(
            "Contextual (K={K}, C={C})  ctx={ctx}  best@ctx={best_arm} (q*={best_mean:.2})  step={}/{}",
            self.steps, self.config.max_steps
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        let line = self.render_ascii();
        crate::render::StyledFrame {
            lines: vec![super::k_armed::style_bandit_line(&line)],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::environment::Snapshot;

    #[test]
    fn default_config_validates() {
        assert!(ContextualBanditConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_max_steps() {
        let bad = ContextualBanditConfig {
            max_steps: 0,
            seed: 0,
        };
        assert!(ContextualBandit::<4, 10>::with_config(bad).is_err());
    }

    type TestBackend = burn::backend::Flex;
    const C: usize = 4;
    const K: usize = 10;

    #[test]
    fn observation_round_trips_through_tensor() {
        let device = Default::default();
        for ctx in 0..C {
            let obs = ContextualBanditObservation::<C> { context: ctx };
            let tensor =
                <ContextualBanditObservation<C> as TensorConvertible<1, TestBackend>>::to_tensor(
                    &obs, &device,
                );
            let back =
                <ContextualBanditObservation<C> as TensorConvertible<1, TestBackend>>::from_tensor(
                    tensor,
                )
                .expect("round-trip should succeed");
            assert_eq!(back.context, ctx);
        }
    }

    #[test]
    fn observation_from_tensor_rejects_wrong_shape() {
        use burn::tensor::{Tensor, TensorData as TD};
        let device = Default::default();
        let data = TD::new(vec![0.0_f32; 2], [2]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let err =
            <ContextualBanditObservation<C> as TensorConvertible<1, TestBackend>>::from_tensor(
                tensor,
            )
            .expect_err("shape [2] should be rejected");
        assert!(err.message.contains("expected shape"));
    }

    #[test]
    fn environment_reset_yields_running_snapshot_with_zero_reward() {
        let mut env = ContextualBandit::<C, K>::with_config(ContextualBanditConfig::default())
            .expect("valid config");
        let snap =
            <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
        assert!(!snap.is_done());
        assert_eq!(f32::from(*snap.reward()), 0.0);
        assert!(snap.observation().context < C);
    }

    #[test]
    fn step_observation_matches_revealed_context_after_step() {
        let mut env = ContextualBandit::<C, K>::with_config(ContextualBanditConfig::default())
            .expect("valid config");
        <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let action = KArmedBanditAction::<K>::from_index(0);
        let snap =
            <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        // After `step`, the observation reflects the *next* context that the
        // env has just sampled.
        assert_eq!(snap.observation().context, env.current_context());
    }

    #[test]
    fn environment_step_terminates_at_max_steps() {
        let mut env = ContextualBandit::<C, K>::with_config(ContextualBanditConfig {
            max_steps: 3,
            seed: 1,
        })
        .expect("valid config");
        let action = KArmedBanditAction::<K>::from_index(0);
        let s1 = <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s1.is_done());
        let s2 = <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(!s2.is_done());
        let s3 = <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut env, action).unwrap();
        assert!(s3.is_terminated());
    }

    #[test]
    fn same_seed_produces_identical_trajectories() {
        let cfg = ContextualBanditConfig {
            max_steps: 32,
            seed: 11,
        };
        let mut a = ContextualBandit::<C, K>::with_config(cfg.clone()).expect("valid config");
        let mut b = ContextualBandit::<C, K>::with_config(cfg).expect("valid config");
        <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut a).unwrap();
        <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut b).unwrap();
        assert_eq!(a.arm_means(), b.arm_means());
        assert_eq!(a.current_context(), b.current_context());

        for step in 0..32 {
            let action = KArmedBanditAction::<K>::from_index(step % K);
            let snap_a =
                <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut a, action).unwrap();
            let snap_b =
                <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut b, action).unwrap();
            assert_eq!(f32::from(*snap_a.reward()), f32::from(*snap_b.reward()));
            assert_eq!(snap_a.observation().context, snap_b.observation().context);
            assert_eq!(snap_a.status(), snap_b.status());
        }
    }

    #[test]
    fn reset_keeps_stable_arm_means() {
        // The per-context arm means (the problem) are sampled once at
        // construction and preserved across resets.
        let mut env = ContextualBandit::<C, K>::with_config(ContextualBanditConfig {
            max_steps: 32,
            seed: 5,
        })
        .expect("valid config");
        let before = *env.arm_means();
        let action = KArmedBanditAction::<K>::from_index(0);
        for _ in 0..5 {
            let _ = <ContextualBandit<C, K> as Environment<1, 1, 1>>::step(&mut env, action);
        }
        <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        assert_eq!(before, *env.arm_means());
    }

    #[test]
    fn successive_episodes_draw_independent_rewards() {
        // The persistent RNG advances across resets, so the revealed contexts
        // and reward realisations differ between episodes on the same problem.
        let cfg = ContextualBanditConfig {
            max_steps: 100,
            seed: 5,
        };
        let mut env = ContextualBandit::<C, K>::with_config(cfg).expect("valid config");
        let action = KArmedBanditAction::<K>::from_index(0);
        let episode = |env: &mut ContextualBandit<C, K>| -> Vec<f32> {
            (0..16)
                .map(|_| {
                    f32::from(
                        *<ContextualBandit<C, K> as Environment<1, 1, 1>>::step(env, action)
                            .unwrap()
                            .reward(),
                    )
                })
                .collect()
        };
        let episode1 = episode(&mut env);
        <ContextualBandit<C, K> as Environment<1, 1, 1>>::reset(&mut env).unwrap();
        let episode2 = episode(&mut env);
        assert_ne!(
            episode1, episode2,
            "reward realisations must differ across episodes"
        );
    }

    #[test]
    fn arm_means_dimensions_match_const_generics() {
        let env = ContextualBandit::<C, K>::with_config(ContextualBanditConfig::default())
            .expect("valid config");
        assert_eq!(env.arm_means().len(), C);
        for row in env.arm_means() {
            assert_eq!(row.len(), K);
        }
    }

    #[test]
    fn fromstr_simple_number_sets_max_steps() {
        let c: ContextualBanditConfig = "300".parse().unwrap();
        assert_eq!(c.max_steps, 300);
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn fromstr_two_keys() {
        let c: ContextualBanditConfig = "max_steps=50,seed=3".parse().unwrap();
        assert_eq!(c.max_steps, 50);
        assert_eq!(c.seed, 3);
    }

    #[test]
    fn fromstr_unknown_key_errors() {
        let err: String = "wrong=1".parse::<ContextualBanditConfig>().unwrap_err();
        assert!(err.contains("Unknown ContextualBanditConfig key"));
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let env: ContextualBandit<10, 4> = ContextualBandit::with_seed(7);
        let plain = env.render_ascii();
        let styled = env.render_styled();
        assert_eq!(styled.lines.len(), 1);
        assert_eq!(styled.plain_text(), plain);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let env: ContextualBandit<10, 4> = ContextualBandit::with_seed(7);
        let styled = env.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text.starts_with("Contextual"))
            .expect("Contextual label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let env: ContextualBandit<10, 4> = ContextualBandit::with_seed(7);
        for line in env.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
