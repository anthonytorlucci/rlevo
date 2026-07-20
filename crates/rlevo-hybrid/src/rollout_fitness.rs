//! [`RolloutFitness`] — scores flat policy-parameter genomes by environment
//! rollout.
//!
//! This is the RL-coupled fitness boundary for weight-only neuroevolution. A
//! population row is unflattened (via a [`ModuleReshaper`]) into a policy
//! network, the network drives one or more episodes against a freshly
//! constructed environment, and the mean episode return is returned **directly**
//! as fitness (no negation): the engine is maximise-native and policy return is
//! a maximisation, so [`RolloutFitness`] declares
//! [`ObjectiveSense::Maximize`](rlevo_core::objective::ObjectiveSense::Maximize).
//!
//! # One scoring path
//!
//! `RolloutFitness` is, structurally, a
//! [`ModuleEvalFn`] whose scorer
//! happens to be an environment rollout. It therefore **holds** an inner
//! `ModuleEvalFn` and delegates [`evaluate_batch`](BatchFitnessFn::evaluate_batch)
//! to it: the slice/unflatten/collect scaffolding lives once, in
//! `rlevo-evolution`. The rollout scorer is the only caller of `rollout_once`.
//!
//! # Stateful policies
//!
//! The evolved module `M` carries the rollout contract via
//! [`StatefulPolicy`]: `rollout_once` calls
//! [`reset`](StatefulPolicy::reset) once at episode start and threads
//! `&mut Hidden` through the step loop, so **recurrent / memory policies are
//! first-class**. A memoryless classic-control policy is the `Hidden = ()` case
//! supplied for free by [`ReactivePolicy`](crate::policy::ReactivePolicy).
//!
//! # Rank fixing
//!
//! `E: Environment<1, 1, 1>` — weight-only policy neuroevolution v1 targets
//! vector-observation, scalar/discrete-action classic-control environments
//! (`CartPole`, `MountainCar`, Pendulum, Acrobot), all rank-1. Higher-rank
//! observation/action spaces are a future generalization.
//!
//! # Gradient isolation
//!
//! `B: Backend`, not `AutodiffBackend` — rollouts are forward-only. `Hidden` is
//! a plain runtime value, never a tracked leaf.

use std::marker::PhantomData;
use std::sync::Arc;

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::module_eval_fn::ModuleEvalFn;
use rlevo_evolution::param_reshaper::ModuleReshaper;

use crate::policy::StatefulPolicy;

/// Device type for backend `B` (the `Device` associated type lives on the
/// `BackendTypes` supertrait in this Burn version).
type Dev<B> = <B as burn::tensor::backend::BackendTypes>::Device;

/// Constructs a fresh environment per evaluation episode.
type EnvFactory<E> = Arc<dyn Fn() -> E + Send + Sync>;

/// Host-side scorer stored inside the inner [`ModuleEvalFn`].
///
/// `Box<dyn Fn>` (not `Arc`) because `Box<F>` implements `Fn` — satisfying
/// [`ModuleEvalFn`]'s `F: Fn(&R::Module) -> f32 + Send` bound — whereas `Arc<F>`
/// does not. `+ Send` matches the [`BatchFitnessFn`] supertrait; `Sync` is not
/// required.
type RolloutScorer<M> = Box<dyn Fn(&M) -> f32 + Send>;

/// Run one capped episode and return its cumulative reward, threading the
/// policy's hidden state across the step loop.
///
/// [`StatefulPolicy::reset`] is called once at episode start; each step advances
/// the hidden state via [`StatefulPolicy::act`].
fn rollout_once<B, M, E>(
    module: &M,
    env_factory: &EnvFactory<E>,
    max_steps: usize,
    device: &Dev<B>,
) -> f32
where
    B: Backend,
    M: StatefulPolicy<B, E>,
    E: Environment<1, 1, 1>,
{
    let mut env = env_factory();
    let mut snapshot = env.reset().expect("environment reset failed");
    let mut hidden = StatefulPolicy::reset(module, device);
    let mut total = 0.0_f32;
    for _ in 0..max_steps {
        if snapshot.is_done() {
            break;
        }
        let action = StatefulPolicy::act(module, &mut hidden, snapshot.observation(), device);
        snapshot = env.step(action).expect("environment step failed");
        let reward: f32 = snapshot.reward().clone().into();
        total += reward;
    }
    total
}

/// A [`BatchFitnessFn`] that scores flat policy parameters by environment
/// rollout.
///
/// # Type Parameters
///
/// - `B`: Burn backend (non-autodiff).
/// - `M`: the policy network module; must implement [`StatefulPolicy`] (a
///   reactive module gets this for free via [`ReactivePolicy`]).
/// - `E`: a rank-1 [`Environment`].
///
/// [`ReactivePolicy`]: crate::policy::ReactivePolicy
pub struct RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1>,
{
    inner: ModuleEvalFn<B, ModuleReshaper<B, M>, RolloutScorer<M>>,
    episodes_per_eval: usize,
    max_steps_per_episode: usize,
    _env: PhantomData<fn() -> E>,
}

impl<B, M, E> std::fmt::Debug for RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RolloutFitness")
            .field("episodes_per_eval", &self.episodes_per_eval)
            .field("max_steps_per_episode", &self.max_steps_per_episode)
            .finish_non_exhaustive()
    }
}

impl<B, M, E> RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1>,
{
    /// Build a rollout-based fitness function.
    ///
    /// # Arguments
    ///
    /// - `reshaper`: reshaper whose template matches the evolved network `M`.
    /// - `env_factory`: builds a fresh `E` for each episode (independent seeds
    ///   are the factory's responsibility).
    /// - `episodes_per_eval`: episodes averaged per genome (≥ 1).
    /// - `max_steps_per_episode`: hard per-episode step cap. Required because
    ///   environments such as `CartPole` have no intrinsic terminal step limit;
    ///   the cap guarantees evaluation terminates regardless of policy skill.
    /// - `device`: captured into the rollout scorer (the inner [`ModuleEvalFn`]
    ///   scorer takes no device parameter). Must equal the `device` later
    ///   handed to [`evaluate_batch`](BatchFitnessFn::evaluate_batch) — the
    ///   single-device convention `ModuleEvalFn` already documents.
    ///
    /// The policy is the module itself: `M` implements [`StatefulPolicy`]
    /// (recurrent) or [`ReactivePolicy`](crate::policy::ReactivePolicy)
    /// (memoryless), so there is no policy closure argument.
    ///
    /// # Panics
    ///
    /// Panics if `episodes_per_eval` or `max_steps_per_episode` is zero. Both
    /// are rejected at construction rather than yielding a degenerate scorer:
    /// zero episodes would divide by zero, and a zero step cap would score
    /// every genome identically at 0.
    pub fn new<FacFn>(
        reshaper: ModuleReshaper<B, M>,
        env_factory: FacFn,
        episodes_per_eval: usize,
        max_steps_per_episode: usize,
        device: Dev<B>,
    ) -> Self
    where
        FacFn: Fn() -> E + Send + Sync + 'static,
        M: 'static,
        E: 'static,
    {
        assert!(episodes_per_eval >= 1, "episodes_per_eval must be >= 1");
        assert!(
            max_steps_per_episode >= 1,
            "max_steps_per_episode must be >= 1"
        );
        let env_factory: EnvFactory<E> = Arc::new(env_factory);
        let scorer: RolloutScorer<M> = Box::new(move |module: &M| -> f32 {
            #[allow(clippy::cast_precision_loss)]
            let episodes = episodes_per_eval as f32;
            let mut total = 0.0_f32;
            for _ in 0..episodes_per_eval {
                total +=
                    rollout_once::<B, M, E>(module, &env_factory, max_steps_per_episode, &device);
            }
            // Natural mean episode return — the engine is maximise-native and
            // policy return is a maximisation, so there is no hand-negation.
            total / episodes
        });
        Self {
            inner: ModuleEvalFn::new(reshaper, scorer),
            episodes_per_eval,
            max_steps_per_episode,
            _env: PhantomData,
        }
    }
}

impl<B, M, E> BatchFitnessFn<B, Tensor<B, 2>> for RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B> + Sync + StatefulPolicy<B, E>,
    E: Environment<1, 1, 1> + Send,
{
    fn evaluate_batch(&mut self, population: &Tensor<B, 2>, device: &Dev<B>) -> Tensor<B, 1> {
        self.inner.evaluate_batch(population, device)
    }

    fn sense(&self) -> rlevo_core::objective::ObjectiveSense {
        // Mean episode return — higher is better. Declared once, on the inner
        // ModuleEvalFn (default ObjectiveSense::Maximize).
        self.inner.sense()
    }
}
