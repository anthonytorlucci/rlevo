//! [`RolloutFitness`] — scores flat policy-parameter genomes by environment
//! rollout.
//!
//! This is the RL-coupled fitness boundary for weight-only neuroevolution. A
//! population row is unflattened (via a [`ModuleReshaper`]) into a policy
//! network, the network drives one or more episodes against a freshly
//! constructed environment, and the mean episode return is returned as
//! fitness — *negated*, because [`Strategy`](rlevo_evolution::Strategy)
//! minimizes while policy return is maximized.
//!
//! # Rank fixing
//!
//! `E: Environment<1, 1, 1>` — weight-only policy neuroevolution v1 targets
//! vector-observation, scalar/discrete-action classic-control environments
//! (CartPole, MountainCar, Pendulum, Acrobot), all rank-1. Higher-rank
//! observation/action spaces are a future generalization.
//!
//! # The policy closure
//!
//! The bridge from a generic module `M` to a concrete environment action is a
//! caller-supplied closure: `Fn(&M, &E::ObservationType, &Device) ->
//! E::ActionType`. The caller owns the concrete `M` and therefore knows its
//! `forward`; this keeps `RolloutFitness` free of any `Policy` supertrait on
//! `Module`.
//!
//! # Gradient isolation
//!
//! `B: Backend`, not `AutodiffBackend` — rollouts are forward-only.

use std::marker::PhantomData;
use std::sync::Arc;

use burn::module::Module;
use burn::tensor::{Tensor, TensorData, backend::Backend};

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::param_reshaper::{ModuleReshaper, ParamReshaper};

/// Device type for backend `B` (the `Device` associated type lives on the
/// `BackendTypes` supertrait in this Burn version).
type Dev<B> = <B as burn::tensor::backend::BackendTypes>::Device;

/// Constructs a fresh environment per evaluation episode.
type EnvFactory<E> = Arc<dyn Fn() -> E + Send + Sync>;

/// Maps a policy module + observation to an environment action.
type PolicyFn<B, M, E> = Arc<
    dyn Fn(
            &M,
            &<E as Environment<1, 1, 1>>::ObservationType,
            &Dev<B>,
        ) -> <E as Environment<1, 1, 1>>::ActionType
        + Send
        + Sync,
>;

/// A [`BatchFitnessFn`] that scores flat policy parameters by environment
/// rollout.
///
/// # Type Parameters
///
/// - `B`: Burn backend (non-autodiff).
/// - `M`: the policy network module.
/// - `E`: a rank-1 [`Environment`].
pub struct RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B>,
    E: Environment<1, 1, 1>,
{
    reshaper: ModuleReshaper<B, M>,
    env_factory: EnvFactory<E>,
    policy: PolicyFn<B, M, E>,
    episodes_per_eval: usize,
    max_steps_per_episode: usize,
    _backend: PhantomData<fn() -> B>,
}

impl<B, M, E> std::fmt::Debug for RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B>,
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
    M: Module<B>,
    E: Environment<1, 1, 1>,
{
    /// Build a rollout-based fitness function.
    ///
    /// # Arguments
    ///
    /// - `reshaper`: reshaper whose template matches the evolved network `M`.
    /// - `env_factory`: builds a fresh `E` for each episode (independent seeds
    ///   are the factory's responsibility).
    /// - `policy`: maps `(module, observation, device)` to an action.
    /// - `episodes_per_eval`: episodes averaged per genome (≥ 1).
    /// - `max_steps_per_episode`: hard per-episode step cap. Required because
    ///   environments such as CartPole have no intrinsic terminal step limit;
    ///   the cap guarantees evaluation terminates regardless of policy skill.
    pub fn new<FacFn, PolFn>(
        reshaper: ModuleReshaper<B, M>,
        env_factory: FacFn,
        policy: PolFn,
        episodes_per_eval: usize,
        max_steps_per_episode: usize,
    ) -> Self
    where
        FacFn: Fn() -> E + Send + Sync + 'static,
        PolFn: Fn(&M, &E::ObservationType, &Dev<B>) -> E::ActionType + Send + Sync + 'static,
    {
        assert!(episodes_per_eval >= 1, "episodes_per_eval must be >= 1");
        assert!(
            max_steps_per_episode >= 1,
            "max_steps_per_episode must be >= 1"
        );
        Self {
            reshaper,
            env_factory: Arc::new(env_factory),
            policy: Arc::new(policy),
            episodes_per_eval,
            max_steps_per_episode,
            _backend: PhantomData,
        }
    }

    /// Run one capped episode and return its cumulative reward.
    fn rollout_once(&self, module: &M, device: &Dev<B>) -> f32 {
        let mut env = (self.env_factory)();
        let mut snapshot = env.reset().expect("environment reset failed");
        let mut total = 0.0_f32;
        for _ in 0..self.max_steps_per_episode {
            if snapshot.is_done() {
                break;
            }
            let action = (self.policy)(module, snapshot.observation(), device);
            snapshot = env.step(action).expect("environment step failed");
            let reward: f32 = snapshot.reward().clone().into();
            total += reward;
        }
        total
    }
}

impl<B, M, E> BatchFitnessFn<B, Tensor<B, 2>> for RolloutFitness<B, M, E>
where
    B: Backend,
    M: Module<B> + Sync,
    E: Environment<1, 1, 1> + Send,
{
    fn evaluate_batch(&mut self, population: &Tensor<B, 2>, device: &Dev<B>) -> Tensor<B, 1> {
        let [pop_size, num_params] = population.dims();
        debug_assert_eq!(num_params, self.reshaper.num_params());
        #[allow(clippy::cast_precision_loss)]
        let episodes = self.episodes_per_eval as f32;
        let mut fitness: Vec<f32> = Vec::with_capacity(pop_size);
        for row in 0..pop_size {
            #[allow(clippy::single_range_in_vec_init)]
            let genome: Tensor<B, 1> = population
                .clone()
                .slice([row..row + 1])
                .reshape([num_params]);
            let module = self.reshaper.unflatten(genome);
            let mut total = 0.0_f32;
            for _ in 0..self.episodes_per_eval {
                total += self.rollout_once(&module, device);
            }
            // Negate: the strategy minimizes, policy return is maximized.
            fitness.push(-total / episodes);
        }
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }
}
