//! Integration smoke test: an `EvolutionaryLoop` scored by `RolloutFitness`
//! evolves a `CartPole` policy.
//!
//! This verifies correctness (compiles, wires `CartPole` through the full
//! weight-only stack, runs without panic), **not** convergence — two
//! generations on a tiny population is far too little to balance the pole.

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};

use rlevo_core::action::DiscreteAction;
use rlevo_environments::classic::cartpole::{CartPole, CartPoleAction, CartPoleObservation};
use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo_evolution::param_reshaper::ModuleReshaper;
use rlevo_evolution::strategy::EvolutionaryLoop;
use rlevo_hybrid::{RolloutFitness, StatefulPolicy};

type TestBackend = Flex;

/// `CartPole` policy: `obs(4) -> 8 -> logits(2)`.
///
/// Backend generic must be `B` — Burn's `#[derive(Module)]` references
/// `B::Device` literally.
#[derive(Module, Debug)]
struct PolicyMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> PolicyMlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 8).init(device),
            l2: LinearConfig::new(8, 2).init(device),
        }
    }

    fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::tanh(self.l1.forward(obs));
        self.l2.forward(h)
    }
}

/// Greedy reactive policy: encode the observation, forward, argmax → discrete
/// action. `PolicyMlp` is memoryless, so its `Hidden` is `()`.
impl StatefulPolicy<CartPoleObservation, CartPoleAction> for PolicyMlp<TestBackend> {
    type Hidden = ();

    fn reset(&self) {}

    fn act(&self, _hidden: &mut (), obs: &CartPoleObservation) -> CartPoleAction {
        let data = TensorData::new(
            vec![obs.cart_pos, obs.cart_vel, obs.pole_angle, obs.pole_ang_vel],
            [1, 4],
        );
        let x = Tensor::<TestBackend, 2>::from_data(data, &self.l1.weight.device());
        let logits = self.forward(x);
        let idx = logits.argmax(1).into_data().into_vec::<i32>().unwrap()[0];
        CartPoleAction::from_index(usize::try_from(idx).unwrap())
    }
}

#[test]
fn rollout_fitness_drives_two_generations_on_cartpole() {
    let device = Default::default();
    let template = PolicyMlp::<TestBackend>::new(&device);
    // Single-source width: build ONE reshaper, read its width, then move it into
    // the fitness so both agree on `num_params` by construction.
    let reshaper = ModuleReshaper::new(template);
    let num_params = reshaper.num_params();
    // 4*8 + 8 + 8*2 + 2 = 58
    assert_eq!(num_params, 58);

    let fitness = RolloutFitness::new(
        reshaper,
        CartPole::default,
        1,  // episodes_per_eval
        50, // max_steps_per_episode (CartPole has no intrinsic cap)
    );

    let params = GaConfig::default_for(16, num_params);
    let mut evo_loop = EvolutionaryLoop::new(
        GeneticAlgorithm::<TestBackend>::new(),
        params,
        fitness,
        7,
        device,
        2,
    )
    .expect("valid params");

    evo_loop.reset();
    evo_loop.step();
    assert!(
        !evo_loop.is_done(),
        "first generation should not exhaust the budget"
    );
    evo_loop.step();
    assert!(
        evo_loop.is_done(),
        "second generation should exhaust the 2-generation budget"
    );
    assert_eq!(evo_loop.generation(), 2);
    assert!(
        evo_loop.best().is_some(),
        "a best individual should exist after evaluation"
    );
}
