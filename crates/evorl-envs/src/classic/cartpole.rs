// use evorl_core::action::{Action, DiscreteAction};
// use evorl_core::environment::Environment;
// use evorl_core::state::{FlattenedState, State, StateError};
// use rand::RngExt;

// /// CartPole state: [x, x_dot, theta, theta_dot]
// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// pub struct CartPoleState {
//     data: [OrderedFloat<f32>; 4], // Use ordered float for Eq/Hash
// }

// impl State for CartPoleState {
//     fn is_valid(&self) -> bool {
//         self.data.iter().all(|f| f.is_finite())
//     }

//     fn numel(&self) -> usize {
//         4
//     }

//     fn shape(&self) -> Vec<usize> {
//         vec![4]
//     }
// }

// impl FlattenedState for CartPoleState {
//     fn flatten(&self) -> Vec<f32> {
//         self.data.iter().map(|f| f.0).collect()
//     }

//     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
//         if data.len() != 4 {
//             return Err(StateError::InvalidSize {
//                 expected: 4,
//                 got: data.len(),
//             });
//         }
//         Ok(Self {
//             data: [
//                 OrderedFloat(data[0]),
//                 OrderedFloat(data[1]),
//                 OrderedFloat(data[2]),
//                 OrderedFloat(data[3]),
//             ],
//         })
//     }
// }

// /// CartPole actions: Left (0) or Right (1)
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum CartPoleAction {
//     Left,
//     Right,
// }

// impl Action for CartPoleAction {
//     fn is_valid(&self) -> bool {
//         true
//     }
// }

// impl DiscreteAction for CartPoleAction {
//     const ACTION_COUNT: usize = 2;

//     fn from_index(index: usize) -> Self {
//         match index {
//             0 => Self::Left,
//             1 => Self::Right,
//             _ => panic!("Invalid CartPole action index: {}", index),
//         }
//     }

//     fn to_index(&self) -> usize {
//         match self {
//             Self::Left => 0,
//             Self::Right => 1,
//         }
//     }
// }

// /// CartPole environment
// #[derive(Debug, Clone)]
// pub struct CartPole {
//     state: CartPoleState,
//     steps: usize,
//     done: bool,
//     config: CartPoleConfig,
// }

// #[derive(Debug, Clone)]
// pub struct CartPoleConfig {
//     pub gravity: f32,
//     pub mass_cart: f32,
//     pub mass_pole: f32,
//     pub length: f32,
//     pub force_mag: f32,
//     pub tau: f32, // Time step
//     pub theta_threshold: f32,
//     pub x_threshold: f32,
//     pub max_steps: usize,
// }

// impl Default for CartPoleConfig {
//     fn default() -> Self {
//         Self {
//             gravity: 9.8,
//             mass_cart: 1.0,
//             mass_pole: 0.1,
//             length: 0.5,
//             force_mag: 10.0,
//             tau: 0.02,
//             theta_threshold: 12.0 * (std::f32::consts::PI / 180.0),
//             x_threshold: 2.4,
//             max_steps: 500,
//         }
//     }
// }

// impl Environment for CartPole {
//     type State = CartPoleState;
//     type Action = CartPoleAction;
//     type Error = CartPoleError;

//     fn reset(&mut self) -> Result<Self::State, Self::Error> {
//         let mut rng = rand::rng();
//         self.state = CartPoleState {
//             data: [
//                 OrderedFloat(rng.gen_range(-0.05..0.05)),
//                 OrderedFloat(rng.gen_range(-0.05..0.05)),
//                 OrderedFloat(rng.gen_range(-0.05..0.05)),
//                 OrderedFloat(rng.gen_range(-0.05..0.05)),
//             ],
//         };
//         self.steps = 0;
//         self.done = false;
//         Ok(self.state.clone())
//     }

//     fn step(&mut self, action: &Self::Action) -> Result<StepResult<Self::State>, Self::Error> {
//         // Physics simulation (simplified)
//         // ... implement dynamics ...

//         // Check termination conditions
//         self.done = self.is_terminal();
//         self.steps += 1;

//         let reward = if self.done { 0.0 } else { 1.0 };
//         let truncated = self.steps >= self.config.max_steps;

//         Ok(StepResult {
//             state: self.state.clone(),
//             reward,
//             done: self.done,
//             truncated,
//             info: EnvInfo::default(),
//         })
//     }

//     fn state(&self) -> &Self::State {
//         &self.state
//     }

//     fn is_done(&self) -> bool {
//         self.done
//     }

//     fn metadata(&self) -> EnvMetadata {
//         EnvMetadata {
//             name: "CartPole-v1".to_string(),
//             state_shape: vec![4],
//             action_space_description: "Discrete(2)".to_string(),
//             max_episode_steps: Some(self.config.max_steps),
//         }
//     }
// }
