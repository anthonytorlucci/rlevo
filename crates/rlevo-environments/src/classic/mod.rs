//! Classic-control environments.
//!
//! Five canonical RL benchmarks ported from the Gymnasium catalogue as
//! native Rust implementations with no Python or C-library dependencies,
//! plus a generic [`bandit`] family covering the standard Sutton & Barto §2
//! testbed and three sibling variants.
//!
//! | Environment | Action | Obs dim | Terminated by |
//! |---|---|---|---|
//! | [`acrobot`] | Discrete(3) | 6 | tip height |
//! | [`cartpole`] | Discrete(2) | 4 | angle / position |
//! | [`mountain_car`] | Discrete(3) | 2 | goal position |
//! | [`mountain_car_continuous`] | Continuous(1) | 2 | goal position |
//! | [`pendulum`] | Continuous(1) | 3 | never (truncated by wrapper) |
//! | [`bandit::k_armed`] | Discrete(K) | 1 | `max_steps` reached |
//! | [`bandit::contextual`] | Discrete(K) | C | `max_steps` reached |
//! | [`bandit::non_stationary`] | Discrete(K) | 1 | `max_steps` reached |
//! | [`bandit::adversarial`] | Discrete(K) | 1 | `max_steps` reached |
//! | [`santa_fe_ant`] | Discrete(3) | 1 | all 89 pellets / `max_steps` |
//!
//! Wrap any env with [`crate::wrappers::TimeLimit`] to impose a step cap.
//!
//! Unlike the others, [`santa_fe_ant`] is a **POMDP**: its single food-ahead
//! bit does not summarise the history needed to act optimally
//! ([`MarkovState::is_markov`](rlevo_core::state::MarkovState::is_markov) is
//! `false`).
pub mod acrobot;
pub mod bandit;
pub mod cartpole;
pub mod mountain_car;
pub mod mountain_car_continuous;
pub mod pendulum;
pub mod santa_fe_ant;

pub use acrobot::{
    Acrobot, AcrobotAction, AcrobotConfig, AcrobotConfigBuilder, AcrobotDynamicsFn,
    AcrobotObservation, AcrobotState, BookDynamics, NipsDynamics,
};
pub use bandit::{
    AdversarialBandit, AdversarialBanditConfig, ContextualBandit, ContextualBanditConfig,
    ContextualBanditObservation, ContextualBanditState, KArmedBandit, KArmedBanditAction,
    KArmedBanditConfig, KArmedBanditObservation, KArmedBanditState, NonStationaryBandit,
    NonStationaryBanditConfig, TenArmedBandit, TenArmedBanditAction, TenArmedBanditConfig,
    TenArmedBanditObservation, TenArmedBanditState,
};
pub use cartpole::{
    CartPole, CartPoleAction, CartPoleConfig, CartPoleConfigBuilder, CartPoleObservation,
    CartPoleState, Integrator,
};
pub use mountain_car::{
    MountainCar, MountainCarAction, MountainCarConfig, MountainCarObservation, MountainCarState,
};
pub use mountain_car_continuous::{
    MountainCarContinuous, MountainCarContinuousAction, MountainCarContinuousConfig,
    MountainCarContinuousObservation, MountainCarContinuousState,
};
pub use pendulum::{
    Pendulum, PendulumAction, PendulumConfig, PendulumObservation, PendulumState, angle_normalize,
};
pub use santa_fe_ant::{
    SantaFeAnt, SantaFeAntAction, SantaFeAntConfig, SantaFeAntObservation, SantaFeAntState,
};
