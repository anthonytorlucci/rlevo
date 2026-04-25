//! Multi-armed bandit family — stationary, contextual, non-stationary, and
//! adversarial.
//!
//! All four variants share the [`KArmedBanditAction<K>`] action type and a
//! common configuration shape (`max_steps`, `seed`, plus per-variant knobs).
//! See the per-module docs for the precise reward dynamics.
//!
//! | Module | Environment | Const generics | Reward |
//! |---|---|---|---|
//! | [`k_armed`] | [`KArmedBandit<K>`] | `K` | `N(q*(a), 1)`; means fixed |
//! | [`contextual`] | [`ContextualBandit<C, K>`] | `C` (contexts), `K` | `N(q*(c, a), 1)`; means fixed |
//! | [`non_stationary`] | [`NonStationaryBandit<K>`] | `K` | `N(q*(a), 1)`; `q*(a)` random-walks each step |
//! | [`adversarial`] | [`AdversarialBandit<K>`] | `K` | Deterministic periodic schedule in `[0, amplitude]` |
//!
//! The 10-armed Sutton & Barto §2 testbed is exposed as the
//! [`TenArmedBandit`] type alias for the canonical instance — existing
//! consumers (the `tabular_bandit` benchmark example, the
//! `ten_armed_bandit_training` example, the `bench::suites` factory) are
//! unaffected by the generalisation.

pub mod adversarial;
pub mod contextual;
pub mod k_armed;
pub mod non_stationary;

pub use adversarial::{AdversarialBandit, AdversarialBanditConfig};
pub use contextual::{
    ContextualBandit, ContextualBanditConfig, ContextualBanditObservation, ContextualBanditState,
};
pub use k_armed::{
    KArmedBandit, KArmedBanditAction, KArmedBanditConfig, KArmedBanditObservation,
    KArmedBanditState,
};
pub use non_stationary::{NonStationaryBandit, NonStationaryBanditConfig};

/// Canonical Sutton & Barto §2 ten-armed testbed.
pub type TenArmedBandit = KArmedBandit<10>;
/// Action type for the canonical ten-armed bandit.
pub type TenArmedBanditAction = KArmedBanditAction<10>;
/// State type for the canonical ten-armed bandit.
pub type TenArmedBanditState = KArmedBanditState;
/// Observation type for the canonical ten-armed bandit.
pub type TenArmedBanditObservation = KArmedBanditObservation;
/// Configuration type for the canonical ten-armed bandit.
pub type TenArmedBanditConfig = KArmedBanditConfig;
