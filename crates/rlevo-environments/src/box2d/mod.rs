//! Box2D-style physics environments using the Rapier2D pure-Rust physics engine.
//!
//! Enable with the `box2d` cargo feature:
//! ```toml
//! rlevo-envs = { features = ["box2d"] }
//! ```
//!
//! | Environment | Action | Obs dim | Terminated by |
//! |---|---|---|---|
//! | [`bipedal_walker`] | Continuous(4) | 24 | hull contact / reward < -100 |
//! | [`lunar_lander`] `LunarLanderDiscrete` | Discrete(4) | 8 | crash / out of bounds |
//! | [`lunar_lander`] `LunarLanderContinuous` | Continuous(2) | 8 | crash / out of bounds |
//! | [`car_racing`] | Continuous(3) | 96×96×3 | lap complete |

#[cfg(feature = "box2d")]
pub mod bipedal_walker;
#[cfg(feature = "box2d")]
pub mod car_racing;
#[cfg(feature = "box2d")]
pub mod lunar_lander;
#[cfg(feature = "box2d")]
pub mod physics;
