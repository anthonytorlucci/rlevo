//! Environment wrappers that augment existing environments without modifying them.
//!
//! The only wrapper currently shipped is [`TimeLimit`], which truncates episodes
//! after a configurable step cap. More wrappers (frame-stacking, observation
//! normalization, reward clipping) may be added incrementally.

pub mod time_limit;

pub use time_limit::TimeLimit;
