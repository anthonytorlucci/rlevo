//! Objective direction: the [`ObjectiveSense`] primitive.
//!
//! `rlevo` spans three fields whose native optimisation conventions disagree —
//! reinforcement learning **maximises** return, evolutionary computation
//! **maximises** fitness, and gradient descent **minimises** loss. To keep the
//! library coherent, the *internal* engine convention is **maximise (higher =
//! better)**, and an objective declares its *natural* direction with an
//! [`ObjectiveSense`].
//!
//! # Two value spaces
//!
//! The contract separates two spaces and confines the mapping between them to a
//! single chokepoint (the evolutionary harness / fitness adapters):
//!
//! - **User space** — the sense the problem declares. A cost/loss/landscape is
//!   `Minimize`; a reward/fitness/accuracy is `Maximize`.
//! - **Canonical (engine) space** — always *maximise, higher = better*. Every
//!   strategy, operator, shaping rule, and metric aggregation works purely here
//!   and never sees an `ObjectiveSense`.
//!
//! [`to_canonical`](crate::objective::ObjectiveSense::to_canonical) maps user space → canonical
//! space (negate iff `Minimize`); [`from_canonical`](crate::objective::ObjectiveSense::from_canonical)
//! is its inverse, used to report results back in the user's sense (a `Minimize`
//! landscape's `best_fitness` reads as its natural cost — Sphere → 0).
//!
//! The mapping is an **involution**: applying it twice is the identity, so the
//! same negate-iff-`Minimize` operation serves both directions.
//!
//! # Multi-objective seam
//!
//! `ObjectiveSense` is the `K = 1` atom of a future per-objective sense vector.
//! Multi-objective dominance canonicalises every objective to maximise space and
//! then applies "≥ on all, > on at least one" with no per-objective branching —
//! the same chokepoint philosophy scaled to a vector.

use serde::{Deserialize, Serialize};

/// The direction in which an objective is optimised.
///
/// This is the typed direction primitive that reconciles the library's
/// maximise-native engine with cost objectives (the benchmark landscapes). See
/// the [module documentation](crate::objective) for the two-space model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveSense {
    /// Lower is better (cost, loss, error, distance-to-target).
    Minimize,
    /// Higher is better (reward, fitness, accuracy, score).
    Maximize,
}

impl ObjectiveSense {
    /// Maps a raw user-space objective value into the engine's maximise-native
    /// canonical space.
    ///
    /// `Maximize` passes the value through unchanged; `Minimize` negates it so a
    /// cost surface is optimised as `−cost` by the maximise-native engine.
    /// Applied **once**, at the harness / fitness-adapter boundary.
    ///
    /// ```
    /// use rlevo_core::objective::ObjectiveSense;
    /// assert_eq!(ObjectiveSense::Maximize.to_canonical(3.0), 3.0);
    /// assert_eq!(ObjectiveSense::Minimize.to_canonical(3.0), -3.0);
    /// ```
    #[must_use]
    pub fn to_canonical(self, raw: f32) -> f32 {
        match self {
            ObjectiveSense::Maximize => raw,
            ObjectiveSense::Minimize => -raw,
        }
    }

    /// Inverse of [`to_canonical`](Self::to_canonical): maps an engine-space
    /// (canonical, maximise) value back to the user's declared sense for
    /// reporting (`best_fitness`, records, showcases).
    ///
    /// Because the mapping is an involution (negate iff `Minimize`), this is the
    /// same operation as [`to_canonical`](Self::to_canonical).
    ///
    /// ```
    /// use rlevo_core::objective::ObjectiveSense;
    /// // A Minimize landscape optimised as -cost in canonical space reads back
    /// // as its natural cost.
    /// let canonical = ObjectiveSense::Minimize.to_canonical(2.5); // -2.5
    /// assert_eq!(ObjectiveSense::Minimize.from_canonical(canonical), 2.5);
    /// ```
    #[must_use]
    pub fn from_canonical(self, canonical: f32) -> f32 {
        // Involution: negate iff Minimize.
        self.to_canonical(canonical)
    }
}

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    #![allow(clippy::float_cmp)]
    use super::ObjectiveSense;

    #[test]
    fn maximize_is_pass_through() {
        assert_eq!(ObjectiveSense::Maximize.to_canonical(7.5), 7.5);
        assert_eq!(ObjectiveSense::Maximize.from_canonical(7.5), 7.5);
    }

    #[test]
    fn minimize_negates() {
        assert_eq!(ObjectiveSense::Minimize.to_canonical(7.5), -7.5);
        assert_eq!(ObjectiveSense::Minimize.from_canonical(-7.5), 7.5);
    }

    #[test]
    fn round_trip_is_identity_for_both_senses() {
        for sense in [ObjectiveSense::Minimize, ObjectiveSense::Maximize] {
            for raw in [-3.0_f32, 0.0, 1.5, 42.0] {
                let canonical = sense.to_canonical(raw);
                approx::assert_relative_eq!(sense.from_canonical(canonical), raw, epsilon = 1e-9);
            }
        }
    }
}
