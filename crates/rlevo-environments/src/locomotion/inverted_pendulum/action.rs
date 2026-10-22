//! Action type for [`super::InvertedPendulum`].

use rlevo_core::action::ContinuousAction;
use rlevo_core::base::Action;
use serde::{Deserialize, Serialize};

/// 1D continuous action — horizontal force target on the cart, in Gymnasium's
/// pre-gear units. Bounds: `[-3.0, 3.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedPendulumAction(pub [f32; 1]);

impl InvertedPendulumAction {
    /// Construct from a scalar convenience.
    #[must_use]
    pub const fn new(force: f32) -> Self {
        Self([force])
    }
}

impl Action<1> for InvertedPendulumAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        self.0[0].is_finite() && self.0[0].abs() <= 3.0
    }
}

impl ContinuousAction<1> for InvertedPendulumAction {
    const COMPONENTS: usize = 1;

    /// Returns the raw force value as a single-element slice.
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Constructs an action from the first element of `values`.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty.
    fn from_slice(values: &[f32]) -> Self {
        Self([values[0]])
    }
}
