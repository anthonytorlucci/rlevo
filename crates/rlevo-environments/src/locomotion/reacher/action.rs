//! Action type for the [`super::Reacher`] environment.
//!
//! [`ReacherAction`] carries a 2-element `[shoulder, elbow]` torque command in
//! pre-gear units. The environment clips each element to `[-1.0, 1.0]` before
//! multiplying by the gear ratio (default `[200, 200]`) and applying the
//! resulting torques to the Rapier rigid bodies.

use burn::prelude::{Backend, Tensor};
use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::{Action, HostRow, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 2D continuous action — `[shoulder, elbow]` torque targets in
/// pre-gear units. Bounds: `[-1.0, 1.0]` per element.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReacherAction(pub [f32; 2]);

impl ReacherAction {
    /// Construct an action from explicit shoulder and elbow torque targets.
    ///
    /// Values outside `[-1.0, 1.0]` are accepted here but will be clamped by
    /// the environment before the gear ratio is applied. Prefer values in range
    /// to match the declared action space.
    #[must_use]
    pub const fn new(shoulder: f32, elbow: f32) -> Self {
        Self([shoulder, elbow])
    }
}

impl Action<1> for ReacherAction {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.0.iter().all(|v| v.is_finite() && v.abs() <= 1.0)
    }
}

impl ContinuousAction<1> for ReacherAction {
    const COMPONENTS: usize = 2;

    /// Returns the raw `[shoulder, elbow]` slice.
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Returns a new action with both elements clamped to `[min, max]`.
    fn clip(&self, min: f32, max: f32) -> Self {
        Self([self.0[0].clamp(min, max), self.0[1].clamp(min, max)])
    }

    /// Construct from a slice of exactly `COMPONENTS` values
    /// `[shoulder, elbow]`.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != Self::COMPONENTS`.
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(
            values.len(),
            Self::COMPONENTS,
            "ReacherAction needs exactly {} components, got {}",
            Self::COMPONENTS,
            values.len(),
        );
        Self([values[0], values[1]])
    }

    /// Sample a uniformly-random action in `[-1.0, 1.0]²` using the global
    /// thread-local RNG.
    fn random() -> Self {
        Self([
            rand::random::<f32>() * 2.0 - 1.0,
            rand::random::<f32>() * 2.0 - 1.0,
        ])
    }
}

/// Per-component bounds `[-1, -1] .. [1, 1]`, in pre-gear units.
///
/// Source: Gymnasium `Reacher-v5` declares `Box(-1, 1, (2,), float32)`, and
/// this type's own [`is_valid`](Action::is_valid) rejects any element with
/// `abs() > 1.0`. Spec and in-repo action contract agree.
///
/// # These are the *type's* bounds, not the environment instance's
///
/// [`ReacherConfig::action_clip`] is a runtime field that defaults to
/// `[-1.0, 1.0]` — matching these bounds — but a caller may narrow it. A
/// narrowed `action_clip` does not change what this type accepts (`is_valid`
/// is unconditionally `abs() <= 1.0`); it only means the environment squeezes
/// the command further before applying the gear ratio, so an agent driven by
/// these bounds stays within the declared space and merely saturates earlier.
/// `low()`/`high()` take no `self` and cannot observe a config (ADR 0053 §3);
/// per-instance bounds are a separate seam that would need its own ADR.
///
/// [`ReacherConfig::action_clip`]: super::config::ReacherConfig::action_clip
impl BoundedAction<1> for ReacherAction {
    fn low() -> &'static [f32] {
        &[-1.0, -1.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0]
    }
}

impl HostRow<1> for ReacherAction {
    fn row_shape() -> [usize; 1] {
        [2]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.0);
    }
}

impl<B: Backend> TensorConvertible<1, B> for ReacherAction {
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 action tensor: {e:?}"),
        })?;
        if slice.len() != 2 {
            return Err(TensorConversionError {
                message: format!("expected 2 action elements, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0], slice[1]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_match_components_and_is_valid() {
        let low = ReacherAction::low();
        let high = ReacherAction::high();
        assert_eq!(low.len(), ReacherAction::COMPONENTS);
        assert_eq!(high.len(), ReacherAction::COMPONENTS);
        for i in 0..ReacherAction::COMPONENTS {
            assert!(low[i] < high[i], "component {i} must have low < high");
        }
        assert!(ReacherAction::from_slice(low).is_valid());
        assert!(ReacherAction::from_slice(high).is_valid());
        assert!(!ReacherAction::new(1.1, 0.0).is_valid());
        assert!(!ReacherAction::new(0.0, -1.1).is_valid());
    }

    #[test]
    #[should_panic(expected = "needs exactly 2 components, got 3")]
    fn from_slice_rejects_an_over_long_slice() {
        // `ContinuousAction::from_slice` accepts *exactly* `COMPONENTS` values
        // (docs/rules.md §3). Before ADR 0053's follow-up there was no length
        // check at all here: a long slice was silently truncated and a short
        // one panicked with a bare index-out-of-bounds.
        let _ = ReacherAction::from_slice(&[0.1, 0.2, 0.3]);
    }

    #[test]
    #[should_panic(expected = "needs exactly 2 components, got 1")]
    fn from_slice_rejects_a_short_slice() {
        let _ = ReacherAction::from_slice(&[0.1]);
    }

    #[test]
    fn bounds_agree_with_the_default_action_clip() {
        // The static bounds are the *type's* declared space; the environment's
        // default `action_clip` is the same interval, so the default
        // configuration cannot narrow an agent's reachable actions.
        let (lo, hi): (f32, f32) = super::super::config::ReacherConfig::default()
            .action_clip
            .into();
        assert!((lo - ReacherAction::low()[0]).abs() < f32::EPSILON);
        assert!((hi - ReacherAction::high()[0]).abs() < f32::EPSILON);
    }
}
