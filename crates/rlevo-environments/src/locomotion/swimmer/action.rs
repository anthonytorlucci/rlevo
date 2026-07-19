//! Action type for the [`super::Swimmer`] environment.
//!
//! [`SwimmerAction`] holds two pre-gear joint-torque targets, one per revolute
//! joint, in the range `[-1.0, 1.0]`. The environment multiplies each by the
//! configured `gear` scalar before applying the torque to the physics body.

use burn::prelude::{Backend, Tensor};
use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::{Action, HostRow, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 2D continuous action — `[joint1, joint2]` torque targets in pre-gear
/// units. Bounds: `[-1.0, 1.0]` per element.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwimmerAction(pub [f32; 2]);

impl SwimmerAction {
    /// Construct an action from explicit joint-torque targets.
    ///
    /// Both values are expected to lie in `[-1.0, 1.0]`. Values outside that
    /// range are accepted here but will fail [`Action::is_valid`] and will be
    /// clamped by the environment before the gear multiplication is applied.
    #[must_use]
    pub const fn new(joint1: f32, joint2: f32) -> Self {
        Self([joint1, joint2])
    }
}

impl Action<1> for SwimmerAction {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.0.iter().all(|v| v.is_finite() && v.abs() <= 1.0)
    }
}

impl ContinuousAction<1> for SwimmerAction {
    const COMPONENTS: usize = 2;

    /// Returns the raw `[joint1, joint2]` slice.
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Returns a new action with both torque targets clamped to `[min, max]`.
    fn clip(&self, min: f32, max: f32) -> Self {
        Self([self.0[0].clamp(min, max), self.0[1].clamp(min, max)])
    }

    /// Construct from a slice. Panics if `values.len() < 2`.
    fn from_slice(values: &[f32]) -> Self {
        Self([values[0], values[1]])
    }

    /// Sample both torque targets uniformly from `[-1.0, 1.0]` using the
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
/// Source: Gymnasium `Swimmer-v5` declares `Box(-1, 1, (2,), float32)`, and
/// this type's own [`is_valid`](Action::is_valid) rejects any element with
/// `abs() > 1.0`. Spec and in-repo action contract agree. (The `gear` scalar
/// this crate applies after clipping is known to differ from Swimmer-v5's
/// `[150, 150]` — see [`Swimmer::joint_torques`] — but that is a torque-scale
/// question downstream of the action space, and does not move these bounds.)
///
/// As with [`ReacherAction`](super::super::reacher::action::ReacherAction),
/// [`SwimmerConfig::action_clip`] is a runtime field defaulting to
/// `[-1.0, 1.0]`. A narrowed value squeezes the command inside the
/// environment; it does not change the space this *type* declares, which
/// `low()`/`high()` cannot observe anyway (ADR 0053 §3).
///
/// [`Swimmer::joint_torques`]: super::env::Swimmer
/// [`SwimmerConfig::action_clip`]: super::config::SwimmerConfig::action_clip
impl BoundedAction<1> for SwimmerAction {
    fn low() -> &'static [f32] {
        &[-1.0, -1.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0]
    }
}

impl HostRow<1> for SwimmerAction {
    fn row_shape() -> [usize; 1] {
        [2]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.0);
    }
}

impl<B: Backend> TensorConvertible<1, B> for SwimmerAction {
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
        let low = SwimmerAction::low();
        let high = SwimmerAction::high();
        assert_eq!(low.len(), SwimmerAction::COMPONENTS);
        assert_eq!(high.len(), SwimmerAction::COMPONENTS);
        for i in 0..SwimmerAction::COMPONENTS {
            assert!(low[i] < high[i], "component {i} must have low < high");
        }
        assert!(SwimmerAction::from_slice(low).is_valid());
        assert!(SwimmerAction::from_slice(high).is_valid());
        assert!(!SwimmerAction::new(1.1, 0.0).is_valid());
        assert!(!SwimmerAction::new(0.0, -1.1).is_valid());
    }

    #[test]
    fn bounds_agree_with_the_default_action_clip() {
        let (lo, hi): (f32, f32) = super::super::config::SwimmerConfig::default()
            .action_clip
            .into();
        assert!((lo - SwimmerAction::low()[0]).abs() < f32::EPSILON);
        assert!((hi - SwimmerAction::high()[0]).abs() < f32::EPSILON);
    }
}
