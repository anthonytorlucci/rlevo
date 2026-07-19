//! Observation type for the [`super::Reacher`] environment.
//!
//! [`ReacherObservation`] is a 10-element `f32` vector. Named accessors are
//! provided for each logical group so callers do not need to know the raw
//! index layout. The full layout is documented on the struct itself.

use burn::prelude::{Backend, Tensor};
use rlevo_core::base::{HostRow, Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 10-dim observation. Layout:
/// `[cos θ₁, cos θ₂, sin θ₁, sin θ₂, target_x, target_y, θ̇₁, θ̇₂,
///   (finger − target)_x, (finger − target)_y]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReacherObservation(pub [f32; 10]);

impl ReacherObservation {
    /// Cosine of the shoulder (link 1) absolute angle θ₁. Index 0.
    #[must_use]
    pub const fn theta1_cos(&self) -> f32 {
        self.0[0]
    }

    /// Cosine of the elbow (link 2) **relative** angle θ₂ = θ_world2 − θ₁,
    /// wrapped to `(-π, π]`. Index 1.
    #[must_use]
    pub const fn theta2_cos(&self) -> f32 {
        self.0[1]
    }

    /// Sine of the shoulder angle θ₁. Index 2.
    #[must_use]
    pub const fn theta1_sin(&self) -> f32 {
        self.0[2]
    }

    /// Sine of the elbow relative angle θ₂. Index 3.
    #[must_use]
    pub const fn theta2_sin(&self) -> f32 {
        self.0[3]
    }

    /// World-frame target position `[x, y]` in metres. Constant within an
    /// episode; sampled at reset from a disk of radius
    /// `config.target_disk_radius`. Indices 4–5.
    #[must_use]
    pub const fn target_xy(&self) -> [f32; 2] {
        [self.0[4], self.0[5]]
    }

    /// Shoulder angular velocity θ̇₁ in rad s⁻¹ (world-z component). Index 6.
    #[must_use]
    pub const fn theta1_dot(&self) -> f32 {
        self.0[6]
    }

    /// Elbow **relative** angular velocity θ̇₂ = ω_link2 − ω_link1 in
    /// rad s⁻¹. Index 7.
    #[must_use]
    pub const fn theta2_dot(&self) -> f32 {
        self.0[7]
    }

    /// Vector from the fingertip to the target in world frame: `[finger_x −
    /// target_x, finger_y − target_y]` in metres. The L2 norm of this vector
    /// is the distance term in the reward. Indices 8–9.
    #[must_use]
    pub const fn finger_minus_target_xy(&self) -> [f32; 2] {
        [self.0[8], self.0[9]]
    }

    /// Returns `true` if every element of the observation vector is finite
    /// (i.e. neither `NaN` nor ±∞).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for ReacherObservation {
    fn default() -> Self {
        Self([0.0; 10])
    }
}

impl Observation<1> for ReacherObservation {
    fn shape() -> [usize; 1] {
        [10]
    }
}

impl HostRow<1> for ReacherObservation {
    fn row_shape() -> [usize; 1] {
        [10]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.0);
    }
}

impl<B: Backend> TensorConvertible<1, B> for ReacherObservation {
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 10 {
            return Err(TensorConversionError {
                message: format!("expected 10 observation elements, got {}", slice.len()),
            });
        }
        let mut arr = [0.0f32; 10];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}
