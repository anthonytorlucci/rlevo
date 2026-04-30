//! Observation type for [`super::Swimmer`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::base::{Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 8-dim observation. Layout matches Gymnasium's `qpos[2:5]` + `qvel`:
/// `[body_angle, joint1_angle, joint2_angle, vx_com, vy_com,
///   ω_body, joint1_dot, joint2_dot]`.
///
/// * `body_angle` — absolute z-rotation of segment0 (wrapped to `(-π, π]`).
/// * `joint{1,2}_angle` — **relative** angle between adjacent segments
///   (child − parent in world-z), wrapped.
/// * `vx_com, vy_com, ω_body` — segment0's linear/angular velocity.
/// * `joint{k}_dot` — relative angular rate `ω_child − ω_parent`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwimmerObservation(pub [f32; 8]);

impl SwimmerObservation {
    #[must_use]
    pub const fn body_angle(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn joint1_angle(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn joint2_angle(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn vx_com(&self) -> f32 {
        self.0[3]
    }
    #[must_use]
    pub const fn vy_com(&self) -> f32 {
        self.0[4]
    }
    #[must_use]
    pub const fn omega_body(&self) -> f32 {
        self.0[5]
    }
    #[must_use]
    pub const fn joint1_dot(&self) -> f32 {
        self.0[6]
    }
    #[must_use]
    pub const fn joint2_dot(&self) -> f32 {
        self.0[7]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for SwimmerObservation {
    fn default() -> Self {
        Self([0.0; 8])
    }
}

impl Observation<1> for SwimmerObservation {
    fn shape() -> [usize; 1] {
        [8]
    }
}

impl<B: Backend> TensorConvertible<1, B> for SwimmerObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 8 {
            return Err(TensorConversionError {
                message: format!("expected 8 observation elements, got {}", slice.len()),
            });
        }
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}
