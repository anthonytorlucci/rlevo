//! Observation type for [`super::InvertedDoublePendulum`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::base::{HostRow, Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 9-dim observation:
/// `$[\text{cart\_x}, \sin\theta_1, \sin\theta_2, \cos\theta_1, \cos\theta_2, \text{cart\_vx},$`
/// `$\dot{\theta}_1, \dot{\theta}_2, \text{constraint\_force\_x}]$`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedDoublePendulumObservation(pub [f32; 9]);

impl InvertedDoublePendulumObservation {
    /// `obs[0]` — cart position along world-x in metres.
    #[must_use]
    pub const fn cart_position(&self) -> f32 {
        self.0[0]
    }
    /// `obs[1]` — sine of pole1's rotation angle about world-y (`$\theta_1$`).
    #[must_use]
    pub const fn sin_theta1(&self) -> f32 {
        self.0[1]
    }
    /// `obs[2]` — sine of the **relative** elbow angle (`$\theta_2$` = pole2 world angle `$-$`
    /// pole1 world angle, wrapped to `$(-\pi, \pi]$`).
    #[must_use]
    pub const fn sin_theta2(&self) -> f32 {
        self.0[2]
    }
    /// `obs[3]` — cosine of `$\theta_1$`.
    #[must_use]
    pub const fn cos_theta1(&self) -> f32 {
        self.0[3]
    }
    /// `obs[4]` — cosine of `$\theta_2$` (relative elbow angle).
    #[must_use]
    pub const fn cos_theta2(&self) -> f32 {
        self.0[4]
    }
    /// `obs[5]` — cart velocity along world-x in m/s.
    #[must_use]
    pub const fn cart_velocity(&self) -> f32 {
        self.0[5]
    }
    /// `obs[6]` — angular velocity of pole1 about world-y (`$\dot{\theta}_1$`) in rad/s.
    #[must_use]
    pub const fn theta1_dot(&self) -> f32 {
        self.0[6]
    }
    /// `obs[7]` — world-frame angular velocity of pole2 about world-y (`$\dot{\theta}_2$`,
    /// absolute, not relative to pole1) in rad/s.
    #[must_use]
    pub const fn theta2_dot(&self) -> f32 {
        self.0[7]
    }
    /// `obs[8]` — approximated constraint force along world-x at the tip of
    /// pole2, sampled from Rapier's contact-force accumulator. See the
    /// [module documentation](super) for the divergence from `MuJoCo`'s
    /// `cfrc_inv`.
    #[must_use]
    pub const fn constraint_force_x(&self) -> f32 {
        self.0[8]
    }

    /// Returns `true` if every element of the observation is a finite floating-point number (i.e.
    /// not `NaN` or `$\pm\infty$`).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for InvertedDoublePendulumObservation {
    fn default() -> Self {
        Self([0.0; 9])
    }
}

impl Observation<1> for InvertedDoublePendulumObservation {
    fn shape() -> [usize; 1] {
        [9]
    }
}

impl HostRow<1> for InvertedDoublePendulumObservation {
    fn row_shape() -> [usize; 1] {
        [9]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.0);
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedDoublePendulumObservation {
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 9 {
            return Err(TensorConversionError {
                message: format!("expected 9 observation elements, got {}", slice.len()),
            });
        }
        let mut out = [0.0f32; 9];
        out.copy_from_slice(slice);
        Ok(Self(out))
    }
}
