//! Observation type for CarRacing: a 96×96×3 RGB pixel buffer.
//!
//! [`CarRacingObservation`] wraps the raw pixel output of the software
//! rasterizer. The buffer is stored row-major (top to bottom), with three `u8`
//! bytes per pixel in RGB order. The camera is fixed on the car's current
//! position, so the car always appears near the centre of the frame.
//!
//! [`TensorConvertible`] produces a Burn
//! `Tensor<B, 3>` with HWC layout `[96, 96, 3]`, with each pixel byte normalised
//! by `÷255` to `[0.0, 1.0]` *inside* the conversion — no external normalisation
//! step is needed or expected. Consumers that feed a Burn `conv2d` must permute
//! the frame from HWC to CHW first.

use std::sync::Arc;

use rlevo_core::base::{Observation, TensorConversionError, TensorConvertible};

use burn::tensor::{Tensor, backend::Backend};

use super::rasterizer::{FRAME_SIZE, PIXEL_BYTES};

/// 96×96×3 pixel observation for CarRacing.
///
/// Pixel values are stored as `u8` in `[0, 255]`, row-major, RGB.
///
/// The buffer is held behind an [`Arc`] so cloning an observation — which the
/// hot path does for the cached `last_obs` and on every `observe()` — is a
/// refcount bump rather than a 27 KB deep copy. The `Arc` also makes the
/// observation `Send + Sync`, which parallel EA rollouts require. Construct one
/// on the render path with [`from_boxed`](Self::from_boxed), which moves the
/// rasterizer's owned buffer in.
///
/// When converted to tensors via
/// [`TensorConvertible`], the buffer becomes
/// a Burn `Tensor<B, 3>` with HWC layout `[96, 96, 3]`, each byte normalised by
/// `÷255` to `[0.0, 1.0]`. [`from_tensor`](TensorConvertible::from_tensor)
/// reconstructs the buffer by scaling back (`×255`) and rounding, so the
/// round-trip is exact for every `u8` value. This convention follows ADR 0020
/// (synthetic pixel over grid).
#[derive(Clone)]
pub struct CarRacingObservation {
    /// Raw pixel buffer: 96 × 96 × 3 = 27 648 bytes, shared via `Arc` so clones
    /// are cheap refcount bumps.
    pub pixels: Arc<[u8; PIXEL_BYTES]>,
}

impl CarRacingObservation {
    /// Construct from a raw pixel array.
    pub fn new(pixels: [u8; PIXEL_BYTES]) -> Self {
        Self {
            pixels: Arc::new(pixels),
        }
    }

    /// Construct by moving an owned boxed buffer in — the zero-copy hand-off from
    /// the rasterizer. Wraps the buffer in an `Arc` so later clones (the cached
    /// `last_obs` and every `observe()`) are refcount bumps rather than 27 KB deep
    /// copies. The `Box` → `Arc` conversion is the single residual 27 KB copy on
    /// the render *hot path* (an `Arc` must prepend a refcount header, so the box
    /// allocation cannot be reused). The cold [`Deserialize`](serde::Deserialize)
    /// and [`from_tensor`](TensorConvertible::from_tensor) paths each perform one
    /// such `Box` → `Arc` copy too, but neither runs per render step.
    pub fn from_boxed(pixels: Box<[u8; PIXEL_BYTES]>) -> Self {
        Self {
            pixels: Arc::from(pixels),
        }
    }

    /// Returns `true` (pixels are always valid `u8` values).
    pub fn is_finite(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for CarRacingObservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CarRacingObservation({}×{}×3)", FRAME_SIZE, FRAME_SIZE)
    }
}

impl Default for CarRacingObservation {
    fn default() -> Self {
        Self {
            pixels: Arc::new([0u8; PIXEL_BYTES]),
        }
    }
}

impl serde::Serialize for CarRacingObservation {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut seq = serializer.serialize_tuple(PIXEL_BYTES)?;
        for byte in self.pixels.iter() {
            seq.serialize_element(byte)?;
        }
        seq.end()
    }
}

impl<'de> serde::Deserialize<'de> for CarRacingObservation {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PixelVisitor;
        impl<'de> serde::de::Visitor<'de> for PixelVisitor {
            type Value = CarRacingObservation;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a {PIXEL_BYTES}-byte pixel buffer")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<CarRacingObservation, A::Error> {
                let mut pixels = Box::new([0u8; PIXEL_BYTES]);
                for i in 0..PIXEL_BYTES {
                    pixels[i] = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(CarRacingObservation {
                    pixels: Arc::from(pixels),
                })
            }
        }
        deserializer.deserialize_tuple(PIXEL_BYTES, PixelVisitor)
    }
}

impl Observation<3> for CarRacingObservation {
    fn shape() -> [usize; 3] {
        [FRAME_SIZE, FRAME_SIZE, 3]
    }
}

impl<B: Backend> TensorConvertible<3, B> for CarRacingObservation {
    fn row_shape() -> [usize; 3] {
        [FRAME_SIZE, FRAME_SIZE, 3]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        // Normalize bytes to [0, 1] so a Burn policy can consume the frame directly.
        buf.extend(self.pixels.iter().map(|&b| f32::from(b) / 255.0));
    }

    /// Reconstructs the frame from a normalized `[FRAME_SIZE, FRAME_SIZE, 3]`
    /// tensor by scaling back to `0..=255` and rounding.
    ///
    /// Round-trips exactly for any `u8` payload: `b / 255.0 * 255.0` rounds back
    /// to `b`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not
    /// `[FRAME_SIZE, FRAME_SIZE, 3]`, the backend fails to materialize its data,
    /// or any value lies outside `[0, 1]` after scaling to the `u8` range.
    fn from_tensor(tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [FRAME_SIZE, FRAME_SIZE, 3] {
            return Err(TensorConversionError {
                message: format!("expected shape [{FRAME_SIZE}, {FRAME_SIZE}, 3], got {dims:?}"),
            });
        }
        let flat: Vec<f32> =
            tensor
                .into_data()
                .into_vec::<f32>()
                .map_err(|e| TensorConversionError {
                    message: format!("failed to read tensor data: {e:?}"),
                })?;
        let mut pixels: Vec<u8> = Vec::with_capacity(PIXEL_BYTES);
        for (idx, &value) in flat.iter().enumerate() {
            let scaled: f32 = value * 255.0;
            if !scaled.is_finite() || scaled < -0.5 || scaled > f32::from(u8::MAX) + 0.5 {
                return Err(TensorConversionError {
                    message: format!("value at index {idx} out of u8 range: {value}"),
                });
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            pixels.push(scaled.round() as u8);
        }
        let arr: Box<[u8; PIXEL_BYTES]> =
            pixels
                .into_boxed_slice()
                .try_into()
                .map_err(|_| TensorConversionError {
                    message: "wrong element count".into(),
                })?;
        Ok(Self {
            pixels: Arc::from(arr),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time guard for the struct doc's claim that observations cross EA
    // rollout threads: adding a non-`Send`/`Sync` field (e.g. swapping `Arc` for
    // `Rc`) would make this fail to compile.
    fn _assert_send_sync() {
        fn f<T: Send + Sync>() {}
        f::<CarRacingObservation>();
    }

    #[test]
    fn test_shape() {
        assert_eq!(CarRacingObservation::shape(), [96, 96, 3]);
    }

    #[test]
    fn test_default_is_zeroed() {
        let obs = CarRacingObservation::default();
        assert!(obs.pixels.iter().all(|&p| p == 0));
    }

    #[test]
    fn serde_round_trip() {
        let mut px: [u8; PIXEL_BYTES] = [0u8; PIXEL_BYTES];
        for (i, p) in px.iter_mut().enumerate() {
            *p = (i % 256) as u8;
        }
        let obs = CarRacingObservation::new(px);

        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&obs, cfg).unwrap();
        let (back, _len): (CarRacingObservation, usize) =
            bincode::serde::decode_from_slice(&bytes, cfg).unwrap();

        // End-to-end: custom Serialize → Deserialize visitor (Box scratch → Arc).
        assert_eq!(&*obs.pixels, &*back.pixels);
    }

    #[test]
    fn from_boxed_preserves_pixels() {
        let mut src = Box::new([0u8; PIXEL_BYTES]);
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let expected = *src.clone();
        let obs = CarRacingObservation::from_boxed(src);
        assert_eq!(&*obs.pixels, &expected);
    }

    #[test]
    fn tensor_round_trip() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let mut px: [u8; PIXEL_BYTES] = [0u8; PIXEL_BYTES];
        for (i, p) in px.iter_mut().enumerate() {
            *p = (i % 256) as u8;
        }
        let obs: CarRacingObservation = CarRacingObservation::new(px);
        let tensor =
            <CarRacingObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let back: CarRacingObservation =
            <CarRacingObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
                .unwrap();
        assert_eq!(&*obs.pixels, &*back.pixels);
    }

    #[test]
    fn tensor_boundary_values() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let mut px: [u8; PIXEL_BYTES] = [0u8; PIXEL_BYTES];
        px[0] = 0;
        px[1] = 255;
        let obs: CarRacingObservation = CarRacingObservation::new(px);
        let tensor =
            <CarRacingObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let flat: Vec<f32> = tensor.clone().into_data().into_vec::<f32>().unwrap();
        assert!((flat[0] - 0.0).abs() < 1e-6);
        assert!((flat[1] - 1.0).abs() < 1e-6);

        let back: CarRacingObservation =
            <CarRacingObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
                .unwrap();
        assert_eq!(back.pixels[0], 0);
        assert_eq!(back.pixels[1], 255);
    }

    #[test]
    fn from_tensor_rejects_wrong_shape() {
        use burn::backend::Flex;
        use burn::tensor::{Tensor, TensorData};
        type TestBackend = Flex;
        let device = Default::default();

        let data: TensorData = TensorData::new(vec![0.0f32; 2 * 2 * 2], [2, 2, 2]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let err = <CarRacingObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
            .unwrap_err();
        assert!(err.message.contains("expected shape"));
    }
}
