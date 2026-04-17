//! Observation type for CarRacing.

use evorl_core::base::Observation;

use super::rasterizer::{FRAME_SIZE, PIXEL_BYTES};

/// 96×96×3 pixel observation for CarRacing.
///
/// Pixel values are `u8` in `[0, 255]` (row-major, RGB).
/// When converted to tensors via [`TensorConvertible`], values are
/// normalised to `[0.0, 1.0]`.
#[derive(Clone)]
pub struct CarRacingObservation {
    /// Raw pixel buffer: 96 × 96 × 3 = 27 648 bytes.
    pub pixels: Box<[u8; PIXEL_BYTES]>,
}

impl CarRacingObservation {
    /// Construct from a raw pixel array.
    pub fn new(pixels: [u8; PIXEL_BYTES]) -> Self {
        Self { pixels: Box::new(pixels) }
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
        Self { pixels: Box::new([0u8; PIXEL_BYTES]) }
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
                Ok(CarRacingObservation { pixels })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(CarRacingObservation::shape(), [96, 96, 3]);
    }

    #[test]
    fn test_default_is_zeroed() {
        let obs = CarRacingObservation::default();
        assert!(obs.pixels.iter().all(|&p| p == 0));
    }
}
