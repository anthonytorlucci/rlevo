//! Observation type for `CarRacing`: a `$96 \times 96 \times 3$` RGB pixel buffer.
//!
//! [`CarRacingObservation`] wraps the raw pixel output of the software
//! rasterizer. The buffer is stored row-major (top to bottom), with three `u8`
//! bytes per pixel in RGB order. The camera is fixed on the car's current
//! position, so the car always appears near the centre of the frame.
//!
//! The host row ([`HostRow::write_host_row`]) has HWC layout `[96, 96, 3]`,
//! with each pixel byte normalised by `$\div 255$` to `[0.0, 1.0]` *inside* the
//! row writer — no external normalisation step is needed or expected. Uploaded
//! as a Burn tensor, the row keeps that shape; consumers that feed a Burn
//! `conv2d` must permute the frame from HWC to CHW first.

// The `$96 \times 96 \times 3$` pixel array (27,648 bytes) exceeds clippy's 16 KiB stack-array
// threshold, but every construction site immediately moves it into a `Box` or `Arc`, and the array
// type is part of this module's public observation API (`Arc<[u8; PIXEL_BYTES]>`). Switching to a
// heap-allocated slice to satisfy the lint would change that public type for no runtime benefit.
#![allow(clippy::large_stack_arrays)]

use std::sync::Arc;

use rlevo_core::base::{HostRow, Observation, TensorConvertible};

use burn::tensor::backend::Backend;

use super::rasterizer::{FRAME_SIZE, PIXEL_BYTES};

/// `$96 \times 96 \times 3$` pixel observation for `CarRacing`.
///
/// Pixel values are stored as `u8` in `[0, 255]`, row-major, RGB.
///
/// The buffer is held behind an [`Arc`] so cloning an observation — which
/// snapshot handling and parallel rollouts may do — is a refcount bump rather
/// than a 27 KB deep copy. The `Arc` also makes the observation `Send + Sync`,
/// which parallel EA rollouts require. Construct one on the render path with
/// [`from_boxed`](Self::from_boxed), which moves the rasterizer's owned buffer
/// in.
///
/// Its host row has HWC layout `[96, 96, 3]`, each byte normalised by
/// `$\div 255$` to `[0.0, 1.0]`. This convention follows ADR 0020 (synthetic
/// pixel over grid).
#[derive(Clone)]
pub struct CarRacingObservation {
    /// Raw pixel buffer: `$96 \times 96 \times 3$` = 27 648 bytes, shared via `Arc` so clones are
    /// cheap refcount bumps.
    pub pixels: Arc<[u8; PIXEL_BYTES]>,
}

impl CarRacingObservation {
    /// Construct from a raw pixel array.
    #[must_use]
    pub fn new(pixels: [u8; PIXEL_BYTES]) -> Self {
        Self {
            pixels: Arc::new(pixels),
        }
    }

    /// Construct by moving an owned boxed buffer in — the zero-copy hand-off from
    /// the rasterizer. Wraps the buffer in an `Arc` so later clones (snapshot
    /// handling, parallel rollouts) are refcount bumps rather than 27 KB deep
    /// copies. The `Box` → `Arc` conversion is the single residual 27 KB copy on
    /// the render *hot path* (an `Arc` must prepend a refcount header, so the box
    /// allocation cannot be reused). The cold [`Deserialize`](serde::Deserialize)
    /// path performs one such `Box` → `Arc` copy too, but it does not run per
    /// render step.
    #[must_use]
    pub fn from_boxed(pixels: Box<[u8; PIXEL_BYTES]>) -> Self {
        Self {
            pixels: Arc::from(pixels),
        }
    }

    /// Returns `true` (pixels are always valid `u8` values).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for CarRacingObservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CarRacingObservation({FRAME_SIZE}×{FRAME_SIZE}×3)")
    }
}

impl Default for CarRacingObservation {
    fn default() -> Self {
        Self {
            pixels: Arc::new([0u8; PIXEL_BYTES]),
        }
    }
}

/// Hand-written `Serialize` — retained because serde has no derive for arrays longer than 32
/// elements, and `pixels` holds `PIXEL_BYTES` (`$96 \times 96 \times 3$` = 27,648) of them.
///
/// This is not a leftover of a trait bound: `Observation<R>` no longer requires
/// serde (ADR 0064). The impl stays because [`CarRacingObservation`] is part of
/// the public API and consumers persist frames; `#[derive(Serialize)]` on the
/// struct simply does not compile at this array length.
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

/// Hand-written [`Visitor`](serde::de::Visitor)-based `Deserialize` — the mirror of the `Serialize`
/// impl above, and retained for the same reason: serde has no derive for a `PIXEL_BYTES`-long array
/// (`$96 \times 96 \times 3$` = `$27{,}648 > 32$`).
///
/// It exists because of that array-length limit, **not** because of any trait
/// bound — `Observation<R>` no longer requires serde (ADR 0064). Do not try to
/// "simplify" it into a derive; there is no derive to simplify it into.
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

impl HostRow<3> for CarRacingObservation {
    fn row_shape() -> [usize; 3] {
        [FRAME_SIZE, FRAME_SIZE, 3]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        // Normalize bytes to [0, 1] so a Burn policy can consume the frame directly.
        buf.extend(self.pixels.iter().map(|&b| f32::from(b) / 255.0));
    }

    /// Always `true`: this row is structurally incapable of carrying a
    /// non-finite value (ADR 0067 §Decision 2).
    ///
    /// This override is *both* a structural assertion and a performance
    /// decision, and here the performance half is the larger one: the default
    /// body would convert `PIXEL_BYTES` (27 648) bytes into `f32` on **every
    /// env step** — the measured worst case in ADR 0067 — to prove something
    /// the payload type already guarantees.
    fn row_is_finite(&self, _scratch: &mut Vec<f32>) -> bool {
        // Compile-time witness for the structural claim below. If the payload
        // type ever changes, THIS LINE fails to compile and the override must
        // be re-derived, not re-asserted.
        let _: &[u8; PIXEL_BYTES] = &self.pixels;
        // `u8 -> f32` (and a divide by a finite non-zero constant) is total: no element of this row
        // can be `NaN` or `$\pm\infty$`.
        true
    }
}

impl<B: Backend> TensorConvertible<3, B> for CarRacingObservation {}

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

    /// The row is HWC in buffer order, `PIXEL_BYTES` long, and each byte is
    /// divided by 255, so `0` maps to `0.0` and `255` to `1.0`.
    #[test]
    fn host_row_normalises_each_byte_in_order() {
        let mut px: [u8; PIXEL_BYTES] = [0u8; PIXEL_BYTES];
        for (i, p) in px.iter_mut().enumerate() {
            *p = (i % 256) as u8;
        }
        let obs = CarRacingObservation::new(px);
        let mut row = Vec::new();
        obs.write_host_row(&mut row);

        assert_eq!(
            row.len(),
            CarRacingObservation::shape().iter().product::<usize>()
        );
        assert_eq!(row.len(), PIXEL_BYTES);
        for (i, (&value, &byte)) in row.iter().zip(px.iter()).enumerate() {
            assert_eq!(value, f32::from(byte) / 255.0, "element {i}");
        }
        assert_eq!(row[0], 0.0);
        assert_eq!(row[255], 1.0);
    }
}
