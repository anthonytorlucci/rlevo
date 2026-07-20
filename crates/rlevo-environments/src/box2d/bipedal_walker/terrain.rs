//! Terrain generation for the `BipedalWalker` environment.
//!
//! The [`TerrainGenerator`] trait is the extension point for ground geometry.
//! Three concrete generators ship with the crate:
//!
//! | Type | Description |
//! |---|---|
//! | [`FlatTerrain`] | Constant y = 0 over 200 world units |
//! | [`RoughTerrain`] | Random height variation after a flat spawn pad |
//! | [`HardcoreTerrain`] | Pits and stumps with tunable frequency |
//!
//! All generators hold a flat spawn pad that **spans the spawn point** (the
//! walker's hull spawns at world x = 0). The pad is `SPAWN_PAD` one-unit
//! steps wide and the walker sits at its middle, matching the reference
//! Gymnasium `TERRAIN_STARTPAD` convention, so obstacles and roughness never
//! begin at the spawn point regardless of the difficulty setting.
//!
//! Points are in **world space** (not scaled by `SCALE`). Scaling is applied
//! inside `BipedalWalker::build_ground` when constructing `Rapier2D` colliders.

use rand::RngExt;
use rand::rngs::StdRng;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

/// World-space x at which terrain generation ends (the terminal cap point).
const TERRAIN_END_X: f32 = 190.0;
/// Horizontal inset of a pit/stump vertical wall from its nominal edge, so the
/// near-vertical face has a small but non-zero run (avoids a zero-length seg).
const WALL_INSET: f32 = 0.1;
/// Depth (world y) of a pit floor.
const PIT_DEPTH: f32 = -2.0;
/// Minimum pit width (world units, inclusive lower bound of the sampled range).
const PIT_WIDTH_MIN: f32 = 2.0;
/// Maximum pit width (world units, exclusive upper bound of the sampled range).
const PIT_WIDTH_MAX: f32 = 5.0;
/// Height (world y) of a stump top.
const STUMP_HEIGHT: f32 = 1.5;
/// Width (world units) of a stump.
const STUMP_WIDTH: f32 = 0.5;
/// Number of one-unit flat steps in the spawn pad. The walker spawns at world
/// x = 0, the middle of this pad, matching Gymnasium `TERRAIN_STARTPAD = 20`.
const SPAWN_PAD: usize = 20;

/// Rejects a value that is not both non-negative and finite.
///
/// Neither [`config::positive`] (allows `+∞`) nor [`config::in_range`] with an
/// infinite upper bound (also allows `+∞`) expresses "finite and `>= 0`", so
/// terrain validation needs this local check.
fn nonneg_finite(config: &'static str, field: &'static str, got: f32) -> Result<(), ConfigError> {
    if got.is_finite() && got >= 0.0 {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::OutOfRange {
                lo: 0.0,
                hi: f64::INFINITY,
                got: f64::from(got),
            },
        })
    }
}

/// Rejects a value that is not both strictly positive and finite.
///
/// [`config::positive`] alone accepts `+∞` (`+∞ > 0`), so it cannot guard a
/// step size that must be a usable finite length.
fn positive_finite(config: &'static str, field: &'static str, got: f32) -> Result<(), ConfigError> {
    if got.is_finite() && got > 0.0 {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotPositive {
                got: f64::from(got),
            },
        })
    }
}

/// Pluggable terrain generator.
///
/// Implementations produce a polyline of (x, y) height samples that are used to
/// build the ground collider in the rapier world.
///
/// # Invariants
///
/// Every implementation's [`generate`](TerrainGenerator::generate) output MUST
/// satisfy, for the polyline `pts`:
///
/// - `pts.len() >= 2` — at least one segment exists.
/// - `pts[i][0] <= pts[i + 1][0]` for all `i` — x is **non-decreasing**
///   left-to-right (a decreasing step would build a backwards, overlapping
///   ground collider).
/// - `pts[0][0] < 0.0` — the first point is left of the spawn point (world
///   x = 0), so the agent spawns over solid ground.
/// - Points are in **world space** — not pre-scaled by `SCALE`.
///
/// These are re-checked at the `BipedalWalker::rebuild_world` chokepoint
/// (ADR 0040): a generator that violates them is rejected with a `ConfigError`
/// rather than silently building invalid geometry.
///
/// # Examples
///
/// ```
/// use rlevo_environments::box2d::bipedal_walker::{FlatTerrain, TerrainGenerator};
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let mut rng = StdRng::seed_from_u64(0);
/// let pts = FlatTerrain.generate(&mut rng);
/// assert!(pts.len() >= 2);
/// assert!(pts[0][0] < 0.0, "must spawn over solid ground");
/// for w in pts.windows(2) {
///     assert!(w[0][0] <= w[1][0], "x must be non-decreasing");
/// }
/// ```
pub trait TerrainGenerator: std::fmt::Debug + Send + Sync {
    /// Generate terrain height samples as a list of world-space (x, y) points.
    ///
    /// Points are ordered left-to-right and satisfy the trait
    /// [`# Invariants`](TerrainGenerator#invariants): at least two points,
    /// non-decreasing x, and a first point with negative x.
    fn generate(&self, rng: &mut StdRng) -> Vec<[f32; 2]>;
}

/// Flat terrain: a straight horizontal ground plane at y = 0.
///
/// Produces 201 evenly spaced points spanning x ∈ `[-10, 190]`.
/// No randomness is used; `rng` is ignored.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatTerrain;

impl TerrainGenerator for FlatTerrain {
    fn generate(&self, _rng: &mut StdRng) -> Vec<[f32; 2]> {
        // 200-unit-wide flat surface at y = 0, spanning x ∈ [-10, 190].
        (0..=200).map(|i| [i as f32 - 10.0, 0.0]).collect()
    }
}

/// Rough terrain: randomly varying height after a flat spawn pad.
///
/// The first `SPAWN_PAD` points are flat so the walker spawns on solid
/// ground; every point beyond the pad receives a y coordinate sampled uniformly
/// from `[-roughness, roughness]`. The default values are `roughness = 1.5` and
/// `step = 1.0`.
///
/// Fields are private and validated at construction: use
/// [`RoughTerrain::new`] for caller-supplied values, or [`Default`] for the
/// standard settings.
#[derive(Debug, Clone)]
pub struct RoughTerrain {
    /// Half-amplitude of height variation in world units.
    roughness: f32,
    /// Horizontal distance between consecutive height samples (world units).
    step: f32,
}

impl Default for RoughTerrain {
    fn default() -> Self {
        Self {
            roughness: 1.5,
            step: 1.0,
        }
    }
}

impl RoughTerrain {
    /// Create a validated [`RoughTerrain`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `roughness` is negative or non-finite
    /// (`roughness == 0.0` is legal — it yields flat terrain), or if `step` is
    /// not strictly positive and finite. Validating here prevents the empty
    /// `Uniform(-roughness, roughness)` range that would otherwise panic inside
    /// [`generate`](TerrainGenerator::generate).
    pub fn new(roughness: f32, step: f32) -> Result<Self, ConfigError> {
        let terrain: Self = Self { roughness, step };
        terrain.validate()?;
        Ok(terrain)
    }

    /// Half-amplitude of the post-pad height variation (world units).
    #[must_use]
    pub fn roughness(&self) -> f32 {
        self.roughness
    }

    /// Horizontal spacing between consecutive height samples (world units).
    #[must_use]
    pub fn step(&self) -> f32 {
        self.step
    }
}

impl Validate for RoughTerrain {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "RoughTerrain";
        // roughness == 0 is legal (flat), so require non-negative + finite,
        // NOT strictly positive.
        nonneg_finite(C, "roughness", self.roughness)?;
        positive_finite(C, "step", self.step)?;
        Ok(())
    }
}

impl TerrainGenerator for RoughTerrain {
    fn generate(&self, rng: &mut StdRng) -> Vec<[f32; 2]> {
        let n: usize = 200;
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(n + 1);
        // First point at world x = -10 (< 0); the first SPAWN_PAD steps are flat
        // so the pad spans the spawn point (world x = 0) and extends ahead of
        // the hull. Random height only beyond the pad.
        for i in 0..=n {
            let x: f32 = i as f32 * self.step - 10.0;
            // `roughness <= 0.0` (validated non-negative, so == 0) skips the
            // draw entirely, avoiding a degenerate `-0.0..=0.0` sample range.
            let y: f32 = if i < SPAWN_PAD || self.roughness <= 0.0 {
                0.0
            } else {
                rng.random_range(-self.roughness..=self.roughness)
            };
            pts.push([x, y]);
        }
        pts
    }
}

/// Hardcore terrain: randomly placed pits and stumps over a flat baseline.
///
/// After the flat spawn pad the generator advances in world space. At each
/// obstacle site a Bernoulli draw inserts a pit, a stump, or a 1-unit flat
/// step; each obstacle advances x by **its own width** (pits by their sampled
/// 2–5 units, stumps by `STUMP_WIDTH`, flat by 1). Pits drop to
/// `PIT_DEPTH`; stumps rise to `STUMP_HEIGHT`.
///
/// The default values are `pit_frequency = 0.5` and `stump_frequency = 0.5`,
/// giving an effective obstacle probability of 0.1 per site (each divided by 10
/// in the generator loop).
///
/// Fields are private and validated at construction: use
/// [`HardcoreTerrain::new`] for caller-supplied values, or [`Default`] for the
/// standard settings.
#[derive(Debug, Clone)]
pub struct HardcoreTerrain {
    /// Controls how often pits appear. At each site the pit branch is taken
    /// when a uniform draw is less than `pit_frequency / 10`.
    pit_frequency: f32,
    /// Controls how often stumps appear. The stump branch fires when the draw
    /// falls in `[pit_frequency / 10, (pit_frequency + stump_frequency) / 10)`.
    stump_frequency: f32,
}

impl Default for HardcoreTerrain {
    fn default() -> Self {
        Self {
            pit_frequency: 0.5,
            stump_frequency: 0.5,
        }
    }
}

impl HardcoreTerrain {
    /// Create a validated [`HardcoreTerrain`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if either frequency is negative or non-finite,
    /// or if `pit_frequency + stump_frequency > 10.0` (which would push the
    /// combined branch probability past 1.0 and break the pit/stump/flat
    /// partition in [`generate`](TerrainGenerator::generate)).
    pub fn new(pit_frequency: f32, stump_frequency: f32) -> Result<Self, ConfigError> {
        let terrain: Self = Self {
            pit_frequency,
            stump_frequency,
        };
        terrain.validate()?;
        Ok(terrain)
    }

    /// Pit branch frequency (probability `pit_frequency / 10` per site).
    #[must_use]
    pub fn pit_frequency(&self) -> f32 {
        self.pit_frequency
    }

    /// Stump branch frequency (probability `stump_frequency / 10` per site).
    #[must_use]
    pub fn stump_frequency(&self) -> f32 {
        self.stump_frequency
    }
}

impl Validate for HardcoreTerrain {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "HardcoreTerrain";
        nonneg_finite(C, "pit_frequency", self.pit_frequency)?;
        nonneg_finite(C, "stump_frequency", self.stump_frequency)?;
        // The two branch probabilities are `freq / 10`; their sum must stay a
        // valid probability so the else-branch (flat) keeps a non-empty share.
        config::in_range(
            C,
            "stump_frequency",
            0.0,
            10.0,
            f64::from(self.pit_frequency + self.stump_frequency),
        )?;
        Ok(())
    }
}

impl TerrainGenerator for HardcoreTerrain {
    fn generate(&self, rng: &mut StdRng) -> Vec<[f32; 2]> {
        /// World-space x at which the obstacle region begins (ahead of the pad).
        const OBSTACLE_START_X: f32 = 10.0;

        let span: f32 = TERRAIN_END_X - OBSTACLE_START_X;
        // The smallest obstacle advance is a stump (STUMP_WIDTH), so bound the
        // iteration count by that; each iteration pushes at most 4 points, plus
        // the pad and the terminal cap (#120 §2.1 — the old `n + 20` estimate
        // under-reserved).
        let max_iters: usize = (span / STUMP_WIDTH).ceil() as usize;
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(SPAWN_PAD + max_iters * 4 + 1);

        // Flat spawn pad: SPAWN_PAD one-unit steps ending just before the
        // obstacle region, so the walker (spawn x = 0) sits at the pad's middle
        // and the first obstacle begins ahead of the hull at OBSTACLE_START_X.
        for i in 0..SPAWN_PAD {
            pts.push([i as f32 - 10.0, 0.0]);
        }

        // Obstacle region, iterated in WORLD space (points pushed directly in
        // world coordinates — no -10 offset — so the non-decreasing-x invariant
        // is legible and cannot be hidden by a dual-offset convention).
        let mut x: f32 = OBSTACLE_START_X;
        while x < TERRAIN_END_X {
            let p: f32 = rng.random();
            if p < self.pit_frequency / 10.0 {
                // Pit: drop to PIT_DEPTH for a sampled width, then return.
                let width: f32 = rng.random_range(PIT_WIDTH_MIN..PIT_WIDTH_MAX);
                pts.push([x, 0.0]);
                pts.push([x + WALL_INSET, PIT_DEPTH]);
                pts.push([x + width - WALL_INSET, PIT_DEPTH]);
                pts.push([x + width, 0.0]);
                x += width;
            } else if p < (self.pit_frequency + self.stump_frequency) / 10.0 {
                // Stump: rise to STUMP_HEIGHT for STUMP_WIDTH.
                pts.push([x, 0.0]);
                pts.push([x + WALL_INSET, STUMP_HEIGHT]);
                pts.push([x + STUMP_WIDTH - WALL_INSET, STUMP_HEIGHT]);
                pts.push([x + STUMP_WIDTH, 0.0]);
                x += STUMP_WIDTH;
            } else {
                pts.push([x, 0.0]);
                x += 1.0;
            }
        }

        // Terminal cap: append the end point ONLY if the last obstacle did not
        // already reach or overshoot TERRAIN_END_X. A wide pit near the boundary
        // can push the last point past 190; unconditionally appending [190, 0]
        // then would land LEFT of it and build a backwards, overlapping cuboid
        // (#120 §1.1 — the regression this whole refactor targets).
        if pts
            .last()
            .is_some_and(|&[last_x, _]| last_x < TERRAIN_END_X)
        {
            pts.push([TERRAIN_END_X, 0.0]);
        }
        pts
    }
}

#[cfg(test)]
mod tests {
    // Terrain x-coordinates are exact literals produced by a fixed step, and
    // the determinism checks compare two runs that must agree bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_flat_terrain_is_flat() {
        let mut rng = StdRng::seed_from_u64(0);
        let pts = FlatTerrain.generate(&mut rng);
        assert!(!pts.is_empty());
        for p in &pts {
            assert_eq!(p[1], 0.0, "flat terrain must have y=0 everywhere");
        }
    }

    #[test]
    fn test_flat_terrain_extent() {
        // §7.2: exact extent — 201 points spanning world x ∈ [-10, 190].
        let mut rng = StdRng::seed_from_u64(0);
        let pts = FlatTerrain.generate(&mut rng);
        assert_eq!(pts.len(), 201, "flat terrain must have 201 points");
        assert_eq!(pts[0][0], -10.0, "flat terrain must start at x = -10");
        assert_eq!(pts[200][0], 190.0, "flat terrain must end at x = 190");
    }

    #[test]
    fn test_rough_terrain_spawn_zone() {
        let mut rng = StdRng::seed_from_u64(42);
        let pts = RoughTerrain::default().generate(&mut rng);
        // The first SPAWN_PAD points are the flat spawn pad.
        for p in pts.iter().take(SPAWN_PAD) {
            assert_eq!(p[1], 0.0, "spawn pad must be flat");
        }
    }

    #[test]
    fn test_rough_terrain_has_variation_past_pad() {
        // §7.2: a default (roughness 1.5) rough terrain must produce at least
        // one non-flat point beyond the spawn pad.
        let mut rng = StdRng::seed_from_u64(42);
        let pts = RoughTerrain::default().generate(&mut rng);
        assert!(
            pts.iter().any(|p| p[1] != 0.0),
            "rough terrain must vary in height past the pad"
        );
    }

    #[test]
    fn test_hardcore_terrain_has_points() {
        let mut rng = StdRng::seed_from_u64(7);
        let pts = HardcoreTerrain::default().generate(&mut rng);
        assert!(pts.len() >= 10);
        for w in pts.windows(2) {
            assert!(w[0][0] <= w[1][0], "terrain x must be non-decreasing");
        }
    }

    #[test]
    fn test_hardcore_x_non_decreasing_all_seeds() {
        // §7.1: the regression test for #120 §1.1. A wide pit near the boundary
        // could previously push x past 190; the unconditional terminal push of
        // [190, 0] then landed LEFT of it, breaking monotonicity. Sweep many
        // seeds so at least one hits that boundary-overshoot case.
        for seed in 0..512u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let pts = HardcoreTerrain::default().generate(&mut rng);
            assert!(pts.len() >= 2, "seed {seed}: need at least 2 points");
            for w in pts.windows(2) {
                assert!(
                    w[0][0] <= w[1][0],
                    "seed {seed}: x decreased ({} -> {})",
                    w[0][0],
                    w[1][0]
                );
            }
        }
    }

    #[test]
    fn test_flat_pad_spans_spawn_point_rough() {
        // §1.2: every point at or before the spawn point (and a margin ahead)
        // must be flat, so the hull never spawns into an obstacle.
        let mut rng = StdRng::seed_from_u64(3);
        let pts = RoughTerrain::default().generate(&mut rng);
        assert!(pts[0][0] < 0.0, "first point must be left of spawn (x < 0)");
        for p in &pts {
            if p[0] <= 5.0 {
                assert_eq!(p[1], 0.0, "spawn pad point at x={} must be flat", p[0]);
            }
        }
    }

    #[test]
    fn test_flat_pad_spans_spawn_point_hardcore() {
        // §1.2: same invariant for hardcore across several seeds.
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let pts = HardcoreTerrain::default().generate(&mut rng);
            assert!(
                pts[0][0] < 0.0,
                "seed {seed}: first point must be left of spawn (x < 0)"
            );
            for p in &pts {
                if p[0] <= 5.0 {
                    assert_eq!(
                        p[1], 0.0,
                        "seed {seed}: spawn pad point at x={} must be flat",
                        p[0]
                    );
                }
            }
        }
    }

    #[test]
    fn test_rough_new_rejects_invalid_roughness() {
        // §7.3: negative and non-finite roughness are rejected; zero is legal.
        assert!(
            RoughTerrain::new(-1.0, 1.0).is_err(),
            "negative roughness must be rejected"
        );
        assert!(
            RoughTerrain::new(f32::NAN, 1.0).is_err(),
            "NaN roughness must be rejected"
        );
        assert!(
            RoughTerrain::new(f32::INFINITY, 1.0).is_err(),
            "infinite roughness must be rejected"
        );
        assert!(
            RoughTerrain::new(0.0, 1.0).is_ok(),
            "zero roughness is legal (flat terrain)"
        );
    }

    #[test]
    fn test_rough_new_rejects_invalid_step() {
        // §7.3: non-positive and non-finite step are rejected.
        assert_eq!(
            RoughTerrain::new(1.0, 0.0).unwrap_err().field,
            "step",
            "zero step must be rejected"
        );
        assert!(
            RoughTerrain::new(1.0, -0.5).is_err(),
            "negative step must be rejected"
        );
        assert!(
            RoughTerrain::new(1.0, f32::INFINITY).is_err(),
            "infinite step must be rejected"
        );
    }

    #[test]
    fn test_rough_new_accessors_roundtrip() {
        let t = RoughTerrain::new(2.0, 0.5).expect("valid rough terrain");
        assert_eq!(t.roughness(), 2.0);
        assert_eq!(t.step(), 0.5);
    }

    #[test]
    fn test_rough_zero_roughness_is_flat() {
        // §7.3 corollary: zero roughness yields all-flat terrain (no panic).
        let mut rng = StdRng::seed_from_u64(1);
        let pts = RoughTerrain::new(0.0, 1.0)
            .expect("valid")
            .generate(&mut rng);
        for p in &pts {
            assert_eq!(p[1], 0.0, "zero-roughness terrain must be flat");
        }
    }

    #[test]
    fn test_hardcore_new_rejects_invalid_frequencies() {
        assert!(
            HardcoreTerrain::new(-1.0, 0.5).is_err(),
            "negative pit_frequency must be rejected"
        );
        assert!(
            HardcoreTerrain::new(0.5, f32::NAN).is_err(),
            "NaN stump_frequency must be rejected"
        );
        assert!(
            HardcoreTerrain::new(6.0, 6.0).is_err(),
            "pit + stump > 10 must be rejected"
        );
        assert!(
            HardcoreTerrain::new(0.5, 0.5).is_ok(),
            "default frequencies must validate"
        );
    }

    #[test]
    fn test_hardcore_new_accessors_roundtrip() {
        let t = HardcoreTerrain::new(0.3, 0.7).expect("valid hardcore terrain");
        assert_eq!(t.pit_frequency(), 0.3);
        assert_eq!(t.stump_frequency(), 0.7);
    }

    #[test]
    fn test_rough_default_validates() {
        // ADR 0026: a library default must pass its own validate().
        assert!(RoughTerrain::default().validate().is_ok());
    }

    #[test]
    fn test_hardcore_default_validates() {
        // ADR 0026: a library default must pass its own validate().
        assert!(HardcoreTerrain::default().validate().is_ok());
    }

    #[test]
    fn test_hardcore_reaches_pit_and_stump() {
        // Obstacle-geometry regression: over a seed sweep, at least one point
        // must reach PIT_DEPTH and at least one must reach STUMP_HEIGHT, so a
        // change that flattens obstacles or flips a sign fails loudly.
        let mut saw_pit: bool = false;
        let mut saw_stump: bool = false;
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let pts = HardcoreTerrain::default().generate(&mut rng);
            for p in &pts {
                if (p[1] - PIT_DEPTH).abs() < 1e-6 {
                    saw_pit = true;
                }
                if (p[1] - STUMP_HEIGHT).abs() < 1e-6 {
                    saw_stump = true;
                }
            }
        }
        assert!(
            saw_pit,
            "no point reached PIT_DEPTH ({PIT_DEPTH}) across seeds"
        );
        assert!(
            saw_stump,
            "no point reached STUMP_HEIGHT ({STUMP_HEIGHT}) across seeds"
        );
    }

    #[test]
    fn test_rough_step_spacing_non_unit() {
        // §7.3 extra: a non-unit step must set the x-spacing exactly and keep
        // x monotonic.
        let step: f32 = 0.5;
        let mut rng = StdRng::seed_from_u64(9);
        let pts = RoughTerrain::new(1.0, step)
            .expect("valid rough terrain")
            .generate(&mut rng);
        assert!(pts[0][0] < 0.0, "first point must be left of spawn (x < 0)");
        for w in pts.windows(2) {
            assert!(w[0][0] <= w[1][0], "x must be non-decreasing");
            approx::assert_relative_eq!(w[1][0] - w[0][0], step, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_new_rejects_nan_fields() {
        // §7.3 extra: field-specific NaN cases for both generators.
        assert!(
            RoughTerrain::new(1.0, f32::NAN).is_err(),
            "NaN step must be rejected"
        );
        assert!(
            HardcoreTerrain::new(f32::NAN, 0.5).is_err(),
            "NaN pit_frequency must be rejected"
        );
    }
}
