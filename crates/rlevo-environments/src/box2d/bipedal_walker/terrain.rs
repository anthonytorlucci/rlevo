//! Terrain generation for the BipedalWalker environment.
//!
//! The [`TerrainGenerator`] trait is the extension point for ground geometry.
//! Three concrete generators ship with the crate:
//!
//! | Type | Description |
//! |---|---|
//! | [`FlatTerrain`] | Constant y = 0 over 200 world units |
//! | [`RoughTerrain`] | Random height variation after a flat spawn zone |
//! | [`HardcoreTerrain`] | Pits and stumps with tunable frequency |
//!
//! All generators preserve a flat 10-unit spawn zone at the left end so the
//! walker starts on solid ground regardless of the difficulty setting.
//!
//! Points are in **world space** (not scaled by `SCALE`). Scaling is applied
//! inside `BipedalWalker::build_ground` when constructing Rapier2D colliders.

use rand::RngExt;
use rand::rngs::StdRng;

/// Pluggable terrain generator.
///
/// Implementations produce a polyline of (x, y) height samples that are
/// used to build the ground collider in the rapier world.
pub trait TerrainGenerator: std::fmt::Debug + Send + Sync {
    /// Generate terrain height samples as a list of world-space (x, y) points.
    ///
    /// Points are ordered left-to-right. The first point should start at
    /// a negative x so the agent spawns over solid ground.
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
        // 200-unit-wide flat surface at y = 0
        (0..=200).map(|i| [i as f32 - 10.0, 0.0]).collect()
    }
}

/// Rough terrain: randomly varying height after a flat spawn zone.
///
/// Each point beyond the 10-unit spawn zone receives a y coordinate sampled
/// uniformly from `[-roughness, roughness]`. The default values are
/// `roughness = 1.5` and `step = 1.0`.
#[derive(Debug, Clone)]
pub struct RoughTerrain {
    /// Half-amplitude of height variation in world units. The y coordinate of
    /// each post-spawn point is drawn from `Uniform(-roughness, roughness)`.
    pub roughness: f32,
    /// Horizontal distance between consecutive height samples (world units).
    pub step: f32,
}

impl Default for RoughTerrain {
    fn default() -> Self {
        Self {
            roughness: 1.5,
            step: 1.0,
        }
    }
}

impl TerrainGenerator for RoughTerrain {
    fn generate(&self, rng: &mut StdRng) -> Vec<[f32; 2]> {
        let n = 200usize;
        let mut pts = Vec::with_capacity(n + 1);
        // Start flat for 10 units so the agent can spawn
        for i in 0..10 {
            pts.push([i as f32 * self.step - 10.0, 0.0]);
        }
        // Random height after the spawn zone
        for i in 10..=n {
            let y: f32 = rng.random_range(-self.roughness..=self.roughness);
            pts.push([i as f32 * self.step - 10.0, y]);
        }
        pts
    }
}

/// Hardcore terrain: randomly placed pits and stumps over a flat baseline.
///
/// At each 1-unit step a Bernoulli draw decides whether to insert a pit, a
/// stump, or flat ground. Pits drop to y = −2 for a randomly sampled width of
/// 2–5 units; stumps rise to y = 1.5 for a fixed width of 0.5 units.
///
/// The default values are `pit_frequency = 0.5` and `stump_frequency = 0.5`,
/// giving an effective obstacle probability of 0.1 per unit (each divided by 10
/// in the generator loop).
#[derive(Debug, Clone)]
pub struct HardcoreTerrain {
    /// Controls how often pits appear. At each step the pit branch is taken
    /// when a uniform draw is less than `pit_frequency / 10`.
    pub pit_frequency: f32,
    /// Controls how often stumps appear. The stump branch fires when the draw
    /// falls in `[pit_frequency / 10, (pit_frequency + stump_frequency) / 10)`.
    pub stump_frequency: f32,
}

impl Default for HardcoreTerrain {
    fn default() -> Self {
        Self {
            pit_frequency: 0.5,
            stump_frequency: 0.5,
        }
    }
}

impl TerrainGenerator for HardcoreTerrain {
    fn generate(&self, rng: &mut StdRng) -> Vec<[f32; 2]> {
        let n = 200usize;
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(n + 20);
        // Flat spawn zone
        for i in 0..10 {
            pts.push([i as f32 - 10.0, 0.0]);
        }
        // Spawn zone ends at world x = -1 (i=9 → x_coord = 9-10 = -1).
        // Loop starts at x=10 so first point lands at x_coord = 10-10 = 0,
        // maintaining non-decreasing order.
        let mut x = 10.0_f32;
        while x < 200.0 {
            let p: f32 = rng.random();
            if p < self.pit_frequency / 10.0 {
                // pit: drop to -2 for 2–5 units, then come back
                let width: f32 = rng.random_range(2.0..5.0);
                pts.push([x - 10.0, 0.0]);
                pts.push([x - 10.0 + 0.1, -2.0]);
                pts.push([x + width - 10.0 - 0.1, -2.0]);
                pts.push([x + width - 10.0, 0.0]);
                x += width;
            } else if p < (self.pit_frequency + self.stump_frequency) / 10.0 {
                // stump: rise to 1.5 for 0.5 units
                pts.push([x - 10.0, 0.0]);
                pts.push([x - 10.0 + 0.1, 1.5]);
                pts.push([x + 0.5 - 10.0 - 0.1, 1.5]);
                pts.push([x + 0.5 - 10.0, 0.0]);
                x += 0.5;
            } else {
                pts.push([x - 10.0, 0.0]);
                x += 1.0;
            }
        }
        pts.push([190.0, 0.0]);
        pts
    }
}

#[cfg(test)]
mod tests {
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
    fn test_rough_terrain_spawn_zone() {
        let mut rng = StdRng::seed_from_u64(42);
        let pts = RoughTerrain::default().generate(&mut rng);
        // First 10 points should be flat (spawn zone)
        for p in pts.iter().take(10) {
            assert_eq!(p[1], 0.0, "spawn zone must be flat");
        }
    }

    #[test]
    fn test_hardcore_terrain_has_points() {
        let mut rng = StdRng::seed_from_u64(7);
        let pts = HardcoreTerrain::default().generate(&mut rng);
        assert!(pts.len() >= 10);
        // x coords should be monotonically increasing
        for w in pts.windows(2) {
            assert!(w[0][0] <= w[1][0], "terrain x must be non-decreasing");
        }
    }
}
