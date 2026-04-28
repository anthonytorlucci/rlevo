//! Terrain generation for BipedalWalker.

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

/// Flat terrain: a straight horizontal ground plane.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatTerrain;

impl TerrainGenerator for FlatTerrain {
    fn generate(&self, _rng: &mut StdRng) -> Vec<[f32; 2]> {
        // 200-unit-wide flat surface at y = 0
        (0..=200).map(|i| [i as f32 - 10.0, 0.0]).collect()
    }
}

/// Rough terrain: randomly varying height.
#[derive(Debug, Clone)]
pub struct RoughTerrain {
    /// Amplitude of height variation (world units).
    pub roughness: f32,
    /// Horizontal spacing between height samples.
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

/// Hardcore terrain: stumps, pits, and stair-like features.
#[derive(Debug, Clone)]
pub struct HardcoreTerrain {
    /// Expected number of pits per 10 world units.
    pub pit_frequency: f32,
    /// Expected number of stumps per 10 world units.
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
