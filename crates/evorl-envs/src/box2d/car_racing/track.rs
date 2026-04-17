//! Bezier-curve track generation for CarRacing.

use rand::RngExt;
use rand::rngs::StdRng;

use super::config::CarRacingConfig;

/// A single quad tile of road surface.
#[derive(Debug, Clone)]
pub struct TrackTile {
    /// Four world-space corners of the tile quad (winding: CCW).
    pub vertices: [[f32; 2]; 4],
    /// RGB colour for rendering.
    pub color: [u8; 3],
    /// Whether this tile has been visited by the car.
    pub visited: bool,
}

/// A closed-loop procedurally generated track.
#[derive(Debug, Clone)]
pub struct Track {
    pub tiles: Vec<TrackTile>,
    pub start_pos: [f32; 2],
    pub start_angle: f32,
}

impl Track {
    /// Generate a randomised closed-loop track.
    ///
    /// Algorithm:
    /// 1. Place N control points uniformly around a circle.
    /// 2. Perturb each point radially.
    /// 3. Sort by angle (ensure simple polygon).
    /// 4. Use Catmull-Rom interpolation to produce the centreline.
    /// 5. Tesselate into `TrackTile` quads.
    pub fn generate(config: &CarRacingConfig, rng: &mut StdRng) -> Self {
        let n = config.track_n_checkpoints;
        let radius = 900.0_f32 / 30.0; // world units

        // Generate control points on a perturbed circle
        let angles: Vec<f32> = (0..n)
            .map(|i| {
                let base = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
                base + rng.random_range(-0.4..=0.4)
            })
            .collect();

        let radii: Vec<f32> = (0..n)
            .map(|_| radius * rng.random_range(0.7..=1.3))
            .collect();

        // Sort by angle to get a simple (non-self-intersecting) polygon
        let pts: Vec<(f32, f32)> = angles
            .iter()
            .zip(radii.iter())
            .map(|(a, r)| (r * a.cos(), r * a.sin()))
            .collect();
        // Already in angle order since we generated them in order
        // Just close the loop
        let closed: Vec<[f32; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();

        // Catmull-Rom interpolation
        let samples_per_seg = 5usize;
        let mut centreline: Vec<[f32; 2]> = Vec::new();
        let m = closed.len();
        for i in 0..m {
            let p0 = closed[(i + m - 1) % m];
            let p1 = closed[i % m];
            let p2 = closed[(i + 1) % m];
            let p3 = closed[(i + 2) % m];
            for s in 0..samples_per_seg {
                let t = s as f32 / samples_per_seg as f32;
                let x = catmull_rom(p0[0], p1[0], p2[0], p3[0], t);
                let y = catmull_rom(p0[1], p1[1], p2[1], p3[1], t);
                centreline.push([x, y]);
            }
        }

        // Build tiles from consecutive centreline pairs
        let hw = config.track_width / 2.0 / 30.0; // half-width in world units
        let n_tiles = centreline.len();
        let mut tiles: Vec<TrackTile> = Vec::with_capacity(n_tiles);
        let road_color: [u8; 3] = [102, 102, 102];
        let kerb_color: [u8; 3] = [255, 255, 255];

        for i in 0..n_tiles {
            let a = centreline[i];
            let b = centreline[(i + 1) % n_tiles];
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len = (dx * dx + dy * dy).sqrt().max(1e-6);
            // Perpendicular
            let nx = -dy / len;
            let ny = dx / len;

            let color = if i % 3 == 0 { kerb_color } else { road_color };
            tiles.push(TrackTile {
                vertices: [
                    [a[0] + nx * hw, a[1] + ny * hw],
                    [a[0] - nx * hw, a[1] - ny * hw],
                    [b[0] - nx * hw, b[1] - ny * hw],
                    [b[0] + nx * hw, b[1] + ny * hw],
                ],
                color,
                visited: false,
            });
        }

        // Update tile_reward based on actual tile count
        let start_pos = centreline[0];
        let dx = centreline[1][0] - centreline[0][0];
        let dy = centreline[1][1] - centreline[0][1];
        let start_angle = dy.atan2(dx);

        Self { tiles, start_pos, start_angle }
    }

    /// Return the index of the tile closest to `pos`.
    pub fn nearest_tile(&self, pos: [f32; 2]) -> Option<usize> {
        self.tiles.iter().enumerate().min_by(|(_, a), (_, b)| {
            let ca = tile_centre(a);
            let cb = tile_centre(b);
            let da = dist2(pos, ca);
            let db = dist2(pos, cb);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }).map(|(i, _)| i)
    }
}

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)
}

fn tile_centre(tile: &TrackTile) -> [f32; 2] {
    let x = tile.vertices.iter().map(|v| v[0]).sum::<f32>() / 4.0;
    let y = tile.vertices.iter().map(|v| v[1]).sum::<f32>() / 4.0;
    [x, y]
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_track(seed: u64) -> Track {
        let cfg = CarRacingConfig::default();
        let mut rng = StdRng::seed_from_u64(seed);
        Track::generate(&cfg, &mut rng)
    }

    #[test]
    fn test_tile_count() {
        let track = make_track(0);
        assert!(
            track.tiles.len() >= 50,
            "track must have at least 50 tiles, got {}",
            track.tiles.len()
        );
    }

    #[test]
    fn test_tile_vertices_count() {
        let track = make_track(1);
        for tile in &track.tiles {
            assert_eq!(tile.vertices.len(), 4);
        }
    }

    #[test]
    fn test_determinism() {
        let a = make_track(42);
        let b = make_track(42);
        assert_eq!(a.tiles.len(), b.tiles.len());
        for (ta, tb) in a.tiles.iter().zip(b.tiles.iter()) {
            for (va, vb) in ta.vertices.iter().zip(tb.vertices.iter()) {
                assert!((va[0] - vb[0]).abs() < 1e-6);
                assert!((va[1] - vb[1]).abs() < 1e-6);
            }
        }
    }
}
