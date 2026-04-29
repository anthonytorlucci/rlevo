//! Software rasterizer for the 96×96×3 CarRacing observation.
//!
//! No external image crate dependency. Uses scan-line fill for convex polygons.

/// Width and height of the rendered frame (pixels).
pub const FRAME_SIZE: usize = 96;

/// Total pixel buffer size (width × height × 3 RGB channels).
pub const PIXEL_BYTES: usize = FRAME_SIZE * FRAME_SIZE * 3;

/// Software rasterizer producing a 96×96 RGB pixel buffer.
pub struct Rasterizer {
    buffer: Box<[u8; PIXEL_BYTES]>,
}

impl std::fmt::Debug for Rasterizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rasterizer")
            .field("size", &FRAME_SIZE)
            .finish()
    }
}

impl Rasterizer {
    /// Create a new rasterizer (buffer initialised to black).
    pub fn new() -> Self {
        Self { buffer: Box::new([0u8; PIXEL_BYTES]) }
    }

    /// Clear the buffer to `color` (RGB).
    pub fn clear(&mut self, color: [u8; 3]) {
        for i in 0..FRAME_SIZE * FRAME_SIZE {
            self.buffer[i * 3] = color[0];
            self.buffer[i * 3 + 1] = color[1];
            self.buffer[i * 3 + 2] = color[2];
        }
    }

    /// Fill a convex polygon defined by `vertices` in pixel space.
    ///
    /// Vertices are `[pixel_x, pixel_y]` floats. The polygon is assumed convex;
    /// non-convex polygons may render incorrectly.
    pub fn fill_polygon(&mut self, vertices: &[[f32; 2]], color: [u8; 3]) {
        if vertices.len() < 3 {
            return;
        }
        // Find bounding box
        let (mut min_y, mut max_y) = (FRAME_SIZE as i32, 0i32);
        for v in vertices {
            let y = v[1] as i32;
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }
        min_y = min_y.clamp(0, FRAME_SIZE as i32 - 1);
        max_y = max_y.clamp(0, FRAME_SIZE as i32 - 1);

        for py in min_y..=max_y {
            let y = py as f32 + 0.5;
            let mut xs: Vec<f32> = Vec::new();
            let n = vertices.len();
            for i in 0..n {
                let a = vertices[i];
                let b = vertices[(i + 1) % n];
                if (a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y) {
                    let t = (y - a[1]) / (b[1] - a[1]);
                    xs.push(a[0] + t * (b[0] - a[0]));
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for chunk in xs.chunks_exact(2) {
                let x0 = (chunk[0] as i32).clamp(0, FRAME_SIZE as i32 - 1);
                let x1 = (chunk[1] as i32).clamp(0, FRAME_SIZE as i32 - 1);
                for px in x0..=x1 {
                    let idx = (py as usize * FRAME_SIZE + px as usize) * 3;
                    self.buffer[idx] = color[0];
                    self.buffer[idx + 1] = color[1];
                    self.buffer[idx + 2] = color[2];
                }
            }
        }
    }

    /// Draw a filled rectangle in pixel space.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: [u8; 3]) {
        let x0 = x.clamp(0, FRAME_SIZE as i32 - 1);
        let y0 = y.clamp(0, FRAME_SIZE as i32 - 1);
        let x1 = (x + w).clamp(0, FRAME_SIZE as i32);
        let y1 = (y + h).clamp(0, FRAME_SIZE as i32);
        for py in y0..y1 {
            for px in x0..x1 {
                let idx = (py as usize * FRAME_SIZE + px as usize) * 3;
                self.buffer[idx] = color[0];
                self.buffer[idx + 1] = color[1];
                self.buffer[idx + 2] = color[2];
            }
        }
    }

    /// Return the raw pixel buffer (row-major, RGB).
    pub fn pixels(&self) -> &[u8; PIXEL_BYTES] {
        &self.buffer
    }
}

impl Default for Rasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_sets_color() {
        let mut r = Rasterizer::new();
        r.clear([255, 0, 0]);
        for i in 0..FRAME_SIZE * FRAME_SIZE {
            assert_eq!(r.buffer[i * 3], 255);
            assert_eq!(r.buffer[i * 3 + 1], 0);
            assert_eq!(r.buffer[i * 3 + 2], 0);
        }
    }

    #[test]
    fn test_fill_polygon_no_panic() {
        let mut r = Rasterizer::new();
        r.fill_polygon(
            &[[10.0, 10.0], [50.0, 10.0], [30.0, 50.0]],
            [0, 255, 0],
        );
        // Triangle was rasterised — at least one pixel should be green
        let has_green = (0..FRAME_SIZE * FRAME_SIZE).any(|i| r.buffer[i * 3 + 1] == 255);
        assert!(has_green);
    }

    #[test]
    fn test_fill_polygon_degenerate_does_not_panic() {
        let mut r = Rasterizer::new();
        r.fill_polygon(&[[10.0, 10.0], [20.0, 10.0]], [255, 0, 0]);
        r.fill_polygon(&[], [255, 0, 0]);
    }

    #[test]
    fn test_fill_rect() {
        let mut r = Rasterizer::new();
        r.fill_rect(5, 5, 10, 10, [0, 0, 255]);
        let idx = (5 * FRAME_SIZE + 5) * 3;
        assert_eq!(r.buffer[idx + 2], 255);
    }
}
