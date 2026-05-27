//! Shared ASCII / styled renderer for fitness-landscape envs.
//!
//! Each of the three landscape envs (Sphere, Rastrigin, Ackley) is a pure
//! N-D fitness evaluator. For visualisation we sample the surface on a
//! `GRID_WIDTH × GRID_HEIGHT` grid spanning the env's `bounds()` along its
//! first two coordinates (others fixed at zero) and project each cell's
//! fitness into a quintile of the ramp ` `, `░`, `▒`, `▓`, `█`. The styled
//! variant additionally paints each block with a colour from the
//! `DarkGray → Blue → Cyan → Yellow → LightYellow` perceptual ramp; the
//! glyph itself carries the same information so colour-blind users still
//! read the surface correctly.
//!
//! Candidate-overlay markers (`×` current, `*` best-so-far) are
//! intentionally absent at this stage — the env type is stateless and the
//! candidate is owned by the EA layer. When `rlevo-evolution` starts
//! emitting `FrameRecord`s for landscape rollouts, a `LandscapeView`
//! wrapper that carries `&Landscape + Option<current> + Option<best>` will
//! be added and the overlay rendered there.

use crate::render::palette::{BEST_FG, BEST_MODIFIER};
use crate::render::{Color, SpanStyle, StyledFrame, StyledLine, StyledSpan};

/// Width (in cells / characters) of the rendered surface.
pub const GRID_WIDTH: usize = 40;
/// Height (in cells / characters) of the rendered surface.
pub const GRID_HEIGHT: usize = 20;

const RAMP: [char; 5] = [' ', '░', '▒', '▓', '█'];
const COLOURS: [Color; 5] = [
    Color::DarkGray,
    Color::Blue,
    Color::Cyan,
    Color::Yellow,
    Color::LightYellow,
];

/// Sample the landscape on the rendering grid and return `(values, fmin, fmax)`.
fn sample<F: Fn(f64, f64) -> f64>(f: F, bounds: (f64, f64)) -> (Vec<f64>, f64, f64) {
    let (lo, hi) = bounds;
    let mut values = Vec::with_capacity(GRID_WIDTH * GRID_HEIGHT);
    let mut fmin = f64::INFINITY;
    let mut fmax = f64::NEG_INFINITY;

    for row in 0..GRID_HEIGHT {
        // Flip Y so row 0 is the top of the rendered surface and rows
        // increase downward; this matches typical contour-plot orientation.
        #[allow(clippy::cast_precision_loss)]
        let ty = 1.0 - (row as f64 + 0.5) / GRID_HEIGHT as f64;
        let y = lo + ty * (hi - lo);
        for col in 0..GRID_WIDTH {
            #[allow(clippy::cast_precision_loss)]
            let tx = (col as f64 + 0.5) / GRID_WIDTH as f64;
            let x = lo + tx * (hi - lo);
            let v = f(x, y);
            if v < fmin {
                fmin = v;
            }
            if v > fmax {
                fmax = v;
            }
            values.push(v);
        }
    }
    (values, fmin, fmax)
}

/// Map a sample value to a quintile index in `0..5`.
fn quintile(v: f64, fmin: f64, fmax: f64) -> usize {
    let span = fmax - fmin;
    if span <= 0.0 {
        return 0;
    }
    let t = ((v - fmin) / span).clamp(0.0, 1.0);
    let idx = (t * 5.0).floor() as usize;
    idx.min(4)
}

fn header_line(label: &str, bounds: (f64, f64), fmin: f64, fmax: f64) -> String {
    format!(
        "{label}  bounds=[{:.2}, {:.2}]  fmin={fmin:.3}  fmax={fmax:.3}",
        bounds.0, bounds.1
    )
}

/// Render the landscape as a plain string: one header line + `GRID_HEIGHT`
/// body lines of `GRID_WIDTH` block-shading glyphs each.
#[must_use]
pub fn render_landscape_ascii<F: Fn(f64, f64) -> f64>(
    f: F,
    bounds: (f64, f64),
    label: &str,
) -> String {
    let (values, fmin, fmax) = sample(&f, bounds);
    let mut out = String::with_capacity(GRID_WIDTH * GRID_HEIGHT + 80);
    out.push_str(&header_line(label, bounds, fmin, fmax));
    out.push('\n');
    for row in 0..GRID_HEIGHT {
        for col in 0..GRID_WIDTH {
            let v = values[row * GRID_WIDTH + col];
            out.push(RAMP[quintile(v, fmin, fmax)]);
        }
        if row + 1 < GRID_HEIGHT {
            out.push('\n');
        }
    }
    out
}

/// Render the landscape as a styled frame.
///
/// The label carries [`BEST_FG`] + [`BEST_MODIFIER`] because the
/// rendered surface is the "thing we're optimising against" — the visual
/// equivalent of the goal marker in a grid env. Body glyphs carry the
/// quintile colour from the `DarkGray → LightYellow` ramp.
#[must_use]
pub fn render_landscape_styled<F: Fn(f64, f64) -> f64>(
    f: F,
    bounds: (f64, f64),
    label: &str,
) -> StyledFrame {
    let (values, fmin, fmax) = sample(&f, bounds);

    let mut lines: Vec<StyledLine> = Vec::with_capacity(GRID_HEIGHT + 1);
    let label_style = SpanStyle::default()
        .fg(BEST_FG)
        .with_modifier(BEST_MODIFIER);
    let header = header_line(label, bounds, fmin, fmax);
    let header_line_spans = if let Some(rest) = header.strip_prefix(label) {
        vec![
            StyledSpan::new(label, label_style),
            StyledSpan::raw(rest.to_string()),
        ]
    } else {
        vec![StyledSpan::raw(header)]
    };
    lines.push(StyledLine::from_spans(header_line_spans));

    for row in 0..GRID_HEIGHT {
        let mut spans: Vec<StyledSpan> = Vec::new();
        let mut current_style = SpanStyle::default();
        let mut current_text = String::with_capacity(GRID_WIDTH);
        for col in 0..GRID_WIDTH {
            let q = quintile(values[row * GRID_WIDTH + col], fmin, fmax);
            let style = SpanStyle::default().fg(COLOURS[q]);
            if style != current_style && !current_text.is_empty() {
                spans.push(StyledSpan::new(
                    std::mem::take(&mut current_text),
                    current_style,
                ));
            }
            current_style = style;
            current_text.push(RAMP[q]);
        }
        if !current_text.is_empty() {
            spans.push(StyledSpan::new(current_text, current_style));
        }
        lines.push(StyledLine::from_spans(spans));
    }
    StyledFrame { lines }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_like_surface_renders_within_budget() {
        let frame = render_landscape_ascii(|x, y| x * x + y * y, (-5.0, 5.0), "Sphere");
        for line in frame.lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
        // Header + body lines.
        assert_eq!(frame.lines().count(), 1 + GRID_HEIGHT);
    }

    #[test]
    fn styled_matches_ascii_glyphs() {
        let plain =
            render_landscape_ascii(|x, y| x * x + y * y, (-5.0, 5.0), "Sphere");
        let styled =
            render_landscape_styled(|x, y| x * x + y * y, (-5.0, 5.0), "Sphere");
        let plain_no_trailing: String = plain.lines().collect::<Vec<_>>().join("\n");
        assert_eq!(styled.plain_text(), plain_no_trailing);
    }

    #[test]
    fn quintile_endpoints_clamp() {
        assert_eq!(quintile(0.0, 0.0, 1.0), 0);
        assert_eq!(quintile(1.0, 0.0, 1.0), 4);
        // Degenerate (flat surface) collapses to quintile 0.
        assert_eq!(quintile(0.5, 1.0, 1.0), 0);
    }

    #[test]
    fn label_styled_with_best_palette() {
        use crate::render::palette::{BEST_FG, BEST_MODIFIER};

        let styled =
            render_landscape_styled(|x, y| x * x + y * y, (-5.0, 5.0), "Sphere");
        let header = &styled.lines[0];
        let label = header
            .spans
            .iter()
            .find(|s| s.text == "Sphere")
            .expect("label span present");
        assert_eq!(label.style.fg, Some(BEST_FG));
        assert!(label.style.modifier.contains(BEST_MODIFIER));
    }
}
