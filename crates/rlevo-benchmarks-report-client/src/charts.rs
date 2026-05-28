//! Convergence-plot panels built on `leptos-chartistry`.
//!
//! Pure-Rust SVG line charts per umbrella spec §3 constraint #6 (no JS
//! interop). Each panel pairs a thin raw line with a thicker smoothed
//! line; the width difference is the hue-redundant signal so a B/W
//! screenshot still distinguishes the two per the project a11y contract.

use leptos::prelude::*;
use leptos_chartistry::{AspectRatio, Chart, Line, Series};

use crate::series::{
    BoxStats, available_metric_names, diversity_series, episode_length_series,
    episode_reward_series, fitness_range_series, metric_series, population_box_data, rolling_mean,
    selection_pressure_series,
};
use crate::wire::{EnvFamily, EpisodeRecord, PopulationSample};

/// Two-coordinate datum the chart consumes. `leptos-chartistry`'s
/// [`Tick`](leptos_chartistry::Tick) trait is implemented for `f64`
/// and `DateTime<Tz>` — `u32` step values are widened at the chart
/// boundary.
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

/// Default rolling-mean window for per-episode panels. Falls back to
/// `len/4` when the run is shorter than the window, so even a 4-episode
/// run still shows a smoothed overlay.
const EPISODE_WINDOW: usize = 50;

/// Rolling-mean window for per-update metric panels (loss / entropy /
/// kl / clip-frac). Narrower than the episode window because metric
/// samples already fire per PPO update, not per env step.
const METRIC_WINDOW: usize = 20;

/// One panel: thin raw line at low alpha plus a thicker smoothed
/// overlay. The pairing keeps the panel legible in B/W per the a11y
/// contract — the width step (1.0 vs 2.5) survives without colour.
///
/// `smoothed = None` draws only the raw line (used for series that are
/// already aggregated per-generation, e.g. `best_fitness`).
#[must_use]
pub fn line_chart_view(
    title: String,
    y_label: String,
    raw: &[(u32, f64)],
    smoothed: Option<&[(u32, f64)]>,
) -> AnyView {
    if raw.is_empty() {
        return view! {
            <figure class="rlevo-chart-card rlevo-chart-empty">
                <figcaption>{title}</figcaption>
                <p class="rlevo-chart-no-data">"no samples"</p>
            </figure>
        }
        .into_any();
    }

    let raw_points: Vec<Point> = raw
        .iter()
        .map(|&(x, y)| Point {
            x: f64::from(x),
            y,
        })
        .collect();
    let smoothed_points: Option<Vec<Point>> = smoothed.map(|s| {
        s.iter()
            .map(|&(x, y)| Point {
                x: f64::from(x),
                y,
            })
            .collect()
    });

    let raw_signal: Signal<Vec<Point>> = Signal::derive(move || raw_points.clone());

    let chart = match smoothed_points {
        Some(sm) => {
            let sm_signal: Signal<Vec<Point>> = Signal::derive(move || sm.clone());
            view! {
                <div class="rlevo-chart-stack">
                    <Chart
                        aspect_ratio=AspectRatio::from_outer_ratio(420.0, 200.0)
                        series=Series::new(|p: &Point| p.x)
                            .line(Line::new(|p: &Point| p.y).with_name("raw").with_width(1.0))
                        data=raw_signal
                    />
                    <Chart
                        aspect_ratio=AspectRatio::from_outer_ratio(420.0, 200.0)
                        series=Series::new(|p: &Point| p.x)
                            .line(Line::new(|p: &Point| p.y).with_name("smoothed").with_width(2.5))
                        data=sm_signal
                    />
                </div>
            }
            .into_any()
        }
        None => view! {
            <Chart
                aspect_ratio=AspectRatio::from_outer_ratio(420.0, 200.0)
                series=Series::new(|p: &Point| p.x)
                    .line(Line::new(|p: &Point| p.y).with_name("value").with_width(2.0))
                data=raw_signal
            />
        }
        .into_any(),
    };

    view! {
        <figure class="rlevo-chart-card">
            <figcaption>{title}</figcaption>
            {chart}
            <span class="rlevo-chart-y">{y_label}</span>
        </figure>
    }
    .into_any()
}

/// Compose the panel grid for a run. RL panels appear when the
/// corresponding metric is present in the record; EA panels appear
/// when the `*_fitness` metrics are present. Empty panels are
/// suppressed entirely.
///
/// `family` is currently unused — the suppression rule keys off
/// per-metric presence, which already distinguishes RL from EA runs.
/// Kept in the signature so future per-family panel choices (e.g. an
/// EA-only diversity panel) can branch without changing the call site.
#[must_use]
pub fn convergence_panel_view(records: &[EpisodeRecord], _family: EnvFamily) -> AnyView {
    if records.is_empty() {
        return view! {
            <section class="rlevo-convergence rlevo-convergence-empty">
                <h2>"Convergence"</h2>
                <p class="rlevo-chart-no-data">"no episodes recorded"</p>
            </section>
        }
        .into_any();
    }

    let episode_count = records.len();
    let window = EPISODE_WINDOW.min(episode_count.max(1) * 4 / 4).max(1);

    let reward = episode_reward_series(records);
    let reward_smoothed = rolling_mean(&reward, window);
    let length = episode_length_series(records);
    let length_smoothed = rolling_mean(&length, window);

    let mut panels: Vec<AnyView> = Vec::new();
    panels.push(line_chart_view(
        "Episode reward".to_string(),
        "reward".to_string(),
        &reward,
        Some(&reward_smoothed),
    ));
    panels.push(line_chart_view(
        "Episode length".to_string(),
        "frames".to_string(),
        &length,
        Some(&length_smoothed),
    ));

    for name in available_metric_names(records) {
        let raw = metric_series(records, &name);
        if raw.is_empty() {
            continue;
        }
        let title = pretty_metric_title(&name);
        let panel = if is_per_generation(&name) {
            line_chart_view(title, name.clone(), &raw, None)
        } else {
            let smoothed = rolling_mean(&raw, METRIC_WINDOW);
            line_chart_view(title, name.clone(), &raw, Some(&smoothed))
        };
        panels.push(panel);
    }

    view! {
        <section class="rlevo-convergence">
            <h2>"Convergence"</h2>
            <div class="rlevo-chart-grid">{panels}</div>
        </section>
    }
    .into_any()
}

fn is_per_generation(name: &str) -> bool {
    matches!(
        name,
        "best_fitness" | "mean_fitness" | "worst_fitness" | "best_fitness_ever"
    )
}

fn pretty_metric_title(name: &str) -> String {
    match name {
        "policy_loss" => "Policy loss".into(),
        "value_loss" => "Value loss".into(),
        "loss" => "Loss".into(),
        "entropy" => "Policy entropy".into(),
        "approx_kl" => "Approx KL".into(),
        "clip_frac" => "Clip fraction".into(),
        "best_fitness" => "Best fitness".into(),
        "mean_fitness" => "Mean fitness".into(),
        "worst_fitness" => "Worst fitness".into(),
        "best_fitness_ever" => "Best fitness (ever)".into(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// EA Population panel (box plot + diversity trace + selection pressure).
//
// `leptos-chartistry` 0.2.3 ships `Line` and `Bar` only — no box plot, no
// scatter, no area-band primitive. The box plot is hand-rolled SVG so the
// project a11y contract (distinct dash patterns on the best/median/worst
// overlay lines) survives a B/W screenshot. The diversity trace and
// selection-pressure indicator reuse `line_chart_view`.
// ---------------------------------------------------------------------------

/// SVG viewport for the box plot.
const BOX_VB_W: f64 = 640.0;
const BOX_VB_H: f64 = 300.0;
const BOX_M_L: f64 = 56.0;
const BOX_M_R: f64 = 16.0;
const BOX_M_T: f64 = 20.0;
const BOX_M_B: f64 = 32.0;

/// Hand-rolled per-generation fitness box plot.
///
/// Inside each generation: filled rect for `[Q1, Q3]`, horizontal
/// median tick, vertical whiskers clipped at the Tukey 1.5×IQR fence,
/// outliers as small open circles. Three overlay polylines (best,
/// median, worst) pair colour with distinct dash patterns so the a11y
/// contract survives a B/W screenshot.
#[must_use]
pub fn population_box_view(
    stats: &[BoxStats],
    overlays: (Vec<(u32, f64)>, Vec<(u32, f64)>, Vec<(u32, f64)>),
) -> AnyView {
    if stats.is_empty() {
        return view! {
            <figure class="rlevo-chart-card rlevo-chart-empty">
                <figcaption>"Fitness distribution per generation"</figcaption>
                <p class="rlevo-chart-no-data">"no population samples"</p>
            </figure>
        }
        .into_any();
    }

    let (best, median, worst) = overlays;

    // x range
    #[allow(clippy::cast_precision_loss)]
    let x_min = f64::from(stats.first().map_or(0, |s| s.generation));
    #[allow(clippy::cast_precision_loss)]
    let x_max_raw = f64::from(stats.last().map_or(1, |s| s.generation));
    let x_max = if (x_max_raw - x_min).abs() < f64::EPSILON {
        x_min + 1.0
    } else {
        x_max_raw
    };

    // y range — include outliers + overlay traces.
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let update = |y_min: &mut f64, y_max: &mut f64, v: f64| {
        if v < *y_min {
            *y_min = v;
        }
        if v > *y_max {
            *y_max = v;
        }
    };
    for s in stats {
        update(&mut y_min, &mut y_max, s.min);
        update(&mut y_min, &mut y_max, s.max);
        for o in &s.outliers {
            update(&mut y_min, &mut y_max, *o);
        }
    }
    for series in [&best, &median, &worst] {
        for (_, v) in series {
            update(&mut y_min, &mut y_max, *v);
        }
    }
    let span = (y_max - y_min).abs();
    let pad = if span < f64::EPSILON {
        0.5
    } else {
        span * 0.05
    };
    y_min -= pad;
    y_max += pad;
    if (y_max - y_min).abs() < f64::EPSILON {
        y_max = y_min + 1.0;
    }

    let plot_w = BOX_VB_W - BOX_M_L - BOX_M_R;
    let plot_h = BOX_VB_H - BOX_M_T - BOX_M_B;
    let scale_x = move |g: f64| -> f64 {
        BOX_M_L + (g - x_min) / (x_max - x_min) * plot_w
    };
    let scale_y = move |v: f64| -> f64 {
        BOX_M_T + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h
    };

    // Per-generation box width: a fraction of the per-generation slice,
    // capped so dense runs do not produce hairlines and sparse runs do
    // not bleed into neighbours.
    #[allow(clippy::cast_precision_loss)]
    let slice = plot_w / stats.len().max(1) as f64;
    let box_w = (slice * 0.6).clamp(2.0, 20.0);

    let mut box_elems: Vec<AnyView> = Vec::new();
    for s in stats {
        let cx = scale_x(f64::from(s.generation));
        let y_q1 = scale_y(s.q1);
        let y_q3 = scale_y(s.q3);
        let y_med = scale_y(s.median);
        let y_lo = scale_y(s.min);
        let y_hi = scale_y(s.max);
        let top = y_q3.min(y_q1);
        let bot = y_q3.max(y_q1);
        box_elems.push(
            view! {
                <line class="rlevo-boxplot-whisker"
                    x1={cx} y1={y_lo} x2={cx} y2={y_hi} />
            }
            .into_any(),
        );
        box_elems.push(
            view! {
                <rect class="rlevo-boxplot-fill"
                    x={cx - box_w / 2.0} y={top}
                    width={box_w} height={(bot - top).max(0.5)} />
            }
            .into_any(),
        );
        box_elems.push(
            view! {
                <line class="rlevo-boxplot-median"
                    x1={cx - box_w / 2.0} y1={y_med}
                    x2={cx + box_w / 2.0} y2={y_med} />
            }
            .into_any(),
        );
        for o in &s.outliers {
            box_elems.push(
                view! {
                    <circle class="rlevo-boxplot-outlier"
                        cx={cx} cy={scale_y(*o)} r=2.0 />
                }
                .into_any(),
            );
        }
    }

    let polyline_str = |series: &[(u32, f64)]| -> String {
        series
            .iter()
            .map(|(x, y)| format!("{:.2},{:.2}", scale_x(f64::from(*x)), scale_y(*y)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let best_pts = polyline_str(&best);
    let median_pts = polyline_str(&median);
    let worst_pts = polyline_str(&worst);

    let view_box = format!("0 0 {BOX_VB_W} {BOX_VB_H}");
    let x_axis_y = BOX_VB_H - BOX_M_B;
    let y_min_label = format!("{y_min:.3}");
    let y_max_label = format!("{y_max:.3}");
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let x_min_label = format!("gen {}", x_min as u32);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let x_max_label = format!("gen {}", x_max as u32);

    view! {
        <figure class="rlevo-chart-card rlevo-boxplot-card">
            <figcaption>"Fitness distribution per generation"</figcaption>
            <svg class="rlevo-svg-frame rlevo-boxplot-svg"
                viewBox=view_box
                role="img"
                aria-label="per-generation fitness box plot">
                // axes
                <line class="rlevo-boxplot-axis"
                    x1={BOX_M_L} y1={x_axis_y}
                    x2={BOX_VB_W - BOX_M_R} y2={x_axis_y} />
                <line class="rlevo-boxplot-axis"
                    x1={BOX_M_L} y1={BOX_M_T}
                    x2={BOX_M_L} y2={x_axis_y} />
                // boxes + whiskers + medians + outliers
                {box_elems}
                // overlay reference lines (best / median trace / worst)
                <polyline class="rlevo-boxplot-best" points={best_pts} />
                <polyline class="rlevo-boxplot-median-trace" points={median_pts} />
                <polyline class="rlevo-boxplot-worst" points={worst_pts} />
                // axis labels
                <text class="rlevo-boxplot-axis-label"
                    x={BOX_M_L - 6.0} y={BOX_M_T + 4.0}
                    text-anchor="end">{y_max_label}</text>
                <text class="rlevo-boxplot-axis-label"
                    x={BOX_M_L - 6.0} y={x_axis_y}
                    text-anchor="end">{y_min_label}</text>
                <text class="rlevo-boxplot-axis-label"
                    x={BOX_M_L} y={x_axis_y + 16.0}>{x_min_label}</text>
                <text class="rlevo-boxplot-axis-label"
                    x={BOX_VB_W - BOX_M_R} y={x_axis_y + 16.0}
                    text-anchor="end">{x_max_label}</text>
            </svg>
            <figcaption class="rlevo-chart-y">
                "lines: best (solid) · median (dashed) · worst (dotted) — lower is better"
            </figcaption>
        </figure>
    }
    .into_any()
}

/// Compose the EA Population section: box plot + diversity trace +
/// selection-pressure indicator. Section suppresses entirely when no
/// population samples are present.
#[must_use]
pub fn population_panel_view(samples: &[PopulationSample]) -> AnyView {
    if samples.is_empty() {
        return view! { <span></span> }.into_any();
    }

    let box_stats = population_box_data(samples);
    let overlays = fitness_range_series(samples);
    let diversity = diversity_series(samples);
    let pressure = selection_pressure_series(samples);

    let mut panels: Vec<AnyView> = Vec::new();
    panels.push(population_box_view(&box_stats, overlays));
    if !diversity.is_empty() {
        panels.push(line_chart_view(
            "Diversity".to_string(),
            "diversity".to_string(),
            &diversity,
            None,
        ));
    }
    if !pressure.is_empty() {
        panels.push(line_chart_view(
            "Selection pressure (best / median)".to_string(),
            "ratio".to_string(),
            &pressure,
            None,
        ));
    }

    view! {
        <section class="rlevo-population">
            <h2>"Population"</h2>
            <div class="rlevo-chart-grid">{panels}</div>
        </section>
    }
    .into_any()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_generation_flag() {
        assert!(is_per_generation("best_fitness"));
        assert!(is_per_generation("mean_fitness"));
        assert!(!is_per_generation("policy_loss"));
        assert!(!is_per_generation("entropy"));
        assert!(!is_per_generation("custom"));
    }

    #[test]
    fn pretty_metric_titles_known_names() {
        assert_eq!(pretty_metric_title("policy_loss"), "Policy loss");
        assert_eq!(pretty_metric_title("approx_kl"), "Approx KL");
        assert_eq!(pretty_metric_title("entropy"), "Policy entropy");
    }

    #[test]
    fn pretty_metric_title_unknown_passes_through() {
        assert_eq!(pretty_metric_title("my_custom_metric"), "my_custom_metric");
    }
}
