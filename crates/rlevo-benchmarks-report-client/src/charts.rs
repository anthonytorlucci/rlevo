//! Convergence-plot panels built on `leptos-chartistry`.
//!
//! Pure-Rust SVG line charts per umbrella spec §3 constraint #6 (no JS
//! interop). Each panel pairs a thin raw line with a thicker smoothed
//! line; the width difference is the hue-redundant signal so a B/W
//! screenshot still distinguishes the two per the project a11y contract.

use leptos::prelude::*;
use leptos_chartistry::{AspectRatio, Chart, Line, Series};
use rlevo_metrics_registry::{MetricKind, descriptor, is_per_generation, title_for};

use crate::series::{
    AxisMode, BandPoint, BoxStats, available_metric_names, distinct_seed_count, diversity_series,
    downsample_minmax, episode_axis, episode_length_series, episode_reward_series,
    fitness_range_series, metric_band, metric_series, nearest_by_x, population_box_data,
    remap_episode_series, rolling_mean, selection_pressure_series,
};
use crate::wire::{EnvFamily, EpisodeRecord, PopulationSample};

/// Two-coordinate datum passed to `leptos-chartistry`.
///
/// `leptos-chartistry`'s `Tick` trait is implemented for `f64`; `u32` step
/// and generation values are widened to `f64` at the chart boundary.
#[derive(Debug, Clone, Copy)]
struct Point {
    /// Horizontal axis value (episode index or generation number, widened to `f64`).
    x: f64,
    /// Vertical axis value (reward, loss, fitness, etc.).
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
    let raw_xy: Vec<(f64, f64)> = raw.iter().map(|&(x, y)| (f64::from(x), y)).collect();
    let smoothed_xy: Option<Vec<(f64, f64)>> =
        smoothed.map(|s| s.iter().map(|&(x, y)| (f64::from(x), y)).collect());
    line_chart_view_xy(title, y_label, &raw_xy, smoothed_xy.as_deref())
}

/// Like [`line_chart_view`] but with floating-point x-coordinates, used when
/// the x-axis is a remapped continuous quantity (e.g. wall-clock seconds under
/// the axis-mode toggle).
#[must_use]
pub fn line_chart_view_xy(
    title: String,
    y_label: String,
    raw: &[(f64, f64)],
    smoothed: Option<&[(f64, f64)]>,
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

    let raw_points: Vec<Point> = raw.iter().map(|&(x, y)| Point { x, y }).collect();
    let smoothed_points: Option<Vec<Point>> =
        smoothed.map(|s| s.iter().map(|&(x, y)| Point { x, y }).collect());

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

/// Hand-rolled line panel with a hover crosshair that reports the *raw*
/// (un-decimated) sample under the cursor.
///
/// `decimated` drives the drawn path (kept light for long runs); `raw_full` is
/// the full-resolution series used only for the tooltip lookup, so the readout
/// is exact even when the path is decimated (M8.2 accuracy requirement). An
/// optional `smoothed` overlay is drawn at greater width for B/W legibility.
#[must_use]
pub fn interactive_line_view(
    title: String,
    raw_full: Vec<(f64, f64)>,
    decimated: Vec<(f64, f64)>,
    smoothed: Option<Vec<(f64, f64)>>,
) -> AnyView {
    use std::fmt::Write as _;
    if decimated.is_empty() {
        return view! {
            <figure class="rlevo-chart-card rlevo-chart-empty">
                <figcaption>{title}</figcaption>
                <p class="rlevo-chart-no-data">"no samples"</p>
            </figure>
        }
        .into_any();
    }

    let x_min = decimated.first().map_or(0.0, |p| p.0);
    let x_max_raw = decimated.last().map_or(1.0, |p| p.0);
    let x_max = if (x_max_raw - x_min).abs() < f64::EPSILON { x_min + 1.0 } else { x_max_raw };
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(_, y) in decimated.iter().chain(smoothed.iter().flatten()) {
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let span = (y_max - y_min).abs();
    let pad = if span < f64::EPSILON { 0.5 } else { span * 0.05 };
    y_min -= pad;
    y_max += pad;
    if (y_max - y_min).abs() < f64::EPSILON {
        y_max = y_min + 1.0;
    }

    let plot_w = BOX_VB_W - BOX_M_L - BOX_M_R;
    let plot_h = BOX_VB_H - BOX_M_T - BOX_M_B;
    // Pixel→data and data→pixel maps share these scalars (all Copy), so both
    // the static path build and the mousemove handler can use them.
    let sx_of = move |x: f64| BOX_M_L + (x - x_min) / (x_max - x_min) * plot_w;
    let sy_of = move |y: f64| BOX_M_T + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h;

    let mut raw_path = String::new();
    for &(x, y) in &decimated {
        if y.is_finite() {
            let _ = write!(raw_path, "{:.2},{:.2} ", sx_of(x), sy_of(y));
        }
    }
    let smoothed_path = smoothed.as_ref().map(|s| {
        let mut p = String::new();
        for &(x, y) in s {
            if y.is_finite() {
                let _ = write!(p, "{:.2},{:.2} ", sx_of(x), sy_of(y));
            }
        }
        p
    });

    let (hover, set_hover) = signal::<Option<(f64, f64, f64, f64)>>(None);

    let raw_for_move = raw_full;
    let on_move = move |ev: leptos::ev::MouseEvent| {
        use wasm_bindgen::JsCast as _;
        let Some(target) = ev.current_target() else { return };
        let Ok(elem) = target.dyn_into::<web_sys::Element>() else { return };
        let width = elem.get_bounding_client_rect().width();
        if width <= 0.0 {
            return;
        }
        let data_x = x_min + (f64::from(ev.offset_x()) / width) * (x_max - x_min);
        if let Some((rx, ry)) = nearest_by_x(&raw_for_move, data_x) {
            set_hover.set(Some((sx_of(rx), sy_of(ry), rx, ry)));
        }
    };
    let on_leave = move |_| set_hover.set(None);

    let x_axis_y = BOX_VB_H - BOX_M_B;
    let view_box = format!("0 0 {BOX_VB_W} {BOX_VB_H}");
    let smoothed_poly = smoothed_path.map(|pts| {
        view! { <polyline class="rlevo-line-smoothed" points={pts} fill="none" /> }.into_any()
    });

    let overlay = move || match hover.get() {
        Some((sx, sy, dx, dy)) => {
            // Clamp the label x so it stays inside the viewBox.
            let label_x = sx.min(BOX_VB_W - 90.0).max(BOX_M_L);
            let label = format!("{dx:.2}, {dy:.4}");
            view! {
                <line class="rlevo-crosshair" x1={sx} y1={BOX_M_T} x2={sx} y2={x_axis_y} />
                <circle class="rlevo-crosshair-dot" cx={sx} cy={sy} r=3.0 />
                <text class="rlevo-crosshair-label" x={label_x} y={BOX_M_T + 10.0}>{label}</text>
            }
            .into_any()
        }
        None => ().into_any(),
    };

    view! {
        <figure class="rlevo-chart-card">
            <figcaption>{title}</figcaption>
            <svg class="rlevo-line" viewBox={view_box} preserveAspectRatio="none"
                role="img" on:mousemove=on_move on:mouseleave=on_leave>
                <line class="rlevo-boxplot-axis" x1={BOX_M_L} y1={BOX_M_T} x2={BOX_M_L} y2={x_axis_y} />
                <polyline class="rlevo-line-raw" points={raw_path} fill="none" />
                {smoothed_poly}
                {overlay}
            </svg>
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

    // Episode-outcome panels (reward, length) get a global x-axis toggle:
    // episode index / cumulative env step / cumulative wall-clock. Per-update
    // metric panels below keep their native training-step axis.
    let episode_outcomes = episode_outcome_panels(records, window);

    let mut shared_panels: Vec<AnyView> = Vec::new();

    // Remaining metrics split into RL diagnostics and EO diagnostics by the
    // shared registry's `MetricKind` (ADR-0015 / [[2026-06-05-rl-vs-eo-learning]]),
    // so the report distinguishes gradient-based RL signals (losses, KL,
    // explained variance) from population-based EO signals (fitness, diversity).
    let mut rl_panels: Vec<AnyView> = Vec::new();
    let mut eo_panels: Vec<AnyView> = Vec::new();
    for name in available_metric_names(records) {
        let full = metric_series(records, &name);
        if full.is_empty() {
            continue;
        }
        // Decimate long series so the SVG path stays light; min/max bucketing
        // keeps peaks. The full series feeds the hover crosshair so the raw
        // value under the cursor is exact even when the path is decimated.
        let decimated = downsample_minmax(&full);
        let raw_full: Vec<(f64, f64)> =
            full.iter().map(|&(x, y)| (f64::from(x), y)).collect();
        let dec_xy: Vec<(f64, f64)> =
            decimated.iter().map(|&(x, y)| (f64::from(x), y)).collect();
        let title = title_for(&name).to_string();
        let panel = if is_per_generation(&name) {
            interactive_line_view(title, raw_full, dec_xy, None)
        } else {
            let smoothed: Vec<(f64, f64)> = rolling_mean(&decimated, METRIC_WINDOW)
                .iter()
                .map(|&(x, y)| (f64::from(x), y))
                .collect();
            interactive_line_view(title, raw_full, dec_xy, Some(smoothed))
        };
        match descriptor(&name).map(|d| d.kind) {
            Some(MetricKind::Eo) => eo_panels.push(panel),
            Some(MetricKind::Rl) => rl_panels.push(panel),
            // Shared metrics (per-episode terminal triple) and any unknown
            // metric land alongside the episode-outcome panels.
            _ => shared_panels.push(panel),
        }
    }

    let rl_section = group_section("RL diagnostics", "rlevo-group-rl", rl_panels);
    let eo_section = group_section("EO diagnostics", "rlevo-group-eo", eo_panels);
    let seed_section = multi_seed_section(records);

    view! {
        <section class="rlevo-convergence">
            <h2>"Convergence"</h2>
            {episode_outcomes}
            <div class="rlevo-chart-grid">{shared_panels}</div>
            {rl_section}
            {eo_section}
            {seed_section}
        </section>
    }
    .into_any()
}

/// Reactive episode-outcome block: a step/episode/wallclock x-axis toggle plus
/// the reward and length panels, which re-render synchronously when the mode
/// changes. The three axis vectors are precomputed once; switching only remaps.
fn episode_outcome_panels(records: &[EpisodeRecord], window: usize) -> AnyView {
    let reward = episode_reward_series(records);
    let reward_smoothed = rolling_mean(&reward, window);
    let length = episode_length_series(records);
    let length_smoothed = rolling_mean(&length, window);
    let axis_episode = episode_axis(records, AxisMode::Episode);
    let axis_step = episode_axis(records, AxisMode::Step);
    let axis_wall = episode_axis(records, AxisMode::Wallclock);

    let (mode, set_mode) = signal(AxisMode::Episode);

    let panels = move || {
        let axis = match mode.get() {
            AxisMode::Episode => &axis_episode,
            AxisMode::Step => &axis_step,
            AxisMode::Wallclock => &axis_wall,
        };
        let x_label = mode.get().label().to_string();
        let reward_xy = remap_episode_series(&reward, axis);
        let reward_sm = remap_episode_series(&reward_smoothed, axis);
        let length_xy = remap_episode_series(&length, axis);
        let length_sm = remap_episode_series(&length_smoothed, axis);
        view! {
            <div class="rlevo-chart-grid">
                {line_chart_view_xy(
                    format!("Episode reward (x: {x_label})"),
                    "reward".to_string(),
                    &reward_xy,
                    Some(&reward_sm),
                )}
                {line_chart_view_xy(
                    format!("Episode length (x: {x_label})"),
                    "frames".to_string(),
                    &length_xy,
                    Some(&length_sm),
                )}
            </div>
        }
    };

    let mk_button = move |m: AxisMode| {
        view! {
            <button
                class="rlevo-axis-btn"
                class:active=move || mode.get() == m
                on:click=move |_| set_mode.set(m)
            >
                {m.label()}
            </button>
        }
    };

    view! {
        <div class="rlevo-axis-toggle" role="group" aria-label="x-axis mode">
            <span class="rlevo-axis-label">"x-axis:"</span>
            {mk_button(AxisMode::Step)}
            {mk_button(AxisMode::Episode)}
            {mk_button(AxisMode::Wallclock)}
        </div>
        {panels}
    }
    .into_any()
}

/// Renders the cross-seed mean±std band section when the record set spans two
/// or more distinct seeds, or nothing for a single-seed run.
///
/// One band panel per metric that has at least one step with ≥2 contributing
/// seeds; per-generation EA metrics are excluded (population panels cover them).
fn multi_seed_section(records: &[EpisodeRecord]) -> AnyView {
    if distinct_seed_count(records) < 2 {
        return ().into_any();
    }
    let mut panels: Vec<AnyView> = Vec::new();
    for name in available_metric_names(records) {
        if is_per_generation(&name) {
            continue;
        }
        let band = metric_band(records, &name);
        if !band.iter().any(|p| p.n >= 2) {
            continue;
        }
        panels.push(band_chart_view(title_for(&name).to_string(), &band));
    }
    if panels.is_empty() {
        return ().into_any();
    }
    let n = distinct_seed_count(records);
    view! {
        <div class="rlevo-metric-group rlevo-group-seed">
            <h3>{format!("Multi-seed aggregation (mean ± std, n={n})")}</h3>
            <div class="rlevo-chart-grid">{panels}</div>
        </div>
    }
    .into_any()
}

/// Hand-rolled SVG mean±std band panel.
///
/// `leptos-chartistry` 0.2 has no area primitive, so the ±std envelope is a
/// filled `<polygon>` (upper edge left→right, lower edge right→left) under a
/// solid mean `<polyline>`. The fill/line pairing keeps the band legible in
/// B/W per the a11y contract.
#[must_use]
pub fn band_chart_view(title: String, band: &[BandPoint]) -> AnyView {
    use std::fmt::Write as _;
    if band.is_empty() {
        return view! {
            <figure class="rlevo-chart-card rlevo-chart-empty">
                <figcaption>{title}</figcaption>
                <p class="rlevo-chart-no-data">"no data"</p>
            </figure>
        }
        .into_any();
    }

    #[allow(clippy::cast_precision_loss)]
    let x_min = f64::from(band.first().map_or(0, |p| p.step));
    #[allow(clippy::cast_precision_loss)]
    let x_max_raw = f64::from(band.last().map_or(1, |p| p.step));
    let x_max = if (x_max_raw - x_min).abs() < f64::EPSILON {
        x_min + 1.0
    } else {
        x_max_raw
    };

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for p in band {
        y_min = y_min.min(p.mean - p.std);
        y_max = y_max.max(p.mean + p.std);
    }
    let span = (y_max - y_min).abs();
    let pad = if span < f64::EPSILON { 0.5 } else { span * 0.05 };
    y_min -= pad;
    y_max += pad;
    if (y_max - y_min).abs() < f64::EPSILON {
        y_max = y_min + 1.0;
    }

    let plot_w = BOX_VB_W - BOX_M_L - BOX_M_R;
    let plot_h = BOX_VB_H - BOX_M_T - BOX_M_B;
    let scale_x = move |g: f64| -> f64 { BOX_M_L + (g - x_min) / (x_max - x_min) * plot_w };
    let scale_y = move |v: f64| -> f64 { BOX_M_T + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h };

    // Band polygon: upper edge forward, lower edge back.
    let mut poly = String::new();
    for p in band {
        let _ = write!(
            poly,
            "{:.2},{:.2} ",
            scale_x(f64::from(p.step)),
            scale_y(p.mean + p.std)
        );
    }
    for p in band.iter().rev() {
        let _ = write!(
            poly,
            "{:.2},{:.2} ",
            scale_x(f64::from(p.step)),
            scale_y(p.mean - p.std)
        );
    }
    let mut mean_pts = String::new();
    for p in band {
        let _ = write!(
            mean_pts,
            "{:.2},{:.2} ",
            scale_x(f64::from(p.step)),
            scale_y(p.mean)
        );
    }

    let view_box = format!("0 0 {BOX_VB_W} {BOX_VB_H}");
    let y_min_label = format!("{y_min:.3}");
    let y_max_label = format!("{y_max:.3}");
    let x_axis_y = BOX_VB_H - BOX_M_B;

    view! {
        <figure class="rlevo-chart-card">
            <figcaption>{title}</figcaption>
            <svg class="rlevo-band" viewBox={view_box} preserveAspectRatio="none" role="img">
                <polygon class="rlevo-band-fill" points={poly} />
                <polyline class="rlevo-band-mean" points={mean_pts} fill="none" />
                <line class="rlevo-boxplot-axis"
                    x1={BOX_M_L} y1={BOX_M_T} x2={BOX_M_L} y2={x_axis_y} />
                <text class="rlevo-boxplot-axis-label" x=4.0 y={BOX_M_T + 8.0}>{y_max_label}</text>
                <text class="rlevo-boxplot-axis-label" x=4.0 y={x_axis_y}>{y_min_label}</text>
            </svg>
        </figure>
    }
    .into_any()
}

/// Wraps a paradigm-specific panel group under its own heading, or renders
/// nothing when the group is empty (so a pure-RL run shows no EO section and
/// vice versa).
fn group_section(heading: &str, class: &str, panels: Vec<AnyView>) -> AnyView {
    if panels.is_empty() {
        return ().into_any();
    }
    let heading = heading.to_string();
    let class = format!("rlevo-metric-group {class}");
    view! {
        <div class=class>
            <h3>{heading}</h3>
            <div class="rlevo-chart-grid">{panels}</div>
        </div>
    }
    .into_any()
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

/// SVG viewBox width for the hand-rolled box plot, in user units.
const BOX_VB_W: f64 = 640.0;
/// SVG viewBox height for the hand-rolled box plot, in user units.
const BOX_VB_H: f64 = 300.0;
/// Left margin reserved for the y-axis labels.
const BOX_M_L: f64 = 56.0;
/// Right margin between the last box and the viewBox edge.
const BOX_M_R: f64 = 16.0;
/// Top margin above the plot area.
const BOX_M_T: f64 = 20.0;
/// Bottom margin reserved for the x-axis labels.
const BOX_M_B: f64 = 32.0;

/// Hand-rolled per-generation fitness box plot.
///
/// Inside each generation: filled rect for `[Q1, Q3]`, horizontal
/// median tick, vertical whiskers clipped at the Tukey 1.5×IQR fence,
/// outliers as small open circles. Three overlay polylines (best,
/// median, worst) pair colour with distinct dash patterns so the a11y
/// contract survives a B/W screenshot.
/// Deterministic jitter in `[-1, 1)` from an index — a cheap integer hash so
/// the strip-plot scatter is stable across renders (no RNG; `Math.random` is
/// unavailable and determinism keeps screenshots reproducible).
fn jitter_unit(i: u64) -> f64 {
    let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // Top 53 bits → [0, 1), then map to [-1, 1).
    let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
    unit.mul_add(2.0, -1.0)
}

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

    // Strip-plot points: one jittered (cx, cy) per sampled individual, computed
    // once. The jitter is a deterministic hash of the index (no RNG — keeps the
    // render reproducible), spread across ~90% of the box width.
    let mut strip_pts: Vec<(f64, f64)> = Vec::new();
    for s in stats {
        let cx = scale_x(f64::from(s.generation));
        for (i, v) in s.points.iter().enumerate() {
            strip_pts.push((cx + jitter_unit(i as u64) * box_w * 0.45, scale_y(*v)));
        }
    }
    let (show_strip, set_show_strip) = signal(false);
    let strip_view = move || {
        if !show_strip.get() {
            return ().into_any();
        }
        let dots: Vec<AnyView> = strip_pts
            .iter()
            .map(|&(cx, cy)| view! { <circle class="rlevo-strip-dot" cx={cx} cy={cy} r=1.5 /> }.into_any())
            .collect();
        view! { <g class="rlevo-strip">{dots}</g> }.into_any()
    };

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
            <div class="rlevo-boxplot-toolbar">
                <button
                    type="button"
                    class="rlevo-strip-toggle"
                    class:active=move || show_strip.get()
                    aria-pressed=move || if show_strip.get() { "true" } else { "false" }
                    on:click=move |_| set_show_strip.update(|v| *v = !*v)
                >
                    "Individual points"
                </button>
            </div>
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
                // strip-plot jitter cloud beneath the boxes (toggle)
                {strip_view}
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

/// Composes the EA population section: box plot, diversity trace, and selection-pressure panel.
///
/// Derives per-generation [`BoxStats`] and overlay traces from `samples`, then
/// builds up to three panels: the hand-rolled fitness box plot (always), a
/// diversity line chart (when data is present), and a selection-pressure ratio
/// chart (when data is present).  Returns an empty `<span>` when `samples` is
/// empty so the section disappears cleanly from RL-only runs.
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
        // Titles now come from the shared registry (ADR-0015).
        assert_eq!(title_for("policy_loss"), "Policy loss");
        assert_eq!(title_for("approx_kl"), "Approx KL");
        assert_eq!(title_for("entropy"), "Policy entropy");
    }

    #[test]
    fn pretty_metric_title_unknown_passes_through() {
        assert_eq!(title_for("my_custom_metric"), "my_custom_metric");
    }
}
