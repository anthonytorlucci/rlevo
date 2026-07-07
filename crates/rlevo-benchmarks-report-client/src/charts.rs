//! Convergence-plot panels built on `leptos-chartistry`.
//!
//! Pure-Rust SVG line charts per umbrella spec §3 constraint #6 (no JS
//! interop). Each panel pairs a thin raw line with a thicker smoothed
//! line; the width difference is the hue-redundant signal so a B/W
//! screenshot still distinguishes the two per the project a11y contract.

use leptos::prelude::*;
use rlevo_metrics_registry::{MetricKind, descriptor, is_per_generation, title_for};

use crate::series::{
    AxisMode, BandPoint, BoxStats, available_metric_names, distinct_seed_count, diversity_series,
    downsample_minmax, ensure_svg_header, episode_axis, episode_length_series,
    episode_reward_series, fitness_range_series, low_diversity_threshold, metric_band,
    metric_series, nearest_by_x, population_box_data, remap_episode_series, rolling_mean,
    selection_pressure_series,
};
use crate::wire::{EnvFamily, EpisodeRecord, ObjectiveSense, PopulationSample};

/// Default rolling-mean window for per-episode panels. Falls back to
/// `len/4` when the run is shorter than the window, so even a 4-episode
/// run still shows a smoothed overlay.
const EPISODE_WINDOW: usize = 50;

/// Rolling-mean window for per-update metric panels (loss / entropy /
/// kl / clip-frac). Narrower than the episode window because metric
/// samples already fire per PPO update, not per env step.
const METRIC_WINDOW: usize = 20;

/// Hand-rolled line panel with labeled axes and a hover crosshair that reports
/// the *raw* (un-decimated) sample under the cursor.
///
/// This is the single line renderer for the whole report: episode-outcome
/// curves, per-step RL/EA metrics, and the selection-pressure trace all flow
/// through it. `decimated` drives the drawn path (kept light for long runs);
/// `raw_full` is the full-resolution series used only for the tooltip lookup,
/// so the readout is exact even when the path is decimated (M8.2 accuracy
/// requirement). An optional `smoothed` overlay is drawn at greater width for
/// B/W legibility. `x_title` / `y_title` label the axes; pass `x_as_int = true`
/// for discrete x axes (episode / generation / step) and `false` for
/// continuous ones (wall-clock seconds).
#[must_use]
pub fn interactive_line_view(
    title: String,
    x_title: &str,
    y_title: &str,
    x_as_int: bool,
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
    let x_max = if (x_max_raw - x_min).abs() < f64::EPSILON {
        x_min + 1.0
    } else {
        x_max_raw
    };
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

    let plot_w = PLOT_RIGHT - BOX_M_L;
    let plot_h = PLOT_BOTTOM - BOX_M_T;
    // Pixel→data and data→pixel maps share these scalars (all Copy), so both
    // the static path build and the mousemove handler can use them. They match
    // `axis_layer`'s maps exactly so the line registers with its tick marks.
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
        let Some(target) = ev.current_target() else {
            return;
        };
        let Ok(elem) = target.dyn_into::<web_sys::Element>() else {
            return;
        };
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

    let view_box = format!("0 0 {BOX_VB_W} {BOX_VB_H}");
    let axes = axis_layer(x_min, x_max, y_min, y_max, x_title, y_title, x_as_int);
    let smoothed_poly = smoothed_path.map(|pts| {
        view! { <polyline class="rlevo-line-smoothed" points={pts} fill="none" /> }.into_any()
    });

    let overlay = move || match hover.get() {
        Some((sx, sy, dx, dy)) => {
            // Clamp the label x so it stays inside the viewBox.
            let label_x = sx.min(BOX_VB_W - 90.0).max(BOX_M_L);
            let label = format!("{dx:.2}, {dy:.4}");
            view! {
                <line class="rlevo-crosshair" x1={sx} y1={BOX_M_T} x2={sx} y2={PLOT_BOTTOM} />
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
            {export_button("rlevo-metric.svg")}
            <svg class="rlevo-line" viewBox={view_box} preserveAspectRatio="none"
                role="img" on:mousemove=on_move on:mouseleave=on_leave>
                {axes}
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
/// `records` may arrive in any order and may span multiple seeds; the
/// x-axis toggle lets readers choose episode-index, cumulative env-step,
/// or wall-clock time without page reload.  Multi-seed runs also get a
/// mean±std band section rendered by `multi_seed_section`.
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
    let window = EPISODE_WINDOW.min(episode_count.max(1)).max(1);

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
        // `episode_return` / `episode_length` are already drawn by the
        // dedicated episode-outcome panels above (with the step/episode/
        // wallclock x-axis toggle). Skip them here so we don't render a
        // second, redundant copy on the native step axis.
        if matches!(name.as_str(), "episode_return" | "episode_length") {
            continue;
        }
        let full = metric_series(records, &name);
        if full.is_empty() {
            continue;
        }
        // Decimate long series so the SVG path stays light; min/max bucketing
        // keeps peaks. The full series feeds the hover crosshair so the raw
        // value under the cursor is exact even when the path is decimated.
        let decimated = downsample_minmax(&full);
        let mut raw_full: Vec<(f64, f64)> = full.iter().map(|&(x, y)| (f64::from(x), y)).collect();
        // `interactive_line_view`'s hover lookup (nearest_by_x) requires x-sorted
        // input. Step counters are monotone in practice, but sort here so the
        // contract holds regardless of emission order.
        raw_full.sort_by(|a, b| a.0.total_cmp(&b.0));
        let dec_xy: Vec<(f64, f64)> = decimated.iter().map(|&(x, y)| (f64::from(x), y)).collect();
        let title = title_for(&name).to_string();
        let y_title = unit_for(&name);
        // Per-generation EA metrics run over the generation axis; per-update RL
        // metrics over the training-step axis. Both are discrete integers.
        let x_title = if is_per_generation(&name) {
            "generation"
        } else {
            "step"
        };
        let panel = if is_per_generation(&name) {
            interactive_line_view(title, x_title, &y_title, true, raw_full, dec_xy, None)
        } else {
            let smoothed: Vec<(f64, f64)> = rolling_mean(&decimated, METRIC_WINDOW)
                .iter()
                .map(|&(x, y)| (f64::from(x), y))
                .collect();
            interactive_line_view(
                title,
                x_title,
                &y_title,
                true,
                raw_full,
                dec_xy,
                Some(smoothed),
            )
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
        // Episode / env-step axes are discrete integers; wall-clock is continuous.
        let x_as_int = mode.get() != AxisMode::Wallclock;
        let reward_xy = remap_episode_series(&reward, axis);
        let reward_sm = remap_episode_series(&reward_smoothed, axis);
        let length_xy = remap_episode_series(&length, axis);
        let length_sm = remap_episode_series(&length_smoothed, axis);
        view! {
            <div class="rlevo-chart-grid">
                {interactive_line_view(
                    format!("Episode reward (x: {x_label})"),
                    &x_label,
                    "reward",
                    x_as_int,
                    reward_xy.clone(),
                    reward_xy,
                    Some(reward_sm),
                )}
                {interactive_line_view(
                    format!("Episode length (x: {x_label})"),
                    &x_label,
                    "frames",
                    x_as_int,
                    length_xy.clone(),
                    length_xy,
                    Some(length_sm),
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
        panels.push(band_chart_view(
            title_for(&name).to_string(),
            &unit_for(&name),
            &band,
        ));
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
pub fn band_chart_view(title: String, y_title: &str, band: &[BandPoint]) -> AnyView {
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

    let plot_w = PLOT_RIGHT - BOX_M_L;
    let plot_h = PLOT_BOTTOM - BOX_M_T;
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
    let axes = axis_layer(x_min, x_max, y_min, y_max, "step", y_title, true);

    view! {
        <figure class="rlevo-chart-card">
            <figcaption>{title}</figcaption>
            {export_button("rlevo-band.svg")}
            <svg class="rlevo-band" viewBox={view_box} preserveAspectRatio="none" role="img">
                {axes}
                <polygon class="rlevo-band-fill" points={poly} />
                <polyline class="rlevo-band-mean" points={mean_pts} fill="none" />
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

/// SVG viewBox width shared by every hand-rolled panel, in user units.
const BOX_VB_W: f64 = 640.0;
/// SVG viewBox height shared by every hand-rolled panel, in user units.
const BOX_VB_H: f64 = 300.0;
/// Left margin reserved for the rotated y-axis title + y tick labels.
const BOX_M_L: f64 = 64.0;
/// Right margin between the plot area and the viewBox edge.
const BOX_M_R: f64 = 18.0;
/// Top margin above the plot area.
const BOX_M_T: f64 = 20.0;
/// Bottom margin reserved for the x tick labels + x-axis title.
const BOX_M_B: f64 = 46.0;

/// Right edge of the plot area (pixels in the shared viewBox).
const PLOT_RIGHT: f64 = BOX_VB_W - BOX_M_R;
/// Bottom edge of the plot area — the y-pixel of the x-axis line.
const PLOT_BOTTOM: f64 = BOX_VB_H - BOX_M_B;

/// "Nice" axis ticks: at most `target`-ish round values spanning `[min, max]`.
///
/// Picks a 1/2/5×10ⁿ step so labels land on human-readable numbers (0, 100,
/// 200 … rather than 0, 137, 274 …). Degenerate ranges collapse to a single
/// tick so callers can still render an axis without dividing by zero.
fn nice_ticks(min: f64, max: f64, target: usize) -> Vec<f64> {
    let target = target.max(2);
    if !(min.is_finite() && max.is_finite()) || (max - min).abs() < f64::EPSILON {
        return vec![min];
    }
    let raw_step = (max - min) / target as f64;
    let mag = 10f64.powf(raw_step.abs().log10().floor());
    let norm = raw_step / mag;
    let nice = if norm < 1.5 {
        1.0
    } else if norm < 3.0 {
        2.0
    } else if norm < 7.0 {
        5.0
    } else {
        10.0
    };
    let step = nice * mag;
    let mut ticks = Vec::new();
    let mut t = (min / step).ceil() * step;
    while t <= max + step * 0.5 && ticks.len() <= target + 2 {
        if t >= min - step * 0.5 {
            ticks.push(t);
        }
        t += step;
    }
    if ticks.is_empty() {
        ticks.push(min);
    }
    ticks
}

/// Formats a tick value for axis labels.
///
/// When `as_int` is `true` the value is rounded to the nearest integer, which
/// keeps episode/generation/step axes clean (e.g. `100` instead of `100.00`).
/// When `false`, adaptive decimal precision is used: magnitude ≥10 → no
/// decimals, ≥1 → two, ≥0.01 → three, smaller → scientific notation.  This
/// keeps both `500` and `0.012` legible without trailing-zero noise.
fn fmt_tick(v: f64, as_int: bool) -> String {
    if as_int {
        return format!("{:.0}", v.round());
    }
    let a = v.abs();
    if a < f64::EPSILON {
        "0".to_string()
    } else if a >= 10.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else if a >= 0.01 {
        format!("{v:.3}")
    } else {
        format!("{v:.1e}")
    }
}

/// y-axis title for a metric panel: its registry unit (e.g. `steps`, `s`), or
/// an empty string when the metric is unitless.
fn unit_for(name: &str) -> String {
    descriptor(name)
        .and_then(|d| d.unit)
        .unwrap_or("")
        .to_string()
}

/// Shared axis furniture for every hand-rolled SVG panel: the x/y axis lines,
/// "nice" tick marks with numeric labels, and rotated axis titles.
///
/// All panels share the `BOX_*` layout constants, so this function derives its
/// own data→pixel maps from the supplied ranges.  Each calling panel also
/// holds its own scale closures (built from the same constants and ranges) so
/// that data points and axis labels are always in register.  `x_as_int` formats
/// the x ticks as whole numbers (episode / generation / step axes); pass
/// `false` for continuous axes (wall-clock).
fn axis_layer(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x_title: &str,
    y_title: &str,
    x_as_int: bool,
) -> AnyView {
    let plot_w = PLOT_RIGHT - BOX_M_L;
    let plot_h = PLOT_BOTTOM - BOX_M_T;
    let x_span = if (x_max - x_min).abs() < f64::EPSILON {
        1.0
    } else {
        x_max - x_min
    };
    let y_span = if (y_max - y_min).abs() < f64::EPSILON {
        1.0
    } else {
        y_max - y_min
    };
    let sx = move |x: f64| BOX_M_L + (x - x_min) / x_span * plot_w;
    let sy = move |y: f64| BOX_M_T + (1.0 - (y - y_min) / y_span) * plot_h;

    let x_ticks: Vec<AnyView> = nice_ticks(x_min, x_max, 6)
        .into_iter()
        .filter(|&t| t >= x_min - f64::EPSILON && t <= x_max + f64::EPSILON)
        .map(|t| {
            let px = sx(t);
            let label = fmt_tick(t, x_as_int);
            view! {
                <line class="rlevo-axis-tick" x1={px} y1={PLOT_BOTTOM} x2={px} y2={PLOT_BOTTOM + 4.0} />
                <text class="rlevo-axis-num" x={px} y={PLOT_BOTTOM + 15.0}
                    text-anchor="middle">{label}</text>
            }
            .into_any()
        })
        .collect();

    let y_ticks: Vec<AnyView> = nice_ticks(y_min, y_max, 4)
        .into_iter()
        .filter(|&t| t >= y_min - f64::EPSILON && t <= y_max + f64::EPSILON)
        .map(|t| {
            let py = sy(t);
            let label = fmt_tick(t, false);
            view! {
                <line class="rlevo-axis-tick" x1={BOX_M_L - 4.0} y1={py} x2={BOX_M_L} y2={py} />
                <text class="rlevo-axis-num" x={BOX_M_L - 7.0} y={py + 3.0}
                    text-anchor="end">{label}</text>
            }
            .into_any()
        })
        .collect();

    let cx = f64::midpoint(BOX_M_L, PLOT_RIGHT);
    let cy = f64::midpoint(BOX_M_T, PLOT_BOTTOM);
    let x_title = x_title.to_string();
    let y_title = y_title.to_string();
    let y_title_view = (!y_title.is_empty()).then(|| {
        view! {
            <text class="rlevo-axis-title" x=12.0 y={cy} text-anchor="middle"
                transform={format!("rotate(-90 12 {cy})")}>{y_title}</text>
        }
        .into_any()
    });
    let x_title_view = (!x_title.is_empty()).then(|| {
        view! {
            <text class="rlevo-axis-title" x={cx} y={BOX_VB_H - 4.0}
                text-anchor="middle">{x_title}</text>
        }
        .into_any()
    });

    view! {
        <g class="rlevo-axes">
            <line class="rlevo-axis-line" x1={BOX_M_L} y1={BOX_M_T} x2={BOX_M_L} y2={PLOT_BOTTOM} />
            <line class="rlevo-axis-line" x1={BOX_M_L} y1={PLOT_BOTTOM} x2={PLOT_RIGHT} y2={PLOT_BOTTOM} />
            {x_ticks}
            {y_ticks}
            {y_title_view}
            {x_title_view}
        </g>
    }
    .into_any()
}

/// Returns a small "⤓ SVG" toolbar button that downloads the panel's SVG.
///
/// The click handler is DOM-generic: it walks up to `.rlevo-chart-card` and
/// serializes the contained `<svg>`, so the same button works for every panel
/// type — hand-rolled or `leptos-chartistry`.
fn export_button(filename: &'static str) -> AnyView {
    let on_click = move |ev: leptos::ev::MouseEvent| export_panel_svg(&ev, filename);
    view! {
        <button type="button" class="rlevo-export-btn"
            title="Download this panel as SVG" on:click=on_click>
            "⤓ SVG"
        </button>
    }
    .into_any()
}

/// Serializes the clicked panel's `<svg>` and triggers a download. Silently
/// no-ops if any DOM step is unavailable (e.g. headless contexts).
fn export_panel_svg(ev: &leptos::ev::MouseEvent, filename: &str) {
    use wasm_bindgen::JsCast as _;
    let Some(target) = ev.current_target() else {
        return;
    };
    let Ok(btn) = target.dyn_into::<web_sys::Element>() else {
        return;
    };
    let Ok(Some(card)) = btn.closest(".rlevo-chart-card") else {
        return;
    };
    let Ok(Some(svg)) = card.query_selector("svg") else {
        return;
    };
    let markup = ensure_svg_header(&svg.outer_html());
    let encoded: String = js_sys::encode_uri_component(&markup).into();
    let href = format!("data:image/svg+xml;charset=utf-8,{encoded}");
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Ok(anchor) = doc.create_element("a") else {
        return;
    };
    let Ok(anchor) = anchor.dyn_into::<web_sys::HtmlAnchorElement>() else {
        return;
    };
    anchor.set_href(&href);
    anchor.set_download(filename);
    anchor.click();
}

/// Deterministic jitter in `[-1, 1)` from an index — a cheap integer hash so
/// the strip-plot scatter is stable across renders (no RNG; `Math.random` is
/// unavailable and determinism keeps screenshots reproducible).
///
/// The multiplier `0x9E3779B97F4A7C15` is the 64-bit Fibonacci / golden-ratio
/// mixing constant (Knuth multiplicative hashing).  It is the standard choice
/// for turning a sequential index into a well-spread bit pattern; changing it
/// would alter all jitter positions and break screenshot stability.
fn jitter_unit(i: u64) -> f64 {
    let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // Top 53 bits → [0, 1), then map to [-1, 1).
    let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
    unit.mul_add(2.0, -1.0)
}

/// Renders the hand-rolled per-generation fitness box plot SVG panel.
///
/// Inside each generation slot: a filled rect for `[Q1, Q3]`, a horizontal
/// median tick, vertical whiskers clipped at the Tukey 1.5×IQR fence, and
/// outliers as small open circles. Three overlay polylines (best, median,
/// worst) pair colour with distinct dash patterns so the a11y contract
/// survives a B/W screenshot. An optional strip-plot scatter is toggled by an
/// "Individual points" button in the toolbar.
///
/// `overlays` is the `(best, median, worst)` triple returned by
/// [`crate::series::fitness_range_series`].
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
    let scale_x = move |g: f64| -> f64 { BOX_M_L + (g - x_min) / (x_max - x_min) * plot_w };
    let scale_y = move |v: f64| -> f64 { BOX_M_T + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h };

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
            .map(|&(cx, cy)| {
                view! { <circle class="rlevo-strip-dot" cx={cx} cy={cy} r=1.5 /> }.into_any()
            })
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
    let axes = axis_layer(x_min, x_max, y_min, y_max, "generation", "fitness", true);

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
                {export_button("rlevo-fitness-boxplot.svg")}
            </div>
            <svg class="rlevo-svg-frame rlevo-boxplot-svg"
                viewBox=view_box
                role="img"
                aria-label="per-generation fitness box plot">
                {axes}
                // strip-plot jitter cloud beneath the boxes (toggle)
                {strip_view}
                // boxes + whiskers + medians + outliers
                {box_elems}
                // overlay reference lines (best / median trace / worst)
                <polyline class="rlevo-boxplot-best" points={best_pts} />
                <polyline class="rlevo-boxplot-median-trace" points={median_pts} />
                <polyline class="rlevo-boxplot-worst" points={worst_pts} />
            </svg>
            <figcaption class="rlevo-chart-y">
                "lines: best (solid) · median (dashed) · worst (dotted) — lower is better"
            </figcaption>
        </figure>
    }
    .into_any()
}

/// Diversity trace panel with a configurable low-diversity guideline.
///
/// A dashed horizontal guideline marks the low-diversity threshold (default:
/// 5th percentile of the first ten generations, [`low_diversity_threshold`]);
/// the user can override it via a number input. When the trace dips below the
/// guideline the card border pulses and the title gains a ⚠ glyph — both
/// colour-redundant so the alert survives a B/W screenshot.
#[must_use]
pub fn diversity_panel_view(diversity: &[(u32, f64)]) -> AnyView {
    use std::fmt::Write as _;
    if diversity.is_empty() {
        return ().into_any();
    }

    let x_min = f64::from(diversity.first().map_or(0, |p| p.0));
    let x_max_raw = f64::from(diversity.last().map_or(1, |p| p.0));
    let x_max = if (x_max_raw - x_min).abs() < f64::EPSILON {
        x_min + 1.0
    } else {
        x_max_raw
    };
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(_, y) in diversity {
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }
    if !y_min.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
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

    let plot_w = PLOT_RIGHT - BOX_M_L;
    let plot_h = PLOT_BOTTOM - BOX_M_T;
    let sx_of = move |x: f64| BOX_M_L + (x - x_min) / (x_max - x_min) * plot_w;
    let sy_of = move |y: f64| BOX_M_T + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h;

    let mut path = String::new();
    for &(x, y) in diversity {
        if y.is_finite() {
            let _ = write!(path, "{:.2},{:.2} ", sx_of(f64::from(x)), sy_of(y));
        }
    }

    let default_t = low_diversity_threshold(diversity).unwrap_or(y_min);
    let values: Vec<f64> = diversity.iter().map(|&(_, y)| y).collect();
    let (threshold, set_threshold) = signal(default_t);

    // Memo so both the title and the card-pulse class can read the breach state
    // (a plain `Fn` closure would be moved by the first consumer).
    let breached = Memo::new(move |_| {
        let t = threshold.get();
        values.iter().any(|&y| y.is_finite() && y < t)
    });
    let guide_y = move || sy_of(threshold.get().clamp(y_min, y_max));
    let view_box = format!("0 0 {BOX_VB_W} {BOX_VB_H}");
    let right_x = PLOT_RIGHT;
    let axes = axis_layer(x_min, x_max, y_min, y_max, "generation", "diversity", true);
    let title = move || {
        if breached.get() {
            "⚠ Diversity"
        } else {
            "Diversity"
        }
    };
    let on_threshold = move |ev: leptos::ev::Event| {
        if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
            set_threshold.set(v);
        }
    };

    view! {
        <figure class="rlevo-chart-card" class:rlevo-pulse=move || breached.get()>
            <figcaption>{title}</figcaption>
            <div class="rlevo-diversity-toolbar">
                <label>"low-diversity threshold: "
                    <input type="number" step="any" class="rlevo-threshold-input"
                        prop:value=move || threshold.get().to_string()
                        on:input=on_threshold />
                </label>
                {export_button("rlevo-diversity.svg")}
            </div>
            <svg class="rlevo-line" viewBox={view_box} preserveAspectRatio="none" role="img"
                aria-label="population diversity over generations">
                {axes}
                <polyline class="rlevo-line-smoothed" points={path} fill="none" />
                <line class="rlevo-diversity-guide"
                    x1={BOX_M_L} y1=guide_y x2={right_x} y2=guide_y />
            </svg>
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
///
/// `sense` is the run's declared objective direction (the report passes the
/// manifest's `objective_sense`, treating `None` as
/// [`ObjectiveSense::Maximize`]). It orients the best/worst overlay traces and
/// the selection-pressure ratio.
#[must_use]
pub fn population_panel_view(samples: &[PopulationSample], sense: ObjectiveSense) -> AnyView {
    if samples.is_empty() {
        return view! { <span></span> }.into_any();
    }

    let box_stats = population_box_data(samples);
    let overlays = fitness_range_series(samples, sense);
    let diversity = diversity_series(samples);
    let pressure = selection_pressure_series(samples, sense);

    let mut panels: Vec<AnyView> = Vec::new();
    panels.push(population_box_view(&box_stats, overlays));
    if !diversity.is_empty() {
        panels.push(diversity_panel_view(&diversity));
    }
    if !pressure.is_empty() {
        let pressure_xy: Vec<(f64, f64)> =
            pressure.iter().map(|&(x, y)| (f64::from(x), y)).collect();
        panels.push(interactive_line_view(
            "Selection pressure (best / median)".to_string(),
            "generation",
            "ratio",
            true,
            pressure_xy.clone(),
            pressure_xy,
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
