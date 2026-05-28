//! Convergence-plot panels built on `leptos-chartistry`.
//!
//! Pure-Rust SVG line charts per umbrella spec §3 constraint #6 (no JS
//! interop). Each panel pairs a thin raw line with a thicker smoothed
//! line; the width difference is the hue-redundant signal so a B/W
//! screenshot still distinguishes the two per the project a11y contract.

use leptos::prelude::*;
use leptos_chartistry::{AspectRatio, Chart, Line, Series};

use crate::series::{
    available_metric_names, episode_length_series, episode_reward_series, metric_series,
    rolling_mean,
};
use crate::wire::{EnvFamily, EpisodeRecord};

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
