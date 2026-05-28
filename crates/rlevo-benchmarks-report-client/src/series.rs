//! Pure-Rust series-extraction helpers for the M8 convergence panel.
//!
//! These transform decoded [`EpisodeRecord`]s into `(x, y)` arrays that
//! the [`crate::charts`] module feeds into `leptos-chartistry`. No DOM,
//! no Leptos — every function here is testable as a native unit test.

use crate::wire::EpisodeRecord;

/// Canonical metric names, ordered for a stable panel layout across
/// runs. Mirrors `rlevo_benchmarks::record::tracing_layer::CANONICAL_METRICS`
/// — drift is caught by the cross-crate wire-format compat test on the
/// host side, and any name not in this list still surfaces via
/// [`available_metric_names`] (just at the end).
pub const CANONICAL_METRICS: &[&str] = &[
    "policy_loss",
    "value_loss",
    "loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "best_fitness",
    "mean_fitness",
    "worst_fitness",
    "best_fitness_ever",
];

/// Per-episode total reward — sum of every frame's reward, indexed by
/// episode position in the input slice.
#[must_use]
pub fn episode_reward_series(records: &[EpisodeRecord]) -> Vec<(u32, f64)> {
    records
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let total: f64 = r.frames.iter().map(|f| f64::from(f.reward)).sum();
            (u32::try_from(i).unwrap_or(u32::MAX), total)
        })
        .collect()
}

/// Per-episode frame count.
#[must_use]
pub fn episode_length_series(records: &[EpisodeRecord]) -> Vec<(u32, f64)> {
    records
        .iter()
        .enumerate()
        .map(|(i, r)| {
            #[allow(clippy::cast_precision_loss)]
            let n = r.frames.len() as f64;
            (u32::try_from(i).unwrap_or(u32::MAX), n)
        })
        .collect()
}

/// Every `MetricSample` matching `name`, in record-then-emission order.
/// x = `MetricSample::step`, y = `MetricSample::value`.
#[must_use]
pub fn metric_series(records: &[EpisodeRecord], name: &str) -> Vec<(u32, f64)> {
    records
        .iter()
        .flat_map(|r| r.metrics.iter())
        .filter(|m| m.name == name)
        .map(|m| (m.step, m.value))
        .collect()
}

/// Distinct metric names present anywhere in the run, ordered by
/// [`CANONICAL_METRICS`] first (so the panel grid is stable across
/// runs), then any non-canonical names appended in lexical order.
#[must_use]
pub fn available_metric_names(records: &[EpisodeRecord]) -> Vec<String> {
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for r in records {
        for m in &r.metrics {
            seen.insert(m.name.as_str());
        }
    }
    let mut out: Vec<String> = Vec::new();
    for canonical in CANONICAL_METRICS {
        if seen.remove(*canonical) {
            out.push((*canonical).to_string());
        }
    }
    for leftover in seen {
        out.push(leftover.to_string());
    }
    out
}

/// Trailing rolling mean with window `w`. Output length equals the input;
/// the first `w-1` points use a shrinking window so the curve starts
/// from the first sample instead of jumping in at index `w`. Empty
/// input returns empty output; `w == 0` is treated as `w == 1`.
#[must_use]
pub fn rolling_mean(samples: &[(u32, f64)], window: usize) -> Vec<(u32, f64)> {
    if samples.is_empty() {
        return Vec::new();
    }
    let w = window.max(1);
    let mut out = Vec::with_capacity(samples.len());
    let mut sum = 0.0_f64;
    for (i, &(x, y)) in samples.iter().enumerate() {
        sum += y;
        if i >= w {
            sum -= samples[i - w].1;
        }
        #[allow(clippy::cast_precision_loss)]
        let count = (i + 1).min(w) as f64;
        out.push((x, sum / count));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{
        EnvFamily, EpisodeRecordHeader, FamilyPayload, FrameRecord, MetricSample, RunId,
    };

    fn frame(step: u32, reward: f32) -> FrameRecord {
        FrameRecord {
            step,
            action: vec![],
            reward,
            ascii: None,
            styled: None,
            family_payload: FamilyPayload::Ascii,
        }
    }

    fn metric(step: u32, name: &str, value: f64) -> MetricSample {
        MetricSample {
            step,
            name: name.into(),
            value,
        }
    }

    fn record(frames: Vec<FrameRecord>, metrics: Vec<MetricSample>) -> EpisodeRecord {
        EpisodeRecord {
            header: EpisodeRecordHeader {
                format_version: 2,
                run_id: RunId("x".into()),
                seed: 0,
                env_family: EnvFamily::Classic,
                created_at: 0,
            },
            frames,
            metrics,
        }
    }

    #[test]
    fn episode_reward_sums_each_record() {
        let recs = vec![
            record(vec![frame(0, 1.0), frame(1, 2.0), frame(2, -0.5)], vec![]),
            record(vec![frame(0, 0.25)], vec![]),
        ];
        let series = episode_reward_series(&recs);
        assert_eq!(series, vec![(0, 2.5), (1, 0.25)]);
    }

    #[test]
    fn episode_length_counts_frames() {
        let recs = vec![
            record(vec![frame(0, 0.0); 3], vec![]),
            record(vec![frame(0, 0.0); 7], vec![]),
        ];
        assert_eq!(episode_length_series(&recs), vec![(0, 3.0), (1, 7.0)]);
    }

    #[test]
    fn metric_series_filters_and_preserves_order() {
        let recs = vec![
            record(
                vec![],
                vec![
                    metric(0, "policy_loss", 0.5),
                    metric(0, "entropy", 1.2),
                    metric(1, "policy_loss", 0.4),
                ],
            ),
            record(
                vec![],
                vec![
                    metric(2, "policy_loss", 0.3),
                    metric(2, "value_loss", 0.7),
                ],
            ),
        ];
        let pl = metric_series(&recs, "policy_loss");
        assert_eq!(pl, vec![(0, 0.5), (1, 0.4), (2, 0.3)]);
        let unknown = metric_series(&recs, "no_such_metric");
        assert!(unknown.is_empty());
    }

    #[test]
    fn available_metric_names_canonical_then_lexical() {
        let recs = vec![record(
            vec![],
            vec![
                metric(0, "entropy", 0.0),
                metric(0, "policy_loss", 0.0),
                metric(0, "zzz_custom", 0.0),
                metric(0, "aaa_custom", 0.0),
                metric(0, "value_loss", 0.0),
            ],
        )];
        let names = available_metric_names(&recs);
        assert_eq!(
            names,
            vec![
                "policy_loss".to_string(),
                "value_loss".to_string(),
                "entropy".to_string(),
                "aaa_custom".to_string(),
                "zzz_custom".to_string(),
            ],
            "canonical names appear first in CANONICAL_METRICS order; remainder lexical"
        );
    }

    #[test]
    fn available_metric_names_empty_when_no_records() {
        assert!(available_metric_names(&[]).is_empty());
    }

    #[test]
    fn rolling_mean_constant_series_is_constant() {
        let s: Vec<(u32, f64)> = (0..10).map(|i| (i, 5.0)).collect();
        let out = rolling_mean(&s, 3);
        assert_eq!(out.len(), s.len());
        for (_, y) in out {
            assert!((y - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn rolling_mean_handles_short_series() {
        let s = vec![(0_u32, 1.0), (1, 2.0)];
        let out = rolling_mean(&s, 50);
        assert_eq!(out, vec![(0, 1.0), (1, 1.5)]);
    }

    #[test]
    fn rolling_mean_window_three() {
        let s: Vec<(u32, f64)> = vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)];
        let out = rolling_mean(&s, 3);
        // shrinking window for first two: 1, (1+2)/2=1.5, (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        assert_eq!(out, vec![(0, 1.0), (1, 1.5), (2, 2.0), (3, 3.0), (4, 4.0)]);
    }

    #[test]
    fn rolling_mean_empty_input() {
        assert!(rolling_mean(&[], 5).is_empty());
    }

    #[test]
    fn rolling_mean_window_zero_treated_as_one() {
        let s = vec![(0_u32, 1.0), (1, 2.0), (2, 3.0)];
        let out = rolling_mean(&s, 0);
        assert_eq!(out, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
    }
}
