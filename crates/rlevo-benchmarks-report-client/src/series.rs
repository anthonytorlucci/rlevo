//! Pure-Rust series-extraction helpers for the RL convergence panel.
//!
//! These transform decoded [`EpisodeRecord`]s into `(x, y)` arrays that
//! the [`crate::charts`] module feeds into `leptos-chartistry`. No DOM,
//! no Leptos — every function here is testable as a native unit test.

//! The canonical metric table is the shared [`rlevo_metrics_registry`] crate
//! (ADR-0015) — there is no client-side copy to keep in sync.

use rlevo_metrics_registry::CANONICAL_METRICS;

use crate::wire::{EpisodeRecord, PopulationSample};

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

/// Series length above which [`downsample_minmax`] decimates before charting.
pub const DOWNSAMPLE_THRESHOLD: usize = 10_000;
/// Target bucket count for [`downsample_minmax`]; the decimated series has at
/// most `2 * DOWNSAMPLE_BUCKETS` points (a min and a max per bucket).
pub const DOWNSAMPLE_BUCKETS: usize = 2_048;

/// Min/max-per-bucket decimation that preserves visual extremes.
///
/// Series at or below [`DOWNSAMPLE_THRESHOLD`] points are returned unchanged.
/// Longer series are split into `DOWNSAMPLE_BUCKETS` equal-width index buckets;
/// each bucket contributes its minimum-y and maximum-y points, emitted in x
/// order, so peaks and troughs survive decimation (a plain stride would alias
/// them away). Non-finite (`NaN`/`±Inf`) y-values are dropped. The first and
/// last finite points are always retained so the curve spans its full domain.
///
/// This is a *display* transform — keep the raw series for exact-value tooltips.
#[must_use]
pub fn downsample_minmax(points: &[(u32, f64)]) -> Vec<(u32, f64)> {
    if points.len() <= DOWNSAMPLE_THRESHOLD {
        return points.iter().copied().filter(|(_, y)| y.is_finite()).collect();
    }
    let n = points.len();
    let buckets = DOWNSAMPLE_BUCKETS;
    let mut out: Vec<(u32, f64)> = Vec::with_capacity(buckets * 2 + 2);
    for b in 0..buckets {
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1).min(n);
        let mut lo: Option<(u32, f64)> = None;
        let mut hi: Option<(u32, f64)> = None;
        for &(x, y) in &points[start..end] {
            if !y.is_finite() {
                continue;
            }
            if lo.is_none_or(|(_, ly)| y < ly) {
                lo = Some((x, y));
            }
            if hi.is_none_or(|(_, hy)| y > hy) {
                hi = Some((x, y));
            }
        }
        // Emit min then max in x order so the rendered path stays monotone in x.
        match (lo, hi) {
            (Some(a), Some(b)) if a.0 <= b.0 => {
                out.push(a);
                if a != b {
                    out.push(b);
                }
            }
            (Some(a), Some(b)) => {
                out.push(b);
                out.push(a);
            }
            _ => {}
        }
    }
    out
}

/// One aggregated point of a multi-seed metric band: the cross-seed mean and
/// (population) standard deviation at a given step, plus how many distinct
/// seeds contributed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandPoint {
    /// Training step / generation the aggregate is computed at.
    pub step: u32,
    /// Mean across the seeds present at this step.
    pub mean: f64,
    /// Population standard deviation across those seeds (`0.0` for one seed).
    pub std: f64,
    /// Number of distinct seeds that contributed a value at this step.
    pub n: usize,
}

/// Number of distinct run seeds (`EpisodeRecordHeader::seed`) in the record
/// set. `>= 2` means the report can draw a cross-seed mean±std band.
#[must_use]
pub fn distinct_seed_count(records: &[EpisodeRecord]) -> usize {
    let mut seeds: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for r in records {
        seeds.insert(r.header.seed);
    }
    seeds.len()
}

/// Cross-seed aggregation of one metric into a mean±std band.
///
/// Records are grouped by `header.seed`; within a seed, repeated samples at the
/// same step are averaged. At each step the per-seed values are reduced to a
/// cross-seed mean and population standard deviation (`std = 0` when a single
/// seed has data there). Steps are returned in ascending order. Non-finite
/// values are ignored. Returns empty when no seed has a finite sample.
#[must_use]
pub fn metric_band(records: &[EpisodeRecord], name: &str) -> Vec<BandPoint> {
    use std::collections::BTreeMap;
    // step -> seed -> (sum, count) for averaging within a seed.
    let mut by_step: BTreeMap<u32, BTreeMap<u64, (f64, u32)>> = BTreeMap::new();
    for r in records {
        let seed = r.header.seed;
        for m in r.metrics.iter().filter(|m| m.name == name) {
            if !m.value.is_finite() {
                continue;
            }
            let entry = by_step.entry(m.step).or_default().entry(seed).or_insert((0.0, 0));
            entry.0 += m.value;
            entry.1 += 1;
        }
    }
    by_step
        .into_iter()
        .map(|(step, per_seed)| {
            let means: Vec<f64> = per_seed
                .values()
                .map(|&(sum, count)| sum / f64::from(count))
                .collect();
            let n = means.len();
            let mean = means.iter().sum::<f64>() / n as f64;
            let var = means.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            BandPoint {
                step,
                mean,
                std: var.sqrt(),
                n,
            }
        })
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
        if seen.remove(canonical.name) {
            out.push(canonical.name.to_string());
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

// ---------------------------------------------------------------------------
// EA population-panel series extraction.
// ---------------------------------------------------------------------------

/// Per-generation summary statistics for the Population box plot.
///
/// Quartiles use linear interpolation between order statistics
/// (Tukey-style positions: `(n - 1) * p`). Outliers follow Tukey's
/// 1.5×IQR rule: anything below `Q1 - 1.5·IQR` or above `Q3 + 1.5·IQR`
/// lands in `outliers` and the whiskers (`min` / `max`) clip to the
/// nearest sample inside the fence.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxStats {
    /// EA generation index; used as the x-axis position in the box plot.
    pub generation: u32,
    /// Lower whisker — minimum fitness value inside the Tukey 1.5×IQR fence.
    pub min: f64,
    /// First quartile (25th percentile), linearly interpolated.
    pub q1: f64,
    /// Median (50th percentile), linearly interpolated.
    pub median: f64,
    /// Third quartile (75th percentile), linearly interpolated.
    pub q3: f64,
    /// Upper whisker — maximum fitness value inside the Tukey 1.5×IQR fence.
    pub max: f64,
    /// Values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`; rendered as open circles.
    pub outliers: Vec<f64>,
}

/// Per-generation box-plot summary, one entry per population sample in
/// the input slice. Samples with empty `fitnesses` are skipped.
#[must_use]
pub fn population_box_data(samples: &[PopulationSample]) -> Vec<BoxStats> {
    samples
        .iter()
        .filter(|s| !s.fitnesses.is_empty())
        .map(|s| box_stats_for(s.generation, &s.fitnesses))
        .collect()
}

/// Computes [`BoxStats`] for one generation from a raw fitness slice.
///
/// Sorts values ascending, derives Q1/median/Q3 via [`quantile`], applies
/// Tukey's 1.5×IQR rule to split whisker values from outliers.  When every
/// value is an outlier (e.g. a constant series with IQR = 0), falls back to
/// the raw min/max so the box still renders.
fn box_stats_for(generation: u32, fitnesses: &[f32]) -> BoxStats {
    let mut sorted: Vec<f64> = fitnesses.iter().map(|f| f64::from(*f)).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = quantile(&sorted, 0.25);
    let median = quantile(&sorted, 0.5);
    let q3 = quantile(&sorted, 0.75);
    let iqr = q3 - q1;
    let lo_fence = q1 - 1.5 * iqr;
    let hi_fence = q3 + 1.5 * iqr;
    let mut outliers: Vec<f64> = Vec::new();
    let mut whisker_lo = f64::INFINITY;
    let mut whisker_hi = f64::NEG_INFINITY;
    for v in &sorted {
        if *v < lo_fence || *v > hi_fence {
            outliers.push(*v);
        } else {
            if *v < whisker_lo {
                whisker_lo = *v;
            }
            if *v > whisker_hi {
                whisker_hi = *v;
            }
        }
    }
    // Degenerate case: every value flagged as outlier (e.g. constant
    // series after IQR=0). Fall back to the raw min / max so the box
    // still renders something the eye can read.
    if !whisker_lo.is_finite() {
        whisker_lo = *sorted.first().unwrap_or(&0.0);
        whisker_hi = *sorted.last().unwrap_or(&0.0);
    }
    BoxStats {
        generation,
        min: whisker_lo,
        q1,
        median,
        q3,
        max: whisker_hi,
        outliers,
    }
}

/// Linear-interpolation quantile (`p` in `[0.0, 1.0]`). Input must be
/// non-empty and pre-sorted ascending.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty(), "quantile of empty slice");
    if sorted.len() == 1 {
        return sorted[0];
    }
    #[allow(clippy::cast_precision_loss)]
    let pos = p * (sorted.len() - 1) as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo = pos.floor() as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    #[allow(clippy::cast_precision_loss)]
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Per-generation diversity scalar, skipping `None` entries. Empty
/// result when no sample carries a diversity value.
#[must_use]
pub fn diversity_series(samples: &[PopulationSample]) -> Vec<(u32, f64)> {
    samples
        .iter()
        .filter_map(|s| s.diversity.map(|d| (s.generation, f64::from(d))))
        .collect()
}

/// Selection-pressure indicator: `best / median` per generation. Skips
/// generations whose median is zero (degenerate, all-zero fitness) and
/// generations with empty fitness vectors.
#[must_use]
pub fn selection_pressure_series(samples: &[PopulationSample]) -> Vec<(u32, f64)> {
    samples
        .iter()
        .filter_map(|s| {
            if s.fitnesses.is_empty() {
                return None;
            }
            let mut sorted: Vec<f64> = s.fitnesses.iter().map(|f| f64::from(*f)).collect();
            sorted
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = quantile(&sorted, 0.5);
            if median.abs() < f64::EPSILON {
                return None;
            }
            let best = sorted[0];
            Some((s.generation, best / median))
        })
        .collect()
}

/// `(best, median, worst)` overlay traces for the box-plot reference
/// lines. One tuple of three series, all sharing the same x-axis
/// (generation). Empty samples produce three empty vectors.
#[must_use]
pub fn fitness_range_series(
    samples: &[PopulationSample],
) -> (Vec<(u32, f64)>, Vec<(u32, f64)>, Vec<(u32, f64)>) {
    let mut best = Vec::new();
    let mut median = Vec::new();
    let mut worst = Vec::new();
    for s in samples {
        if s.fitnesses.is_empty() {
            continue;
        }
        let mut sorted: Vec<f64> = s.fitnesses.iter().map(|f| f64::from(*f)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        best.push((s.generation, sorted[0]));
        median.push((s.generation, quantile(&sorted, 0.5)));
        worst.push((s.generation, *sorted.last().unwrap_or(&0.0)));
    }
    (best, median, worst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{
        EnvFamily, EpisodeKind, EpisodeRecordHeader, FamilyPayload, FrameRecord, MetricSample,
        RunId, FORMAT_VERSION,
    };

    #[test]
    fn downsample_passthrough_below_threshold() {
        let pts: Vec<(u32, f64)> = (0..100).map(|i| (i, f64::from(i))).collect();
        assert_eq!(downsample_minmax(&pts), pts);
    }

    #[test]
    fn downsample_drops_non_finite_below_threshold() {
        let pts = vec![(0, 1.0), (1, f64::NAN), (2, f64::INFINITY), (3, 2.0)];
        let out = downsample_minmax(&pts);
        assert_eq!(out, vec![(0, 1.0), (3, 2.0)]);
        assert!(out.iter().all(|(_, y)| y.is_finite()));
    }

    #[test]
    fn downsample_preserves_global_extremes() {
        // A long ramp with a single planted spike and dip; decimation must keep both.
        let mut pts: Vec<(u32, f64)> = (0..50_000u32).map(|i| (i, f64::from(i % 7))).collect();
        pts[12_345].1 = 9_999.0; // global max
        pts[37_000].1 = -9_999.0; // global min
        let out = downsample_minmax(&pts);
        assert!(out.len() <= DOWNSAMPLE_BUCKETS * 2 + 2, "bounded point count");
        let max = out.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
        let min = out.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        assert!((max - 9_999.0).abs() < 1e-9, "global max survived: {max}");
        assert!((min + 9_999.0).abs() < 1e-9, "global min survived: {min}");
    }

    #[test]
    fn downsample_constant_series_stays_constant() {
        let pts: Vec<(u32, f64)> = (0..20_000u32).map(|i| (i, 3.0)).collect();
        let out = downsample_minmax(&pts);
        assert!(!out.is_empty());
        assert!(out.iter().all(|(_, y)| (*y - 3.0).abs() < 1e-12));
    }

    #[test]
    fn downsample_x_is_non_decreasing() {
        let pts: Vec<(u32, f64)> = (0..30_000u32)
            .map(|i| (i, ((f64::from(i) * 0.01).sin())))
            .collect();
        let out = downsample_minmax(&pts);
        assert!(out.windows(2).all(|w| w[0].0 <= w[1].0), "x monotone");
    }

    /// Builds a one-episode record under `seed` carrying the given
    /// `(step, value)` samples for metric `name`.
    fn seeded_metric_record(seed: u64, name: &str, samples: &[(u32, f64)]) -> EpisodeRecord {
        let header = EpisodeRecordHeader {
            format_version: FORMAT_VERSION,
            run_id: RunId("r".into()),
            seed,
            env_family: EnvFamily::Classic,
            created_at: 0,
            trial: None,
            kind: EpisodeKind::Training,
        };
        let metrics = samples
            .iter()
            .map(|&(step, value)| MetricSample {
                step,
                name: name.to_string(),
                value,
            })
            .collect();
        EpisodeRecord {
            header,
            frames: vec![],
            metrics,
            population_samples: vec![],
        }
    }

    #[test]
    fn distinct_seed_count_counts_unique_seeds() {
        let recs = vec![
            seeded_metric_record(1, "loss", &[(0, 1.0)]),
            seeded_metric_record(1, "loss", &[(1, 1.0)]),
            seeded_metric_record(2, "loss", &[(0, 1.0)]),
        ];
        assert_eq!(distinct_seed_count(&recs), 2);
    }

    #[test]
    fn metric_band_computes_cross_seed_mean_and_std() {
        // Three seeds at step 0 with values 1, 2, 3 → mean 2, popvar = 2/3.
        let recs = vec![
            seeded_metric_record(1, "loss", &[(0, 1.0)]),
            seeded_metric_record(2, "loss", &[(0, 2.0)]),
            seeded_metric_record(3, "loss", &[(0, 3.0)]),
        ];
        let band = metric_band(&recs, "loss");
        assert_eq!(band.len(), 1);
        let p = band[0];
        assert_eq!(p.step, 0);
        assert_eq!(p.n, 3);
        assert!((p.mean - 2.0).abs() < 1e-12);
        assert!((p.std - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn metric_band_single_seed_has_zero_std() {
        let recs = vec![seeded_metric_record(7, "loss", &[(0, 5.0), (1, 6.0)])];
        let band = metric_band(&recs, "loss");
        assert_eq!(band.len(), 2);
        assert!(band.iter().all(|p| p.n == 1 && p.std == 0.0));
    }

    #[test]
    fn metric_band_averages_within_seed_then_across() {
        // Seed 1 has two samples at step 0 (avg 2.0); seed 2 has one (4.0).
        // Cross-seed mean = 3.0.
        let recs = vec![
            seeded_metric_record(1, "loss", &[(0, 1.0), (0, 3.0)]),
            seeded_metric_record(2, "loss", &[(0, 4.0)]),
        ];
        let band = metric_band(&recs, "loss");
        assert_eq!(band.len(), 1);
        assert_eq!(band[0].n, 2);
        assert!((band[0].mean - 3.0).abs() < 1e-12);
    }

    #[test]
    fn metric_band_skips_non_finite() {
        let recs = vec![
            seeded_metric_record(1, "loss", &[(0, f64::NAN)]),
            seeded_metric_record(2, "loss", &[(0, 2.0)]),
        ];
        let band = metric_band(&recs, "loss");
        assert_eq!(band.len(), 1);
        assert_eq!(band[0].n, 1);
        assert!((band[0].mean - 2.0).abs() < 1e-12);
    }

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
                format_version: FORMAT_VERSION,
                run_id: RunId("x".into()),
                seed: 0,
                env_family: EnvFamily::Classic,
                created_at: 0,
                trial: None,
                kind: EpisodeKind::Training,
            },
            frames,
            metrics,
            population_samples: Vec::new(),
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

    // ---- M8.1: population-panel series extraction ---------------------

    fn pop_sample(
        generation: u32,
        fitnesses: Vec<f32>,
        diversity: Option<f32>,
    ) -> PopulationSample {
        PopulationSample {
            generation,
            fitnesses,
            diversity,
            best_index: 0,
            best_genome_digest: None,
            parents_of_best: Vec::new(),
            inner_rl_returns: None,
        }
    }

    #[test]
    fn quantile_linear_interpolates_on_known_series() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&sorted, 0.0), 1.0);
        assert_eq!(quantile(&sorted, 0.5), 3.0);
        assert_eq!(quantile(&sorted, 1.0), 5.0);
        // 0.25 → position 1.0 → exact element index 1 = 2.0.
        assert_eq!(quantile(&sorted, 0.25), 2.0);
        // 0.75 → position 3.0 → exact element index 3 = 4.0.
        assert_eq!(quantile(&sorted, 0.75), 4.0);
    }

    #[test]
    fn population_box_data_no_outliers_on_uniform_series() {
        let samples = vec![pop_sample(7, vec![1.0, 2.0, 3.0, 4.0, 5.0], None)];
        let boxes = population_box_data(&samples);
        assert_eq!(boxes.len(), 1);
        let b = &boxes[0];
        assert_eq!(b.generation, 7);
        assert_eq!(b.q1, 2.0);
        assert_eq!(b.median, 3.0);
        assert_eq!(b.q3, 4.0);
        assert_eq!(b.min, 1.0);
        assert_eq!(b.max, 5.0);
        assert!(b.outliers.is_empty());
    }

    #[test]
    fn population_box_data_flags_iqr_outliers() {
        // 7-element series with one extreme value at the high end. The
        // bulk [1..7] gives IQR=3, hi_fence = 5.5 + 1.5*3 = 10.0. The
        // 50.0 outlier overshoots; the whisker clamps to the in-fence
        // maximum.
        let samples = vec![pop_sample(0, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0], None)];
        let boxes = population_box_data(&samples);
        let b = &boxes[0];
        assert_eq!(b.outliers.len(), 1);
        assert_eq!(b.outliers[0], 50.0);
        // Whisker high clips to 6.0 (max inside fence), not 50.0.
        assert_eq!(b.max, 6.0);
    }

    #[test]
    fn population_box_data_skips_empty_fitness_vector() {
        let samples = vec![
            pop_sample(0, vec![], None),
            pop_sample(1, vec![1.0, 2.0, 3.0], None),
        ];
        let boxes = population_box_data(&samples);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].generation, 1);
    }

    #[test]
    fn diversity_series_skips_none_and_carries_generation_x_axis() {
        let samples = vec![
            pop_sample(0, vec![1.0], Some(0.5)),
            pop_sample(1, vec![1.0], None),
            pop_sample(2, vec![1.0], Some(0.25)),
        ];
        let out = diversity_series(&samples);
        assert_eq!(out, vec![(0, 0.5), (2, 0.25)]);
    }

    #[test]
    fn diversity_series_empty_when_no_sample_provides_diversity() {
        let samples = vec![pop_sample(0, vec![1.0, 2.0], None)];
        assert!(diversity_series(&samples).is_empty());
    }

    #[test]
    fn selection_pressure_skips_zero_median() {
        let samples = vec![
            pop_sample(0, vec![1.0, 2.0, 3.0], None),
            pop_sample(1, vec![0.0, 0.0, 0.0], None),
        ];
        let out = selection_pressure_series(&samples);
        // gen 0: best=1, median=2 → 0.5; gen 1 skipped (median=0).
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[0].1, 0.5);
    }

    #[test]
    fn fitness_range_returns_three_aligned_series() {
        let samples = vec![
            pop_sample(0, vec![1.0, 2.0, 3.0, 4.0, 5.0], None),
            pop_sample(1, vec![0.5, 1.5, 2.5, 3.5, 4.5], None),
        ];
        let (best, median, worst) = fitness_range_series(&samples);
        assert_eq!(best, vec![(0, 1.0), (1, 0.5)]);
        assert_eq!(median, vec![(0, 3.0), (1, 2.5)]);
        assert_eq!(worst, vec![(0, 5.0), (1, 4.5)]);
    }

    #[test]
    fn fitness_range_empty_input_yields_three_empty_vectors() {
        let (best, median, worst) = fitness_range_series(&[]);
        assert!(best.is_empty());
        assert!(median.is_empty());
        assert!(worst.is_empty());
    }
}
