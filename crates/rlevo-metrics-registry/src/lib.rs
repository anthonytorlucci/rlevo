//! Single source of truth for the canonical training-metric registry.
//!
//! Both the live-TUI capture layer and the on-disk recording layer in
//! `rlevo-benchmarks`, *and* the WASM report client
//! (`rlevo-benchmarks-report-client`), classify metric field names against the
//! same table. Before ADR-0015 this list was a flat `&[&str]` that was
//! hand-copied into the report client with no compile-time or test-time guard,
//! and the client re-derived metric *semantics* (RL-vs-EO grouping, per-update
//! vs per-generation cadence, display titles) in three disconnected hardcoded
//! places. This crate replaces all of that with one typed
//! [`MetricDescriptor`] table.
//!
//! # Why a leaf crate
//!
//! The report client compiles to `wasm32` and cannot depend on
//! `rlevo-benchmarks` (which transitively pulls in `burn` → `rand` →
//! `getrandom`). This crate is `#![no_std]` with no dependencies, so it builds
//! everywhere and lets both consumers share one definition instead of mirroring
//! it. See ADR-0015.
//!
//! # Extending the registry
//!
//! Add one [`MetricDescriptor`] row to [`CANONICAL_METRICS`] and emit a
//! matching `tracing::info!` field from the algorithm. The single edit makes
//! the field visible in the live TUI sparklines, the on-disk recording stream,
//! and the report's RL/EO panel grouping — no client-side change required.

#![no_std]

/// Which optimisation paradigm a metric belongs to.
///
/// Drives the report client's panel grouping: RL diagnostics and EO
/// diagnostics render in separate sections, and [`MetricKind::Shared`] metrics
/// (per-episode outcomes emitted regardless of paradigm) render in the common
/// section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricKind {
    /// Reinforcement-learning diagnostic (losses, KL, entropy, value health…).
    Rl,
    /// Evolutionary-optimisation diagnostic (population fitness statistics).
    Eo,
    /// Emitted by both paradigms (per-episode terminal outcomes).
    Shared,
}

/// How often a metric is sampled, which decides whether the report applies a
/// rolling-mean overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cadence {
    /// Once per optimiser update / training iteration (RL). Smoothed.
    PerUpdate,
    /// Once per evolutionary generation (EO). Plotted raw, no smoothing.
    PerGeneration,
    /// Once per completed episode (terminal outcome). Smoothed.
    PerEpisode,
}

/// Typed description of one canonical metric.
///
/// The `name` is the exact `tracing` field name the recorder matches against;
/// it is the wire contract between the algorithm crates and the recorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricDescriptor {
    /// Exact `tracing` field name (the wire contract).
    pub name: &'static str,
    /// Optimisation paradigm this metric belongs to.
    pub kind: MetricKind,
    /// Sampling cadence (controls report smoothing).
    pub cadence: Cadence,
    /// Human-readable panel title for the report.
    pub title: &'static str,
    /// Optional unit suffix (e.g. `"steps/s"`), `None` when dimensionless.
    pub unit: Option<&'static str>,
}

/// The canonical metric table, ordered for a stable report panel layout.
///
/// Order matters: the report client lays out panels in this order, RL
/// diagnostics first, then per-episode/shared, then EO.
pub const CANONICAL_METRICS: &[MetricDescriptor] = &[
    // ---- RL training stats (per update) ----
    d("policy_loss", MetricKind::Rl, Cadence::PerUpdate, "Policy loss"),
    d("value_loss", MetricKind::Rl, Cadence::PerUpdate, "Value loss"),
    d("loss", MetricKind::Rl, Cadence::PerUpdate, "Loss"),
    d("entropy", MetricKind::Rl, Cadence::PerUpdate, "Policy entropy"),
    d("approx_kl", MetricKind::Rl, Cadence::PerUpdate, "Approx KL"),
    d("clip_frac", MetricKind::Rl, Cadence::PerUpdate, "Clip fraction"),
    // ---- RL value/policy diagnostics (v6) ----
    d(
        "explained_variance",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Explained variance",
    ),
    d(
        "old_approx_kl",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Approx KL (pre-update)",
    ),
    // ---- Per-iteration training stats (v6) ----
    d(
        "episode_return_mean",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Episode return (mean)",
    ),
    d(
        "episode_return_std",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Episode return (std)",
    ),
    d(
        "episode_return_min",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Episode return (min)",
    ),
    d(
        "episode_return_max",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Episode return (max)",
    ),
    d(
        "episode_length_mean",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Episode length (mean)",
    ),
    du(
        "env_steps_sampled",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Env steps sampled",
        "steps",
    ),
    du(
        "steps_per_sec",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Throughput",
        "steps/s",
    ),
    d(
        "learning_rate",
        MetricKind::Rl,
        Cadence::PerUpdate,
        "Learning rate",
    ),
    // ---- Per-episode terminal triple (shared, v6) ----
    d(
        "episode_return",
        MetricKind::Shared,
        Cadence::PerEpisode,
        "Episode return",
    ),
    du(
        "episode_length",
        MetricKind::Shared,
        Cadence::PerEpisode,
        "Episode length",
        "steps",
    ),
    du(
        "episode_wall_clock_secs",
        MetricKind::Shared,
        Cadence::PerEpisode,
        "Episode wall-clock",
        "s",
    ),
    // ---- DQN family (v6) ----
    d("td_loss", MetricKind::Rl, Cadence::PerUpdate, "TD loss"),
    d("q_values", MetricKind::Rl, Cadence::PerUpdate, "Q-values (mean)"),
    // ---- SAC family (v6) ----
    d("qf1_loss", MetricKind::Rl, Cadence::PerUpdate, "Critic 1 loss"),
    d("qf2_loss", MetricKind::Rl, Cadence::PerUpdate, "Critic 2 loss"),
    d("actor_loss", MetricKind::Rl, Cadence::PerUpdate, "Actor loss"),
    d("alpha", MetricKind::Rl, Cadence::PerUpdate, "Entropy temperature α"),
    d("alpha_loss", MetricKind::Rl, Cadence::PerUpdate, "Alpha loss"),
    // ---- Schedules (v6) ----
    d("clip_range", MetricKind::Rl, Cadence::PerUpdate, "Clip range"),
    du("n_updates", MetricKind::Rl, Cadence::PerUpdate, "Update count", "updates"),
    // ---- Evolution training stats (per generation) ----
    d(
        "best_fitness",
        MetricKind::Eo,
        Cadence::PerGeneration,
        "Best fitness",
    ),
    d(
        "mean_fitness",
        MetricKind::Eo,
        Cadence::PerGeneration,
        "Mean fitness",
    ),
    d(
        "worst_fitness",
        MetricKind::Eo,
        Cadence::PerGeneration,
        "Worst fitness",
    ),
    d(
        "best_fitness_ever",
        MetricKind::Eo,
        Cadence::PerGeneration,
        "Best fitness (ever)",
    ),
];

/// Const constructor for a unit-less descriptor — keeps the table terse.
const fn d(
    name: &'static str,
    kind: MetricKind,
    cadence: Cadence,
    title: &'static str,
) -> MetricDescriptor {
    MetricDescriptor {
        name,
        kind,
        cadence,
        title,
        unit: None,
    }
}

/// Const constructor for a descriptor carrying a unit suffix.
const fn du(
    name: &'static str,
    kind: MetricKind,
    cadence: Cadence,
    title: &'static str,
    unit: &'static str,
) -> MetricDescriptor {
    MetricDescriptor {
        name,
        kind,
        cadence,
        title,
        unit: Some(unit),
    }
}

/// Looks up the descriptor for a metric field name, if it is canonical.
#[must_use]
pub fn descriptor(name: &str) -> Option<&'static MetricDescriptor> {
    CANONICAL_METRICS.iter().find(|d| d.name == name)
}

/// `true` if `name` is a recognised canonical metric field name.
#[must_use]
pub fn is_canonical_metric(name: &str) -> bool {
    descriptor(name).is_some()
}

/// Human-readable title for a metric field name, falling back to the raw name
/// (so an as-yet-undescribed metric still surfaces without a code change).
#[must_use]
pub fn title_for(name: &str) -> &str {
    match descriptor(name) {
        Some(d) => d.title,
        None => name,
    }
}

/// `true` if the metric is sampled once per evolutionary generation, so the
/// report should plot it raw (no rolling-mean overlay).
#[must_use]
pub fn is_per_generation(name: &str) -> bool {
    matches!(
        descriptor(name),
        Some(MetricDescriptor {
            cadence: Cadence::PerGeneration,
            ..
        })
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_are_recognised() {
        for desc in CANONICAL_METRICS {
            assert!(is_canonical_metric(desc.name), "{} canonical", desc.name);
        }
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert!(!is_canonical_metric("batch_size"));
        assert!(!is_canonical_metric("not_a_metric"));
        assert!(!is_canonical_metric(""));
    }

    #[test]
    fn names_are_unique() {
        for (i, a) in CANONICAL_METRICS.iter().enumerate() {
            for b in &CANONICAL_METRICS[i + 1..] {
                assert_ne!(a.name, b.name, "duplicate metric name {}", a.name);
            }
        }
    }

    #[test]
    fn fitness_metrics_are_eo_per_generation() {
        for name in ["best_fitness", "mean_fitness", "worst_fitness", "best_fitness_ever"] {
            let desc = descriptor(name).expect("present");
            assert_eq!(desc.kind, MetricKind::Eo);
            assert_eq!(desc.cadence, Cadence::PerGeneration);
            assert!(is_per_generation(name));
        }
    }

    #[test]
    fn rl_diagnostics_are_rl_kind() {
        for name in ["policy_loss", "explained_variance", "td_loss", "qf1_loss", "alpha"] {
            assert_eq!(descriptor(name).unwrap().kind, MetricKind::Rl, "{name}");
            assert!(!is_per_generation(name));
        }
    }

    #[test]
    fn terminal_triple_is_shared_per_episode() {
        for name in ["episode_return", "episode_length", "episode_wall_clock_secs"] {
            let desc = descriptor(name).unwrap();
            assert_eq!(desc.kind, MetricKind::Shared, "{name}");
            assert_eq!(desc.cadence, Cadence::PerEpisode, "{name}");
        }
    }

    #[test]
    fn title_falls_back_to_raw_name() {
        assert_eq!(title_for("policy_loss"), "Policy loss");
        assert_eq!(title_for("unknown_metric"), "unknown_metric");
    }
}
