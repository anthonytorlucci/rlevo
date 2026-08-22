//! Pins BYOE-1 blockers B2 and B10: the harness is reachable through the
//! umbrella, and the umbrella's viz features actually forward into it.
//!
//! # What was wrong
//!
//! `rlevo-benchmarks` was a **dev**-dependency of `rlevo`. Inside the
//! workspace that resolves fine, so nothing here failed; outside it, the
//! harness was invisible and `rlevo`'s `viz-tui` / `viz-report` features
//! forwarded into a crate the consumer could not see. A researcher following
//! the README's `cargo add rlevo` had to discover, name, and version-match a
//! second crate whose own description says "internal crate — use `rlevo` for
//! the full API". Measured: `cargo tree -p rlevo -e normal --features viz-tui`
//! reported **zero** ratatui nodes before the fix and three after.
//!
//! # Why these tests are compile-shaped
//!
//! [`rlevo::benchmarks`] is a `pub use` in the umbrella's library. Rust cannot
//! re-export a dev-dependency from a library target, so the mere existence of
//! that path forces `rlevo-benchmarks` into `[dependencies]` — the B10 defect
//! is structurally unreachable while these paths resolve. The bodies are
//! therefore thin: what is being asserted is that each *path* exists, through
//! the umbrella, under the feature set named on the block.

/// The preset suites arrive with the umbrella, no extra feature required.
///
/// `rlevo` enables `rlevo-benchmarks/fixtures` unconditionally, so
/// `cargo add rlevo` is enough to run a built-in environment on the harness.
#[test]
fn fixtures_are_reachable_from_the_umbrella() {
    use rlevo::benchmarks::evaluator::EvaluatorConfig;

    let cfg = EvaluatorConfig::default();
    let suite = rlevo::benchmarks::fixtures::suites::cartpole_suite(cfg);
    assert_eq!(suite.name, "cartpole");
}

/// A user environment reaches the harness by the same path, with no
/// `rlevo-benchmarks` line in their manifest.
#[test]
fn a_suite_can_be_built_over_an_umbrella_env() {
    use rlevo::benchmarks::evaluator::EvaluatorConfig;
    use rlevo::benchmarks::suite::Suite;
    use rlevo::envs::classic::{Pendulum, PendulumConfig};

    let suite = Suite::new("pendulum-via-umbrella", EvaluatorConfig::default()).with_env(
        "default",
        |seed| {
            Pendulum::with_config(PendulumConfig {
                seed,
                ..PendulumConfig::default()
            })
            .expect("valid config")
        },
    );
    assert_eq!(suite.envs.len(), 1);
}

/// `viz-report` forwards all the way to the recording tier **and** to the
/// per-env family impls, which now live on the harness side.
///
/// `for_env::<CartPole>()` is the load-bearing call: it resolves only if the
/// `RecordedEnvFamily` impl for a `rlevo-environments` type is visible from
/// `rlevo-benchmarks`, which is the flipped edge working.
#[cfg(feature = "viz-report")]
#[test]
fn viz_report_reaches_the_recording_tier() {
    use rlevo::benchmarks::record::{EnvFamily, RecordingConfig};
    use rlevo::envs::classic::CartPole;

    let cfg = RecordingConfig::for_env::<CartPole>(7);
    assert_eq!(cfg.env_family, EnvFamily::Classic);
}

/// `box2d` turns on both halves of its pair, so a physics env still has a
/// recording family.
///
/// The two features are separate crates' features and cargo will not link
/// them for you; the umbrella's `box2d = [envs/box2d, benchmarks/fixtures-box2d]`
/// is what keeps them together. Enabling only the first compiles a
/// `BipedalWalker` with no `RecordedEnvFamily` impl, and this call stops
/// resolving.
#[cfg(all(feature = "viz-report", feature = "box2d"))]
#[test]
fn the_box2d_feature_pair_stays_linked() {
    use rlevo::benchmarks::record::{EnvFamily, RecordingConfig};
    use rlevo::envs::box2d::bipedal_walker::BipedalWalker;

    let cfg = RecordingConfig::for_env::<BipedalWalker>(7);
    assert_eq!(cfg.env_family, EnvFamily::Box2d);
}

/// `viz-tui` forwards to the live dashboard.
#[cfg(feature = "viz-tui")]
#[test]
fn viz_tui_reaches_the_dashboard() {
    // Naming the module is the assertion; `viz-tui` not forwarding is a
    // compile error here rather than a silently inert feature flag.
    let _ = std::any::type_name::<rlevo::benchmarks::env_wrappers::TuiEnvTap<(), 1, 1, 1>>();
}
