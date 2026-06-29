//! Declarative macros that generate the standard algorithm-test suite.
//!
//! Every algorithm integration test converges on the same two universal checks
//! — "a learned policy beats/reaches a threshold" and "a seeded run is
//! reproducible" — wrapped in identical `#[test]` + [`flex_guard`] scaffolding.
//! These macros generate that scaffolding so each test file is left with a
//! single `run(seed, total) -> TrainOutcome` function plus a couple of macro
//! invocations, instead of hand-written `#[test]` bodies.
//!
//! The `run` function is the only algorithm-specific glue: it builds the agent,
//! trains it for `total` steps, and returns a [`TrainOutcome`]. It must **not**
//! claim the backend lock itself — the generated test holds [`flex_guard`] for
//! the whole body, then calls `run` (twice, for reproducibility) underneath it.
//!
//! Attach `#[ignore = "..."]` (or nothing) before the test name: the leading
//! `$(#[$attr:meta])*` capture forwards it to the generated function, so the
//! same macro serves both default-run and `#[ignore]`d tests.
//!
//! [`flex_guard`]: crate::flex::flex_guard
//! [`TrainOutcome`]: crate::TrainOutcome

/// Generates a learning/convergence test.
///
/// Two assertion modes:
/// - `improves_over_random(margin = $margin)` — the trained score must beat a
///   *measured* random-policy baseline by `$margin`. Requires a `random =
///   <fn(seed) -> f32>` callback that rolls a random policy over the same
///   environment (see [`crate::baseline`]). Used for the continuous tracking
///   tasks and the on-policy Pendulum / `CartPole` learning checks.
/// - `reaches($threshold)` — `>=` an absolute floor (the discrete value-based
///   `_converges` / `_acceptance` checks, where the target is a fixed bar
///   rather than a margin over random).
///
/// ```ignore
/// rl_learning_test! {
///     #[ignore = "8 000-step LinearEnv convergence check; run with `cargo test -- --ignored`"]
///     td3_linear_improves_over_random,
///     improves_over_random(margin = 2.0),
///     seed = 42,
///     total = 8_000,
///     run = run_linear,
///     random = random_linear,
/// }
/// ```
#[macro_export]
macro_rules! rl_learning_test {
    (
        $(#[$attr:meta])*
        $name:ident,
        improves_over_random(margin = $margin:expr),
        seed = $seed:expr,
        total = $total:expr,
        run = $run:expr,
        random = $random:expr $(,)?
    ) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            let _guard = $crate::flex::flex_guard();
            let random_avg: f32 = ($random)($seed);
            let outcome: $crate::TrainOutcome = ($run)($seed, $total);
            $crate::assert::assert_improves_over_random(outcome.avg_score, random_avg, $margin);
        }
    };
    (
        $(#[$attr:meta])*
        $name:ident,
        reaches($threshold:expr),
        seed = $seed:expr,
        total = $total:expr,
        run = $run:expr $(,)?
    ) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            let _guard = $crate::flex::flex_guard();
            let outcome: $crate::TrainOutcome = ($run)($seed, $total);
            $crate::assert::assert_reaches(outcome.avg_score, $threshold);
        }
    };
}

/// Generates a seeded-reproducibility test: runs `run(seed, total)` twice and
/// asserts the results match.
///
/// Two comparison modes:
/// - `bits` — the two `avg_score` scalars must be bit-identical (continuous).
/// - `seq` — the two `rewards` sequences must match element-for-element
///   (discrete).
///
/// ```ignore
/// rl_reproducibility_test! {
///     td3_reproducibility_flex,
///     bits,
///     seed = 42,
///     total = 1_000,
///     run = run_linear,
/// }
/// ```
#[macro_export]
macro_rules! rl_reproducibility_test {
    (
        $(#[$attr:meta])*
        $name:ident,
        bits,
        seed = $seed:expr,
        total = $total:expr,
        run = $run:expr $(,)?
    ) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            let _guard = $crate::flex::flex_guard();
            let a: $crate::TrainOutcome = ($run)($seed, $total);
            let b: $crate::TrainOutcome = ($run)($seed, $total);
            $crate::assert::assert_reproducible_bits(a.avg_score, b.avg_score);
        }
    };
    (
        $(#[$attr:meta])*
        $name:ident,
        seq,
        seed = $seed:expr,
        total = $total:expr,
        run = $run:expr $(,)?
    ) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            let _guard = $crate::flex::flex_guard();
            let a: $crate::TrainOutcome = ($run)($seed, $total);
            let b: $crate::TrainOutcome = ($run)($seed, $total);
            $crate::assert::assert_reproducible_seq(&a.rewards, &b.rewards);
        }
    };
}
