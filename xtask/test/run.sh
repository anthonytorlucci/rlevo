#!/usr/bin/env bash
#
# Tiered test runner — `cargo xtask test <tier> [selector] [extra args...]`
#
#   fast   Mirrors the pull-request gate. Every crate's default test set.
#          Seconds per crate. This is the one to run after any change.
#   heavy  The `#[ignore]`d long-running tests, in release, MINUS the
#          acceptance runs. Minutes. This is the merge-gate tier.
#   accep- Full-solve convergence runs, hours each. Split out so `heavy` does
#   tance  not silently inherit a 500 000-step run.
#   gpu    Backend-parity tests needing a wgpu adapter. Local only — every CI
#          runner here is `ubuntu-latest` with no GPU, where `cubecl-wgpu`
#          aborts on device init rather than failing gracefully.
#
# WHY THIS EXISTS: the tiers were previously folklore — a workflow file, a
# gitignored zsh script, and the `#[ignore]` attribute messages. With no named
# fast tier, the obvious-looking invention is `cargo test --workspace
# --all-targets`, which is the catastrophic one: `--all-targets` includes
# `--benches`, and with `harness = false` criterion targets that *executes* 19
# benchmark binaries at full measurement (~14 hours) while still not running
# the `#[ignore]`d tests, which need `-- --ignored` to lift. Strictly worse
# coverage than `fast`, for four orders of magnitude more time.
#
# NEVER add `--all-targets` here. Benchmarks belong to `cargo bench`, run
# deliberately (docs/rules.md, Bench placement rule).
#
# Usage:
#   cargo xtask test fast                    # every crate
#   cargo xtask test fast rlevo-core         # one crate (what CI's matrix runs)
#   cargo xtask test heavy                   # every heavy unit
#   cargo xtask test heavy ppo_integration   # one unit
#   cargo xtask test acceptance              # the hours-long runs, explicitly
#   cargo xtask test gpu                     # needs a real adapter

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 90

TIER="${1:-}"
shift || true

# A leading `-` means "no selector, these are passthrough args".
SELECTOR=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    SELECTOR="$1"
    shift
fi
EXTRA=("$@")
# Expanded as ${EXTRA[@]+"${EXTRA[@]}"} throughout: bash 3.2, which is what
# macOS ships, treats "${arr[@]}" on an empty array as an unbound variable
# under `set -u`. The `+` guard expands to nothing instead of erroring.

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }

FAILED=()
PASSED=0

# run <label> <cargo args...>
run() {
    local label="$1"
    shift
    info "$label"
    if cargo test "$@"; then
        PASSED=$((PASSED + 1))
    else
        FAILED+=("$label")
    fi
}

# ---------------------------------------------------------------------------
# fast — the pull-request gate
# ---------------------------------------------------------------------------
# Crates are enumerated from the filesystem, never from a hard-coded list, so
# a new crate joins this tier by existing. The workflow's matrix still names
# them literally (GitHub Actions cannot expand a glob), and
# `crate-tests.yml`'s `test-matrix-coverage` job is what keeps that list
# honest — it is the one place a crate can still be silently dropped.
#
# Only the feature overrides live here, because those are what actually drifted:
# both exist because a bare `cargo test -p <crate>` compiles the gated target
# to nothing and reports success.
features_for() {
    case "$1" in
        # `tests/{cartpole_report_smoke,recording_episode_count}.rs` are
        # `required-features = ["viz-report"]`, absent from `rlevo`'s defaults.
        rlevo) echo "--features viz-report" ;;
        # `rlevo-benchmarks` defaults to `json` only, so a bare run compiles
        # neither the record tier nor the fixtures.
        #
        # `fixtures` is here because leaving it out cost a real miss: ADR 0081
        # deleted the `*Payload` wire mirrors, and `src/fixtures/family.rs`'s
        # test module still named five of them. It compiled nowhere in CI, so
        # four workflows went green over code that did not build. The two
        # passthroughs add no new dependency -- `rlevo-environments` already
        # defaults to `box2d` + `locomotion`, so this tier builds rapier
        # either way.
        rlevo-benchmarks) echo "--features record,fixtures,fixtures-box2d,fixtures-locomotion" ;;
        *) echo "" ;;
    esac
}

tier_fast() {
    log "fast — pull-request gate"
    local crates=()
    if [[ -n "$SELECTOR" ]]; then
        [[ -d "crates/$SELECTOR" ]] || { echo "no such crate: $SELECTOR" >&2; exit 90; }
        crates=("$SELECTOR")
    else
        for dir in crates/*/; do crates+=("$(basename "$dir")"); done
    fi
    for crate in "${crates[@]}"; do
        # Unquoted on purpose: empty expands to no argument, non-empty to two.
        # shellcheck disable=SC2046
        run "$crate" --package "$crate" $(features_for "$crate") ${EXTRA[@]+"${EXTRA[@]}"}
    done
}

# ---------------------------------------------------------------------------
# heavy — the `#[ignore]`d suite, in release
# ---------------------------------------------------------------------------
# Folded in from the former `scripts/cargo_test_ignored.zsh`, which was zsh,
# gitignored (so unavailable to CI and to every other contributor), lacked
# `--release`, and had drifted from the weekly workflow it shadowed — 24
# binaries against weekly's 8.
#
# `--release` is not a speed preference. Several of these are convergence runs
# that do not finish in debug within any sane budget, and debug's overflow
# checks mask release wrapping behaviour, so a debug-only pass proves less than
# it appears to.
#
# Binaries with no `#[ignore]`d test are harmless here: they report 0 tests.
# Enumerating rather than curating means a newly-ignored test is picked up
# without anyone remembering to add it.
heavy_units() {
    for f in crates/rlevo/tests/*.rs; do
        echo "rlevo:$(basename "$f" .rs)"
    done
    # Ignored tests outside `crates/rlevo/tests`. GPU parity binaries are
    # deliberately absent — they are the `gpu` tier.
    echo "rlevo-examples:neuroevolution_santa_fe_ant"
    echo "rlevo-evolution:coevolution_forgetting"
}

# Only the binaries that genuinely gate on it. Passing `--features viz-report`
# to all of `rlevo` would change what the weekly workflow compiles.
#
# Every name carrying `required-features = ["viz-report"]` in
# `crates/rlevo/Cargo.toml` must appear here: `heavy_units` enumerates the
# `tests/` directory, and cargo *errors* on a target whose required features
# are absent rather than skipping it, so a missing entry fails the tier.
heavy_features_for() {
    case "$1" in
        cartpole_report_smoke | recording_episode_count | payload_forwarding_completeness)
            echo "--features viz-report"
            ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Acceptance tests — full-solve runs measured in hours, not minutes.
# ---------------------------------------------------------------------------
# `crate:binary:test` triples. `heavy` skips these; the `acceptance` tier runs
# exactly them. The split mirrors `weekly-tests.yml`, which has always run them
# as a separate job with a 350-minute timeout — without the split, a plain
# `cargo xtask test heavy` silently inherits that runtime, which is the same
# unnamed-scope trap this runner exists to close.
#
# CURATED, not enumerated — deliberately, and against the rule used everywhere
# else in this file. These tests are generated by `rl_learning_test!`, so the
# name never appears as `fn <name>` in any source file and no grep can discover
# them. A new acceptance test must be added here by hand; nothing will notice
# if it is not, and it will simply run inside `heavy` instead.
acceptance_units() {
    echo "rlevo:qrdqn_integration:qrdqn_cartpole_acceptance"
}

# `--skip <name>` for every acceptance test living in binary $1.
acceptance_skips_for() {
    while IFS=: read -r _crate bin test; do
        [[ "$bin" == "$1" ]] && printf -- '--skip %s ' "$test"
    done < <(acceptance_units)
}

tier_heavy() {
    log "heavy — ignored tests, release profile"
    local matched=0
    while IFS=: read -r crate bin; do
        [[ -n "$SELECTOR" && "$SELECTOR" != "$bin" ]] && continue
        matched=1
        # shellcheck disable=SC2046
        run "$crate/$bin" --release --package "$crate" \
            $(heavy_features_for "$bin") --test "$bin" \
            -- --ignored $(acceptance_skips_for "$bin") ${EXTRA[@]+"${EXTRA[@]}"}
    done < <(heavy_units)
    if [[ -n "$SELECTOR" && "$matched" -eq 0 ]]; then
        echo "no heavy unit named: $SELECTOR" >&2
        exit 90
    fi
}

# ---------------------------------------------------------------------------
# acceptance — full-solve runs, hours
# ---------------------------------------------------------------------------
# Deliberately not part of `heavy`: a merge-gate validation should not inherit
# a 500 000-step convergence run. Selector matches the binary, so
# `cargo xtask test acceptance qrdqn_integration` narrows to one.
tier_acceptance() {
    log "acceptance — full-solve runs (hours)"
    info "Not included in 'heavy'. Expect one test to take hours, not minutes."
    local matched=0
    while IFS=: read -r crate bin test; do
        [[ -n "$SELECTOR" && "$SELECTOR" != "$bin" && "$SELECTOR" != "$test" ]] && continue
        matched=1
        # shellcheck disable=SC2046
        run "$crate/$bin::$test" --release --package "$crate" \
            $(heavy_features_for "$bin") --test "$bin" \
            -- --ignored "$test" ${EXTRA[@]+"${EXTRA[@]}"}
    done < <(acceptance_units)
    if [[ -n "$SELECTOR" && "$matched" -eq 0 ]]; then
        echo "no acceptance unit named: $SELECTOR" >&2
        exit 90
    fi
}

# ---------------------------------------------------------------------------
# gpu — backend parity, local only
# ---------------------------------------------------------------------------
# These three are run by nothing today: `#[ignore]` hides them from the gate,
# and the weekly matrix only reaches `-p rlevo --test <bin>`, which is the
# wrong crate for all of them. They verify that the wgpu backend agrees with
# flex — precisely the class of defect no other test can catch.
tier_gpu() {
    log "gpu — backend parity (requires a wgpu adapter)"
    info "No CI runner here has a GPU; this tier is local-only by construction."
    run "rlevo-evolution/backend_parity" \
        --release --package rlevo-evolution --test backend_parity -- --ignored ${EXTRA[@]+"${EXTRA[@]}"}
    run "rlevo-reinforcement-learning/c51_projection_backend_parity" \
        --release --package rlevo-reinforcement-learning \
        --test c51_projection_backend_parity -- --ignored ${EXTRA[@]+"${EXTRA[@]}"}
    # The only `#[ignore]`d lib test in the crate is the wgpu clamp probe, so
    # `--lib -- --ignored` selects exactly it.
    run "rlevo-reinforcement-learning/clamp_preserving_nan_matches_flex_on_wgpu" \
        --release --package rlevo-reinforcement-learning --lib -- --ignored ${EXTRA[@]+"${EXTRA[@]}"}
}

case "$TIER" in
    fast)       tier_fast ;;
    heavy)      tier_heavy ;;
    acceptance) tier_acceptance ;;
    gpu)        tier_gpu ;;
    *)
        cat >&2 <<'USAGE'
usage: cargo xtask test <tier> [selector] [extra cargo-test args...]

tiers:
  fast        pull-request gate — every crate's default tests (seconds per crate)
  heavy       #[ignore]d suite in release, minus acceptance runs (minutes)
  acceptance  full-solve convergence runs (hours) — excluded from heavy
  gpu         wgpu/flex backend parity — needs a real adapter, local only

Benchmarks are never run here. Use `cargo bench`.
USAGE
        exit 90
        ;;
esac

log "Summary"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    printf '\033[1;32m%s unit(s) passed.\033[0m\n' "$PASSED"
    exit 0
fi
printf '\033[1;31m%s passed, %s failed:\033[0m\n' "$PASSED" "${#FAILED[@]}"
for f in ${FAILED[@]+"${FAILED[@]}"}; do info "FAILED  $f"; done
exit 1
