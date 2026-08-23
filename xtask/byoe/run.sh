#!/usr/bin/env bash
#
# BYOE-1 — bring-your-own-environment acceptance test.
# Spec: docs/.private/specs/2026-08-21-byoe-first-class-citizen/
#
# Walks the path a researcher takes who has never seen this repository:
#
#   1. cargo new, outside the workspace
#   2. add rlevo (and whatever else turns out to be required)
#   3. implement Environment from scratch
#   4. satisfy the post-terminal contract
#   5. implement BenchableAgent
#   6. drive it through Evaluator::run_suite
#   7. record the run
#   8. emit a static HTML report - with a rich payload, not the ASCII fallback
#   9. emit a report with the interactive client
#
# The probe consumes `cargo package` tarballs rather than crates.io, so it runs
# with no publish and catches pre-publish regressions. Only files that
# `cargo package` would actually ship are present, so a missing `include`, an
# un-`pub`'d item, or an uncommitted file surfaces here.
#
# Because the packaged manifests have their intra-workspace `path` keys
# stripped (cargo rewrites them to bare `version = "x.y.z"` registry
# requirements), the extracted manifests are rewritten to point at their
# siblings in the staging directory. Nothing else about them is touched.
#
# THIS SCRIPT IS EXPECTED TO FAIL until the spec's blockers are resolved.
# It reports the first failing step; that number is the deliverable.
#
# Usage:
#   cargo xtask byoe                      # package, stage, build, run
#   BYOE_KEEP=1 cargo xtask byoe          # keep the staging dir for poking
#   BYOE_TRUNK_DIST=<dir> cargo xtask byoe   # supply client assets
#
# Invoking `xtask/byoe/run.sh` directly works too; the cargo alias just saves
# remembering the path.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/xtask/byoe/probe"

# Publish-boundary simulation, shared with BYOA-1. See xtask/common/stage.sh.
# shellcheck source=../common/stage.sh
. "$REPO_ROOT/xtask/common/stage.sh"

# Crates the probe's dependency cone needs, in no particular order. Excludes
# the three `publish = false` crates, which a consumer can never obtain.
PUBLISHABLE=(
    rlevo-metrics-registry
    rlevo-core
    rlevo-evolution
    rlevo-reinforcement-learning
    rlevo-environments
    rlevo-hybrid
    rlevo-benchmarks
    rlevo
)

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\n\033[1;31mBYOE-1 ABORTED: %s\033[0m\n' "$*" >&2; exit 90; }

VERSION="$(workspace_version "$REPO_ROOT")"
[[ -n "$VERSION" ]] || die "could not read version from [workspace.package]"

# Everything lives outside the repo so cargo cannot adopt the probe or the
# staged crates into the rlevo workspace.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/byoe-XXXXXX")" || die "mktemp failed"
STAGE="$WORK/stage"
PROBE="$WORK/byoe-probe"
mkdir -p "$STAGE" "$PROBE/src"

cleanup() {
    if [[ "${BYOE_KEEP:-0}" == "1" ]]; then
        printf '\n   staging kept at %s\n' "$WORK"
    else
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

log "BYOE-1 acceptance test  (workspace version $VERSION)"
info "staging: $WORK"

# ---------------------------------------------------------------------------
# Package and stage. This is the publish boundary standing in for crates.io.
# ---------------------------------------------------------------------------
stage_crates "$STAGE" "$VERSION" "$REPO_ROOT" "$WORK" "${PUBLISHABLE[@]}"

# ---------------------------------------------------------------------------
# Steps 1 & 2 — scaffold the probe and declare its dependencies.
# ---------------------------------------------------------------------------
log "Steps 1-2: scaffolding the probe crate outside the workspace"
cp "$TEMPLATE_DIR/src/main.rs" "$PROBE/src/main.rs"
sed -e "s|@@STAGE@@|$STAGE|g" -e "s|@@VERSION@@|$VERSION|g" \
    "$TEMPLATE_DIR/Cargo.toml.template" > "$PROBE/Cargo.toml"

echo "BYOE-STEP 1 PASS scaffold :: probe created at $PROBE (outside the workspace)"

# Count the crates a researcher must add by hand. Anything past `rlevo` is a
# step 2 finding, per the spec. Enabling a *feature* on `rlevo` is not a
# finding — that is what feature flags are for.
#
# `rlevo-benchmarks` stays in the pattern although the template no longer lists
# it: it is the regression guard for B2, and a count of 4 here means the
# harness has drifted back out of the umbrella.
EXTRA=$(sed -n '/^\[dependencies\]/,/^\[\[/p' "$PROBE/Cargo.toml" |
        grep -cE '^(rlevo-benchmarks|parking_lot|rand|serde) ' || true)
if [[ "$EXTRA" -gt 0 ]]; then
    echo "BYOE-STEP 2 FAIL add-deps :: rlevo alone is insufficient; $EXTRA additional crate(s) required by hand (B2 resolved; the remainder are version-matching hazards on parking_lot / rand / serde)"
    STEP2=fail
else
    echo "BYOE-STEP 2 PASS add-deps :: rlevo alone sufficed"
    STEP2=pass
fi

# ---------------------------------------------------------------------------
# Steps 3 & 5 — the probe compiles against the packaged public API.
# ---------------------------------------------------------------------------
log "Steps 3 & 5: building the probe against packaged crates"
if ! cargo build --quiet --manifest-path "$PROBE/Cargo.toml" 2>"$WORK/build.err"; then
    sed 's/^/   /' "$WORK/build.err" >&2
    echo "BYOE-STEP 3 FAIL implement-environment :: probe does not compile against the packaged public API"
    printf '\n\033[1;31mBYOE-1 reached step 3 of 9.\033[0m\n'
    exit 3
fi
echo "BYOE-STEP 3 PASS implement-environment :: Environment implemented using public API only"
echo "BYOE-STEP 5 PASS implement-agent :: BenchableAgent implemented using public API only"

# ---------------------------------------------------------------------------
# Steps 4, 6-9 — runtime behaviour.
# ---------------------------------------------------------------------------
log "Steps 4, 6-9: running the probe"
cargo run --quiet --manifest-path "$PROBE/Cargo.toml" 2>"$WORK/run.err"
RUN_STATUS=$?
if [[ -s "$WORK/run.err" ]]; then
    sed 's/^/   /' "$WORK/run.err" >&2
fi

# ---------------------------------------------------------------------------
# Verdict. The failing step number is the deliverable.
# ---------------------------------------------------------------------------
log "Verdict"
if [[ "$RUN_STATUS" -eq 0 && "$STEP2" == "pass" ]]; then
    printf '\033[1;32mBYOE-1 PASSES all 9 steps.\033[0m\n'
    exit 0
fi

if [[ "$RUN_STATUS" -eq 0 ]]; then
    printf '\033[1;33mBYOE-1 reached step 9, but step 2 failed.\033[0m\n'
    info "The path works; discovering it does not. See blocker B2."
    exit 2
fi

printf '\033[1;31mBYOE-1 fails at step %s of 9.\033[0m\n' "$RUN_STATUS"
info "This is Phase 0's deliverable, not a broken script."
info "Record the finding in the spec's blocker section, then fix and re-run."
exit "$RUN_STATUS"
