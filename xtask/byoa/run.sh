#!/usr/bin/env bash
#
# BYOA-1 — bring-your-own-algorithm acceptance test.
# Spec: docs/.private/specs/2026-08-21-byoe-first-class-citizen/
#
# The mirror of BYOE-1. BYOE-1 varies the environment and holds the algorithm
# fixed; BYOA-1 varies the **algorithm** and uses first-party environments —
# the leaderboard-submitter persona, who brings an existing or novel method and
# wants it benched against other submissions and a random baseline.
#
# Walks the path a submitter takes who has never seen this repository:
#
#   1. cargo new, outside the workspace
#   2. add rlevo (and whatever else turns out to be required)
#   3. implement a learning algorithm from scratch
#   4. reach it through the documented harness entry point
#   5. train it against a shipped environment
#   6. compare it with a random baseline under identical config
#   7. record the run
#   8. get the submitter's own metrics into the report
#   9. prove two submissions faced the same task
#
# Steps 2 and 4 are *discoverability* findings: reported, folded into the
# verdict, but non-fatal. They answer "was this findable?", which is a different
# question from the one the other steps answer ("was this possible?"). Both
# matter and conflating them loses information — BYOE-1 makes the same split at
# its own step 2.
#
# The probe consumes `cargo package` tarballs rather than crates.io, so it runs
# with no publish and catches pre-publish regressions.
#
# THIS SCRIPT IS EXPECTED TO FAIL until the spec's blockers are resolved.
# It reports the first fatal failing step; that number is the deliverable.
#
# Usage:
#   cargo xtask byoa                  # package, stage, build, run
#   BYOA_KEEP=1 cargo xtask byoa      # keep the staging dir for poking
#
# Invoking `xtask/byoa/run.sh` directly works too; the cargo alias just saves
# remembering the path.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/xtask/byoa/probe"

# Publish-boundary simulation, shared with BYOE-1. See xtask/common/stage.sh.
# shellcheck source=../common/stage.sh
. "$REPO_ROOT/xtask/common/stage.sh"

# Crates the probe's dependency cone needs, in dependency order. Excludes the
# `publish = false` crates, which a consumer can never obtain.
#
# Identical to BYOE-1's list, and deliberately so: a submitter who never writes
# an environment still receives the whole umbrella cone, because `rlevo` is one
# crate. If that list ever diverges between the two probes, the umbrella has
# grown a consumer-visible seam that one persona sees and the other does not.
PUBLISHABLE=(
    rlevo-metrics-registry
    rlevo-scene
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
die()  { printf '\n\033[1;31mBYOA-1 ABORTED: %s\033[0m\n' "$*" >&2; exit 90; }

VERSION="$(workspace_version "$REPO_ROOT")"
[[ -n "$VERSION" ]] || die "could not read version from [workspace.package]"

# Everything lives outside the repo so cargo cannot adopt the probe or the
# staged crates into the rlevo workspace.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/byoa-XXXXXX")" || die "mktemp failed"
STAGE="$WORK/stage"
PROBE="$WORK/byoa-probe"
mkdir -p "$STAGE" "$PROBE/src"

cleanup() {
    if [[ "${BYOA_KEEP:-0}" == "1" ]]; then
        printf '\n   staging kept at %s\n' "$WORK"
    else
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

log "BYOA-1 acceptance test  (workspace version $VERSION)"
info "staging: $WORK"

stage_crates "$STAGE" "$VERSION" "$REPO_ROOT" "$WORK" "${PUBLISHABLE[@]}"

# ---------------------------------------------------------------------------
# Steps 1 & 2 — scaffold the probe and declare its dependencies.
# ---------------------------------------------------------------------------
log "Steps 1-2: scaffolding the probe crate outside the workspace"
cp "$TEMPLATE_DIR/src/main.rs" "$PROBE/src/main.rs"
sed -e "s|@@STAGE@@|$STAGE|g" -e "s|@@VERSION@@|$VERSION|g" \
    "$TEMPLATE_DIR/Cargo.toml.template" > "$PROBE/Cargo.toml"

echo "BYOA-STEP 1 PASS scaffold :: probe created at $PROBE (outside the workspace)"

# Count the crates a submitter must add by hand. Anything past `rlevo` is a
# step 2 finding. Enabling a *feature* on `rlevo` is not a finding — that is
# what feature flags are for.
EXTRA=$(sed -n '/^\[dependencies\]/,/^\[\[/p' "$PROBE/Cargo.toml" |
        grep -cE '^(parking_lot|rand|tracing|tracing-subscriber|serde) ' || true)
if [[ "$EXTRA" -gt 0 ]]; then
    echo "BYOA-STEP 2 FAIL add-deps :: rlevo alone is insufficient; $EXTRA additional crate(s) required by hand, each an unversioned guess (parking_lot appears in a public trait signature; tracing + tracing-subscriber are the only route to the record)"
    STEP2=fail
else
    echo "BYOA-STEP 2 PASS add-deps :: rlevo alone sufficed"
    STEP2=pass
fi

# ---------------------------------------------------------------------------
# Step 3 — the probe compiles against the packaged public API.
# ---------------------------------------------------------------------------
log "Step 3: building the probe against packaged crates"
if ! cargo build --quiet --manifest-path "$PROBE/Cargo.toml" 2>"$WORK/build.err"; then
    sed 's/^/   /' "$WORK/build.err" >&2
    echo "BYOA-STEP 3 FAIL implement-algorithm :: probe does not compile against the packaged public API"
    printf '\n\033[1;31mBYOA-1 reached step 3 of 9.\033[0m\n'
    exit 3
fi
echo "BYOA-STEP 3 PASS implement-algorithm :: learning algorithm implemented using public API only"

# ---------------------------------------------------------------------------
# Steps 4-9 — runtime behaviour.
# ---------------------------------------------------------------------------
log "Steps 4-9: running the probe"
cargo run --quiet --manifest-path "$PROBE/Cargo.toml" 2>"$WORK/run.err"
RUN_STATUS=$?
if [[ -s "$WORK/run.err" ]]; then
    sed 's/^/   /' "$WORK/run.err" >&2
fi

# Step 4 is emitted by the probe as a non-fatal finding; recover it for the
# verdict rather than re-deriving the condition here.
STEP4=pass
if grep -q '^BYOA-STEP 4 FAIL' "$WORK/run.err" 2>/dev/null; then
    STEP4=fail
fi

# ---------------------------------------------------------------------------
# Verdict. The failing step number is the deliverable.
# ---------------------------------------------------------------------------
log "Verdict"
if [[ "$RUN_STATUS" -eq 0 && "$STEP2" == "pass" ]]; then
    printf '\033[1;32mBYOA-1 PASSES all 9 steps.\033[0m\n'
    exit 0
fi

if [[ "$RUN_STATUS" -eq 0 ]]; then
    printf '\033[1;33mBYOA-1 reached step 9, but a discoverability step failed.\033[0m\n'
    info "The path works; discovering it does not."
    exit 2
fi

printf '\033[1;31mBYOA-1 fails at step %s of 9.\033[0m\n' "$RUN_STATUS"
info "This is the deliverable, not a broken script."
info "Record the finding in the spec, then fix and re-run."
exit "$RUN_STATUS"
