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

VERSION="$(
    sed -n '/^\[workspace.package\]/,/^\[/p' "$REPO_ROOT/Cargo.toml" |
    sed -n 's/^version *= *"\(.*\)"/\1/p' | head -1
)"
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
log "Packaging ${#PUBLISHABLE[@]} publishable crates"
for crate in "${PUBLISHABLE[@]}"; do
    if ! cargo package --quiet -p "$crate" --no-verify --allow-dirty \
            --manifest-path "$REPO_ROOT/Cargo.toml" 2>"$WORK/pkg-$crate.err"; then
        cat "$WORK/pkg-$crate.err" >&2
        die "cargo package failed for $crate (this is itself a finding)"
    fi
    tar -xf "$REPO_ROOT/target/package/$crate-$VERSION.crate" -C "$STAGE" ||
        die "could not extract $crate-$VERSION.crate"
    info "staged $crate-$VERSION"
done

# Re-point intra-workspace dependencies at the staged siblings. `cargo package`
# strips `path` and leaves `version = "x.y.z"`, which cannot resolve without a
# registry. Only rlevo-* entries are touched; third-party deps still resolve
# from crates.io exactly as a real consumer's would.
log "Re-pointing staged rlevo-* dependencies at siblings"
python3 - "$STAGE" "$VERSION" <<'PY' || die "manifest rewrite failed"
import pathlib, re, sys

stage, version = pathlib.Path(sys.argv[1]), sys.argv[2]
# Matches `[dependencies.rlevo-core]`, `[dev-dependencies.rlevo-x]`,
# `[target.'cfg(..)'.dependencies.rlevo-y]`, ...
header = re.compile(r"^\[[^\]]*dependencies\.(rlevo[-\w]*)\]\s*$")
touched = 0

for manifest in sorted(stage.glob("*/Cargo.toml")):
    out, in_dep, dep = [], None, None
    for line in manifest.read_text().splitlines():
        m = header.match(line)
        if m:
            in_dep, dep = True, m.group(1)
            out.append(line)
            continue
        if in_dep and line.startswith("["):
            in_dep, dep = False, None
        if in_dep and line.startswith("version"):
            out.append(line)
            out.append(f'path = "{stage / f"{dep}-{version}"}"')
            touched += 1
            continue
        out.append(line)
    manifest.write_text("\n".join(out) + "\n")

print(f"   rewrote {touched} dependency entries")
if touched == 0:
    raise SystemExit("no rlevo-* dependency entries found - packaging layout changed")
PY

# ---------------------------------------------------------------------------
# Steps 1 & 2 — scaffold the probe and declare its dependencies.
# ---------------------------------------------------------------------------
log "Steps 1-2: scaffolding the probe crate outside the workspace"
cp "$TEMPLATE_DIR/src/main.rs" "$PROBE/src/main.rs"
sed -e "s|@@STAGE@@|$STAGE|g" -e "s|@@VERSION@@|$VERSION|g" \
    "$TEMPLATE_DIR/Cargo.toml.template" > "$PROBE/Cargo.toml"

echo "BYOE-STEP 1 PASS scaffold :: probe created at $PROBE (outside the workspace)"

# Count the crates a researcher must add by hand. Anything past `rlevo` is a
# step 2 finding, per the spec.
EXTRA=$(sed -n '/^\[dependencies\]/,/^\[\[/p' "$PROBE/Cargo.toml" |
        grep -cE '^(rlevo-benchmarks|parking_lot|rand|serde) ' || true)
if [[ "$EXTRA" -gt 0 ]]; then
    echo "BYOE-STEP 2 FAIL add-deps :: rlevo alone is insufficient; $EXTRA additional crate(s) required by hand (blocker B2)"
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
