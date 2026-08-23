#!/usr/bin/env bash
#
# Shared staging helpers for the outside-in acceptance probes (BYOE-1, BYOA-1).
#
# Sourced, never executed. Both probes need the identical publish-boundary
# simulation: package every publishable crate, extract the tarballs, and
# re-point their intra-workspace dependencies at the staged siblings. That is
# roughly fifty lines of subtle cargo behaviour, and a second hand-maintained
# copy of it would go stale silently — the staged tree would stop resembling
# what a consumer gets, and the probe built on it would keep passing.
#
# Provides:
#   stage_crates <stage_dir> <version> <repo_root> <work_dir> <crate>...
#       Packages each crate in dependency order, extracts it into <stage_dir>,
#       and rewrites the extracted manifests. Calls `die` on any failure.
#
# Expects the sourcing script to define: log, info, die.

# Packages the listed crates and stages them so a probe crate can depend on
# them by path, exactly as if they had come from a registry.
#
# `$@` (the crate list) is in dependency order, and it has to be: after
# packaging each crate we add a `[patch.crates-io]` entry pointing at its
# freshly staged copy, so the *next* crate's packaging resolves siblings from
# this working tree.
#
# Without that patch, `cargo package` resolves internal `path + version` deps
# against the real crates.io index whenever a satisfying version is already
# published there — a blind spot documented in the maintainer's publishing
# guide, and one both probes inherit. It bites hardest exactly when the
# workspace has *changed*: adding a feature to a member crate makes packaging
# its dependents abort with "does not have that feature", naming the published
# version's feature list, because the local manifest that does have it was
# never consulted. Nothing about the working tree is being tested at that
# point, which is the opposite of what these scripts exist for.
stage_crates() {
    local stage="$1" version="$2" repo_root="$3" work="$4"
    shift 4
    local crates=("$@")

    local patch_args=()
    log "Packaging ${#crates[@]} publishable crates"
    for crate in "${crates[@]}"; do
        if ! cargo package --quiet -p "$crate" --no-verify --allow-dirty \
                --manifest-path "$repo_root/Cargo.toml" \
                ${patch_args[@]+"${patch_args[@]}"} 2>"$work/pkg-$crate.err"; then
            cat "$work/pkg-$crate.err" >&2
            die "cargo package failed for $crate (this is itself a finding)"
        fi
        tar -xf "$repo_root/target/package/$crate-$version.crate" -C "$stage" ||
            die "could not extract $crate-$version.crate"
        patch_args+=(--config "patch.crates-io.$crate.path=\"$stage/$crate-$version\"")
        info "staged $crate-$version"
    done

    # Re-point intra-workspace dependencies at the staged siblings.
    # `cargo package` strips `path` and leaves `version = "x.y.z"`, which cannot
    # resolve without a registry. Only rlevo-* entries are touched; third-party
    # deps still resolve from crates.io exactly as a real consumer's would.
    log "Re-pointing staged rlevo-* dependencies at siblings"
    python3 - "$stage" "$version" <<'PY' || die "manifest rewrite failed"
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
}

# Reads the workspace version from `[workspace.package]`.
workspace_version() {
    local repo_root="$1"
    sed -n '/^\[workspace.package\]/,/^\[/p' "$repo_root/Cargo.toml" |
        sed -n 's/^version *= *"\(.*\)"/\1/p' | head -1
}
