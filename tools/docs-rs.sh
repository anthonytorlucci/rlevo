#!/usr/bin/env bash
# Document each published library crate as docs.rs does, on a nightly.
#
#   tools/docs-rs.sh                         # every crate, on `nightly`
#   tools/docs-rs.sh nightly-2026-11-01      # on a dated nightly
#   tools/docs-rs.sh nightly rlevo-core      # the named crates only
#
# docs.rs documents a release with a current nightly rustdoc, not with the
# toolchain `rust-toolchain.toml` pins, so docs that pass every other gate
# can fail to build there. 0.6.0's `rlevo-core` and `rlevo-harness` did: the
# 1.101.0 nightly's rustdoc resolves a module's own `//!` links in the
# module's scope, where 1.99 resolved them in the parent's when the
# `pub mod` line carries docs too, and the workspace's `deny` rustdoc lints
# made each link that moved an error.
#
# Each crate is built as docs.rs's builder runs cargo: `cargo rustdoc --lib
# -Zrustdoc-scrape-examples` with `DOCS_RS=1`, the features its
# `[package.metadata.docs.rs]` names, and rustdoc flags `--cfg docsrs`
# followed by that table's `rustdoc-args`. A crate with no table gets its
# default features, as on docs.rs. For `rlevo` that is not `--all-features`,
# which the 1.101.0 nightly cannot build: it fails to compile the dependency
# `turso_parser`, reached through `wgpu`. docs.rs runs rustdoc in the package
# root and cargo runs it in the workspace root, so a `rustdoc-args` entry
# naming a file in the package is given its path from the workspace root. A
# key of the table that this script does not apply is refused, not ignored.
#
# Not reproduced: docs.rs builds the packaged crate, resolved against
# crates.io, for `x86_64-unknown-linux-gnu`, and adds flags that only shape
# the HTML (`--emit`, `--resource-suffix`, `--static-root-path`,
# `-Zrustdoc-map`). A file the package leaves out is not missed here, since
# the checkout has it.
#
# Every crate is documented even after one fails, and the exit status is
# non-zero if any failed or warned. docs.rs fails on an error only, but a
# rustdoc warning is a defect on the page it publishes: three of 0.6.0's
# doc tables were split at a `|` in a cell, which rustdoc warns of and
# docs.rs rendered anyway. The closing table gives each crate's result and
# warning count, and appends to `$GITHUB_STEP_SUMMARY` when it is set.
# Requires jq and the toolchain (`rustup toolchain install nightly --profile
# minimal`).
set -euo pipefail
cd "$(dirname "$0")/.."

toolchain=${1:-nightly}
if [ "$#" -gt 0 ]; then
  shift
fi
requested=" $* "

meta=$(mktemp "${TMPDIR:-/tmp}/docs-rs-meta.XXXXXX")
log=$(mktemp "${TMPDIR:-/tmp}/docs-rs-log.XXXXXX")
trap 'rm -f "$meta" "$log"' EXIT

version=$(rustc +"$toolchain" --version)
cargo metadata --no-deps --format-version 1 > "$meta"

# The keys of a docs.rs table this script applies.
applied='["features", "all-features", "no-default-features", "rustdoc-args"]'

# `publish` is null when unrestricted and a registry list otherwise, `[]`
# for `publish = false`.
crates=$(jq -r '
  .packages[]
  | select(.publish == null or any(.publish[]; . == "crates-io"))
  | select([.targets[].kind[]] | any(. == "lib" or . == "rlib"))
  | .name' "$meta")
if [ -z "$crates" ]; then
  echo "error: no published library crate; this checks nothing" >&2
  exit 1
fi
for name in $requested; do
  if ! printf '%s\n' "$crates" | grep -qx "$name"; then
    echo "error: $name is not a published library crate" >&2
    exit 1
  fi
done

# Refuse a table this script would build differently from docs.rs, before
# building anything.
for name in $crates; do
  refused=$(jq -r --arg name "$name" --argjson applied "$applied" '
    .packages[] | select(.name == $name)
    | (.metadata.docs.rs // {}) | keys - $applied | join(", ")' "$meta")
  if [ -n "$refused" ]; then
    echo "error: $name's [package.metadata.docs.rs] sets $refused, which $0 does not apply" >&2
    exit 1
  fi
done

esc=$(printf '\033')
rows=""
failed=""
for name in $crates; do
  case $requested in
    "  " | *" $name "*) ;;
    *) continue ;;
  esac

  dir=$(jq -r --arg name "$name" '
    .workspace_root as $root
    | .packages[] | select(.name == $name)
    | .manifest_path | ltrimstr($root + "/") | rtrimstr("/Cargo.toml")' "$meta")
  features=$(jq -r --arg name "$name" '
    .packages[] | select(.name == $name) | (.metadata.docs.rs // {})
    | [ (if (.features // []) != [] then "--features=" + (.features | join(",")) else empty end),
        (if ."all-features" then "--all-features" else empty end),
        (if ."no-default-features" then "--no-default-features" else empty end) ]
    | join(" ")' "$meta")
  args=$(jq -r --arg name "$name" '
    .packages[] | select(.name == $name)
    | (.metadata.docs.rs // {})."rustdoc-args" // [] | .[]' "$meta")

  flags="--cfg docsrs"
  while IFS= read -r arg; do
    if [ -z "$arg" ]; then
      continue
    fi
    case $arg in
      *[[:space:]]*)
        echo "error: $name's rustdoc-args entry '$arg' holds whitespace, which RUSTDOCFLAGS would split" >&2
        exit 1
        ;;
    esac
    if [ -f "$dir/$arg" ]; then
      arg="$dir/$arg"
    fi
    flags="$flags $arg"
  done <<< "$args"

  echo
  echo "== $name: RUSTDOCFLAGS='$flags' cargo +$toolchain rustdoc -p $name --lib -Zrustdoc-scrape-examples $features"
  # `$features` is unquoted on purpose: it is zero or more flags.
  if DOCS_RS=1 RUSTDOCFLAGS="$flags" \
    cargo +"$toolchain" rustdoc -p "$name" --lib -Zrustdoc-scrape-examples $features \
    2>&1 | tee "$log"; then
    result="ok"
  else
    result="**failed**"
    failed="$failed $name"
  fi
  # cargo's own count for the library's doc build; colour codes stripped.
  warnings=$(sed -e "s/${esc}\[[0-9;]*m//g" "$log" \
    | sed -n 's/^warning: `[^`]*` (lib doc) generated \([0-9][0-9]*\) warning.*/\1/p')
  warnings=${warnings:-0}
  if [ "$result" = ok ] && [ "$warnings" -gt 0 ]; then
    result="**warned**"
    failed="$failed $name"
  fi
  rows="$rows| \`$name\` | ${features:-default} | $result | $warnings |
"
done

summary="Documented as docs.rs does, on \`$version\`.

| crate | features | result | warnings |
|---|---|---|---|
$rows"
echo
printf '%s' "$summary"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '%s' "$summary" >> "$GITHUB_STEP_SUMMARY"
fi

if [ -n "$failed" ]; then
  echo "error: rustdoc failed or warned for$failed on $version; docs.rs would fail to build them or publish the defects warned of" >&2
  exit 1
fi
