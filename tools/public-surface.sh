#!/usr/bin/env bash
# List the public surface of one workspace crate, one item per line, sorted.
#
# The surface delta a pull request reports is the diff of two listings: the
# base commit's and the branch's.
#
#   tools/public-surface.sh rlevo-core > after.txt
#   tools/public-surface.sh rlevo-environments --no-default-features
#   tools/public-surface.sh target/doc/rlevo_core.json   # an existing JSON file
#
# Feature flags after the crate name go to cargo; the default is --all-features.
#
# Each line is a kind, a tab and a path:
#   trait, struct, enum, variant, function, constant, type_alias, module, ...
#       an item a user can name. Methods and associated items of a public
#       trait are listed where rustdoc records a path for them, which is not
#       every one; the trait's own line is the reliable one
#   method, assoc_const
#       an inherent item on a public type
#   field
#       a public field of a public struct
#   use
#       a re-export, with the file that declares it
#
# The listing is read from rustdoc's JSON output, which is unstable.
# RUSTC_BOOTSTRAP=1 enables it on the pinned toolchain, so the surface is
# taken with the compiler that builds the crate. The script refuses a format
# version other than the one its filter was written for; when the toolchain
# pin moves, re-check the filter and update FORMAT_VERSION. Requires jq.
set -euo pipefail

FORMAT_VERSION=61

target="${1:?usage: public-surface.sh <crate | file.json> [cargo feature flags]}"
shift

if [ "${target##*.}" = "json" ]; then
  json="$target"
else
  if [ "$#" -eq 0 ]; then set -- --all-features; fi
  target_dir=$(cargo metadata --no-deps --format-version 1 | jq -r .target_directory)
  RUSTC_BOOTSTRAP=1 cargo rustdoc --quiet -p "$target" --lib "$@" -- \
    -Z unstable-options --output-format json
  json="$target_dir/doc/$(printf '%s' "$target" | tr - _).json"
fi

found=$(jq -r .format_version "$json")
if [ "$found" != "$FORMAT_VERSION" ]; then
  echo "public-surface.sh: rustdoc JSON format $found, filter written for $FORMAT_VERSION" >&2
  exit 1
fi

listing=$(jq -r '
  .index as $index | .paths as $paths
  | ( $paths | to_entries[] | select(.value.crate_id == 0)
      | "\(.value.kind)\t\(.value.path | join("::"))" ),
    ( $index[] | select(.inner.impl != null and .inner.impl.trait == null)
      | (.inner.impl.for.resolved_path.path // "?") as $type
      | .inner.impl.items[] | $index[tostring]
      | select(. != null and .visibility == "public")
      | "\(if .inner.function != null then "method" else (.inner | keys[0]) end)\t\($type)::\(.name)" ),
    ( $index | to_entries[] | select(.value.inner.struct != null)
      | ($paths[.key].path // [.value.name] | join("::")) as $type
      | .value.inner.struct.kind | objects
      | ((.plain.fields // []) + ((.tuple // []) | map(select(. != null))))[]
      | $index[tostring]
      | select(. != null and .visibility == "public")
      | "field\t\($type).\(.name)" ),
    ( $index[] | select(.inner.use != null and .visibility == "public")
      | "use\t\(.inner.use.source)\(if .inner.use.is_glob then "::*" else "" end) as \(.inner.use.name)\t\(.span.filename)" )
' "$json")
printf '%s\n' "$listing" | sort -u
