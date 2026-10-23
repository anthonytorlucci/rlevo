#!/usr/bin/env bash
# Fail if any Rust source asks Burn for its default device.
#
#   tools/no-default-device.sh
#
# On Burn 0.22 `Device::default()` is whichever backend the build compiled
# first. A test build that compiles wgpu therefore gets a wgpu device, and on
# a runner with no GPU adapter a wgpu device aborts the process. Code names
# its device instead: `Device::flex()`, `Device::flex().autodiff()` or a
# named adapter (ADR 0093). A library takes the device it is given (ADR 0080).
#
# Matched: `Device::default()`, and a binding whose name contains `device`
# initialised with `Default::default()`, typed or inferred. A bare
# `Default::default()` passed straight to a device parameter is not matched.
# The workspace allows clippy's `default_trait_access`; run clippy with
# `-W clippy::default_trait_access` to list those, as `Device::default()`.
set -euo pipefail
cd "$(dirname "$0")/.."

pattern='Device::default\(\)|let[[:space:]]+(mut[[:space:]]+)?[A-Za-z0-9_]*device[A-Za-z0-9_]*[[:space:]]*(:[[:space:]]*Device[[:space:]]*)?=[[:space:]]*Default::default\(\)'
hits=$(git grep -nE "$pattern" -- '*.rs' || true)
if [ -n "$hits" ]; then
  printf '%s\n' "$hits"
  echo "error: a default device; name it: Device::flex(), Device::flex().autodiff() or a named adapter (ADR 0093)" >&2
  exit 1
fi
echo "no default device"
