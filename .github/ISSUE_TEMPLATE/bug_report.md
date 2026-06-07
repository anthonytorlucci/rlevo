---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Affected crate(s)**
Which `rlevo-*` crate(s) are involved? (e.g. `rlevo-core`, `rlevo-environments`, `rlevo-evolution`)

**To Reproduce**
Minimal reproducible example or cargo invocation:

```rust
// paste code here
```

```bash
# or paste the cargo command here, e.g.:
# cargo test -p rlevo-core -- my_test --nocapture
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Actual behavior / error output**
Paste the full compiler error, panic message, or incorrect output verbatim:

```
// paste error here
```

**Environment**
- OS and GPU model (relevant for `wgpu` backend): [e.g. macOS 15.2, Apple M2]
- `rustc --version`: 
- `cargo --version`: 
- Burn backend in use: [e.g. `wgpu`, `ndarray`, `tch`]
- Workspace features enabled: [e.g. `--features tui,metrics`]

**Additional context**
Add any other context about the problem here.
