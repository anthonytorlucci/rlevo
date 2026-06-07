<!--
Before opening this PR, confirm you have:
  1. Opened (or linked) an issue discussing the change — required for anything beyond a one-line typo fix.
  2. Verified all three commands pass locally:
       cargo build --workspace
       cargo test --workspace
       cargo clippy --all-targets --all-features
  3. Confirmed the change is within the current contribution scope (examples, bug fixes,
     documentation corrections). See CONTRIBUTING.md § "What Contributions Are Welcome Right Now".

Draft PRs will not be reviewed. Mark as Ready for Review when the above is complete.
-->

## Summary

<!-- One paragraph: what changed and why. Be specific — e.g. "Fixes off-by-one in CartPole
termination condition that caused episodes to run one step past the pole-angle threshold." -->


## Change Type

<!-- Check all that apply. Remove lines that don't apply. -->

- [ ] Bug fix (non-breaking, includes regression test)
- [ ] New example (self-contained, demonstrates existing API surface)
- [ ] Documentation correction (typo, broken link, misleading doc comment)
- [ ] Other — _describe and justify scope below_

<!--
NOTE: New environments, new RL/evolutionary algorithms, refactors, new dependencies, and
CI/CD changes are out of scope during the alpha phase. If your change falls into one of
those categories, close this PR and open an issue to discuss it first.
-->

## Linked Issue

<!-- Paste the issue URL. If no issue exists and this is more than a typo fix, stop here,
open one, and return when there is prior discussion on record. -->

Closes #


## What Was Changed

<!-- Enumerate the files/modules touched and the specific modifications made.
     Concrete is better than vague: prefer "added `impl Reset for CartPole`" over "fixed the env". -->

-
-


## How It Was Tested

<!-- Describe the exact commands run and their output, or paste the relevant test names.
     For bug fixes, identify the regression test added and the before/after behavior. -->

```
cargo test --workspace
cargo clippy --all-targets --all-features
```

<!-- Add environment details if hardware-backend-specific (CPU / WGPU / CUDA, OS, Rust toolchain). -->

- Rust toolchain: `rustup show` output or channel (e.g. `stable 1.xx.x`)
- Burn backend tested:
- OS:


## AI-Assisted Content

<!-- Per CONTRIBUTING.md § "Change Ownership": you are responsible for every line in this PR
     regardless of origin. If any part of this change was generated or significantly assisted
     by an AI tool, declare it here. You must be able to explain and defend the code on request. -->

- [ ] No AI-generated content in this PR.
- [ ] AI tooling was used. Tool(s): ________________. Scope: ________________.


## Checklist

- [ ] `cargo build --workspace` passes with no errors.
- [ ] `cargo test --workspace` passes with no new test failures.
- [ ] `cargo clippy --all-targets --all-features` passes with no warnings (treated as errors).
- [ ] Public API additions or changes include doc comments explaining *what* and *why*.
- [ ] Bug fixes include a regression test that fails on the previous code and passes on this PR.
- [ ] This PR addresses a single concern. (If it grew into something larger, it has been split.)
- [ ] This PR is marked **Ready for Review** (not Draft).
