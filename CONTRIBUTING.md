# Contributing to rlevo

Thank you for your interest in `rlevo`. This is an early-stage, solo side project — contributions are welcome, but please read this document carefully before opening a pull request. The scope is intentionally narrow right now so the project can grow sustainably.

---

## Project Status

`rlevo` is **alpha software**. Core APIs are still being designed and are subject to breaking changes without notice. The maintainer is a single developer working on this in spare time. Because of that, the capacity to review, merge, and maintain contributions is limited.

This is not a discouragement — it is transparency. A smaller number of high-quality, well-scoped contributions is far more valuable here than a large influx of changes that stall in review.

---

## What Contributions Are Welcome Right Now

To keep the review burden manageable during the alpha phase, contributions are currently scoped to:

- **New examples** — clear, self-contained examples that demonstrate existing functionality; lightweight env/evo/RL examples go in `crates/rlevo/examples/`, heavier viz or record examples go in `crates/rlevo-examples/`
- **Bug fixes** — small, focused fixes with a clear description of the problem and a regression test
- **Documentation corrections** — typos, broken links, misleading doc comments

### Not Yet

The following are out of scope until the core API stabilizes:

- New environments (unless discussed in an issue first)
- New RL algorithms or evolutionary strategies
- Refactors or code style changes
- New dependencies
- CI/CD or workspace configuration changes

If you have an idea that falls outside the current scope, please open an issue to discuss it. That conversation may shape a future milestone.

---

## Before You Open a PR

1. **Open an issue first** for anything beyond a one-line typo fix. Describe what you want to change and why. This avoids wasted effort if the direction does not fit the current roadmap.
2. **Read the codebase.** Run `cargo doc --workspace --no-deps --open` for a navigable API reference. Key traits live in `crates/rlevo-core/src/` (`state.rs`, `action.rs`, `environment.rs`).
3. **Keep it small.** One PR, one concern. If a fix naturally grows into a refactor, stop and discuss.
4. **Verify it builds, tests pass, and formatting is clean.**

```bash
cargo fmt --check
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets --all-features
```

All four must succeed before submitting.

---

## Pull Request Guidelines

- Write a clear title and a short description: what changed, why, and how you tested it.
- Link to the issue (required unless this is a trivial typo fix).
- Convert from draft to ready-for-review only once all four checks pass.
- Do not force-push during active review without a note explaining why.
- PRs with no author response for **14 days** may be closed. You can always reopen them.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <Description>
```

- **Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`
- **Scope**: short crate name of the primary change site (e.g., `rlevo-core`, `rlevo-evolution`); use `workspace` for cross-crate changes; omit for repo-root-only changes
- **Subject**: max 72 characters, imperative mood, no trailing punctuation
- **Body** (optional): explain the *why* when motivation is non-obvious; wrap at 72 characters
- Keep each commit self-contained — the build and tests must pass at every commit

### Change Ownership

You are responsible for every line in your PR — regardless of whether you wrote it yourself, adapted it from another project, or generated it with an AI tool. Be prepared to explain and defend any change during review.

---

## Toolchain

This project uses **Rust edition 2024**. There is no pinned toolchain file — latest stable is expected. Verify with `rustup update stable` before working on the codebase.

---

## Code Quality

- Follow existing code style and Rust idioms.
- Document public APIs with doc comments. Explain *why* for non-obvious logic.
- Bug fixes must include a regression test.
- Prefer clarity. This codebase uses extensive lints — clippy warnings are treated as errors.

---

## Getting Help

Open an issue or start a GitHub Discussion if you are unsure about anything. There is no Discord or chat at this time, but issues are monitored regularly.

---

## License

By submitting a pull request you agree that your contribution is licensed under the same terms as this project.

---

*Thank you for understanding the constraints of a small, early-stage project. Every thoughtful contribution — even a corrected typo — matters here.*
