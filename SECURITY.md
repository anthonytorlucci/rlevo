# Security Policy

## Supported Versions

`rlevo` is currently in **alpha** (`v0.x`). Only the latest published version receives security fixes. No backports are made to earlier releases during this stage.

| Version | Supported |
|---------|-----------|
| 0.1.x (latest) | Yes |
| < 0.1.0 | No |

Once the library reaches `v1.0.0`, this table will be updated to reflect a stable long-term support policy.

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Preferred: GitHub Private Vulnerability Reporting

Use GitHub's built-in private disclosure mechanism:

1. Navigate to the [Security tab](https://github.com/anthonytorlucci/rlevo/security) of this repository.
2. Click **"Report a vulnerability"**.
3. Fill in the advisory form with as much detail as possible (see below).

This keeps the disclosure private between you and the maintainers until a fix is ready.

---

## What to Include in a Report

A high-quality report helps us triage and fix issues faster. Please include:

- A description of the vulnerability and the potential impact
- The affected component(s) and version(s)
- Steps to reproduce or a minimal proof-of-concept
- Any suggested mitigations or patches you have already identified

---

## Response Timeline

| Milestone | Target |
|-----------|--------|
| Acknowledgement of report | Within 72 hours |
| Initial triage and severity assessment | Within 7 days |
| Fix or mitigation available | Within 30 days for critical/high severity |
| Public disclosure | After fix is released, coordinated with reporter |

If a fix requires more than 30 days due to complexity, the maintainers will communicate progress to the reporter and agree on an extended timeline before any public disclosure.

---

## Disclosure Policy

We follow a **coordinated disclosure** model:

1. Reporter submits a private disclosure.
2. Maintainers acknowledge, assess severity, and develop a fix.
3. A patched release is prepared and tested.
4. The fix is published and a GitHub Security Advisory is issued.
5. The reporter is credited in the advisory (unless they prefer to remain anonymous).

We ask reporters to refrain from public disclosure until a fix has been released, or until 90 days have passed from the initial report — whichever comes first. If we cannot meet this window, we will coordinate an earlier disclosure date with the reporter.

---

## Scope

The following are considered in-scope security concerns for `rlevo`:

- **Memory safety issues** in `unsafe` blocks within the library
- **Dependency vulnerabilities** in direct or transitive dependencies that affect users of `rlevo` (report these even if they originate upstream — we will coordinate with the upstream project)
- **Numeric correctness issues** with security implications in RL/evolutionary algorithm implementations (e.g., integer overflow, NaN propagation in reward calculations that could cause undefined behavior in downstream systems)
- **Serialization/deserialization vulnerabilities** in saved model state or environment configurations
- **Sandbox escapes** in environment implementations that allow unintended access to host resources

### Out of Scope

The following are generally **not** in scope for this security policy:

- Theoretical attacks with no practical exploit path
- Vulnerabilities in code that users write using `rlevo` as a library (these are the user's responsibility)
- Issues in example code that are clearly illustrative and not intended for production use
- Denial-of-service via computationally expensive training runs (this is expected behavior for an RL library)
- Bugs that do not have a plausible security impact

If you are unsure whether something is in scope, report it anyway — we would rather triage a non-issue than miss a real one.

---

## Security Best Practices for Users

Because `rlevo` is a training infrastructure library, security considerations extend beyond the library itself:

- **Validate reward functions and fitness functions** before training at scale. A misspecified objective can cause an agent to take unintended actions in deployment.
- **Constrain action spaces** when training agents that will interact with real systems.
- **Audit trained policies** for emergent behaviors before distribution or deployment.
- **Pin dependency versions** in production to avoid unexpected changes from transitive updates. Use `cargo audit` to check for known vulnerabilities in your dependency tree.
- **Do not deserialize untrusted model weights** without validation. Deserialization of foreign data is a common attack surface in ML systems.

---

## Acknowledgements

We are grateful to the security researchers and community members who help keep `rlevo` safe. Reporters who submit valid, in-scope vulnerabilities will be credited in the corresponding GitHub Security Advisory unless they request anonymity.
