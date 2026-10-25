# The `rlevo-algo` template

Three [`cargo-generate`](https://github.com/cargo-generate/cargo-generate)
templates, one for each thing an author implements against `rlevo`:

| Template | You implement | It runs |
|---|---|---|
| `agent` | `Agent`, an agent that learns | yours, a random agent and the shipped DQN, trained and measured on CartPole |
| `environment` | `Environment`, with its observation and action types | a random agent and the shipped DQN, trained on yours |
| `strategy` | `Strategy`, an evolutionary method | yours and the shipped CMA-ES, on the sphere |

```bash
cargo generate --git https://github.com/anthonytorlucci/rlevo --tag v0.5.0 templates/rlevo-algo
```

`cargo generate` asks which template and what to name the crate. Name one
directly with a longer path, `templates/rlevo-algo/agent`, and the crate
with `--name`.

Each generated crate depends on `rlevo` alone and works as it stands:
`cargo run` trains and measures every row under one configuration through
`rlevo::harness` and prints the rows as a Markdown table, and `cargo test`
runs its tests. The file to replace says, at its top, what the harness
calls and what it expects back.

The tag is the release whose `rlevo` the templates are written against;
generate from it, not from `main`, whose templates may use API that release
does not have. The command names the current release, as every version in
this repository's documentation does. The templates are on `main` and first
ship with the next release.
