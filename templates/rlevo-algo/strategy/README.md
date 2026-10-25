# {{project-name}}

An evolutionary strategy of your own, run on the sphere through
[`rlevo`](https://github.com/anthonytorlucci/rlevo)'s evaluation harness,
beside the shipped CMA-ES.

```bash
cargo run --release
```

prints one row per strategy as a Markdown table: the best cost each found,
averaged over its trials. The sphere's minimum is 0. `cargo run` without
`--release` gives the same table more slowly. `cargo test` runs the
strategy's tests.

## What to change

- **`src/strategy.rs`**: `MyStrategy`, random search that works as it
  stands, with its parameters and state. Replace `ask` and `tell` with your
  method and keep the shape. The file's top comment says which direction
  fitness runs in, why `tell` cleans it, and where the randomness comes
  from.
- **`src/main.rs`**: `config`, the one configuration every row runs under;
  the landscape and its dimension; the generation budget; and the rows.
