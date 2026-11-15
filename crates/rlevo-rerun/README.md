# rlevo-rerun

![Alt Text](https://raw.githubusercontent.com/anthonytorlucci/rlevo/main/rlevo-logo.png)

A [rerun](https://rerun.io) recording sink for the `rlevo` evaluation
harness.

## Status

**Unreleased.** `RerunSink` records a run's metrics: every episode's return
and length, every metric an agent or a generation loop reports, each trial's
identity and build, and an entry for a trial that errored. With `UpdateLayer`
installed it also records per-update statistics, such as PPO's losses. Frames
and a default layout are not written yet.

`rlevo-rerun` is the one crate in the workspace that depends on rerun. The
harness (`rlevo-harness`) reports a run through its `Reporter` trait and
knows nothing about rerun; this crate turns those reports into `.rrd`
recordings that the rerun viewer opens. No rerun type appears in its API.

## Use

```sh
cargo add rlevo rlevo-rerun
```

```rust,ignore
use rlevo_rerun::{RerunConfig, RerunSink, UpdateLayer};

// Optional, once per process: per-update statistics.
tracing_subscriber::registry().with(UpdateLayer::new()).init();

let mut sink = RerunSink::new(RerunConfig::new("runs", "ppo"))?;
let report = evaluator.run_suite(&suite, make_agent, &mut sink);
if let Some(how) = sink.viewer_instructions() {
    println!("{how}");
}
```

A run writes one file per trial into `runs/<run ID>/`, all under one
recording ID, so the viewer shows them as one recording. The run ID is in the
harness's report (`BenchmarkReport::run_id`). One sink records run after run,
each in its own directory. `viewer_instructions` names the viewer version
that wrote the files and the command that opens them.

## Versions

rerun guarantees that a recording opens only across adjacent minor
versions. So this crate has its own release line, separate from the other
`rlevo` crates: it releases when rerun moves a minor, and it pins rerun with
a caret inside one minor.

## License

MIT OR Apache-2.0
