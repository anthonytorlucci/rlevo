//! Static-HTML emitter — turns a [`RecordedRun`] into a single
//! self-contained `index.html` file with the manifest + every episode
//! payload inlined as `<script>` blocks.
//!
//! M5 ships the data-transport contract and a placeholder body. The
//! Leptos/WASM client that consumes these inlined blocks lands in a
//! follow-on milestone; per-family playback adapters and convergence
//! plots land in M6+ (umbrella spec §9).
//!
//! Wire layout produced inside the HTML:
//!
//! ```text
//! <script type="application/json" id="rlevo-manifest">{...}</script>
//! <script type="application/json" id="rlevo-warnings">[...]</script>
//! <script type="application/json" id="rlevo-episode-index">[...]</script>
//! <script type="application/octet-stream" id="rlevo-episode-000000">BASE64...</script>
//! ...
//! ```
//!
//! All payloads are inlined; the file has no external references.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::Serialize;

use super::replay::{EpisodeIndex, OpenWarning, RecordedRun};

/// Tunables for the emitter. Defaults match the umbrella spec's "single
/// self-contained file" output (§9): no external assets, every payload
/// inlined.
#[derive(Debug, Clone)]
pub struct EmitConfig {
    /// Soft upper bound on the emitted file size in bytes. Emission
    /// still completes if exceeded, but the outcome carries a
    /// `size_warning` flag so the caller can surface a downsample hint.
    /// Default mirrors `rollout-and-replay` §5: 10 MB.
    pub size_warn_bytes: u64,
    /// Optional title rendered in the placeholder header. `None`
    /// defaults to the manifest's `run_id`.
    pub title: Option<String>,
}

impl Default for EmitConfig {
    fn default() -> Self {
        Self {
            size_warn_bytes: 10 * 1024 * 1024,
            title: None,
        }
    }
}

/// Outcome metadata reported by [`emit_static_html`].
#[derive(Debug, Clone)]
pub struct EmitOutcome {
    pub episode_count: u32,
    pub bytes_written: u64,
    pub size_warning: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error("io error writing output: {0}")]
    Io(#[source] io::Error),
    #[error("could not encode manifest as JSON: {0}")]
    ManifestJson(#[source] serde_json::Error),
    #[error("could not re-encode episode {episode} for inlining: {source}")]
    EpisodeReencode {
        episode: u32,
        #[source]
        source: io::Error,
    },
}

#[derive(Serialize)]
struct EpisodeMeta {
    episode: u32,
    frame_count: u32,
    episode_reward: f64,
    length: u32,
    script_id: String,
}

#[derive(Serialize)]
struct WarningJson<'a> {
    kind: &'static str,
    detail: Option<&'a str>,
    manifest_count: Option<u32>,
    found_count: Option<u32>,
}

/// Emit a single-file HTML report at `out_path`.
///
/// # Errors
///
/// Returns [`EmitError::Io`] if the output file cannot be written,
/// [`EmitError::ManifestJson`] if the manifest cannot be JSON-encoded,
/// or [`EmitError::EpisodeReencode`] if an on-disk episode file cannot
/// be re-read.
pub fn emit_static_html(
    run: &RecordedRun,
    out_path: &Path,
    config: &EmitConfig,
) -> Result<EmitOutcome, EmitError> {
    let manifest_json =
        serde_json::to_string(run.manifest()).map_err(EmitError::ManifestJson)?;
    let warnings_json = warnings_to_json(run.warnings());

    let mut body = String::with_capacity(64 * 1024);
    write_html_head(&mut body, run, config);
    write_placeholder_body(&mut body, run, config);

    push_script_json(&mut body, "rlevo-manifest", &manifest_json);
    push_script_json(&mut body, "rlevo-warnings", &warnings_json);

    let mut episode_metas = Vec::with_capacity(run.episodes().len());
    for ep in run.episodes() {
        let script_id = format!("rlevo-episode-{:06}", ep.episode);
        let bytes = fs::read(&ep.path).map_err(|e| EmitError::EpisodeReencode {
            episode: ep.episode,
            source: e,
        })?;
        let encoded = B64.encode(&bytes);
        push_script_base64(&mut body, &script_id, &encoded);
        episode_metas.push(EpisodeMeta {
            episode: ep.episode,
            frame_count: ep.frame_count,
            episode_reward: ep.episode_reward,
            length: ep.length,
            script_id,
        });
    }
    let index_json = serde_json::to_string(&episode_metas).map_err(EmitError::ManifestJson)?;
    push_script_json(&mut body, "rlevo-episode-index", &index_json);

    body.push_str("</body>\n</html>\n");

    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent).map_err(EmitError::Io)?;
    }
    let tmp = tmp_path(out_path);
    {
        let mut f = fs::File::create(&tmp).map_err(EmitError::Io)?;
        f.write_all(body.as_bytes()).map_err(EmitError::Io)?;
        f.sync_all().map_err(EmitError::Io)?;
    }
    fs::rename(&tmp, out_path).map_err(EmitError::Io)?;

    let bytes_written = body.len() as u64;
    Ok(EmitOutcome {
        episode_count: u32::try_from(run.episodes().len()).unwrap_or(u32::MAX),
        bytes_written,
        size_warning: bytes_written > config.size_warn_bytes,
    })
}

fn tmp_path(out_path: &Path) -> PathBuf {
    let mut name = out_path
        .file_name()
        .map_or_else(|| "index.html".into(), |n| n.to_string_lossy().into_owned());
    name.insert(0, '.');
    name.push_str(".tmp");
    out_path
        .parent()
        .map_or_else(|| PathBuf::from(&name), |p| p.join(&name))
}

fn write_html_head(out: &mut String, run: &RecordedRun, config: &EmitConfig) {
    let title = config
        .title
        .clone()
        .unwrap_or_else(|| run.manifest().run_id.0.clone());
    let escaped_title = escape_html(&title);
    out.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str(
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n",
    );
    out.push_str("<meta name=\"generator\" content=\"rlevo-benchmarks/report (M5)\">\n");
    let _ = writeln!(out, "<title>rlevo report — {escaped_title}</title>");
    out.push_str("<style>\n");
    out.push_str(EMBEDDED_CSS);
    out.push_str("</style>\n");
    out.push_str("</head>\n<body>\n");
}

fn write_placeholder_body(out: &mut String, run: &RecordedRun, config: &EmitConfig) {
    let m = run.manifest();
    let title = config
        .title
        .clone()
        .unwrap_or_else(|| m.run_id.0.clone());
    out.push_str("<header class=\"rlevo-header\">\n");
    let _ = writeln!(out, "<h1>rlevo report &mdash; {}</h1>", escape_html(&title));
    out.push_str("<dl class=\"rlevo-meta\">\n");
    let _ = writeln!(
        out,
        "<dt>run id</dt><dd>{}</dd>",
        escape_html(&m.run_id.0)
    );
    let _ = writeln!(out, "<dt>env family</dt><dd>{:?}</dd>", m.env_family);
    let _ = writeln!(out, "<dt>seed</dt><dd>{}</dd>", m.seed);
    let _ = writeln!(out, "<dt>format version</dt><dd>{}</dd>", m.format_version);
    let _ = writeln!(out, "<dt>frame stride</dt><dd>{}</dd>", m.frame_stride);
    let _ = writeln!(out, "<dt>episodes</dt><dd>{}</dd>", run.episodes().len());
    out.push_str("</dl>\n");
    if !run.warnings().is_empty() {
        out.push_str("<section class=\"rlevo-warnings\" role=\"status\">\n<h2>Warnings</h2>\n<ul>\n");
        for w in run.warnings() {
            let _ = writeln!(out, "<li>{}</li>", escape_html(&format!("{w:?}")));
        }
        out.push_str("</ul>\n</section>\n");
    }
    out.push_str("</header>\n");
    out.push_str("<main class=\"rlevo-main\">\n");
    out.push_str(
        "<p>The per-family playback adapters, convergence plots, and timeline scrubber\n\
         land in Milestones 6&ndash;8. The Leptos/WASM client that consumes the inlined\n\
         payloads ships in M5.1. The data is already embedded in this file &mdash; this\n\
         is the skeleton.</p>\n",
    );
    out.push_str("<h2>Episodes</h2>\n<table class=\"rlevo-episodes\">\n<thead><tr>");
    out.push_str(
        "<th>episode</th><th>frames</th><th>length</th><th>return</th><th>payload id</th>",
    );
    out.push_str("</tr></thead>\n<tbody>\n");
    for ep in run.episodes() {
        let _ = writeln!(
            out,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td><code>rlevo-episode-{:06}</code></td></tr>",
            ep.episode, ep.frame_count, ep.length, ep.episode_reward, ep.episode
        );
    }
    out.push_str("</tbody>\n</table>\n");
    out.push_str("</main>\n");
}

fn push_script_json(out: &mut String, id: &str, body: &str) {
    let _ = writeln!(out, "<script type=\"application/json\" id=\"{id}\">");
    out.push_str(&escape_script(body));
    out.push_str("\n</script>\n");
}

fn push_script_base64(out: &mut String, id: &str, body: &str) {
    let _ = writeln!(out, "<script type=\"application/octet-stream\" id=\"{id}\">");
    out.push_str(body);
    out.push_str("\n</script>\n");
}

fn warnings_to_json(warnings: &[OpenWarning]) -> String {
    let serialised: Vec<WarningJson> = warnings
        .iter()
        .map(|w| match w {
            OpenWarning::ManifestSynthesised => WarningJson {
                kind: "ManifestSynthesised",
                detail: None,
                manifest_count: None,
                found_count: None,
            },
            OpenWarning::EpisodeCountMismatch {
                manifest_count,
                found_count,
            } => WarningJson {
                kind: "EpisodeCountMismatch",
                detail: None,
                manifest_count: Some(*manifest_count),
                found_count: Some(*found_count),
            },
        })
        .collect();
    serde_json::to_string(&serialised).unwrap_or_else(|_| "[]".into())
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Escape only the characters that could prematurely terminate a
/// `<script>` block; the JSON we embed is already well-formed.
fn escape_script(s: &str) -> String {
    s.replace("</", "<\\/")
}

#[allow(unused)]
fn _meta_count(run: &RecordedRun) -> usize {
    run.episodes().len()
}

#[allow(unused)]
fn _index_path(ep: &EpisodeIndex) -> &Path {
    ep.path.as_path()
}

const EMBEDDED_CSS: &str = r"
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 64rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
.rlevo-header h1 { margin-bottom: 0.25rem; }
.rlevo-meta { display: grid; grid-template-columns: max-content 1fr; column-gap: 1rem; row-gap: 0.25rem; margin: 0; }
.rlevo-meta dt { font-weight: 600; }
.rlevo-warnings { border: 1px solid #d4a017; padding: 0.5rem 1rem; margin: 1rem 0; border-radius: 0.5rem; background: #fff8e1; color: #4a3300; }
@media (prefers-color-scheme: dark) {
    .rlevo-warnings { background: #3b2f0a; color: #ffe8a8; border-color: #b88a0d; }
}
.rlevo-episodes { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
.rlevo-episodes th, .rlevo-episodes td { padding: 0.25rem 0.5rem; border-bottom: 1px solid rgba(127,127,127,0.4); text-align: left; }
.rlevo-episodes td:nth-child(4) { font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, monospace; }
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{
        EnvFamily, FamilyPayload, FrameRecord, MetricSample, RecordSink, RecordWriter,
        RecordingConfig, RunId,
    };
    use tempfile::tempdir;

    fn frame(step: u32) -> FrameRecord {
        FrameRecord {
            step,
            action: vec![0],
            reward: 1.0,
            ascii: Some(format!("s{step}")),
            styled: None,
            family_payload: FamilyPayload::Ascii,
        }
    }

    fn write_run(root: &Path) -> PathBuf {
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 99,
            run_id: Some(RunId("emit-test".into())),
        };
        let mut w = RecordWriter::open(root, cfg).unwrap();
        let run_dir = w.run_dir().to_path_buf();
        for ep in 0..2u32 {
            w.on_episode_start(ep);
            for s in 0..3u32 {
                w.on_frame(frame(s));
            }
            w.on_metric(MetricSample {
                step: ep,
                name: "policy_loss".into(),
                value: 0.1,
            });
            w.on_episode_end(3.0, 3);
        }
        w.on_run_end(w.manifest_template());
        run_dir
    }

    #[test]
    fn emit_round_trips_manifest_and_payloads() {
        let dir = tempdir().unwrap();
        let run_dir = write_run(dir.path());
        let run = RecordedRun::open(&run_dir).unwrap();

        let out = dir.path().join("index.html");
        let outcome = emit_static_html(&run, &out, &EmitConfig::default()).unwrap();
        assert_eq!(outcome.episode_count, 2);
        assert!(out.exists());
        let body = fs::read_to_string(&out).unwrap();

        // Manifest JSON block is present with the seed visible.
        assert!(
            body.contains("id=\"rlevo-manifest\""),
            "manifest script tag missing"
        );
        assert!(body.contains("\"seed\":99"), "seed not inlined");
        // Both episode payload blocks are present.
        assert!(body.contains("id=\"rlevo-episode-000000\""));
        assert!(body.contains("id=\"rlevo-episode-000001\""));
        // Episode-index lists both episodes by id.
        assert!(body.contains("\"rlevo-episode-000000\""));
        assert!(body.contains("\"rlevo-episode-000001\""));
        // No external resource references.
        assert!(!body.contains("<link rel=\"stylesheet\""));
        assert!(!body.contains("src=\"http"));
    }

    #[test]
    fn emit_payload_is_valid_base64_of_episode_file() {
        let dir = tempdir().unwrap();
        let run_dir = write_run(dir.path());
        let run = RecordedRun::open(&run_dir).unwrap();
        let out = dir.path().join("index.html");
        emit_static_html(&run, &out, &EmitConfig::default()).unwrap();
        let body = fs::read_to_string(&out).unwrap();

        let marker = "id=\"rlevo-episode-000000\">\n";
        let start = body.find(marker).unwrap() + marker.len();
        let rest = &body[start..];
        let end = rest.find("\n</script>").unwrap();
        let b64 = &rest[..end];

        let decoded = B64.decode(b64.trim()).unwrap();
        let from_disk = fs::read(run_dir.join("episode_000000.rec")).unwrap();
        assert_eq!(decoded, from_disk);
    }

    #[test]
    fn emit_size_warning_fires_when_budget_exceeded() {
        let dir = tempdir().unwrap();
        let run_dir = write_run(dir.path());
        let run = RecordedRun::open(&run_dir).unwrap();
        let out = dir.path().join("index.html");
        let cfg = EmitConfig {
            size_warn_bytes: 1,
            title: None,
        };
        let outcome = emit_static_html(&run, &out, &cfg).unwrap();
        assert!(outcome.size_warning, "expected size warning at 1-byte cap");
    }

    #[test]
    fn emit_includes_warnings_block_when_manifest_is_synthetic() {
        let dir = tempdir().unwrap();
        let run_dir = write_run(dir.path());
        fs::remove_file(run_dir.join("run.toml")).unwrap();
        let run = RecordedRun::open(&run_dir).unwrap();
        assert!(!run.warnings().is_empty());

        let out = dir.path().join("index.html");
        emit_static_html(&run, &out, &EmitConfig::default()).unwrap();
        let body = fs::read_to_string(&out).unwrap();
        assert!(body.contains("ManifestSynthesised"));
    }

    #[test]
    fn emit_escapes_closing_script_in_inlined_json() {
        // Ensure our </ -> <\/ escape is applied: forge a run_id with
        // "</script>" inside and check it survives the round trip.
        let dir = tempdir().unwrap();
        let cfg = RecordingConfig {
            frame_stride: Some(1),
            env_family: EnvFamily::Classic,
            seed: 1,
            run_id: Some(RunId("safe</script>id".into())),
        };
        let mut w = RecordWriter::open(dir.path(), cfg).unwrap();
        let run_dir = w.run_dir().to_path_buf();
        w.on_episode_start(0);
        w.on_frame(frame(0));
        w.on_episode_end(1.0, 1);
        w.on_run_end(w.manifest_template());

        let run = RecordedRun::open(&run_dir).unwrap();
        let out = dir.path().join("index.html");
        emit_static_html(&run, &out, &EmitConfig::default()).unwrap();
        let body = fs::read_to_string(&out).unwrap();
        // Should never contain a raw </script> inside the JSON payload.
        let manifest_marker = "id=\"rlevo-manifest\">";
        let start = body.find(manifest_marker).unwrap() + manifest_marker.len();
        let end = start
            + body[start..]
                .find("\n</script>")
                .expect("manifest script terminator");
        let chunk = &body[start..end];
        assert!(!chunk.contains("</script>"));
        assert!(chunk.contains("<\\/script>"));
    }
}
