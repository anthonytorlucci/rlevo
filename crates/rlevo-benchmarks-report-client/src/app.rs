//! Top-level Leptos application.
//!
//! Owns the page layout: manifest panel, warnings, episode table, and
//! the per-family [`crate::playback::playback_panel`] that provides the
//! scrubber + play/pause + speed controls. Family dispatch happens off
//! `manifest.env_family`.

use leptos::prelude::*;

use crate::charts::{convergence_panel_view, population_panel_view};
use crate::inline_data::{
    EpisodeMeta, InlineError, WarningEntry, read_all_episode_records,
    read_all_population_samples, read_episode_index, read_episode_record, read_manifest,
    read_warnings,
};
use crate::playback::playback_panel;
use crate::wire::{EnvFamily, EpisodeRecord, RunManifest};

/// Root Leptos component that assembles the full report page.
///
/// Reads the run manifest, episode index, and warnings from inline data,
/// then renders the header, episode table, convergence and population charts,
/// and the per-episode playback panel.  Episode selection is managed with a
/// reactive signal; the playback panel re-renders whenever the selection
/// changes.
///
/// # Examples
///
/// ```no_run
/// # use leptos::mount::mount_to_body;
/// # use rlevo_benchmarks_report_client::app::App;
/// mount_to_body(App);
/// ```
#[component]
pub fn App() -> impl IntoView {
    let manifest = read_manifest();
    let family: Option<EnvFamily> = manifest.as_ref().ok().map(|m| m.env_family);
    let episodes = read_episode_index().unwrap_or_default();
    let warnings = read_warnings().unwrap_or_default();

    let (selected, set_selected) = signal::<Option<String>>(None);
    let initial = episodes.first().map(|m| m.script_id.clone());
    if initial.is_some() {
        set_selected.set(initial);
    }

    let selected_meta = {
        let episodes = episodes.clone();
        Memo::new(move |_| {
            let id = selected.get()?;
            episodes.iter().find(|m| m.script_id == id).cloned()
        })
    };

    let selected_record = Memo::new(move |_| -> Option<Result<EpisodeRecord, String>> {
        let id = selected.get()?;
        Some(read_episode_record(&id).map_err(|e| e.to_string()))
    });

    view! {
        <header>
            {match manifest {
                Ok(m) => view! {
                    {manifest_view(&m)}
                    {checkpoints_view(&m)}
                }.into_any(),
                Err(e) => view! { <p class="rlevo-error">"manifest load error: " {e.to_string()}</p> }.into_any(),
            }}
            {warnings_view(warnings)}
        </header>
        <main>
            <h2>"Episodes"</h2>
            {episode_table(episodes, selected, set_selected)}
            {convergence_panel_view(read_all_episode_records(), family.unwrap_or(EnvFamily::Classic))}
            {population_panel_view(read_all_population_samples())}
            <h2>"Selected episode"</h2>
            <div class="rlevo-detail">
                {move || match (selected_meta.get(), selected_record.get()) {
                    (Some(meta), Some(Ok(rec))) => {
                        let fam = family.unwrap_or(EnvFamily::Classic);
                        let episode_no = meta.episode;
                        view! {
                            <h3>{"Episode "}{episode_no.to_string()}</h3>
                            {playback_panel(fam, rec)}
                        }.into_any()
                    },
                    (_, Some(Err(e))) => view! { <p class="rlevo-error">"decode error: " {e}</p> }.into_any(),
                    _ => view! { <p>"no episode selected"</p> }.into_any(),
                }}
            </div>
        </main>
    }
}

/// Produces one `<dt>`/`<dd>` pair for use inside the manifest definition list.
///
/// Extracted as a helper so each field in [`manifest_view`] stays a single
/// call rather than an inline `view!` block.
fn meta_row(label: &str, value: String) -> AnyView {
    let label = label.to_string();
    view! { <dt>{label}</dt><dd>{value}</dd> }.into_any()
}

/// Renders the manifest header block as a `<h1>` and definition list.
///
/// Always shows core fields; the v6 run-provenance fields (algorithm, crate /
/// toolchain / backend versions, git commit, device, seed count, success
/// threshold) render only when present, so older records degrade cleanly.
fn manifest_view(m: &RunManifest) -> impl IntoView {
    let run_id = m.run_id.0.clone();

    let mut rows: Vec<AnyView> = vec![
        meta_row("run id", run_id.clone()),
        meta_row("env family", format!("{:?}", m.env_family)),
        meta_row("seed", m.seed.to_string()),
        meta_row("format version", m.format_version.to_string()),
        meta_row("frame stride", m.frame_stride.to_string()),
        meta_row("episodes", m.episode_count.to_string()),
    ];

    // v6 run provenance — each row appears only when the field is populated.
    if let Some(algo) = &m.algorithm {
        rows.push(meta_row("algorithm", algo.clone()));
    }
    if let Some(n) = m.num_seeds {
        rows.push(meta_row("seeds", n.to_string()));
    }
    if let Some(t) = m.success_threshold {
        rows.push(meta_row("success threshold", format_f64(t)));
    }
    if let Some(d) = &m.device {
        rows.push(meta_row("device", d.clone()));
    }
    if let Some(v) = &m.rlevo_version {
        rows.push(meta_row("rlevo version", v.clone()));
    }
    if let Some(v) = &m.burn_version {
        rows.push(meta_row("burn version", v.clone()));
    }
    if let Some(v) = &m.rustc_version {
        rows.push(meta_row("rustc", v.clone()));
    }
    if let Some(p) = &m.platform {
        rows.push(meta_row("platform", p.clone()));
    }
    if let Some(commit) = &m.git_commit {
        let dirty = if m.git_dirty == Some(true) { " (dirty)" } else { "" };
        rows.push(meta_row("git", format!("{commit}{dirty}")));
    }

    view! {
        <h1>"rlevo report — " {run_id}</h1>
        <dl class="rlevo-meta">{rows}</dl>
    }
}

/// Formats an `f64` for human-readable display, omitting the `.0` suffix on
/// exact integers (e.g. `500` rather than `500.0`).
///
/// Non-finite values (`NaN`, `±∞`) pass through `format!("{v}")` unchanged and
/// are not treated specially; callers are responsible for validating inputs
/// before display.
fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        format!("{v}")
    }
}

/// Renders the learner-checkpoint table from the manifest, or nothing when no
/// checkpoints were registered (EA runs and un-wired RL). Added for v6.
fn checkpoints_view(m: &RunManifest) -> AnyView {
    if m.checkpoints.is_empty() {
        return ().into_any();
    }
    let rows: Vec<AnyView> = m
        .checkpoints
        .iter()
        .map(|c| {
            let kind = format!("{:?}", c.kind);
            let fmt = format!("{:?}", c.format);
            let step = c.step.to_string();
            let path = c.path.clone();
            let metric = c.metric.map_or_else(|| "—".to_string(), format_f64);
            let digest = c.digest.map_or_else(
                || "—".to_string(),
                |d| {
                    // Short 8-hex-char prefix of the 128-bit content digest.
                    use std::fmt::Write as _;
                    d[..4].iter().fold(String::with_capacity(8), |mut s, byte| {
                        let _ = write!(s, "{byte:02x}");
                        s
                    })
                },
            );
            view! {
                <tr>
                    <td>{step}</td>
                    <td>{kind}</td>
                    <td>{fmt}</td>
                    <td class="rlevo-ckpt-path">{path}</td>
                    <td>{metric}</td>
                    <td><code>{digest}</code></td>
                </tr>
            }
            .into_any()
        })
        .collect();
    view! {
        <section class="rlevo-checkpoints">
            <h2>"Checkpoints"</h2>
            <table class="rlevo-ckpt-table">
                <thead>
                    <tr>
                        <th>"step"</th><th>"kind"</th><th>"format"</th>
                        <th>"path"</th><th>"metric"</th><th>"digest"</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </section>
    }
    .into_any()
}

/// Renders the run warnings as a `<section>` block, or an empty `<span>` when
/// there are none.
fn warnings_view(warnings: Vec<WarningEntry>) -> impl IntoView {
    if warnings.is_empty() {
        return view! { <span></span> }.into_any();
    }
    let items: Vec<_> = warnings
        .into_iter()
        .map(|w| {
            let summary = match (w.manifest_count, w.found_count) {
                (Some(m), Some(f)) => format!("{}: manifest_count={m}, found_count={f}", w.kind),
                _ => w.kind.clone(),
            };
            view! { <li>{summary}</li> }
        })
        .collect();
    view! {
        <section class="rlevo-warnings" role="status">
            <h2>"Warnings"</h2>
            <ul>{items}</ul>
        </section>
    }
    .into_any()
}

/// Renders the episode list as an HTML `<table>`.
///
/// Clicking a row updates `set_selected` with that episode's `script_id`.
fn episode_table(
    episodes: Vec<EpisodeMeta>,
    selected: ReadSignal<Option<String>>,
    set_selected: WriteSignal<Option<String>>,
) -> impl IntoView {
    let rows: Vec<_> = episodes
        .into_iter()
        .map(|m| episode_row(m, selected, set_selected))
        .collect();
    view! {
        <table class="rlevo-episodes">
            <thead>
                <tr>
                    <th>"episode"</th>
                    <th>"kind"</th>
                    <th>"frames"</th>
                    <th>"length"</th>
                    <th>"return"</th>
                    <th>"payload id"</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    }
}

/// Renders a single `<tr>` for one episode; adds the `selected` CSS class reactively.
fn episode_row(
    m: EpisodeMeta,
    selected: ReadSignal<Option<String>>,
    set_selected: WriteSignal<Option<String>>,
) -> impl IntoView {
    let script_id = m.script_id.clone();
    let script_id_for_click = script_id.clone();
    let script_id_for_class = script_id.clone();
    let reward_text = format!("{:.3}", m.episode_reward);
    let episode_str = m.episode.to_string();
    let frame_count_str = m.frame_count.to_string();
    let length_str = m.length.to_string();
    // Eval episodes get a distinct badge class so the split survives a B/W
    // screenshot (text label + class, not colour alone) per the a11y contract.
    let is_eval = m.kind.eq_ignore_ascii_case("evaluation");
    let kind_label = if is_eval { "eval" } else { "train" };
    let kind_class = if is_eval {
        "rlevo-kind rlevo-kind-eval"
    } else {
        "rlevo-kind rlevo-kind-train"
    };
    view! {
        <tr
            class:selected=move || selected.get().as_deref() == Some(script_id_for_class.as_str())
            on:click=move |_| set_selected.set(Some(script_id_for_click.clone()))
        >
            <td class="numeric">{episode_str}</td>
            <td><span class=kind_class>{kind_label}</span></td>
            <td class="numeric">{frame_count_str}</td>
            <td class="numeric">{length_str}</td>
            <td class="numeric">{reward_text}</td>
            <td><code>{script_id}</code></td>
        </tr>
    }
}

// Keep the type alive in the module so `read_episode_record`'s error
// type stays public via `inline_data::InlineError`.
#[allow(dead_code)]
fn _typecheck_inline_error(_: InlineError) {}
