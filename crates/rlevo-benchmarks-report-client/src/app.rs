//! Top-level Leptos application.
//!
//! M5.1 shipped the data-binding skeleton (manifest, warnings, episode
//! table, static frame dump). M6 replaces the static detail dump with a
//! per-family [`crate::playback::playback_panel`] that owns scrubber +
//! play/pause + speed controls; family dispatch happens off
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
                Ok(m) => manifest_view(&m).into_any(),
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

fn manifest_view(m: &RunManifest) -> impl IntoView {
    let run_id = m.run_id.0.clone();
    let env_family = format!("{:?}", m.env_family);
    let seed = m.seed;
    let format_version = m.format_version;
    let frame_stride = m.frame_stride;
    let episode_count = m.episode_count;
    view! {
        <h1>"rlevo report — " {run_id.clone()}</h1>
        <dl class="rlevo-meta">
            <dt>"run id"</dt><dd>{run_id}</dd>
            <dt>"env family"</dt><dd>{env_family}</dd>
            <dt>"seed"</dt><dd>{seed.to_string()}</dd>
            <dt>"format version"</dt><dd>{format_version.to_string()}</dd>
            <dt>"frame stride"</dt><dd>{frame_stride.to_string()}</dd>
            <dt>"episodes"</dt><dd>{episode_count.to_string()}</dd>
        </dl>
    }
}

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
    view! {
        <tr
            class:selected=move || selected.get().as_deref() == Some(script_id_for_class.as_str())
            on:click=move |_| set_selected.set(Some(script_id_for_click.clone()))
        >
            <td class="numeric">{episode_str}</td>
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
