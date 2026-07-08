//! Per-episode playback panel: scrubber + play/pause + speed.
//!
//! `PlaybackPanel` is invoked anew each time the parent's selected
//! episode changes — fresh `frame_idx` / `playing` / `speed` signals are
//! created scoped to the current owner, so the previous panel's signals
//! and interval handle are disposed cleanly by Leptos's owner tree.
//!
//! # Module layout
//!
//! - **Pure helpers** (`next_frame_idx`, `clamp_idx`, `play_interval_ms`) —
//!   stateless functions that contain the playback arithmetic.  They are
//!   `pub` so they can be unit-tested outside the WASM environment.
//! - **`SPEEDS`** — the canonical list of speed multipliers shared between
//!   the play-loop `Effect` and the speed-button renderer.
//! - **[`playback_panel`]** — the Leptos component entry point; wires the
//!   helpers and signals into a reactive DOM subtree.

use std::time::Duration;

use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::adapters;
use crate::wire::{EnvFamily, EpisodeRecord};

/// Base interval between play-loop ticks at 1× speed, in milliseconds.
/// Higher speeds divide this value; the result is floored at 20 ms.
const PLAY_BASE_INTERVAL_MS: u64 = 200;

/// Determine the next frame index when the play loop ticks. Returns
/// `None` when the play loop should auto-pause (terminal frame reached
/// or empty record).
#[must_use]
pub fn next_frame_idx(cur: usize, frame_count: usize) -> Option<usize> {
    if frame_count == 0 || cur + 1 >= frame_count {
        None
    } else {
        Some(cur + 1)
    }
}

/// Clamp a user-supplied frame index into `[0, frame_count)`. Returns 0
/// when the record is empty.
#[must_use]
pub fn clamp_idx(requested: usize, frame_count: usize) -> usize {
    if frame_count == 0 {
        0
    } else {
        requested.min(frame_count - 1)
    }
}

/// Compute the interval delay for the play loop given a speed
/// multiplier. Clamps to a floor so 10× doesn't burn the browser.
#[must_use]
pub fn play_interval_ms(speed: u32) -> u64 {
    let s = u64::from(speed.max(1));
    (PLAY_BASE_INTERVAL_MS / s).max(20)
}

/// Available playback speed multipliers surfaced as toggle buttons in the UI.
///
/// This is the canonical source of truth for speed choices.  Both the
/// `Effect` that arms the play-loop interval and the speed-button renderer
/// iterate over this slice, so adding or removing an entry here propagates
/// to both automatically.
const SPEEDS: &[u32] = &[1, 2, 5, 10];

/// Renders the interactive playback panel for one episode.
///
/// Creates three reactive signals scoped to the current Leptos owner —
/// `frame_idx`, `playing`, and `speed` — so that when the parent replaces the
/// selected episode the previous panel's signals and interval handle are
/// disposed cleanly by the owner tree.
///
/// The play loop runs via `set_interval_with_handle`; the handle is stored in
/// a `StoredValue` and cleared whenever `playing` or `speed` changes or when
/// the owner is dropped (`on_cleanup`).  Reaching the terminal frame
/// auto-pauses rather than wrapping.
///
/// Returns an error placeholder when `record.frames` is empty.
///
/// # Panics
///
/// Panics if the scrubber's `<input type="range">` event fires without a target
/// element — an invariant guaranteed by the DOM, so this cannot happen in
/// practice.
#[must_use]
// Single reactive-panel builder wiring signals, the play loop, and the scrubber;
// splitting it would fragment the shared signal graph.
#[allow(clippy::too_many_lines)]
pub fn playback_panel(family: EnvFamily, record: EpisodeRecord) -> AnyView {
    let frame_count = record.frames.len();

    let frame_idx = RwSignal::<usize>::new(0);
    let playing = RwSignal::<bool>::new(false);
    let speed = RwSignal::<u32>::new(1);
    let stored_handle: StoredValue<Option<IntervalHandle>> = StoredValue::new(None);

    let record_for_view = StoredValue::new(record);

    Effect::new(move |_| {
        let on = playing.get();
        let sp = speed.get();
        if let Some(h) = stored_handle.get_value() {
            h.clear();
        }
        stored_handle.set_value(None);

        if !on {
            return;
        }
        let interval = Duration::from_millis(play_interval_ms(sp));
        let handle = set_interval_with_handle(
            move || match next_frame_idx(frame_idx.get_untracked(), frame_count) {
                Some(n) => frame_idx.set(n),
                None => playing.set(false),
            },
            interval,
        );
        if let Ok(h) = handle {
            stored_handle.set_value(Some(h));
        }
    });

    on_cleanup(move || {
        if let Some(h) = stored_handle.get_value() {
            h.clear();
        }
    });

    if frame_count == 0 {
        return view! {
            <div class="rlevo-playback">
                <p class="rlevo-error">"This episode has no recorded frames."</p>
            </div>
        }
        .into_any();
    }

    let max_idx_str = (frame_count - 1).to_string();
    let frame_count_disp = frame_count;

    let readout = move || {
        let i = frame_idx.get();
        let f = record_for_view.with_value(|r| {
            let frame = &r.frames[i];
            (frame.step, frame.reward)
        });
        format!(
            "frame {}/{}  ·  step {}  ·  reward {:+.3}",
            i + 1,
            frame_count_disp,
            f.0,
            f.1,
        )
    };

    let frame_view = move || {
        let i = frame_idx.get();
        record_for_view.with_value(|r| adapters::render(family, &r.frames[i]))
    };

    let on_scrub = move |ev: leptos::ev::Event| {
        let target = ev
            .target()
            .expect("range input must have a target")
            .unchecked_into::<web_sys::HtmlInputElement>();
        let raw = target.value().parse::<usize>().unwrap_or(0);
        frame_idx.set(clamp_idx(raw, frame_count));
    };

    let toggle_play = move |_| {
        if playing.get_untracked() {
            playing.set(false);
        } else {
            // Restart from 0 if we are at the terminal frame.
            if frame_idx.get_untracked() + 1 >= frame_count {
                frame_idx.set(0);
            }
            playing.set(true);
        }
    };

    let restart = move |_| {
        frame_idx.set(0);
        playing.set(false);
    };

    let speed_buttons: Vec<AnyView> = SPEEDS
        .iter()
        .map(|&s| {
            let label = format!("{s}×");
            let is_active = move || speed.get() == s;
            let on_click = move |_| speed.set(s);
            view! {
                <button
                    type="button"
                    aria-pressed=move || if is_active() { "true" } else { "false" }
                    on:click=on_click
                >
                    {label}
                </button>
            }
            .into_any()
        })
        .collect();

    let frame_idx_attr = move || frame_idx.get().to_string();
    let play_label = move || {
        if playing.get() {
            "⏸ Pause"
        } else {
            "▶ Play"
        }
    };

    view! {
        <div class="rlevo-playback">
            <div class="frame">{frame_view}</div>
            <div class="controls">
                <button type="button" on:click=toggle_play>{play_label}</button>
                <button type="button" on:click=restart>"⏮ Restart"</button>
                <input
                    type="range"
                    class="scrubber"
                    min="0"
                    max=max_idx_str
                    step="1"
                    prop:value=frame_idx_attr
                    on:input=on_scrub
                />
                <span class="speed-group">{speed_buttons}</span>
            </div>
            <div class="readout">{readout}</div>
        </div>
    }
    .into_any()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_frame_idx_advances_until_terminal() {
        assert_eq!(next_frame_idx(0, 3), Some(1));
        assert_eq!(next_frame_idx(1, 3), Some(2));
        assert_eq!(next_frame_idx(2, 3), None);
    }

    #[test]
    fn next_frame_idx_empty_returns_none() {
        assert_eq!(next_frame_idx(0, 0), None);
        assert_eq!(next_frame_idx(5, 0), None);
    }

    #[test]
    fn clamp_idx_inside_range_is_identity() {
        assert_eq!(clamp_idx(0, 5), 0);
        assert_eq!(clamp_idx(3, 5), 3);
        assert_eq!(clamp_idx(4, 5), 4);
    }

    #[test]
    fn clamp_idx_caps_at_last_frame() {
        assert_eq!(clamp_idx(5, 5), 4);
        assert_eq!(clamp_idx(99, 5), 4);
    }

    #[test]
    fn clamp_idx_handles_empty() {
        assert_eq!(clamp_idx(0, 0), 0);
        assert_eq!(clamp_idx(99, 0), 0);
    }

    #[test]
    fn play_interval_scales_inversely_with_speed() {
        let one = play_interval_ms(1);
        let two = play_interval_ms(2);
        let ten = play_interval_ms(10);
        assert!(one >= two, "1× should be slower than 2×");
        assert!(two >= ten, "2× should be slower than 10×");
        assert!(ten >= 20, "10× should clamp to the 20ms floor");
    }

    #[test]
    fn play_interval_zero_treated_as_one() {
        assert_eq!(play_interval_ms(0), play_interval_ms(1));
    }
}
