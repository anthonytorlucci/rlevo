//! Toy-text adapter — structured render of a [`FamilyPayload::TabularText`]
//! payload (ADR-0013).
//!
//! Two layouts share one adapter:
//! - **Grid** (`FrozenLake` / `CliffWalking` / `Taxi`) — an SVG tile grid: one
//!   `<rect>` per cell (start / goal / hazard / frozen / empty) with
//!   shape-distinct marker glyphs overlaid (`@` agent, `P` passenger, `D`
//!   destination, `\u{25c6}` named location).
//! - **Cards** (Blackjack) — an HTML card table: the player and dealer hands
//!   as card chips with running totals and a usable-ace badge.
//!
//! Per the a11y contract every coloured element pairs with a redundant glyph
//! or label so a B/W screenshot still reads. Falls back to
//! [`super::fallback::render`] for any non-`TabularText` payload (e.g. a
//! legacy `Ascii` record).
//!
//! [`FamilyPayload::TabularText`]: crate::wire::FamilyPayload::TabularText

use leptos::prelude::*;

use crate::wire::{
    CardTable, FamilyPayload, FrameRecord, TabularCell, TabularGrid, TabularLayout, TabularMarkerKind,
};

/// Side length of one grid cell in SVG user units.
const CELL: f32 = 26.0;
/// Padding inside the viewBox on each edge.
const PAD: f32 = 4.0;

/// Renders one toy-text frame, dispatching on the payload variant.
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::TabularText(payload) => match &payload.layout {
            TabularLayout::Grid(grid) => grid_view(grid),
            TabularLayout::Cards(cards) => cards_view(cards),
        },
        _ => super::fallback::render(crate::wire::EnvFamily::ToyText, frame),
    }
    .into_any()
}

/// Background CSS class for a tabular grid cell.
const fn cell_class(cell: TabularCell) -> &'static str {
    match cell {
        TabularCell::Empty => "rlevo-tab-empty",
        TabularCell::Frozen => "rlevo-tab-frozen",
        TabularCell::Start => "rlevo-tab-start",
        TabularCell::Goal => "rlevo-tab-goal",
        TabularCell::Hazard => "rlevo-tab-hazard",
    }
}

/// Marker glyph + CSS class for an overlaid point of interest.
const fn marker_glyph(kind: TabularMarkerKind) -> (&'static str, &'static str) {
    match kind {
        TabularMarkerKind::Agent => ("@", "rlevo-tab-agent-fg"),
        TabularMarkerKind::Passenger => ("P", "rlevo-tab-passenger-fg"),
        TabularMarkerKind::Destination => ("D", "rlevo-tab-dest-fg"),
        TabularMarkerKind::Location => ("\u{25c6}", "rlevo-tab-loc-fg"),
    }
}

/// SVG tile-grid view (`FrozenLake` / `CliffWalking` / `Taxi`).
fn grid_view(grid: &TabularGrid) -> AnyView {
    let w = grid.width;
    let h = grid.height;
    if w == 0 || h == 0 || grid.cells.len() != usize::from(w) * usize::from(h) {
        return view! {
            <p class="rlevo-warnings">"tabular grid payload has inconsistent dimensions"</p>
        }
        .into_any();
    }

    let vb_w = f32::from(w) * CELL + 2.0 * PAD;
    let vb_h = f32::from(h) * CELL + 2.0 * PAD;
    let view_box = format!("0 0 {vb_w} {vb_h}");

    let mut cells: Vec<AnyView> = Vec::with_capacity(grid.cells.len());
    for y in 0..h {
        for x in 0..w {
            let cell = grid.cells[usize::from(y) * usize::from(w) + usize::from(x)];
            let rx = PAD + f32::from(x) * CELL;
            let ry = PAD + f32::from(y) * CELL;
            cells.push(
                view! { <rect x=rx y=ry width=CELL height=CELL class=cell_class(cell) /> }
                    .into_any(),
            );
        }
    }

    let markers: Vec<AnyView> = grid
        .markers
        .iter()
        .map(|m| {
            let (glyph, cls) = marker_glyph(m.kind);
            let gx = PAD + f32::from(m.x) * CELL + CELL / 2.0;
            let gy = PAD + f32::from(m.y) * CELL + CELL / 2.0;
            view! {
                <text
                    x=gx
                    y=gy
                    class=format!("rlevo-tab-glyph {cls}")
                    text-anchor="middle"
                    dominant-baseline="central"
                >
                    {glyph}
                </text>
            }
            .into_any()
        })
        .collect();

    view! {
        <figure class="rlevo-family-toy-text">
            <svg
                class="rlevo-svg-frame"
                viewBox=view_box
                role="img"
                aria-label="toy-text grid view"
            >
                {cells}
                {markers}
            </svg>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-tab-agent-fg">"@"</span>
                    " agent"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-tab-goal-swatch" />
                    " goal"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-tab-hazard-swatch" />
                    " hazard (hole / cliff)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-tab-passenger-fg">"P"</span>
                    " / "
                    <span class="rlevo-legend-glyph rlevo-tab-dest-fg">"D"</span>
                    " passenger / destination (Taxi)"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}

/// Renders one hand as a row of card chips.
fn hand_view(cards: &[u8]) -> Vec<AnyView> {
    cards
        .iter()
        .map(|&c| {
            let label = match c {
                1 => "A".to_string(),
                n => n.to_string(),
            };
            view! { <span class="rlevo-card">{label}</span> }.into_any()
        })
        .collect()
}

/// HTML card-table view (Blackjack).
fn cards_view(cards: &CardTable) -> AnyView {
    let player = hand_view(&cards.player_cards);
    let dealer = hand_view(&cards.dealer_cards);
    let player_total = cards.player_total;
    let dealer_showing = cards.dealer_showing;
    let ace = cards.usable_ace;

    view! {
        <figure class="rlevo-family-toy-text rlevo-blackjack">
            <div class="rlevo-bj-row">
                <span class="rlevo-bj-label">"Player"</span>
                <span class="rlevo-bj-hand">{player}</span>
                <span class="rlevo-bj-total">"= " {player_total}</span>
                {ace.then(|| view! { <span class="rlevo-bj-ace">"usable ace"</span> })}
            </div>
            <div class="rlevo-bj-row">
                <span class="rlevo-bj-label">"Dealer"</span>
                <span class="rlevo-bj-hand">{dealer}</span>
                <span class="rlevo-bj-total">"showing " {dealer_showing}</span>
            </div>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    "Blackjack — card values (A = ace, counted 1 or 11); totals and "
                    "usable-ace state shown as text so the hand reads without colour."
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
