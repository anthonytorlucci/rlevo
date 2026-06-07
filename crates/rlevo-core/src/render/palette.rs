//! Project-wide semantic palette for styled environment renderings.
//!
//! # Accessibility contract
//!
//! Color alone is never sufficient to convey meaning. Every semantic colour
//! defined here is paired with a `*_MODIFIER` companion that supplies a
//! hue-redundant signal — `BOLD`, `REVERSED`, `UNDERLINED`, etc. The
//! deuteranopia/protanopia red/green collision is the canonical case: a
//! red hazard and a green goal that differ only in hue collapse into a
//! single shade for ~5% of the population. Pairing red with `REVERSED` and
//! green with `BOLD` ensures the two glyphs remain distinguishable even
//! when colour is unavailable.
//!
//! Environments must reach for these constants rather than raw [`Color`]
//! values. The grep test in the milestone verification suite asserts that
//! `Color::Red` and `Color::Green` do not appear outside this module.
//!
//! # Default theme
//!
//! This module ships a single default theme. Theme pluggability
//! (light/dark/colorblind variants) is on the roadmap; the current
//! design choices here are intended not to block that work.

use super::styled::{Color, Modifier};

/// Foreground colour for the controllable agent (cart, car, walker, etc.).
pub const AGENT_FG: Color = Color::Cyan;
/// Modifier that pairs with [`AGENT_FG`] to make the agent salient against
/// the background, irrespective of hue perception.
pub const AGENT_MODIFIER: Modifier = Modifier::BOLD;

/// Foreground colour for a goal cell, flag, or success terminal state.
pub const GOAL_FG: Color = Color::Green;
/// Modifier that pairs with [`GOAL_FG`]; `BOLD` carries the same "this is
/// where you want to be" weight even when the hue is invisible.
pub const GOAL_MODIFIER: Modifier = Modifier::BOLD;

/// Foreground colour for a hazard, lava, or failure terminal state.
pub const HAZARD_FG: Color = Color::Red;
/// Modifier that pairs with [`HAZARD_FG`]. `REVERSED` swaps foreground and
/// background so the cell flashes as a solid block — visible at a glance
/// regardless of red/green discrimination.
///
/// See also [`GOAL_MODIFIER`] for the complementary success signal.
pub const HAZARD_MODIFIER: Modifier = Modifier::REVERSED;

/// Foreground colour for static structural elements (walls, ground line,
/// track outline). Should fade into the background visually.
pub const WALL_FG: Color = Color::DarkGray;
/// Modifier pairing with [`WALL_FG`]. Empty by default — structural elements
/// stay quiet and do not compete with [`AGENT_FG`] or [`GOAL_FG`] for
/// attention.
pub const WALL_MODIFIER: Modifier = Modifier::EMPTY;

/// Foreground colour for the "best-so-far" candidate in landscape and
/// optimisation renders.
pub const BEST_FG: Color = Color::Green;
/// Modifier pairing with [`BEST_FG`]; `BOLD` lifts the marker over the
/// quintile ramp.
pub const BEST_MODIFIER: Modifier = Modifier::BOLD;

/// Foreground colour for the "current" candidate or active state marker.
pub const CURRENT_FG: Color = Color::White;
/// Modifier pairing with [`CURRENT_FG`]; `BOLD` keeps the marker readable
/// against any ramp colour.
pub const CURRENT_MODIFIER: Modifier = Modifier::BOLD;
