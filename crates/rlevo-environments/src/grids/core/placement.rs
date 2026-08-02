//! Uniform random placement of objects and agents into a [`Grid`].
//!
//! Per-episode layout randomization is the same three questions in every grid
//! environment: *which cells are free*, *pick one uniformly*, and *which way is
//! the agent facing*. This module answers all three once so that no env has to
//! hand-roll a rejection loop.
//!
//! ```
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//! use rlevo_environments::grids::core::{Entity, Grid, Rect, no_reject, place_obj};
//!
//! let mut grid = Grid::new(5, 5);
//! grid.draw_walls();
//! let mut rng = StdRng::seed_from_u64(7);
//!
//! // Somewhere in the 3x3 interior, uniformly.
//! let (x, y) = place_obj(
//!     &mut grid,
//!     &mut rng,
//!     Entity::Goal,
//!     Rect::new(1, 1, 3, 3),
//!     &mut no_reject,
//! )?;
//! assert_eq!(grid.get(x, y), Entity::Goal);
//! # Ok::<(), rlevo_environments::grids::core::PlacementError>(())
//! ```
//!
//! # Why free functions
//!
//! These are free functions rather than an extension trait on [`Grid`] or a
//! borrowing `Placer<'a>` helper, and both alternatives were considered and
//! rejected:
//!
//! - **An extension trait on [`Grid`]** would still have to receive the RNG and
//!   the env-specific rejection predicate as arguments, so it is a free
//!   function wearing a costume. Moving the RNG *into* [`Grid`] instead is
//!   worse: [`Grid`] is a pure `Clone` data type that sits on the
//!   observation-extraction path (see
//!   `grid::egocentric_view`), and giving it interior
//!   randomness would make cloning a state semantically load-bearing.
//! - **A `Placer<'a>` holding `&mut Grid` + `&mut R`** has no amortized state to
//!   justify it, and it would fight the borrow checker in the `build` bodies
//!   these functions exist to serve — those interleave placement with direct
//!   [`Grid::set`] calls and with reads of cells that were placed a moment
//!   earlier.
//!
//! # Two deliberate deviations from upstream Minigrid
//!
//! **1. There is no `max_tries` parameter.** Upstream `MiniGrid.place_obj` runs
//! an unbounded rejection loop with a retry budget, because it does not want to
//! materialize the free set. This module materializes the candidate cells into
//! a `Vec` and draws a uniform index instead — the same approach
//! `grids::dynamic_obstacles` already uses for obstacle spawning. The
//! distribution is identical, there is no budget to tune, exhaustion is exactly
//! `candidates.is_empty()` rather than "we gave up", and the cost is `O(area)`
//! once rather than unbounded. Grid environments here are at most `25x25`, so
//! the allocation is noise. Do not "restore" a retry budget.
//!
//! **2. Exhaustion is a [`Result`], never a panic.** Config validation happens
//! at a chokepoint, but that validates *config*; exhaustion depends on the
//! config **and** on the draws already made this episode. A `DoorKey` at its
//! minimum size has a `3x3` interior — place the agent, then place the key with
//! an adjacency-rejecting filter, and the residual free set can legitimately be
//! empty. That is an ordinary outcome of a legal config, not a programming
//! error, so it must not panic (`rules.md` §4).
//!
//! # Propagate with `?`; do not `.expect()`
//!
//! [`PlacementError`] converts into
//! [`ConfigError`] and into
//! [`EnvironmentError`], so a
//! `reset()` body or a fallible `build` writes `?` and nothing else. A `Result`
//! at the sampler paired with `.expect()` at every call site is a panic with
//! extra steps, and it re-introduces exactly the failure mode deviation 2 was
//! written to remove. If a `build` helper is currently infallible, make it
//! return `Result` rather than unwrapping inside it.
//!
//! # The agent is not a grid cell
//!
//! [`AgentState`] lives beside the [`Grid`], not inside it, so [`is_free`]
//! cannot see the agent: the cell the agent stands on reads as
//! [`Entity::Empty`]. A caller that must not drop an object onto the agent
//! passes a `reject` closure saying so.

use std::fmt;
use std::ops::RangeInclusive;

// `Rng` is the object-safe core trait in rand 0.10 (what 0.9 called `RngCore`);
// the `random_range` sugar lives on the blanket-implemented `RngExt`. The
// public bound stays `R: Rng + ?Sized` per `rules.md` §8.
use rand::{Rng, RngExt as _};
use rlevo_core::config::{ConfigError, ConstraintKind};
use rlevo_core::environment::EnvironmentError;

use super::agent::AgentState;
use super::entity::Entity;
use super::grid::Grid;
use crate::direction::Direction;

/// Every [`Direction`], in the canonical [`Direction::to_u8`] order.
///
/// Private: the order is an implementation detail of [`random_direction`]'s
/// draw, and pinning it here keeps that draw reproducible.
const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::East,
    Direction::South,
    Direction::West,
    Direction::North,
];

/// A rectangular sub-region of a [`Grid`], **inclusive on all four sides**.
///
/// `Rect::new(1, 1, 3, 3)` is the nine cells `$x \in \{1, 2, 3\} \times y \in \{1, 2, 3\}$` —
/// both corners are members. Inclusive bounds are the right convention for
/// discrete cells: callers think "the left room spans columns 1 through 5", and
/// the single-cell region `Rect::new(2, 2, 2, 2)` is expressible without a
/// special case. It also makes the type's invariant (`x_min <= x_max` and
/// `y_min <= y_max`) equivalent to "non-empty", so an empty region cannot be
/// built at all — see [`new`](Self::new) / [`try_new`](Self::try_new).
///
/// The corners are *not* required to lie inside any particular grid. Sampling
/// clips the region to the grid it is used against, so a caller may pass a
/// region that overhangs the border and still get only in-bounds cells back.
///
/// Fields are private so the ordering invariant survives construction from
/// anywhere (`rules.md` §2).
///
/// # Examples
///
/// ```
/// use rlevo_environments::grids::core::Rect;
///
/// let room = Rect::new(1, 1, 3, 3);
/// assert!(room.contains(1, 1));
/// assert!(room.contains(3, 3)); // inclusive on the far corner
/// assert!(!room.contains(4, 3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,
}

impl Rect {
    /// Builds an inclusive region from compile-time-known corners.
    ///
    /// This is the constructor for literals and for corners derived from an
    /// already-validated config, mirroring
    /// [`Bounds::new`](rlevo_core::bounds::Bounds::new): the bad value is right
    /// at the call site. Use [`try_new`](Self::try_new) for corners derived
    /// from data that has not been validated.
    ///
    /// # Panics
    ///
    /// Panics when `x_min > x_max` or `y_min > y_max`. An inverted region is
    /// silently empty under any other reading, which would turn a caller's
    /// arithmetic slip into a mysterious [`PlacementError::Exhausted`] much
    /// later.
    #[must_use]
    pub const fn new(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> Self {
        assert!(
            x_min <= x_max && y_min <= y_max,
            "Rect::new: inverted region (x_min > x_max or y_min > y_max)"
        );
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// Builds an inclusive region from runtime-derived corners.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError::InvertedRect`] when `x_min > x_max` or
    /// `y_min > y_max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_environments::grids::core::{PlacementError, Rect};
    ///
    /// assert!(Rect::try_new(1, 1, 3, 3).is_ok());
    /// assert_eq!(
    ///     Rect::try_new(3, 1, 1, 3),
    ///     Err(PlacementError::InvertedRect {
    ///         x_min: 3,
    ///         y_min: 1,
    ///         x_max: 1,
    ///         y_max: 3,
    ///     })
    /// );
    /// ```
    pub const fn try_new(
        x_min: i32,
        y_min: i32,
        x_max: i32,
        y_max: i32,
    ) -> Result<Self, PlacementError> {
        if x_min <= x_max && y_min <= y_max {
            Ok(Self {
                x_min,
                y_min,
                x_max,
                y_max,
            })
        } else {
            Err(PlacementError::InvertedRect {
                x_min,
                y_min,
                x_max,
                y_max,
            })
        }
    }

    /// The region covering every cell of `grid`, walls included.
    ///
    /// [`Grid`] is never zero-sized, so this is always a valid region.
    #[must_use]
    pub fn whole_grid(grid: &Grid) -> Self {
        // `Grid::new` rejects a zero dimension, so both maxima are `>= 0`.
        // A grid wider than `i32::MAX` cannot be indexed by `Grid`'s own `i32`
        // coordinates anyway, so saturating is the only meaningful clamp.
        let x_max = i32::try_from(grid.width()).unwrap_or(i32::MAX).max(1) - 1;
        let y_max = i32::try_from(grid.height()).unwrap_or(i32::MAX).max(1) - 1;
        Self {
            x_min: 0,
            y_min: 0,
            x_max,
            y_max,
        }
    }

    /// The inclusive minimum x coordinate.
    #[must_use]
    pub const fn x_min(&self) -> i32 {
        self.x_min
    }

    /// The inclusive minimum y coordinate.
    #[must_use]
    pub const fn y_min(&self) -> i32 {
        self.y_min
    }

    /// The inclusive maximum x coordinate.
    #[must_use]
    pub const fn x_max(&self) -> i32 {
        self.x_max
    }

    /// The inclusive maximum y coordinate.
    #[must_use]
    pub const fn y_max(&self) -> i32 {
        self.y_max
    }

    /// Whether `(x, y)` lies inside the region, both corners included.
    #[must_use]
    pub const fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }

    /// The region's coordinate ranges intersected with `grid`'s extent.
    ///
    /// Either returned range is empty when the region misses the grid on that
    /// axis; `RangeInclusive` iterates zero times in that case, so callers need
    /// no special handling. Clipping *before* iterating also bounds the scan by
    /// the grid area rather than by the caller's corners, so a deliberately
    /// oversized region costs nothing extra.
    fn clip_to(self, grid: &Grid) -> (RangeInclusive<i32>, RangeInclusive<i32>) {
        let width = i32::try_from(grid.width()).unwrap_or(i32::MAX);
        let height = i32::try_from(grid.height()).unwrap_or(i32::MAX);
        (
            self.x_min.max(0)..=self.x_max.min(width - 1),
            self.y_min.max(0)..=self.y_max.min(height - 1),
        )
    }
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "x={}..={}, y={}..={}",
            self.x_min, self.x_max, self.y_min, self.y_max
        )
    }
}

/// Why a placement could not be made.
///
/// `#[non_exhaustive]` because a future placement mode may fail in a new way;
/// match with a trailing `_` arm (`rules.md` §2).
///
/// # Reaching the environment error path
///
/// Both variants are config-domain failures — a region that a config's
/// arithmetic produced, or a residual free set that a config's geometry made
/// too small — so they convert into
/// [`ConfigError`] and from there into
/// [`EnvironmentError::Config`],
/// the variant whose own documentation anticipates a `reset()` that re-runs
/// construction-time work. A caller therefore writes `?` and nothing else:
///
/// ```
/// use rlevo_core::environment::EnvironmentError;
/// use rlevo_environments::grids::core::PlacementError;
///
/// let err = PlacementError::Exhausted {
///     region: rlevo_environments::grids::core::Rect::new(1, 1, 1, 1),
///     searched: 1,
/// };
/// let as_env: EnvironmentError = err.into();
/// assert!(matches!(as_env, EnvironmentError::Config(_)));
/// ```
///
/// The conversion is lossy in one direction only: `ConfigError` is
/// allocation-free (`&'static str` payloads), so the region and the scanned-cell
/// count survive only on the `PlacementError` itself. Log or match the
/// `PlacementError` where that detail matters, then convert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PlacementError {
    /// A region was built with `x_min > x_max` or `y_min > y_max`.
    #[error("inverted placement region: x={x_min}..={x_max}, y={y_min}..={y_max}")]
    InvertedRect {
        /// The supplied minimum x coordinate.
        x_min: i32,
        /// The supplied minimum y coordinate.
        y_min: i32,
        /// The supplied maximum x coordinate.
        x_max: i32,
        /// The supplied maximum y coordinate.
        y_max: i32,
    },
    /// Every in-bounds cell of the region was occupied or rejected.
    #[error(
        "no free cell in placement region ({region}): all {searched} in-bounds cells were occupied or rejected"
    )]
    Exhausted {
        /// The region that was searched, before clipping to the grid.
        region: Rect,
        /// How many cells of the region actually lie inside the grid. `0` means
        /// the region misses the grid entirely.
        searched: usize,
    },
}

impl From<PlacementError> for ConfigError {
    fn from(err: PlacementError) -> Self {
        // `ConstraintKind` has no variant for "a derived region was unusable",
        // and `Custom` is the documented home for a one-off invariant with a
        // static explanation. The messages state the policy that was violated
        // without asserting a mechanism, per `rules.md` §4.
        let kind = match err {
            PlacementError::InvertedRect { .. } => ConstraintKind::Custom(
                "placement region is inverted (x_min > x_max or y_min > y_max)",
            ),
            PlacementError::Exhausted { .. } => {
                ConstraintKind::Custom("no free cell remained in the placement region")
            }
        };
        Self {
            config: "GridPlacement",
            field: "region",
            kind,
        }
    }
}

impl From<PlacementError> for EnvironmentError {
    fn from(err: PlacementError) -> Self {
        Self::Config(ConfigError::from(err))
    }
}

/// Whether `(x, y)` is a cell an object or agent may be placed on.
///
/// A cell is free exactly when it is in bounds and holds [`Entity::Empty`].
/// [`Entity::Floor`], [`Entity::Goal`], and [`Entity::Lava`] are all *passable*
/// but are **not** free: placing onto them would overwrite an object the env
/// deliberately put there. Out-of-bounds cells read as [`Entity::Wall`] and are
/// never free.
///
/// The agent is invisible to this predicate — see the module docs.
#[must_use]
pub fn is_free(grid: &Grid, x: i32, y: i32) -> bool {
    grid.in_bounds(x, y) && grid.get(x, y) == Entity::Empty
}

/// The identity rejection filter: rejects nothing.
///
/// Pass `&mut no_reject` wherever a placement takes a `reject` argument but the
/// only constraint is [`is_free`]:
///
/// ```
/// # use rand::SeedableRng;
/// # use rand::rngs::StdRng;
/// use rlevo_environments::grids::core::{Grid, Rect, no_reject, sample_pos};
///
/// let mut grid = Grid::new(5, 5);
/// grid.draw_walls();
/// let mut rng = StdRng::seed_from_u64(1);
/// let (x, y) = sample_pos(&grid, &mut rng, Rect::new(1, 1, 3, 3), &mut no_reject)?;
/// assert!((1..=3).contains(&x) && (1..=3).contains(&y));
/// # Ok::<(), rlevo_environments::grids::core::PlacementError>(())
/// ```
#[must_use]
pub fn no_reject(_grid: &Grid, _x: i32, _y: i32) -> bool {
    false
}

/// Collects the free, un-rejected cells of `region`, plus how many in-bounds
/// cells were examined.
///
/// Iteration is row-major (`y` outer, `x` inner) and therefore deterministic,
/// which is what makes a fixed seed reproduce a fixed draw.
fn candidates(
    grid: &Grid,
    region: Rect,
    reject: &mut dyn FnMut(&Grid, i32, i32) -> bool,
) -> (Vec<(i32, i32)>, usize) {
    let (xs, ys) = region.clip_to(grid);
    let mut free = Vec::new();
    let mut searched = 0usize;
    for y in ys {
        for x in xs.clone() {
            searched += 1;
            if is_free(grid, x, y) && !reject(grid, x, y) {
                free.push((x, y));
            }
        }
    }
    (free, searched)
}

/// Draws a uniformly random free cell of `region` without modifying the grid.
///
/// A cell is a candidate when [`is_free`] accepts it and `reject` does **not**
/// return `true` for it. All candidates are materialized and one index is
/// drawn, so the distribution is exactly uniform over the candidate set and
/// there is no retry budget (see the module docs).
///
/// # Arguments
///
/// * `grid` — the board the region is clipped against and read from.
/// * `rng` — the caller's persistent stream; one `random_range` draw is taken
///   on success and none on failure.
/// * `region` — inclusive sub-rectangle; cells outside the grid are ignored.
/// * `reject` — env-specific veto, e.g. "not adjacent to the agent". Pass
///   [`no_reject`] when [`is_free`] is the whole constraint.
///
/// # Errors
///
/// Returns [`PlacementError::Exhausted`] when no cell of `region` survives both
/// filters. This is an ordinary outcome on a small board, not a bug: propagate
/// it with `?`, do not `.expect()` it.
pub fn sample_pos<R: Rng + ?Sized>(
    grid: &Grid,
    rng: &mut R,
    region: Rect,
    reject: &mut dyn FnMut(&Grid, i32, i32) -> bool,
) -> Result<(i32, i32), PlacementError> {
    let (free, searched) = candidates(grid, region, reject);
    if free.is_empty() {
        return Err(PlacementError::Exhausted { region, searched });
    }
    let idx = rng.random_range(0..free.len());
    Ok(free[idx])
}

/// Writes `entity` into a uniformly random free cell of `region` and returns
/// that cell.
///
/// Equivalent to [`sample_pos`] followed by [`Grid::set`]. The grid is left
/// untouched when the draw fails, so a caller that recovers from
/// [`PlacementError::Exhausted`] sees no partial mutation.
///
/// # Arguments
///
/// * `grid` — mutated in place on success only.
/// * `rng` — the caller's persistent stream.
/// * `entity` — what to write. Writing [`Entity::Empty`] is legal but pointless.
/// * `region` — inclusive sub-rectangle; cells outside the grid are ignored.
/// * `reject` — env-specific veto; pass [`no_reject`] for none.
///
/// # Errors
///
/// Returns [`PlacementError::Exhausted`] when no cell of `region` is both free
/// and un-rejected. Propagate with `?`; do not `.expect()`.
///
/// # Examples
///
/// ```
/// # use rand::SeedableRng;
/// # use rand::rngs::StdRng;
/// use rlevo_environments::grids::core::{Entity, Grid, Rect, place_obj};
///
/// let mut grid = Grid::new(5, 5);
/// grid.draw_walls();
/// let mut rng = StdRng::seed_from_u64(3);
///
/// // A key, anywhere in the interior except the column x == 2.
/// let (x, y) = place_obj(
///     &mut grid,
///     &mut rng,
///     Entity::Key(rlevo_environments::grids::core::Color::Yellow),
///     Rect::new(1, 1, 3, 3),
///     &mut |_grid, cx, _cy| cx == 2,
/// )?;
/// assert_ne!(x, 2);
/// assert!(matches!(grid.get(x, y), Entity::Key(_)));
/// # Ok::<(), rlevo_environments::grids::core::PlacementError>(())
/// ```
pub fn place_obj<R: Rng + ?Sized>(
    grid: &mut Grid,
    rng: &mut R,
    entity: Entity,
    region: Rect,
    reject: &mut dyn FnMut(&Grid, i32, i32) -> bool,
) -> Result<(i32, i32), PlacementError> {
    let (x, y) = sample_pos(&*grid, rng, region, reject)?;
    grid.set(x, y, entity);
    Ok((x, y))
}

/// Draws a uniformly random [`Direction`].
///
/// Each of the four facings has probability `1/4`. Consumes exactly one
/// `random_range` draw.
#[must_use]
pub fn random_direction<R: Rng + ?Sized>(rng: &mut R) -> Direction {
    ALL_DIRECTIONS[rng.random_range(0..ALL_DIRECTIONS.len())]
}

/// Draws a start pose — a free cell of `region` **and** a facing — for the
/// agent.
///
/// Every consumer needs both halves, so they are drawn together rather than
/// leaving each env to remember the second call. The grid is not modified: the
/// agent is not a grid cell (see the module docs), so the returned
/// [`AgentState`] is the whole result.
///
/// Draw order is fixed and part of the reproducibility contract: **position
/// first, then direction**. Reordering it would change every seeded layout.
///
/// # Arguments
///
/// * `grid` — read-only; the board the region is clipped against.
/// * `rng` — the caller's persistent stream; consumes two draws on success.
/// * `region` — inclusive sub-rectangle; cells outside the grid are ignored.
/// * `reject` — env-specific veto, e.g. "not on the goal's row". Pass
///   [`no_reject`] for none.
///
/// # Errors
///
/// Returns [`PlacementError::Exhausted`] when no cell of `region` is both free
/// and un-rejected. Propagate with `?`; do not `.expect()`.
///
/// # Examples
///
/// ```
/// # use rand::SeedableRng;
/// # use rand::rngs::StdRng;
/// use rlevo_environments::grids::core::{Grid, Rect, no_reject, place_agent};
///
/// let mut grid = Grid::new(5, 5);
/// grid.draw_walls();
/// let mut rng = StdRng::seed_from_u64(11);
///
/// let agent = place_agent(&grid, &mut rng, Rect::new(1, 1, 3, 3), &mut no_reject)?;
/// assert!((1..=3).contains(&agent.x));
/// assert_eq!(agent.carrying, None);
/// # Ok::<(), rlevo_environments::grids::core::PlacementError>(())
/// ```
pub fn place_agent<R: Rng + ?Sized>(
    grid: &Grid,
    rng: &mut R,
    region: Rect,
    reject: &mut dyn FnMut(&Grid, i32, i32) -> bool,
) -> Result<AgentState, PlacementError> {
    let (x, y) = sample_pos(grid, rng, region, reject)?;
    let direction = random_direction(rng);
    Ok(AgentState::new(x, y, direction))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grids::core::color::Color;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashSet;

    /// A 5x5 board with a wall perimeter, leaving a 3x3 free interior.
    fn walled_5x5() -> Grid {
        let mut grid = Grid::new(5, 5);
        grid.draw_walls();
        grid
    }

    /// The 3x3 interior of [`walled_5x5`].
    const INTERIOR: Rect = Rect::new(1, 1, 3, 3);

    #[test]
    fn rect_is_inclusive_on_both_corners() {
        let r = Rect::new(1, 2, 3, 4);
        assert_eq!(
            (r.x_min(), r.y_min(), r.x_max(), r.y_max()),
            (1, 2, 3, 4),
            "accessors must report the corners as given"
        );
        assert!(r.contains(1, 2), "the min corner is a member");
        assert!(r.contains(3, 4), "the max corner is a member");
        assert!(!r.contains(4, 4), "one past x_max is outside");
        assert!(!r.contains(3, 5), "one past y_max is outside");
    }

    #[test]
    fn rect_single_cell_is_representable() {
        let r = Rect::new(2, 2, 2, 2);
        assert!(r.contains(2, 2), "a degenerate rect holds exactly one cell");
        assert!(!r.contains(2, 3), "and nothing else");
    }

    #[test]
    #[should_panic(expected = "Rect::new: inverted region")]
    fn rect_new_panics_on_inverted_x() {
        let _ = Rect::new(3, 1, 1, 3);
    }

    #[test]
    #[should_panic(expected = "Rect::new: inverted region")]
    fn rect_new_panics_on_inverted_y() {
        let _ = Rect::new(1, 3, 3, 1);
    }

    #[test]
    fn rect_try_new_reports_inversion() {
        assert!(
            Rect::try_new(1, 1, 1, 1).is_ok(),
            "a single-cell region is valid"
        );
        assert_eq!(
            Rect::try_new(3, 1, 1, 3),
            Err(PlacementError::InvertedRect {
                x_min: 3,
                y_min: 1,
                x_max: 1,
                y_max: 3,
            }),
            "try_new must report the offending corners rather than panicking"
        );
    }

    #[test]
    fn rect_whole_grid_covers_every_cell() {
        let grid = Grid::new(4, 6);
        let r = Rect::whole_grid(&grid);
        assert!(r.contains(0, 0), "origin is covered");
        assert!(r.contains(3, 5), "far corner is covered");
        assert!(!r.contains(4, 5), "one past the width is not");
        assert!(!r.contains(3, 6), "one past the height is not");
    }

    #[test]
    fn rect_display_names_both_axes() {
        assert_eq!(Rect::new(1, 2, 3, 4).to_string(), "x=1..=3, y=2..=4");
    }

    #[test]
    fn is_free_accepts_only_empty_cells() {
        let mut grid = Grid::new(3, 3);
        assert!(is_free(&grid, 1, 1), "a fresh cell is Empty and free");

        grid.set(0, 0, Entity::Wall);
        grid.set(0, 1, Entity::Floor);
        grid.set(0, 2, Entity::Goal);
        grid.set(1, 0, Entity::Lava);
        grid.set(1, 2, Entity::Key(Color::Yellow));
        for (x, y) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)] {
            assert!(
                !is_free(&grid, x, y),
                "({x}, {y}) holds an object and must not be free"
            );
        }

        assert!(!is_free(&grid, -1, 0), "out of bounds is never free");
        assert!(!is_free(&grid, 3, 0), "out of bounds is never free");
    }

    #[test]
    fn no_reject_rejects_nothing() {
        let grid = Grid::new(2, 2);
        for y in 0..2 {
            for x in 0..2 {
                assert!(!no_reject(&grid, x, y), "no_reject must never veto");
            }
        }
    }

    #[test]
    fn sample_pos_eventually_draws_every_free_cell() {
        let grid = walled_5x5();
        let mut seen = HashSet::new();
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let pos = sample_pos(&grid, &mut rng, INTERIOR, &mut no_reject)
                .expect("the 3x3 interior is entirely free");
            seen.insert(pos);
        }
        assert_eq!(
            seen.len(),
            9,
            "all nine interior cells must be reachable, saw {seen:?}"
        );
    }

    #[test]
    fn sample_pos_never_returns_an_occupied_cell() {
        let mut grid = walled_5x5();
        // Block the whole interior except (2, 2).
        for y in 1..=3 {
            for x in 1..=3 {
                if (x, y) != (2, 2) {
                    grid.set(x, y, Entity::Ball(Color::Red));
                }
            }
        }
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let pos = sample_pos(&grid, &mut rng, INTERIOR, &mut no_reject)
                .expect("(2, 2) is still free");
            assert_eq!(pos, (2, 2), "only the single free cell may be drawn");
        }
    }

    #[test]
    fn sample_pos_honours_the_reject_closure() {
        let grid = walled_5x5();
        for seed in 0..100u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (x, y) = sample_pos(&grid, &mut rng, INTERIOR, &mut |_g, cx, _cy| cx != 2)
                .expect("column x == 2 survives the filter");
            assert_eq!(x, 2, "cells the closure rejected must never be returned");
            assert!((1..=3).contains(&y), "y stays inside the region");
        }
    }

    #[test]
    fn reject_closure_sees_the_grid() {
        // A filter that vetoes any cell orthogonally adjacent to a wall: on a
        // walled 5x5 that leaves only the centre.
        let grid = walled_5x5();
        let mut rng = StdRng::seed_from_u64(4);
        let pos = sample_pos(&grid, &mut rng, INTERIOR, &mut |g, x, y| {
            [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .any(|(dx, dy)| g.get(x + dx, y + dy) == Entity::Wall)
        })
        .expect("the centre cell has no wall neighbour");
        assert_eq!(pos, (2, 2), "the closure must be able to read the grid");
    }

    #[test]
    fn sample_pos_reports_exhaustion_instead_of_looping() {
        let mut grid = walled_5x5();
        for y in 1..=3 {
            for x in 1..=3 {
                grid.set(x, y, Entity::Wall);
            }
        }
        let mut rng = StdRng::seed_from_u64(0);
        let err = sample_pos(&grid, &mut rng, INTERIOR, &mut no_reject)
            .expect_err("no interior cell is free");
        assert_eq!(
            err,
            PlacementError::Exhausted {
                region: INTERIOR,
                searched: 9,
            },
            "exhaustion must be reported with the region it searched"
        );
    }

    #[test]
    fn exhaustion_can_follow_earlier_draws_this_episode() {
        // The DoorKey-at-minimum-size case: a 3x3 interior, an agent placed,
        // then a key that must not be adjacent to the agent. Placing the agent
        // in a corner and vetoing the whole board afterwards is not needed —
        // the residual set genuinely empties when the filter is strict enough.
        let grid = walled_5x5();
        let mut rng = StdRng::seed_from_u64(9);
        let agent = place_agent(&grid, &mut rng, INTERIOR, &mut no_reject)
            .expect("the interior is free at this point");
        let mut occupied = grid.clone();
        // Veto every interior cell within Chebyshev distance 1 of the agent
        // *and* every remaining cell, mimicking a config whose constraints
        // cannot all be satisfied.
        let err = place_obj(
            &mut occupied,
            &mut rng,
            Entity::Key(Color::Yellow),
            INTERIOR,
            &mut |_g, x, y| (x - agent.x).abs() <= 2 && (y - agent.y).abs() <= 2,
        )
        .expect_err("every interior cell is within the vetoed radius");
        assert!(
            matches!(err, PlacementError::Exhausted { .. }),
            "an over-constrained episode must return Err, not panic: {err:?}"
        );
    }

    #[test]
    fn exhaustion_leaves_the_grid_untouched() {
        let mut grid = walled_5x5();
        let before = (1..=3)
            .flat_map(|y| (1..=3).map(move |x| (x, y)))
            .map(|(x, y)| grid.get(x, y))
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(2);
        let err = place_obj(
            &mut grid,
            &mut rng,
            Entity::Goal,
            INTERIOR,
            &mut |_g, _x, _y| true,
        )
        .expect_err("the filter vetoes everything");
        assert!(matches!(err, PlacementError::Exhausted { .. }));
        let after = (1..=3)
            .flat_map(|y| (1..=3).map(move |x| (x, y)))
            .map(|(x, y)| grid.get(x, y))
            .collect::<Vec<_>>();
        assert_eq!(before, after, "a failed placement must not mutate the grid");
    }

    #[test]
    fn region_is_clipped_to_the_grid() {
        // No walls: every one of the 3x3 = 9 cells is free, and the region
        // overhangs the board by a wide margin on all four sides.
        let grid = Grid::new(3, 3);
        let huge = Rect::new(-1_000, -1_000, 1_000, 1_000);
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (x, y) =
                sample_pos(&grid, &mut rng, huge, &mut no_reject).expect("the whole board is free");
            assert!(
                grid.in_bounds(x, y),
                "({x}, {y}) came back from outside the grid"
            );
        }
    }

    #[test]
    fn clipping_bounds_the_scan_by_grid_area() {
        // Same oversized region, but nothing is free: `searched` proves the
        // scan visited 9 cells rather than the region's 2001x2001.
        let mut grid = Grid::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                grid.set(x, y, Entity::Wall);
            }
        }
        let huge = Rect::new(-1_000, -1_000, 1_000, 1_000);
        let mut rng = StdRng::seed_from_u64(0);
        let err = sample_pos(&grid, &mut rng, huge, &mut no_reject).expect_err("nothing is free");
        assert_eq!(
            err,
            PlacementError::Exhausted {
                region: huge,
                searched: 9,
            },
            "the scan must be clipped to the grid's 9 cells"
        );
    }

    #[test]
    fn region_entirely_off_grid_searches_nothing() {
        let grid = Grid::new(3, 3);
        let off = Rect::new(10, 10, 12, 12);
        let mut rng = StdRng::seed_from_u64(0);
        let err = sample_pos(&grid, &mut rng, off, &mut no_reject)
            .expect_err("the region misses the grid entirely");
        assert_eq!(
            err,
            PlacementError::Exhausted {
                region: off,
                searched: 0,
            },
            "a region off the board searches zero cells"
        );
    }

    #[test]
    fn place_obj_writes_the_entity() {
        let mut grid = walled_5x5();
        let mut rng = StdRng::seed_from_u64(5);
        let (x, y) = place_obj(&mut grid, &mut rng, Entity::Goal, INTERIOR, &mut no_reject)
            .expect("the interior is free");
        assert_eq!(grid.get(x, y), Entity::Goal, "the entity must be written");
        assert!(
            !is_free(&grid, x, y),
            "the cell must no longer be a candidate"
        );
    }

    #[test]
    fn place_obj_fills_the_region_without_collisions() {
        let mut grid = walled_5x5();
        let mut rng = StdRng::seed_from_u64(6);
        let mut placed = HashSet::new();
        for _ in 0..9 {
            let pos = place_obj(
                &mut grid,
                &mut rng,
                Entity::Ball(Color::Blue),
                INTERIOR,
                &mut no_reject,
            )
            .expect("nine cells, nine placements");
            assert!(placed.insert(pos), "{pos:?} was placed twice");
        }
        let err = place_obj(
            &mut grid,
            &mut rng,
            Entity::Ball(Color::Blue),
            INTERIOR,
            &mut no_reject,
        )
        .expect_err("the tenth placement has nowhere to go");
        assert!(matches!(err, PlacementError::Exhausted { .. }));
    }

    #[test]
    fn random_direction_covers_all_four_facings() {
        let mut seen = HashSet::new();
        for seed in 0..100u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            seen.insert(random_direction(&mut rng));
        }
        assert_eq!(
            seen.len(),
            4,
            "every facing must be reachable, saw {seen:?}"
        );
    }

    #[test]
    fn place_agent_yields_a_legal_cell_and_a_varying_facing() {
        let grid = walled_5x5();
        let mut cells = HashSet::new();
        let mut facings = HashSet::new();
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let agent = place_agent(&grid, &mut rng, INTERIOR, &mut no_reject)
                .expect("the interior is free");
            assert!(
                INTERIOR.contains(agent.x, agent.y),
                "({}, {}) is outside the requested region",
                agent.x,
                agent.y
            );
            assert!(
                is_free(&grid, agent.x, agent.y),
                "the agent must start on a free cell"
            );
            assert_eq!(agent.carrying, None, "a fresh agent carries nothing");
            cells.insert((agent.x, agent.y));
            facings.insert(agent.direction);
        }
        assert_eq!(cells.len(), 9, "every interior cell must be reachable");
        assert_eq!(facings.len(), 4, "the facing must vary across seeds");
    }

    #[test]
    fn same_seed_reproduces_the_same_draw_sequence() {
        fn run(seed: u64) -> Vec<(i32, i32, Option<Direction>)> {
            let mut grid = walled_5x5();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut out = Vec::new();
            let agent = place_agent(&grid, &mut rng, INTERIOR, &mut no_reject)
                .expect("the interior is free");
            out.push((agent.x, agent.y, Some(agent.direction)));
            for _ in 0..4 {
                let (x, y) = place_obj(
                    &mut grid,
                    &mut rng,
                    Entity::Ball(Color::Green),
                    INTERIOR,
                    &mut |_g, cx, cy| (cx, cy) == (agent.x, agent.y),
                )
                .expect("free cells remain");
                out.push((x, y, None));
            }
            out
        }

        for seed in 0..20u64 {
            assert_eq!(
                run(seed),
                run(seed),
                "seed {seed} must replay bit-identically"
            );
        }
        assert_ne!(
            run(0),
            run(1),
            "different seeds must produce different layouts"
        );
    }

    #[test]
    fn placement_error_converts_into_environment_error() {
        let err = PlacementError::Exhausted {
            region: INTERIOR,
            searched: 9,
        };
        let cfg: ConfigError = err.into();
        assert_eq!(cfg.config, "GridPlacement");
        assert_eq!(cfg.field, "region");
        assert!(
            cfg.to_string().contains("no free cell"),
            "the ConfigError must explain itself: {cfg}"
        );

        let env: EnvironmentError = err.into();
        assert!(
            matches!(env, EnvironmentError::Config(_)),
            "placement failures land on the config-domain variant: {env:?}"
        );

        let inverted: EnvironmentError = PlacementError::InvertedRect {
            x_min: 3,
            y_min: 1,
            x_max: 1,
            y_max: 3,
        }
        .into();
        assert!(matches!(inverted, EnvironmentError::Config(_)));
    }

    #[test]
    fn placement_error_display_names_the_region() {
        let err = PlacementError::Exhausted {
            region: Rect::new(1, 1, 3, 3),
            searched: 9,
        };
        let text = err.to_string();
        assert!(text.contains("x=1..=3, y=1..=3"), "got: {text}");
        assert!(text.contains('9'), "got: {text}");
    }

    /// A `&mut dyn Rng` must satisfy the `?Sized` bound — this is the reason
    /// the signatures say `R: Rng + ?Sized` rather than naming `StdRng`.
    #[test]
    fn accepts_an_unsized_rng() {
        let grid = walled_5x5();
        let mut concrete = StdRng::seed_from_u64(12);
        let rng: &mut dyn Rng = &mut concrete;
        let pos = sample_pos(&grid, rng, INTERIOR, &mut no_reject).expect("the interior is free");
        assert!(INTERIOR.contains(pos.0, pos.1));
    }
}
