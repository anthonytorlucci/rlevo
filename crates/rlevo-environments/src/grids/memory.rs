//! `Memory`: a cue object is shown once, then matched at a fork from recall.
//!
//! Ports Farama Minigrid's [`MemoryEnv`]. Each episode the environment samples
//! a **cue** object — a [`Key`] or a [`Ball`] — and places it in the start room.
//! The agent walks a one-cell corridor to a fork at the far end, where a Key and
//! a Ball sit facing each other in random order. Emitting [`GridAction::Done`]
//! while facing the fork object **whose type equals the cue** pays
//! [`success_reward`]; emitting it while facing the other fork object, the cue,
//! or anything else ends the episode with reward `0.0`.
//!
//! A successful policy must *remember* what it saw. That is the whole point of
//! this environment, and it is a property the geometry has to earn — see
//! **Invariant M** below.
//!
//! ## What the geometry guarantees — and what it does not
//!
//! The cue does **not** vanish for good once the agent leaves the start room.
//! From any ordinary corridor cell at column `x <= 7` the agent can turn to face
//! **West** and simply re-read the cue. This *leak zone* is set by the view's
//! 6-cell reach from the cue's fixed column (`x = 1`): `$x - 6 \leq 1 \implies x \leq 7$`.
//! It is **independent of `size`** — growing the grid does not shrink it.
//!
//! **Occlusion does not close the leak.** This environment emits an *occluded*
//! observation (`MemoryEnv::VISIBILITY` is [`Visibility::Occluded`], canonical
//! `see_through_walls=False`), and the leak zone is nonetheless exactly the one
//! an unoccluded view would give — measured cell by cell at sizes 11, 13, and 17
//! by `test_memory_env_cue_leak_zone_is_bounded_and_decision_region_is_clean`.
//! The reason is the shape of the canonical algorithm: `Grid.process_vis` is a
//! forward **flood fill**, not a ray cast. The corridor's full-width wall rows do
//! squeeze the lit region down to three view columns while the beam is *inside*
//! the corridor — but the moment it reaches the start room at `x = 3` the room is
//! open across rows `$\text{mid} \pm 1$`, light spreads sideways with no further limit, and
//! the cue at `(1, mid - 1)` is lit again. Whatever hides the cue in this
//! environment, it is not occlusion.
//!
//! What the geometry *does* buy is narrower, and it is the part that matters:
//!
//! 1. **At the decision region the cue is not observable, in any facing**
//!    (Invariant M, below). The correct fork therefore cannot be read off the
//!    observation the agent is acting on.
//! 2. **No single observation anywhere on the board contains both the cue cell
//!    and a fork object cell.** The view spans at most seven columns in any
//!    facing, while the cue (`x = 1`) and the fork (`x = size - 2 >= 9`) are at
//!    least eight columns apart. A memoryless, single-observation policy thus
//!    can never see the cue and the choice at the same time, and cannot beat
//!    chance — which is exactly the property a prior dead-RNG bug destroyed:
//!    the cue type was never drawn from `_rng` in `reset`, so the fork match
//!    didn't depend on the cue at all and a memoryless policy could win by
//!    keying on coordinate alone. Sampling the cue from `_rng` (and the
//!    fork-invariance regression test in this module) closed that gap.
//!
//! Both are pinned by the test
//! `test_memory_env_cue_leak_zone_is_bounded_and_decision_region_is_clean`,
//! which also asserts the leak zone itself so it cannot drift unnoticed.
//!
//! ## The recall horizon scales with `size`
//!
//! Because the leak zone (`x <= 7`) is fixed while the corridor grows with
//! `size`, the cue-free stretch the agent must bridge from memory is
//! `size`-dependent. The cue-free centre-row corridor cells are exactly
//! `$x \in [8, \text{size} - 3]$`, i.e. `size - 10` of them:
//!
//! | `size` | fork column | cue-free corridor cells | recall horizon |
//! |--------|-------------|-------------------------|----------------|
//! | 11     | 9           | `{8}` — the gap alone   | ~1–2 steps     |
//! | 13     | 11          | `{8, 9, 10}`            | 3 cells        |
//! | 17     | 15          | `{8, …, 14}`            | 7 cells        |
//!
//! **`MIN_SIZE == 11` is a *correctness* floor, not a memory benchmark.** It is
//! the smallest size that keeps the decision region clean, and at that size a
//! memoryless policy still provably cannot win — but the agent only has to hold
//! the cue for a step or two, which makes size 11 the *weakest* recall task the
//! layout admits. **The default is therefore `size = 13`** (`max_steps = 845`):
//! it triples the cue-free run to three corridor cells for ~40% more step
//! budget, and is the smallest size that asks for recall rather than merely
//! forbidding perception. Raise it to `17` for a long-horizon variant; drop to
//! the floor of `11` only when the step budget matters more than the horizon.
//!
//! ## What carries the signal (and what deliberately does not)
//!
//! - **Object type is the only signal.** The cue and *both* fork objects are
//!   [`Color::Green`] — one `OBJECT_COLOR` constant covers all three. This is
//!   load-bearing, not cosmetic: if the matching object had a distinct color,
//!   the answer would be readable straight off the observation's color channel
//!   and no recall would be needed.
//! - **The fork always holds one Key and one Ball**; only their order is
//!   randomized. There is never a Key/Key or Ball/Ball fork.
//! - **The cue type is sampled per episode** from the environment's persistent
//!   RNG (ADR 0029: `reset()` draws from the stream and lets it advance — it
//!   never re-seeds). The matching fork position is *derived* from the sampled
//!   cue, never fixed by configuration.
//!
//! ## Invariant M (cue-hiding)
//!
//! > **For every cell in the decision region — the vertical hallway at the far
//! > end of the corridor — and for every one of the four facing directions, the
//! > agent's egocentric view must not contain the cue cell.**
//!
//! This is what makes the *answer* a recall problem rather than a perception
//! problem. Note what it does and does not say: it constrains the **decision
//! region only**. It says nothing about the rest of the corridor, where the cue
//! is in fact still re-readable (see the leak zone above) — and it does not need
//! to, because an agent standing in the leak zone is not yet being asked to
//! answer.
//!
//! Invariant M is a claim about the observation this environment actually
//! **emits** — the occluded one — not about the raw window `egocentric_view`
//! extracts. The tests below therefore drive it through
//! [`observe_grid`]/`mask_view` at `MemoryEnv::VISIBILITY`, because asserting
//! against the raw window would assert a property of a projection no
//! environment emits.
//!
//! Invariant M is *not* satisfied automatically, and — contrary to what ADR
//! 0043 §3 predicted, and contrary to what the earlier occlusion gap implied
//! it would take (`egocentric_view` applied no visibility masking at all, so
//! this environment had to fall back on raw distance instead of walls,
//! forcing `MIN_SIZE = 11` rather than canonical's occlusion-reliant S7/S9) —
//! now that occlusion is actually implemented, **it still does not buy it**. rlevo
//! runs the canonical shadow cast (`see_through_walls=False`), and an executed
//! sweep over every decision-region cell × facing at sizes **7 and 9** finds the
//! cue visible under [`Visibility::Occluded`] in exactly the same (cell, facing)
//! pairs as under [`Visibility::SeeThrough`] — 5 of them at each size. The
//! mechanism is the flood fill described above: the cue is reachable *around*
//! the corridor's walls, through the mouth at `x = 4` and out sideways into the
//! open start room. So here Invariant M is still bought with **distance
//! alone**, and `MIN_SIZE` stays 11. This is pinned, with the sizes and the
//! violating poses, by `test_memory_env_occlusion_does_not_relax_min_size`.
//!
//! (That is a statement about rlevo's port, which is a faithful transcription of
//! `Grid.process_vis`; it has not been checked against a live Python
//! `MiniGrid-MemoryS9-v0` run. If canonical really does hide the cue at S7/S9,
//! the divergence is in the port or the layout, and that test is where it will
//! surface.)
//!
//! The arithmetic: the view window reaches `VIEW_SIZE - 1 == 6` cells backward
//! (the agent sits at `view[6][3]` looking toward row `0`), so from a cell at
//! column `x` no facing can reach a cell at column `x - 6` or nearer than
//! `x - 6`… i.e. column `c` is visible only if `x - c <= 6`. The cue sits at
//! `x = 1` and the fork column is `size - 2`, so Invariant M requires
//!
//! ```math
//! (\text{size} - 2) - 6 > 1 \implies \text{size} > 9 \implies \text{size} \geq 11 \quad (\text{size is odd})
//! ```
//!
//! **`MIN_SIZE == 11` is a consequence of Invariant M, not a magic number.**
//! Lowering it to match canonical Minigrid's S7/S9 configurations would silently
//! re-break the environment: the agent could stand at the fork, face back down
//! the corridor, and simply re-read the cue — solvable by a reactive,
//! memoryless policy. That is not a prediction; it is what the executed sweep
//! reports at both 7 and 9. Do not lower `MIN_SIZE` on the strength of an
//! argument about walls: the only thing that would justify it is a *re-run* of
//! `test_memory_env_occlusion_does_not_relax_min_size` coming back clean, after
//! some change to how sight is computed (ADR 0063 Decision 6). Because the
//! floor is still a distance bound, the compile-time assertion below still
//! derives it from `VIEW_REACH` and needs no revision.
//!
//! ## Layout (default `size = 13`)
//!
//! ```text
//!      x =  0  1  2  3  4  5  6  7  8  9 10 11 12
//! y =  0     #  #  #  #  #  #  #  #  #  #  #  #  #
//!      1     #  .  .  .  .  .  .  .  .  .  #  .  #
//!      2     #  .  .  .  .  .  .  .  .  .  #  .  #
//!      3     #  .  .  .  .  .  .  .  .  .  #  .  #
//!      4     #  #  #  #  #  .  .  .  .  .  #  O  #
//!      5     #  C  .  .  #  #  #  #  #  #  #  d  #
//!      6     #  >  .  .  .  .  .  .  .  .  .  J  #
//!      7     #  .  .  .  #  #  #  #  #  #  #  d  #
//!      8     #  #  #  #  #  .  .  .  .  .  #  O  #
//!      9     #  .  .  .  .  .  .  .  .  .  #  .  #
//!     10     #  .  .  .  .  .  .  .  .  .  #  .  #
//!     11     #  .  .  .  .  .  .  .  .  .  #  .  #
//!     12     #  #  #  #  #  #  #  #  #  #  #  #  #
//! ```
//!
//! - `C` — the **cue** at `(1, height/2 - 1)`, one row *off* the corridor
//!   centerline (canonical placement). Note it is *not* the occlusion that this
//!   offset buys: the shadow cast lights it anyway, via the open start room.
//! - `>` — agent start at `(1, height/2)` facing East, with the cue in view.
//! - `O` — the two **fork objects** at `$(\text{size} - 2, \text{height}/2 \mp 2)$`; one Key, one
//!   Ball, both green, order randomized.
//! - `J` — the fork **junction** `(size - 2, height/2)`, reached by walking East.
//! - `d` — the two **decision cells**: `Done` from here, facing the adjacent
//!   `O`, is how the agent answers.
//! - The corridor cells at `$x \in [8, 10]$` are the **cue-free run**: from none of
//!   them can any facing reach the cue (see the leak zone above). At the floor
//!   size of `11` this run is the single cell `x = 8`.
//! - The blank interior cells in rows 1–3 and 9–11 are sealed dead space, exactly
//!   as in canonical Minigrid.
//!
//! ## Solving it (the oracle script, any valid `size`)
//!
//! ```text
//! Forward × (size - 3)          walk the corridor to the junction
//! TurnLeft  if match_pos is the upper fork object, else TurnRight
//! Forward                       step onto the decision cell
//! Done                          facing the matching object
//! ```
//!
//! ## Deviation from canonical: how success is claimed
//!
//! Canonical Minigrid terminates the moment the agent *steps onto* the cell
//! adjacent to a fork object. rlevo instead pays on an explicit
//! [`GridAction::Done`] issued **while facing** the object — the same
//! interaction protocol every other rlevo grid environment uses. The fork
//! objects are positioned so that a facing-based `Done` is always reachable.
//! This is a deliberate, documented deviation, not an oversight.
//!
//! | Observation | 7 × 7 **occluded** egocentric grid, `[type, color, state]` per cell |
//! |-------------|--------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`, `Done`                         |
//! | Reward      | `success_reward(steps, max_steps)` on correct Done; else `0.0`     |
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::memory::{MemoryConfig, MemoryEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = MemoryConfig::new(13, 845, 0); // == MemoryConfig::default()
//! let mut env = MemoryEnv::with_config(cfg, false).expect("valid config");
//! let _snap = env.reset().unwrap();
//! // The cue is sampled fresh each episode; the matching fork follows from it.
//! println!("cue: {:?}, match at {:?}", env.cue(), env.match_pos());
//! ```
//!
//! [`MemoryEnv`]: https://minigrid.farama.org/environments/minigrid/MemoryEnv/
//! [`Key`]: super::core::entity::Entity::Key
//! [`Ball`]: super::core::entity::Entity::Ball
//! [`Color::Green`]: super::core::color::Color::Green

use super::core::{
    GridSnapshot, VIEW_SIZE, Visibility,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use crate::episode::EpisodeGuard;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Smallest grid side length at which **Invariant M** holds.
///
/// Derived, not chosen: the egocentric view reaches `VIEW_SIZE - 1 == 6` cells
/// backward, the cue sits at `x = 1`, and the fork column is `size - 2`. Hiding
/// the cue from the fork therefore needs `(size - 2) - 6 > 1`, i.e.
/// `size >= 11` for odd `size`.
///
/// The derivation is a **distance** bound even though this environment now runs
/// canonical occlusion: the shadow cast does not hide the cue at any size, so it
/// does not enter the arithmetic. The measured evidence is in
/// `test_memory_env_occlusion_does_not_relax_min_size`; the mechanism is in the
/// module docs.
const MIN_SIZE: usize = 11;

/// Side length used by [`MemoryConfig::default`].
///
/// **Not** [`MIN_SIZE`]. The floor is where the task becomes *correct* (the cue
/// is unreadable at the fork); it is also where the cue-free corridor run is a
/// single cell, so the recall horizon is ~1–2 steps. `13` widens that run to
/// three cells — the cheapest size at which the agent must genuinely *hold* the
/// cue — at a `$5 \cdot \text{size}^2$` budget of `845` rather than `605`. See the recall
/// horizon table in the module docs.
const DEFAULT_SIZE: usize = 13;

/// Backward reach of the egocentric view, in cells.
///
/// The agent sits at `view[VIEW_SIZE - 1][VIEW_SIZE / 2]` and looks toward row
/// `0`, so no facing can reveal a cell more than this many columns behind it.
const VIEW_REACH: usize = VIEW_SIZE - 1;

// Invariant M, enforced at compile time. The cue sits at column 1 and the fork
// at column `size - 2`; hiding the cue from the fork needs
// `(MIN_SIZE - 2) - VIEW_REACH > 1`. If `VIEW_SIZE` ever grows, or someone
// lowers `MIN_SIZE` to match canonical Minigrid's S7/S9, the build fails here
// instead of silently re-opening the "recall task structurally defeated"
// failure mode this environment was fixed for — where a memoryless policy
// can win because the answer is readable from a single observation (there:
// the cue type was never sampled from the RNG, so the fork match didn't
// depend on it; here it would be geometry letting one view see both the cue
// and the fork).
//
// This assertion survived the arrival of occlusion unchanged, and that is the
// correct outcome rather than an oversight: ADR 0063 Decision 6 said it would
// have to be replaced by an occlusion-geometry bound *if* the S7/S9 sweep came
// back clean. It did not (see `test_memory_env_occlusion_does_not_relax_min_size`),
// so the floor is still a pure distance bound and `VIEW_REACH` is still the
// right thing to derive it from.
const _: () = assert!(
    MIN_SIZE > VIEW_REACH + 3,
    "MIN_SIZE no longer hides the cue from the fork decision cell (Invariant M)"
);
const _: () = assert!(MIN_SIZE % 2 == 1, "MIN_SIZE must be odd");

// The shipped default must itself satisfy `Validate` — a default that cannot be
// constructed would panic in `ConstructableEnv::new`. Checked here so the build
// fails rather than the constructor.
const _: () = assert!(
    DEFAULT_SIZE >= MIN_SIZE && DEFAULT_SIZE % 2 == 1,
    "DEFAULT_SIZE must be odd and at or above the Invariant-M floor"
);

/// The one colour every object in this environment wears.
///
/// The cue and **both** fork objects are green, exactly as in canonical
/// Minigrid. Colour must not distinguish the matching object from the
/// distractor, or the observation's colour channel would leak the answer and
/// the recall property would be gone.
const OBJECT_COLOR: Color = Color::Green;

/// Rejection text for `size < MIN_SIZE`. Names the *cause*, not just the bound.
///
/// The cause really is distance, and the message says so: occlusion is enabled
/// here and measurably does not hide the cue at size 7 or 9 (see
/// `test_memory_env_occlusion_does_not_relax_min_size`), so a smaller board is
/// rejected on the only ground that actually holds.
const SIZE_BELOW_MIN: &str = "MemoryEnv requires size >= 11: at smaller sizes the fork is within \
                              backward view reach of the cue, and occlusion does not hide it \
                              either — light reaches the cue around the corridor walls, through \
                              the open start room (Invariant M)";

/// Rejection text for an even `size`.
const SIZE_NOT_ODD: &str = "MemoryEnv requires an odd size so the corridor has a single centre row";

/// Canonical Minigrid step budget: `$5 \cdot \text{size}^2$`.
#[must_use]
const fn default_max_steps(size: usize) -> usize {
    5 * size * size
}

/// Cell coordinates derived from the grid `size`.
///
/// Mirrors canonical Minigrid's `_gen_grid` arithmetic: `hallway_end = size - 3`
/// is the wall column with the single corridor gap, and the fork lives one
/// column further right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// Side length of the (square) grid, as a signed coordinate.
    size: i32,
    /// The wall column pierced by the corridor: `size - 3`.
    hallway_end: i32,
    /// The vertical-hallway column holding the fork objects: `size - 2`.
    fork_x: i32,
    /// Corridor centre row: `size / 2`.
    mid: i32,
    /// The cue cell, one row above the centerline.
    cue_pos: (i32, i32),
    /// Upper fork object.
    top_pos: (i32, i32),
    /// Lower fork object.
    bottom_pos: (i32, i32),
}

impl Layout {
    /// Derive the layout for a validated `size` (odd, `>= MIN_SIZE`).
    fn new(size: usize) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        let size = size as i32;
        let mid = size / 2;
        let hallway_end = size - 3;
        let fork_x = hallway_end + 1;
        Self {
            size,
            hallway_end,
            fork_x,
            mid,
            cue_pos: (1, mid - 1),
            top_pos: (fork_x, mid - 2),
            bottom_pos: (fork_x, mid + 2),
        }
    }

    /// The agent's fixed start cell, at the west end of the corridor.
    const fn start_pos(self) -> (i32, i32) {
        (1, self.mid)
    }

    /// Whether `pos` is one of the two fork object cells.
    fn is_fork(self, pos: (i32, i32)) -> bool {
        pos == self.top_pos || pos == self.bottom_pos
    }
}

/// Configuration for [`MemoryEnv`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::memory::MemoryConfig;
///
/// let cfg = MemoryConfig::new(13, 845, 42);
/// assert_eq!(cfg.size, 13);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Side length of the square grid, including the border walls.
    ///
    /// Must be **odd** (the corridor needs a single centre row) and **at least
    /// 11**. The lower bound is not arbitrary — see **Invariant M** in the
    /// [module docs](self): only distance hides the cue from the fork (the
    /// occlusion this environment runs demonstrably does not), and `size >= 11`
    /// is exactly the distance required.
    ///
    /// The default is `13`, not the floor of `11`: the floor is merely the
    /// smallest *correct* size, at which the cue-free corridor run is a single
    /// cell. `13` gives a three-cell run — a recall horizon worth the name — for
    /// a `$5 \cdot \text{size}^2$` budget of `845` instead of `605`.
    pub size: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    ///
    /// Canonical Minigrid uses `$5 \cdot \text{size}^2$`.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG.
    ///
    /// The RNG is seeded **once** at construction and advances across resets
    /// (ADR 0029), so a fixed seed reproduces a whole *sequence* of episodes,
    /// not one episode over and over. Use [`MemoryEnv::reset_with_seed`] to
    /// replay a specific episode.
    pub seed: u64,
}

impl MemoryConfig {
    /// Creates a [`MemoryConfig`] with the given parameters.
    ///
    /// The value is not validated here; [`MemoryEnv::with_config`] does that.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::memory::MemoryConfig;
    ///
    /// let cfg = MemoryConfig::new(13, 845, 0);
    /// assert_eq!(cfg.max_steps, 845);
    /// assert_eq!(cfg, MemoryConfig::default());
    /// ```
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64) -> Self {
        Self {
            size,
            max_steps,
            seed,
        }
    }
}

impl Default for MemoryConfig {
    /// `size = 13`, `max_steps = 845` (`$5 \cdot \text{size}^2$`), `seed = 0`.
    ///
    /// Deliberately **one step above `MIN_SIZE`**: see the module docs. The
    /// floor is a correctness bound, and shipping it as the default would ship
    /// the weakest recall task the layout supports (a one-cell cue-free run).
    fn default() -> Self {
        let size = DEFAULT_SIZE;
        Self {
            size,
            max_steps: default_max_steps(size),
            seed: 0,
        }
    }
}

impl Validate for MemoryConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MemoryConfig";
        if self.size < MIN_SIZE {
            return Err(ConfigError {
                config: C,
                field: "size",
                kind: ConstraintKind::Custom(SIZE_BELOW_MIN),
            });
        }
        if self.size.is_multiple_of(2) {
            return Err(ConfigError {
                config: C,
                field: "size",
                kind: ConstraintKind::Custom(SIZE_NOT_ODD),
            });
        }
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for MemoryConfig {
    type Err = String;

    /// Parses `"size=13,max_steps=845,seed=0"` (keys in any order) or the
    /// positional form `"13,845,0"`.
    ///
    /// When `max_steps` is omitted it is derived from `size` as `$5 \cdot \text{size}^2$`,
    /// so `"size=13"` yields the canonical budget for a 13×13 grid rather than
    /// the default-size one.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection (which
    /// names *why* small or even sizes are refused).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cfg = Self::default();
        let mut max_steps_set = false;
        for (idx, raw) in s.trim().split(',').map(str::trim).enumerate() {
            if raw.is_empty() {
                continue;
            }
            if let Some((key, value)) = raw.split_once('=') {
                match key.trim() {
                    "size" => cfg.size = value.trim().parse().map_err(|e| format!("size: {e}"))?,
                    "max_steps" => {
                        cfg.max_steps = value
                            .trim()
                            .parse()
                            .map_err(|e| format!("max_steps: {e}"))?;
                        max_steps_set = true;
                    }
                    "seed" => cfg.seed = value.trim().parse().map_err(|e| format!("seed: {e}"))?,
                    other => return Err(format!("unknown key `{other}`")),
                }
            } else {
                match idx {
                    0 => cfg.size = raw.parse().map_err(|e| format!("size: {e}"))?,
                    1 => {
                        cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?;
                        max_steps_set = true;
                    }
                    2 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    _ => return Err(format!("unexpected positional value `{raw}`")),
                }
            }
        }
        if !max_steps_set {
            cfg.max_steps = default_max_steps(cfg.size);
        }
        cfg.validate()
            .map_err(|e| format!("{e} (got size={})", cfg.size))?;
        Ok(cfg)
    }
}

/// Minigrid's `Memory` environment: see a cue, recall it at the fork.
///
/// Each [`Environment::reset`] samples a fresh cue type and fork order from the
/// environment's **persistent** RNG. The matching fork object is derived from
/// the cue, so nothing in the configuration pins the answer to a side — the only
/// way to score above chance is to remember the cue.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`](super::core::GridObservation) / [`GridAction`] /
/// [`ScalarReward`].
///
/// The episode ends on the first [`GridAction::Done`] (or at the step budget);
/// a [`step`](Environment::step) taken afterwards is rejected with
/// [`EnvironmentError::StepAfterEpisodeEnd`] — see the [`EpisodeGuard`] field.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::memory::MemoryEnv;
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = MemoryEnv::new(false);
/// let _snap = env.reset().unwrap();
/// println!("cue: {:?}, match pos: {:?}", env.cue(), env.match_pos());
/// ```
#[derive(Debug)]
pub struct MemoryEnv {
    state: GridState,
    config: MemoryConfig,
    layout: Layout,
    steps: usize,
    render: bool,
    /// The cue entity placed in the start room this episode.
    cue: Entity,
    /// World coordinates of the fork object whose type equals [`Self::cue`].
    match_pos: (i32, i32),
    rng: StdRng,
    /// Rejects a `step()` taken after the agent has already answered.
    ///
    /// This environment is a *matching* task that is supposed to end the instant
    /// the agent commits to one of the two fork objects — one commitment, one
    /// payout. Nothing in the state made that true: `GridAction::Done` returns
    /// [`StepOutcome::DoneAction`] without touching the grid or the agent, so
    /// [`facing_match`](Self::facing_match) is recomputed, unchanged, on every
    /// later call. Without the guard, an unguarded post-terminal `step()` let
    /// the agent:
    ///
    /// - **re-answer correctly, repeatedly.** Standing at the decision cell
    ///   facing the matching object, each further `Done` re-entered the
    ///   `facing_match()` branch and paid another
    ///   `success_reward(self.steps, max_steps)` — an unbounded episode return
    ///   from a single act of recall, decaying only because `self.steps` kept
    ///   climbing.
    /// - **retract a wrong answer and take the other object.** A `Done` facing
    ///   the distractor pays `0.0` and ends the episode; unguarded, the agent
    ///   could then turn around, walk to the *other* fork object, and `Done`
    ///   again for the full success reward. Guessing became free, which destroys
    ///   the recall property the whole layout (Invariant M) exists to protect.
    ///
    /// It also kept `self.steps` advancing past `max_steps`, so a timed-out
    /// episode reported a length longer than its own budget.
    ///
    /// [`StepOutcome::DoneAction`]: super::core::dynamics::StepOutcome::DoneAction
    guard: EpisodeGuard,
}

impl MemoryEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical `minigrid/envs/memory.py`
    /// passes `see_through_walls=False` explicitly — which is also
    /// `MiniGridEnv.__init__`'s own default, so the env restates the canonical
    /// behaviour rather than opting into it. This is the environment where the
    /// choice is load-bearing rather than incidental: the corridor's full-width
    /// wall rows are what hide the cue from the fork, and **Invariant M** (see
    /// the [module docs](self)) is a claim about the observation this const
    /// selects. See ADR 0063
    /// (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs a [`MemoryEnv`] from an explicit configuration.
    ///
    /// Seeds the persistent RNG **once** and samples the first episode from it.
    /// Call [`Environment::reset`] before the first [`Environment::step`] to
    /// obtain an observation (and a freshly sampled cue).
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// the Invariant-M minimum of `11` (the rejection names the cause — view
    /// reach, with occlusion measured not to help), an even `size`, or a zero
    /// `max_steps`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::memory::{MemoryConfig, MemoryEnv};
    ///
    /// let env = MemoryEnv::with_config(
    ///     MemoryConfig::new(13, 845, 0), // the default; 11 is the legal floor
    ///     true,                          // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    ///
    /// // A 7×7 grid is refused: it cannot hide the cue from the fork — not by
    /// // distance, and (measurably) not by occlusion either.
    /// let err = MemoryEnv::with_config(MemoryConfig::new(7, 245, 0), false).unwrap_err();
    /// assert!(err.to_string().contains("view reach"));
    /// ```
    pub fn with_config(config: MemoryConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let layout = Layout::new(config.size);
        let (state, cue, match_pos) = Self::build(layout, &mut rng);
        Ok(Self {
            state,
            config,
            layout,
            steps: 0,
            render,
            cue,
            match_pos,
            rng,
            guard: EpisodeGuard::new(),
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream, so
    /// successive episodes draw independent cues (ADR 0029). Use this when a
    /// *specific* episode must reproduce bit-for-bit — replaying a failure, or
    /// pinning a cue in a test.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) (currently none).
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<GridSnapshot, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Side length of the square grid.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.config.size
    }

    /// Returns the number of steps taken since the last reset.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Returns a reference to the current grid state.
    #[must_use]
    pub const fn state(&self) -> &GridState {
        &self.state
    }

    /// The cue entity shown in the start room for the current episode.
    ///
    /// Either `Key(Green)` or `Ball(Green)`. This is the *oracle*: a policy that
    /// could read it would solve the environment trivially, which is exactly why
    /// it is exposed only as an inherent accessor for tests and scripted
    /// rollouts — it is not part of the observation.
    #[must_use]
    pub const fn cue(&self) -> Entity {
        self.cue
    }

    /// World coordinates of the fork object matching the current cue.
    ///
    /// Derived from the sampled cue type each episode, never from configuration.
    #[must_use]
    pub const fn match_pos(&self) -> (i32, i32) {
        self.match_pos
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Build a fresh episode: walls from `layout`, then a cue and a fork order
    /// drawn from `rng`.
    ///
    /// Returns the state, the sampled cue entity, and the derived position of
    /// the matching fork object.
    fn build(layout: Layout, rng: &mut StdRng) -> (GridState, Entity, (i32, i32)) {
        #[allow(clippy::cast_sign_loss)]
        let side = layout.size as usize;
        let mut grid = Grid::new(side, side);
        grid.draw_walls();

        // Start room: walls two rows above and below the centerline, pinched at
        // x = 4 down to the single-cell corridor mouth.
        for x in 1..5 {
            grid.set(x, layout.mid - 2, Entity::Wall);
            grid.set(x, layout.mid + 2, Entity::Wall);
        }
        grid.set(4, layout.mid - 1, Entity::Wall);
        grid.set(4, layout.mid + 1, Entity::Wall);

        // Horizontal hallway: one cell tall, running East along the centre row.
        for x in 5..layout.hallway_end {
            grid.set(x, layout.mid - 1, Entity::Wall);
            grid.set(x, layout.mid + 1, Entity::Wall);
        }

        // Vertical hallway: a solid wall column with a single gap on the centre
        // row. (Canonical also walls `hallway_end + 2`, which is the right
        // border — already drawn by `draw_walls`.)
        for y in 0..layout.size {
            if y != layout.mid {
                grid.set(layout.hallway_end, y, Entity::Wall);
            }
        }

        // The cue: Key or Ball with equal probability, always green.
        let key = Entity::Key(OBJECT_COLOR);
        let ball = Entity::Ball(OBJECT_COLOR);
        let cue = if rng.random_bool(0.5) { key } else { ball };
        grid.set(layout.cue_pos.0, layout.cue_pos.1, cue);

        // The fork: always one Key and one Ball, order drawn from the same
        // stream — so neither side nor colour predicts the answer.
        let (top, bottom) = if rng.random_bool(0.5) {
            (key, ball)
        } else {
            (ball, key)
        };
        grid.set(layout.top_pos.0, layout.top_pos.1, top);
        grid.set(layout.bottom_pos.0, layout.bottom_pos.1, bottom);

        // The answer is *derived* from the cue, not stored in the config.
        let match_pos = if top == cue {
            layout.top_pos
        } else {
            layout.bottom_pos
        };

        let (sx, sy) = layout.start_pos();
        let agent = AgentState::new(sx, sy, Direction::East);
        (GridState::new(grid, agent), cue, match_pos)
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }

    /// `true` when the agent faces a **fork** object whose type equals the cue.
    ///
    /// The fork-position guard is load-bearing. The cue object is the very same
    /// `Entity` value as its matching fork object, so a bare
    /// `entity_in_front == cue` test would also pay out for standing in the
    /// start room and issuing `Done` at the cue itself.
    fn facing_match(&self) -> bool {
        let front = self.state.agent.front();
        if !self.layout.is_fork(front) {
            return false;
        }
        self.state.grid.get(front.0, front.1) == self.cue
    }
}

impl crate::render::AsciiRenderable for MemoryEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for MemoryEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for MemoryEnv {
    fn new(render: bool) -> Self {
        Self::with_config(MemoryConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for MemoryEnv {
    type Action = GridAction;
    type State = GridState;
    type Observation = GridObservation;

    /// Emission model `$O(a, s')$`. The observation is a function of the resulting
    /// `next_state` alone, so this forwards to the same projection as
    /// [`observe_reset`](Self::observe_reset).
    fn observe(&self, _action: &GridAction, next_state: &GridState) -> GridObservation {
        observe_grid(next_state, Self::VISIBILITY)
    }

    fn observe_reset(&self, state: &GridState) -> GridObservation {
        observe_grid(state, Self::VISIBILITY)
    }
}

impl Environment<3, 3, 1> for MemoryEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    /// Samples a fresh episode from the **persistent** RNG.
    ///
    /// Per ADR 0029 this deliberately does *not* re-seed from `config.seed`:
    /// doing so would replay a bit-identical cue every episode and hand any
    /// reactive policy the answer for free. Use
    /// [`reset_with_seed`](Self::reset_with_seed) for deterministic replay.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.reset();
        let (state, cue, match_pos) = Self::build(self.layout, &mut self.rng);
        self.state = state;
        self.cue = cue;
        self.match_pos = match_pos;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    /// Advances one step and returns the resulting snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended — the agent has answered, or the step budget ran out. Call
    /// [`reset`](Environment::reset) first.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // Guard first: before `steps` advances, before the action is applied,
        // and before anything can touch `self.rng`. A rejected call must leave
        // the grid, the agent, and the step counter untouched — and must not
        // advance the persistent stream, or the cue of every subsequent episode
        // would depend on how many illegal steps a caller made (ADR 0029).
        self.guard.check()?;

        self.steps += 1;
        let outcome = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = if outcome == StepOutcome::DoneAction {
            if self.facing_match() {
                (success_reward(self.steps, self.config.max_steps), true)
            } else {
                (0.0, true)
            }
        } else {
            let done = self.steps >= self.config.max_steps;
            (0.0, done)
        };
        let observation = self.observe(&action, &self.state);
        // Single exit: one snapshot is built, and the guard is fed that
        // snapshot's own status, so no branch can record something the caller
        // was not handed.
        let snapshot = self.emit(observation, reward, done);
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

impl rlevo_scene::payload::GridPayloadSource for MemoryEnv {
    fn grid_snapshot(&self) -> rlevo_scene::payload::GridSnapshot {
        crate::grids::core::render::grid_snapshot(&self.state.grid, &self.state.agent)
    }
}

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values
    // are literals or seeds read back without arithmetic, or two identically
    // seeded runs that must agree bit-for-bit. A tolerance would let a real
    // regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]

    use super::*;
    // Test-only: the `Visibility` dispatch point. Invariant M is a claim about
    // the *masked* view, so the sweeps below go through this rather than the raw
    // `egocentric_view` no environment emits.
    use super::super::core::mask_view;
    // Test-only: the byte a masked cell carries, asserted directly by
    // `test_memory_env_masked_empty_cells_are_encoded_as_unseen`.
    use super::super::core::UNSEEN_TYPE;
    use crate::episode::assert_rejects_post_terminal_step;
    // `Snapshot` arrives through `use super::*` — the module now imports the
    // trait itself, for `snapshot.status()` in `step`.
    use std::collections::HashSet;

    const KEY: Entity = Entity::Key(OBJECT_COLOR);
    const BALL: Entity = Entity::Ball(OBJECT_COLOR);

    fn env_default() -> MemoryEnv {
        MemoryEnv::with_config(MemoryConfig::default(), false).expect("valid config")
    }

    /// Walk East from the start cell to the fork junction `(fork_x, mid)`.
    /// Returns the snapshot observed *at* the junction, still facing East.
    fn drive_to_junction(env: &mut MemoryEnv) -> GridSnapshot {
        let steps = env.layout.fork_x - env.layout.start_pos().0;
        let mut last = None;
        for _ in 0..steps {
            last = Some(env.step(GridAction::Forward).expect("step"));
        }
        let snap = last.expect("corridor must be at least one cell long");
        let agent = env.state().agent;
        assert_eq!(
            (agent.x, agent.y),
            (env.layout.fork_x, env.layout.mid),
            "driver must land on the fork junction"
        );
        assert_eq!(
            agent.direction,
            Direction::East,
            "driver must arrive facing East"
        );
        snap
    }

    /// The turn that a cue-reading oracle would take at the junction.
    fn oracle_turn(env: &MemoryEnv) -> GridAction {
        if env.match_pos() == env.layout.top_pos {
            GridAction::TurnLeft
        } else {
            GridAction::TurnRight
        }
    }

    /// From the junction, commit to `turn` and answer.
    fn answer(env: &mut MemoryEnv, turn: GridAction) -> f32 {
        env.step(turn).expect("turn");
        env.step(GridAction::Forward).expect("forward");
        let snap = env.step(GridAction::Done).expect("done");
        assert!(snap.is_done(), "Done must terminate the episode");
        (*snap.reward()).into()
    }

    /// Every passable cell at or beyond the corridor's mouth — the region from
    /// which the agent commits to an answer.
    fn decision_region(env: &MemoryEnv) -> Vec<(i32, i32)> {
        let l = env.layout;
        (l.hallway_end..l.size - 1)
            .flat_map(|x| (1..l.size - 1).map(move |y| (x, y)))
            .filter(|&(x, y)| env.state().grid.get(x, y).is_passable())
            .collect()
    }

    // ---------------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------------

    #[test]
    fn test_memory_config_default_validates() {
        assert!(
            MemoryConfig::default().validate().is_ok(),
            "the library default must itself be valid"
        );
    }

    #[test]
    fn test_memory_config_default_values() {
        let cfg = MemoryConfig::default();
        assert_eq!(cfg.size, 13, "default size is 13");
        assert_eq!(cfg.max_steps, 845, "default budget is canonical 5 * size^2");
        assert_eq!(cfg.max_steps, 5 * cfg.size * cfg.size, "…which is 5 * 13^2");
        assert_eq!(cfg.seed, 0, "default seed is 0");
    }

    /// The default is deliberately **above** the Invariant-M floor.
    ///
    /// `MIN_SIZE` is a correctness bound: it is the smallest size at which the
    /// cue is unreadable from the decision region. It is *also* the size at which
    /// the cue-free corridor run collapses to one cell, so shipping it as the
    /// default would ship the weakest recall task the layout supports. This pins
    /// the distinction, so a future "simplify: default == the minimum" edit has
    /// to argue with a failing test rather than a comment.
    #[test]
    fn test_memory_config_default_exceeds_the_correctness_floor() {
        let cfg = MemoryConfig::default();
        assert!(
            cfg.size > MIN_SIZE,
            "the default must not be the bare correctness floor of {MIN_SIZE}"
        );

        // The cue-free corridor run is `$x \in [\text{cue\_x} + \text{VIEW\_REACH} + 1, \text{fork\_x} - 1]$`.
        #[allow(clippy::cast_possible_wrap)]
        let reach = VIEW_REACH as i32;
        let cue_free = |size: usize| {
            let l = Layout::new(size);
            l.fork_x - 1 - (l.cue_pos.0 + reach)
        };
        assert_eq!(
            cue_free(MIN_SIZE),
            1,
            "at the floor the run is a single cell"
        );
        assert_eq!(
            cue_free(cfg.size),
            3,
            "the default must buy a genuine cue-free run"
        );
    }

    #[test]
    fn test_memory_config_rejects_size_below_min() {
        // The canonical Minigrid S7 layout: registered upstream, illegal here.
        // rlevo now runs the same shadow cast upstream does, and it measurably
        // does not hide the cue at this size — see
        // `test_memory_env_occlusion_does_not_relax_min_size`.
        let err = MemoryConfig::new(7, 245, 0).validate().unwrap_err();
        assert_eq!(err.field, "size", "the size field must be named");
        let msg = err.to_string();
        assert!(
            msg.contains("view reach"),
            "the rejection must name the cause that actually holds — distance, \
             not occlusion — got: {err}"
        );
        assert!(
            msg.contains("occlusion does not hide it"),
            "the rejection must also foreclose the occlusion argument, since that \
             is the reason a reader would expect size 7 to be legal, got: {err}"
        );
        assert!(
            MemoryEnv::with_config(MemoryConfig::new(9, 405, 0), false).is_err(),
            "size 9 still leaves the cue within backward view reach"
        );
    }

    #[test]
    fn test_memory_config_rejects_even_size() {
        let err = MemoryConfig::new(12, 720, 0).validate().unwrap_err();
        assert_eq!(err.field, "size", "the size field must be named");
        assert!(
            err.to_string().contains("odd"),
            "the rejection must say the size has to be odd, got: {err}"
        );
    }

    #[test]
    fn test_memory_config_rejects_zero_max_steps() {
        let bad = MemoryConfig {
            max_steps: 0,
            ..Default::default()
        };
        assert!(
            MemoryEnv::with_config(bad, false).is_err(),
            "a zero step budget must be refused at construction"
        );
    }

    #[test]
    fn test_memory_config_fromstr_roundtrips() {
        let cfg: MemoryConfig = "size=13,max_steps=845,seed=7".parse().unwrap();
        assert_eq!(
            cfg,
            MemoryConfig::new(13, 845, 7),
            "keyed parse round-trips"
        );

        let positional: MemoryConfig = "13,845,7".parse().unwrap();
        assert_eq!(positional, cfg, "positional parse matches the keyed form");
    }

    #[test]
    fn test_memory_config_fromstr_derives_max_steps_from_size() {
        let cfg: MemoryConfig = "size=13".parse().unwrap();
        assert_eq!(
            cfg.max_steps,
            5 * 13 * 13,
            "an omitted budget follows the parsed size, not the default size"
        );
    }

    #[test]
    fn test_memory_config_fromstr_rejects_swap_fork_key() {
        // `swap_fork` used to pin the answer to a side. It is gone; a config
        // string carrying it must be rejected rather than silently ignored.
        let err = "swap_fork=true".parse::<MemoryConfig>().unwrap_err();
        assert!(
            err.contains("unknown key"),
            "the retired swap_fork key must be rejected, got: {err}"
        );
    }

    #[test]
    fn test_memory_config_fromstr_rejects_small_size() {
        let err = "size=7".parse::<MemoryConfig>().unwrap_err();
        assert!(
            err.contains("view reach") && err.contains("got size=7"),
            "the parse rejection must name the cause and the offending value, got: {err}"
        );
    }

    // ---------------------------------------------------------------------
    // Geometry
    // ---------------------------------------------------------------------

    /// Pins the default (`size = 13`) layout against the canonical arithmetic:
    /// `hallway_end = size - 3`, `fork_x = size - 2`, `mid = size / 2`, cue one
    /// row off the centerline, fork objects two rows off it.
    #[test]
    fn test_memory_env_layout_matches_canonical_geometry() {
        let env = env_default();
        assert_eq!(env.size(), 13, "this test pins the *default* layout");
        let l = env.layout;
        assert_eq!(l.hallway_end, 10, "hallway_end = size - 3");
        assert_eq!(l.fork_x, 11, "fork column = size - 2");
        assert_eq!(l.mid, 6, "corridor centre row = size / 2");
        assert_eq!(l.cue_pos, (1, 5), "cue sits one row off the centerline");
        assert_eq!(l.top_pos, (11, 4), "upper fork object");
        assert_eq!(l.bottom_pos, (11, 8), "lower fork object");
        assert_eq!(
            l.start_pos(),
            (1, 6),
            "agent starts at the corridor's west end"
        );

        let g = &env.state().grid;
        assert_eq!(g.get(4, 5), Entity::Wall, "start room pinches at x = 4");
        assert_eq!(g.get(4, 7), Entity::Wall, "start room pinches at x = 4");
        assert_eq!(g.get(6, 5), Entity::Wall, "hallway roof");
        assert_eq!(g.get(6, 7), Entity::Wall, "hallway floor");
        assert_eq!(g.get(10, 4), Entity::Wall, "hallway_end column is walled");
        assert_eq!(g.get(10, 6), Entity::Empty, "…except for the single gap");
        assert_eq!(
            env.state().agent.direction,
            Direction::East,
            "agent faces down the corridor"
        );
    }

    #[test]
    fn test_memory_env_cue_is_visible_at_episode_start() {
        // Necessary counterpart to Invariant M: the cue has to be *seen* once —
        // in the observation the environment actually emits, which is the
        // occluded one. (The cue is the agent's immediate neighbour, and
        // `process_vis` seeds the agent's own cell and lights its row
        // neighbours, so occlusion cannot touch it.)
        let env = env_default();
        let obs = observe_grid(env.state(), MemoryEnv::VISIBILITY);
        // Facing East from the start cell, the cue one row above it is one cell
        // to the agent's left: view row VIEW_SIZE-1, column VIEW_SIZE/2 - 1.
        // (At the default size that is start (1, 6), cue (1, 5).)
        let cell = obs.view[VIEW_SIZE - 1][VIEW_SIZE / 2 - 1];
        assert_eq!(
            cell[0],
            env.cue().type_u8(),
            "the cue must be inside the starting observation"
        );
        assert_eq!(
            cell[1],
            OBJECT_COLOR.to_u8(),
            "the cue is green like every other object"
        );
    }

    /// The four facings, in the order every sweep below iterates them.
    const DIRS: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    /// Sentinel planted on the cue cell. `Lava` never occurs naturally here.
    ///
    /// It is also *transparent* ([`Entity::see_behind`] is `true` for everything
    /// except `Wall` and shut `Door`), exactly like the `Key`/`Ball` it stands
    /// in for — so planting it cannot perturb the shadow cast and turn a probe
    /// into a measurement of itself.
    const CUE_MARK: Entity = Entity::Lava;

    /// Sentinel planted on both fork cells. Transparent, for the same reason.
    const FORK_MARK: Entity = Entity::Goal;

    /// Does the observation `MemoryEnv` **emits** from `(x, y)` facing `dir`
    /// contain `mark`?
    ///
    /// Deliberately routed through `mask_view` at an explicit [`Visibility`]
    /// rather than through the raw `egocentric_view`: Invariant M is a claim
    /// about what a *policy* can see, and what a policy sees is the masked view.
    /// Passing the policy in as a parameter is what lets the sweeps below
    /// contrast the two and show the shadow cast makes no difference here.
    fn sees(grid: &Grid, x: i32, y: i32, dir: Direction, mark: Entity, vis: Visibility) -> bool {
        mask_view(grid, &AgentState::new(x, y, dir), vis)
            .iter()
            .flatten()
            .any(|cell| *cell == Some(mark))
    }

    /// Every (cell, facing) in the decision region from which the cue is visible
    /// under `vis`. Empty iff Invariant M holds.
    fn invariant_m_violations(
        grid: &Grid,
        region: &[(i32, i32)],
        cue_pos: (i32, i32),
        vis: Visibility,
    ) -> Vec<((i32, i32), Direction)> {
        let mut probe = grid.clone();
        probe.set(cue_pos.0, cue_pos.1, CUE_MARK);
        region
            .iter()
            .flat_map(|&cell| DIRS.map(move |dir| (cell, dir)))
            .filter(|&((x, y), dir)| sees(&probe, x, y, dir, CUE_MARK, vis))
            .collect()
    }

    /// **Invariant M**: from every cell of the decision region, for all four
    /// facings, the emitted observation does not contain the cue cell.
    ///
    /// The check is black-box: a unique, transparent sentinel is planted on the
    /// cue cell of a scratch grid and the real emission path
    /// ([`MemoryEnv::VISIBILITY`], via `mask_view`) is asked whether it can see
    /// it.
    #[test]
    fn test_memory_env_invariant_m_cue_never_visible_from_decision_region() {
        for size in [11usize, 13, 17] {
            let env =
                MemoryEnv::with_config(MemoryConfig::new(size, default_max_steps(size), 0), false)
                    .expect("valid config");
            let region = decision_region(&env);
            assert!(!region.is_empty(), "decision region must be non-empty");

            let violations = invariant_m_violations(
                &env.state().grid,
                &region,
                env.layout.cue_pos,
                MemoryEnv::VISIBILITY,
            );
            assert!(
                violations.is_empty(),
                "Invariant M violated at size {size}: cue {:?} is visible from {violations:?}",
                env.layout.cue_pos
            );
        }
    }

    /// ADR 0063 Decision 6, executed: does occlusion relax `MIN_SIZE` to 7?
    ///
    /// **It does not.** ADR 0043 §3 called the 11 → 7 relaxation "a one-line
    /// change", and the report that tracked the missing-occlusion gap in
    /// `egocentric_view` (walls never blocked line of sight, so this
    /// environment had to rely on distance instead) carried that same 11 → 7
    /// relaxation as an acceptance bullet once occlusion landed; this sweep is
    /// the evidence that refutes it. At sizes 7 and 9 the cue is visible from
    /// the decision region under [`Visibility::Occluded`] in exactly the same
    /// (cell, facing) pairs as under [`Visibility::SeeThrough`] — the shadow
    /// cast changes nothing about cue visibility on this board, because
    /// `process_vis` is a forward flood fill and light reaches the cue *around*
    /// the corridor walls through the open start room.
    ///
    /// Sizes 7 and 9 are built through `MemoryEnv::build` directly rather than
    /// `with_config`, because `Validate` refuses them — which is the very policy
    /// under test.
    ///
    /// # If this test fails
    ///
    /// It means sight computation changed and the cue is now genuinely hidden at
    /// a small size. That is the good outcome, not a regression: reopen ADR 0063
    /// Decision 6, lower `MIN_SIZE`, and replace the `MIN_SIZE > VIEW_REACH + 3`
    /// compile-time assertion with one derived from the occluding geometry.
    #[test]
    fn test_memory_env_occlusion_does_not_relax_min_size() {
        for size in [7usize, 9] {
            let layout = Layout::new(size);
            let (state, _cue, _match_pos) = MemoryEnv::build(layout, &mut StdRng::seed_from_u64(0));
            let grid = &state.grid;
            let region: Vec<(i32, i32)> = (layout.hallway_end..layout.size - 1)
                .flat_map(|x| (1..layout.size - 1).map(move |y| (x, y)))
                .filter(|&(x, y)| grid.get(x, y).is_passable())
                .collect();
            assert!(
                !region.is_empty(),
                "size {size}: decision region must be non-empty for this to mean anything"
            );

            let occluded =
                invariant_m_violations(grid, &region, layout.cue_pos, Visibility::Occluded);
            let see_through =
                invariant_m_violations(grid, &region, layout.cue_pos, Visibility::SeeThrough);

            assert!(
                !occluded.is_empty(),
                "size {size}: the cue is hidden from the whole decision region under \
                 occlusion — the MIN_SIZE = 7 relaxation ADR 0043 predicted is back on \
                 the table; see this test's docs"
            );
            assert_eq!(
                occluded, see_through,
                "size {size}: occlusion is supposed to make no difference to cue \
                 visibility on this board, yet the two policies disagree"
            );
        }
    }

    /// The other half of the same finding: occlusion is not a no-op *in general*
    /// for this environment — it just never touches the cue.
    ///
    /// Without this, `test_memory_env_occlusion_does_not_relax_min_size` would
    /// also pass if [`Visibility::Occluded`] were silently doing nothing at all.
    #[test]
    fn test_memory_env_occlusion_changes_the_emitted_observation() {
        let env = env_default();
        let l = env.layout;
        // From the fork junction facing North the agent looks along the vertical
        // hallway; the wall column at `hallway_end` shadows the dead space
        // beyond it, and those are walls, so the change survives into the bytes.
        let state = GridState::new(
            env.state().grid.clone(),
            AgentState::new(l.fork_x, l.mid, Direction::North),
        );
        let occluded = observe_grid(&state, MemoryEnv::VISIBILITY);
        let see_through = observe_grid(&state, Visibility::SeeThrough);
        assert_ne!(
            occluded.view, see_through.view,
            "occlusion must actually change the emitted observation somewhere, or \
             the MIN_SIZE sweep is vacuous"
        );
    }

    /// Masked `Entity::Empty` cells reach the encoding as *unseen*, pinned at
    /// the pose where it costs the most.
    ///
    /// Facing West from the fork junction — the agent's pose at the moment it
    /// answers — the shadow cast masks a substantial block of the window, and
    /// **every** masked cell happens to be `Entity::Empty`. This is the worst
    /// case for a channel-0 encoding that cannot tell *unseen* from
    /// *confirmed-empty*: while `UNSEEN_TYPE == Entity::Empty.type_u8() == 0`
    /// the occluded observation here was byte-identical to the unoccluded one,
    /// so the entire occlusion signal was erased by the encoder at exactly the
    /// pose the environment is built around.
    ///
    /// This test was the regression guard for that defect (it asserted the
    /// collision, so the renumbering could not land unnoticed) and remains the
    /// regression guard for the fix. ADR 0063 Decision 4 moved `type_u8` to
    /// canonical `OBJECT_TO_IDX` (`Empty` is `1`), so the two views must now
    /// differ, and they must differ *precisely* on the hidden cells: each one
    /// reads `UNSEEN_TYPE` under occlusion and `Entity::Empty.type_u8()` under
    /// see-through, with every other cell untouched.
    #[test]
    fn test_memory_env_masked_empty_cells_are_encoded_as_unseen() {
        let env = env_default();
        let l = env.layout;
        let grid = env.state().grid.clone();
        let agent = AgentState::new(l.fork_x, l.mid, Direction::West);

        let masked = mask_view(&grid, &agent, MemoryEnv::VISIBILITY);
        let raw = mask_view(&grid, &agent, Visibility::SeeThrough);

        let hidden_cells: Vec<(usize, usize)> = (0..VIEW_SIZE)
            .flat_map(|r| (0..VIEW_SIZE).map(move |c| (r, c)))
            .filter(|&(r, c)| masked[r][c].is_none())
            .collect();
        let hidden: Vec<Entity> = hidden_cells
            .iter()
            .map(|&(r, c)| raw[r][c].expect("SeeThrough masks nothing"))
            .collect();

        assert!(
            !hidden.is_empty(),
            "the shadow cast must hide something from the fork junction facing West"
        );
        assert!(
            hidden.iter().all(|&e| e == Entity::Empty),
            "this test's premise is that every hidden cell here is Empty; got {hidden:?}"
        );

        let state = GridState::new(grid, agent);
        let occluded = observe_grid(&state, MemoryEnv::VISIBILITY);
        let see_through = observe_grid(&state, Visibility::SeeThrough);

        assert_ne!(
            occluded.view,
            see_through.view,
            "{} Empty cells are hidden and the encoding must say so; if these are \
             equal again, UNSEEN_TYPE has re-collided with Entity::Empty and the \
             occlusion signal is being erased at the encoder — see ADR 0063 \
             Decision 4",
            hidden.len()
        );

        // Differ *exactly* on the hidden cells, in both directions.
        for r in 0..VIEW_SIZE {
            for c in 0..VIEW_SIZE {
                if hidden_cells.contains(&(r, c)) {
                    assert_eq!(
                        occluded.view[r][c],
                        [UNSEEN_TYPE, 0, 0],
                        "hidden cell ({r}, {c}) must encode as the unseen triple"
                    );
                    assert_eq!(
                        see_through.view[r][c][0],
                        Entity::Empty.type_u8(),
                        "hidden cell ({r}, {c}) is Empty, so see-through must report it as such"
                    );
                } else {
                    assert_eq!(
                        occluded.view[r][c], see_through.view[r][c],
                        "visible cell ({r}, {c}) must be unaffected by occlusion"
                    );
                }
            }
        }
    }

    /// Pins **both halves** of the truth about cue visibility, driving the real
    /// emitted observation ([`MemoryEnv::VISIBILITY`]) rather than re-deriving
    /// the arithmetic.
    ///
    /// 1. The cue **is** re-observable from ordinary centre-row corridor cells
    ///    at `x <= cue_x + VIEW_REACH == 7` if the agent turns to face West.
    ///    Its justification changed when occlusion landed, but the assertion did
    ///    not: this leak was once excused as "a limitation of an occlusion-free
    ///    view", and it is now *measured* to survive occlusion unchanged at
    ///    every size here. `process_vis` is a forward flood fill, so light that
    ///    reaches the start room through the corridor mouth spreads sideways and
    ///    relights the cue's row. The module docs state it; this asserts it so
    ///    the docs cannot silently rot.
    /// 2. The cue is **not** observable from any decision-region cell in any
    ///    facing (Invariant M) — the answer cannot be read at the moment it is
    ///    given.
    /// 3. **No** observation anywhere on the board contains the cue *and* a fork
    ///    object. This — not (2) alone — is what defeats a memoryless policy, so
    ///    it is asserted directly instead of being left to prose.
    ///
    /// Note that the leak zone is fixed by the cue's column and the view reach,
    /// so it does **not** grow with `size`; the cue-free run before the fork
    /// does. A regression that widened the leak zone would trip (1); one that
    /// let the cue back into the decision region would trip (2) or (3).
    #[test]
    fn test_memory_env_cue_leak_zone_is_bounded_and_decision_region_is_clean() {
        #[allow(clippy::cast_possible_wrap)]
        let reach = VIEW_REACH as i32;
        let vis = MemoryEnv::VISIBILITY;

        for size in [11usize, 13, 17] {
            let env =
                MemoryEnv::with_config(MemoryConfig::new(size, default_max_steps(size), 0), false)
                    .expect("valid config");
            let l = env.layout;

            let mut probe = env.state().grid.clone();
            probe.set(l.cue_pos.0, l.cue_pos.1, CUE_MARK);
            probe.set(l.top_pos.0, l.top_pos.1, FORK_MARK);
            probe.set(l.bottom_pos.0, l.bottom_pos.1, FORK_MARK);

            let sees =
                |x: i32, y: i32, dir: Direction, mark: Entity| sees(&probe, x, y, dir, mark, vis);

            // (1) The leak zone. Its far edge is `cue_x + VIEW_REACH`, fixed by
            // the view geometry alone — it must not move when `size` does.
            let leak_end = l.cue_pos.0 + reach;
            assert_eq!(
                leak_end, 7,
                "the leak zone is bounded by view reach from the cue column, \
                 so it must not depend on size (size {size})"
            );
            for x in l.start_pos().0..=leak_end {
                assert!(
                    env.state().grid.get(x, l.mid).is_passable(),
                    "size {size}: ({x}, {}) must be a corridor cell for this \
                     assertion to mean anything",
                    l.mid
                );
                assert!(
                    sees(x, l.mid, Direction::West, CUE_MARK),
                    "size {size}: the cue must still be re-readable from ({x}, {}) \
                     facing West — this is the documented, size-independent leak zone",
                    l.mid
                );
            }
            // One column further East the cue is gone in *every* facing: this is
            // where the cue-free run begins.
            for dir in DIRS {
                assert!(
                    !sees(leak_end + 1, l.mid, dir, CUE_MARK),
                    "size {size}: the leak zone must end at x = {leak_end}, yet the \
                     cue is visible from x = {} facing {dir:?}",
                    leak_end + 1
                );
            }

            // (2) Invariant M: the decision region is clean in every facing.
            for (x, y) in decision_region(&env) {
                for dir in DIRS {
                    assert!(
                        !sees(x, y, dir, CUE_MARK),
                        "size {size}: Invariant M violated — the cue is visible from \
                         the decision cell ({x}, {y}) facing {dir:?}"
                    );
                }
            }

            // (3) The property that actually defeats a memoryless policy: the cue
            // and the fork objects are never in the same 7x7 window, anywhere, in
            // any facing — the view spans at most 7 columns and they are at least
            // 8 apart.
            for y in 0..l.size {
                for x in 0..l.size {
                    for dir in DIRS {
                        assert!(
                            !(sees(x, y, dir, CUE_MARK) && sees(x, y, dir, FORK_MARK)),
                            "size {size}: the observation from ({x}, {y}) facing \
                             {dir:?} contains both the cue and a fork object — a \
                             single-observation policy could read the answer"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_memory_env_min_size_follows_from_view_reach() {
        // Pins the derivation, so a future edit to VIEW_SIZE or the cue column
        // fails here rather than silently re-breaking the recall property.
        let l = Layout::new(MIN_SIZE);
        #[allow(clippy::cast_possible_wrap)]
        let reach = VIEW_REACH as i32;
        assert!(
            l.fork_x - reach > l.cue_pos.0,
            "at MIN_SIZE the fork must be strictly beyond backward view reach of the cue"
        );
        let too_small = Layout::new(MIN_SIZE - 2);
        assert!(
            too_small.fork_x - reach <= too_small.cue_pos.0,
            "MIN_SIZE must be the *smallest* odd size satisfying Invariant M"
        );
    }

    // ---------------------------------------------------------------------
    // The acceptance test for the dead-RNG recall bug: the cue type was never
    // sampled from `_rng` in `reset`, so the fork match didn't depend on the
    // cue and the "memory" task was winnable without remembering anything.
    // ---------------------------------------------------------------------

    /// Locate two seeds whose episodes share a fork order but differ in cue.
    fn contrasting_seeds(env: &mut MemoryEnv) -> (u64, u64) {
        let (mut key_cue, mut ball_cue) = (None, None);
        for seed in 0..512u64 {
            env.reset_with_seed(seed).expect("reset");
            let top = env
                .state()
                .grid
                .get(env.layout.top_pos.0, env.layout.top_pos.1);
            if top != KEY {
                continue; // pin the fork order: Key on top, Ball on the bottom
            }
            match env.cue() {
                KEY if key_cue.is_none() => key_cue = Some(seed),
                BALL if ball_cue.is_none() => ball_cue = Some(seed),
                _ => {}
            }
            if key_cue.is_some() && ball_cue.is_some() {
                break;
            }
        }
        (
            key_cue.expect("a Key-cue episode with Key on top"),
            ball_cue.expect("a Ball-cue episode with Key on top"),
        )
    }

    /// Two episodes that differ **only** in the cue produce byte-identical
    /// observations at the fork, yet demand opposite actions.
    ///
    /// This is a mechanical proof that no reactive (memoryless) policy can beat
    /// chance here: at the decision point the two worlds are indistinguishable
    /// from the observation alone. It fails on the environment as it stood
    /// before the cue was sampled from `_rng` (the fork match didn't depend on
    /// the cue, so both episodes demanded the same action), and it would still
    /// fail after a naive "just sample the cue" fix on the old 7×5
    /// layout — the cue was re-readable from the fork. It also only passes
    /// because every object is green.
    #[test]
    fn test_memory_env_fork_observation_is_cue_invariant() {
        let mut env = env_default();
        let (key_seed, ball_seed) = contrasting_seeds(&mut env);

        let mut observations = Vec::new();
        let mut oracle_actions = Vec::new();
        for seed in [key_seed, ball_seed] {
            env.reset_with_seed(seed).expect("reset");
            let snap = drive_to_junction(&mut env);
            observations.push(*snap.observation());
            oracle_actions.push(oracle_turn(&env));
        }

        // Guard against a degenerate pass: the observations must be identical
        // *and* informative — both fork objects are plainly in view, so the
        // agent can see the choice it has to make, it simply cannot see which
        // arm is right. Facing East from (fork_x, mid), the upper object sits
        // two cells to the agent's left and the lower two to its right.
        let (left, right) = (VIEW_SIZE / 2 - 2, VIEW_SIZE / 2 + 2);
        let types: HashSet<u8> = [
            observations[0].view[VIEW_SIZE - 1][left][0],
            observations[0].view[VIEW_SIZE - 1][right][0],
        ]
        .into_iter()
        .collect();
        assert_eq!(
            types,
            HashSet::from([KEY.type_u8(), BALL.type_u8()]),
            "the fork must show one Key and one Ball in the observation"
        );

        assert_eq!(
            observations[0].view, observations[1].view,
            "the 7x7x3 view at the fork must be byte-identical across cue types"
        );
        assert_eq!(
            observations[0].agent_direction, observations[1].agent_direction,
            "the agent faces the same way in both episodes"
        );
        assert_eq!(
            observations[0], observations[1],
            "the whole observation at the fork must be cue-invariant"
        );
        assert_ne!(
            oracle_actions[0], oracle_actions[1],
            "…yet the correct action differs, so the observation cannot determine it"
        );
    }

    // ---------------------------------------------------------------------
    // The RNG is live (ADR 0029)
    // ---------------------------------------------------------------------

    #[test]
    fn test_memory_env_samples_cue_and_fork_order_uniformly() {
        const N: usize = 400;
        let mut env = env_default();
        let (mut key_cues, mut key_on_top) = (0usize, 0usize);
        for _ in 0..N {
            env.reset().expect("reset");
            if env.cue() == KEY {
                key_cues += 1;
            }
            if env
                .state()
                .grid
                .get(env.layout.top_pos.0, env.layout.top_pos.1)
                == KEY
            {
                key_on_top += 1;
            }
        }
        // Generous bounds: this asserts "the RNG is live and advancing", not the
        // quality of StdRng. A dead RNG lands on 0 or N.
        assert!(
            (N / 3..2 * N / 3).contains(&key_cues),
            "cue type must be ~uniform over {N} resets, saw {key_cues} Keys"
        );
        assert!(
            (N / 3..2 * N / 3).contains(&key_on_top),
            "fork order must be ~uniform over {N} resets, saw {key_on_top} Keys on top"
        );
    }

    /// Catches the ADR-0029 re-seed bug directly: if `reset()` re-seeded from
    /// `config.seed`, every episode would replay the identical cue and fork.
    #[test]
    fn test_memory_env_consecutive_resets_differ() {
        let mut env = env_default();
        let mut episodes = HashSet::new();
        for _ in 0..32 {
            env.reset().expect("reset");
            episodes.insert((env.cue(), env.match_pos()));
        }
        assert!(
            episodes.len() > 1,
            "reset() must draw from the persistent stream, not re-seed from config"
        );
    }

    #[test]
    fn test_memory_env_reset_with_seed_is_reproducible() {
        let mut env = env_default();
        // Read the *upper fork object* — derived from the layout, not a literal,
        // so this keeps testing the fork order if the default size ever moves.
        let top = env.layout.top_pos;
        let top_obj = |e: &MemoryEnv| e.state().grid.get(top.0, top.1);

        env.reset_with_seed(1234).expect("reset");
        let first = (env.cue(), env.match_pos(), top_obj(&env));

        // Advance the stream so a re-seed is doing real work.
        for _ in 0..5 {
            env.reset().expect("reset");
        }

        env.reset_with_seed(1234).expect("reset");
        let second = (env.cue(), env.match_pos(), top_obj(&env));
        assert_eq!(
            first, second,
            "reset_with_seed must replay an episode bit-for-bit"
        );
    }

    // ---------------------------------------------------------------------
    // Policy behaviour
    // ---------------------------------------------------------------------

    #[test]
    fn test_memory_env_oracle_policy_always_wins() {
        const N: usize = 100;
        let mut env = env_default();
        for episode in 0..N {
            env.reset().expect("reset");
            drive_to_junction(&mut env);
            let turn = oracle_turn(&env);
            let reward = answer(&mut env, turn);
            assert!(
                reward > 0.9,
                "a cue-reading oracle must win every episode (failed on {episode})"
            );
        }
    }

    #[test]
    fn test_memory_env_fixed_side_policy_scores_near_chance() {
        const N: usize = 200;
        let mut env = env_default();
        let mut wins = 0usize;
        for _ in 0..N {
            env.reset().expect("reset");
            drive_to_junction(&mut env);
            // The reactive shortcut the old implementation rewarded: always go
            // for the same side. It must now be worth exactly a coin flip.
            let reward = answer(&mut env, GridAction::TurnLeft);
            if reward > 0.0 {
                wins += 1;
            }
        }
        assert!(
            (N / 3..2 * N / 3).contains(&wins),
            "a fixed-side policy must score near chance, got {wins}/{N}"
        );
    }

    #[test]
    fn test_memory_env_match_pos_is_derived_from_cue() {
        let mut env = env_default();
        for _ in 0..32 {
            env.reset().expect("reset");
            let (mx, my) = env.match_pos();
            assert_eq!(
                env.state().grid.get(mx, my),
                env.cue(),
                "match_pos must hold the cue-typed object"
            );
            let other = if (mx, my) == env.layout.top_pos {
                env.layout.bottom_pos
            } else {
                env.layout.top_pos
            };
            let distractor = env.state().grid.get(other.0, other.1);
            assert_ne!(
                distractor,
                env.cue(),
                "the distractor must be the *other* object type"
            );
            assert_eq!(
                distractor.color_u8(),
                OBJECT_COLOR.to_u8(),
                "the distractor must be green too — colour may not leak the answer"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Termination branches
    // ---------------------------------------------------------------------

    #[test]
    fn test_memory_env_done_at_distractor_pays_zero() {
        let mut env = env_default();
        env.reset().expect("reset");
        drive_to_junction(&mut env);
        let wrong = match oracle_turn(&env) {
            GridAction::TurnLeft => GridAction::TurnRight,
            _ => GridAction::TurnLeft,
        };
        let reward = answer(&mut env, wrong);
        assert_eq!(
            reward, 0.0,
            "Done facing the distractor must terminate with zero reward"
        );
    }

    #[test]
    fn test_memory_env_done_in_corridor_pays_zero() {
        let mut env = env_default();
        env.reset().expect("reset");
        let snap = env.step(GridAction::Done).expect("done");
        assert!(snap.is_done(), "Done always terminates");
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0, "Done away from the fork pays nothing");
    }

    #[test]
    fn test_memory_env_done_facing_the_cue_pays_zero() {
        // The cue object is the *same* Entity value as its matching fork object.
        // Rewarding a bare `entity_in_front == cue` compare would let the agent
        // win from the start room without ever walking the corridor.
        let mut env = env_default();
        env.reset().expect("reset");
        env.step(GridAction::TurnLeft).expect("turn"); // East -> North
        let front = env.state().agent.front();
        assert_eq!(front, env.layout.cue_pos, "agent now faces the cue");
        assert_eq!(
            env.state().grid.get(front.0, front.1),
            env.cue(),
            "…and the cell in front really is the cue entity"
        );
        let snap = env.step(GridAction::Done).expect("done");
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(
            reward, 0.0,
            "the cue itself is not a valid answer — only fork objects are"
        );
    }

    #[test]
    fn test_memory_env_times_out_with_zero_reward() {
        let mut env = MemoryEnv::with_config(MemoryConfig::new(11, 3, 0), false).expect("valid");
        env.reset().expect("reset");
        env.step(GridAction::TurnLeft).expect("step");
        env.step(GridAction::TurnLeft).expect("step");
        let snap = env.step(GridAction::TurnLeft).expect("step");
        assert!(snap.is_done(), "the step budget must truncate the episode");
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0, "a timeout pays nothing");
        assert_eq!(env.steps(), 3, "step counter tracks the budget");
    }

    #[test]
    fn test_memory_env_reset_clears_step_counter() {
        let mut env = env_default();
        env.reset().expect("reset");
        env.step(GridAction::Forward).expect("step");
        assert_eq!(env.steps(), 1, "step counter advances");
        env.reset().expect("reset");
        assert_eq!(env.steps(), 0, "reset clears the step counter");
    }

    #[test]
    fn test_memory_env_display_names_size_and_budget() {
        let env = env_default();
        let s = env.to_string();
        assert!(
            s.contains("size=13"),
            "Display must report the size, got {s}"
        );
        assert!(
            s.contains("0/845"),
            "Display must report the budget, got {s}"
        );
    }

    // ---------------------------------------------------------------------
    // Post-terminal step guard (ADR 0044).
    //
    // Per ADR 0044, `step()` after the episode ends must return `Err`, not
    // silently resurrect a finished episode. This closes a real correctness
    // gap, not just step-count hygiene: elsewhere in this family
    // (`dynamic_obstacles.rs`), stepping past a terminal collision without
    // `reset()` let internal state mutate against a stale, already-terminal
    // snapshot. `EpisodeGuard::check()` as the first statement of `step()`
    // makes environment invariants hold unconditionally rather than
    // depending on callers never stepping past a terminal snapshot.
    // ---------------------------------------------------------------------

    /// Reset, walk to the junction, commit to a fork object, and answer.
    ///
    /// Drives the whole episode through real `step()` calls — no hand-written
    /// terminal snapshot, no poking at `env.state`. With `correct = true` the
    /// agent turns toward the matching object (success terminal); with `false`
    /// it turns toward the distractor (failure terminal). Both are genuine
    /// `Done` terminations, reached well inside the step budget, so neither
    /// exercises `step()`'s other exit path: a `max_steps` cutoff (see the
    /// `done` computation in [`MemoryEnv::step`]) currently folds into the
    /// same bare `bool` as a real terminal, and `build_snapshot`
    /// (`grids/core/mod.rs`) has no `Truncated` arm to route it to — so a
    /// timed-out episode is reported as `Terminated`, not `Truncated`. That
    /// mislabels the value-function bootstrap target: a truncated episode
    /// still has future value, a true terminal has none (`docs/rules.md`
    /// §10, `EpisodeStatus`).
    fn drive_to_commit(env: &mut MemoryEnv, correct: bool) -> GridSnapshot {
        env.reset().expect("reset");
        drive_to_junction(env);
        let turn = match (oracle_turn(env), correct) {
            (t, true) => t,
            (GridAction::TurnLeft, false) => GridAction::TurnRight,
            (_, false) => GridAction::TurnLeft,
        };
        env.step(turn).expect("turn");
        env.step(GridAction::Forward).expect("forward");
        let snap = env.step(GridAction::Done).expect("done");
        assert!(snap.is_done(), "Done at the fork must end the episode");
        snap
    }

    /// Conformance: after a **correct** commit the episode is over, and a
    /// further legal `Done` must be refused rather than paying out again.
    #[test]
    fn test_memory_env_rejects_post_terminal_step_after_correct_answer() {
        let mut env = env_default();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| drive_to_commit(env, true),
            GridAction::Done,
        );
    }

    /// Conformance for the other terminal: a **wrong** commit ends the episode
    /// too, and the guard must reject afterwards just as firmly. This is the
    /// half that matters most here — unguarded, the agent could retract a wrong
    /// answer and go take the other object.
    #[test]
    fn test_memory_env_rejects_post_terminal_step_after_wrong_answer() {
        let mut env = env_default();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| drive_to_commit(env, false),
            GridAction::Done,
        );
    }

    /// Regression for the concrete defect: `Done` leaves the grid and the agent
    /// untouched, so `facing_match()` stayed true and every further `Done`
    /// re-paid `success_reward` while `steps` climbed. A rejected step must
    /// change nothing at all.
    #[test]
    fn test_memory_env_post_terminal_step_pays_nothing_and_mutates_nothing() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = env_default();
        let terminal = drive_to_commit(&mut env, true);
        let paid: f32 = (*terminal.reward()).into();
        assert!(paid > 0.0, "the correct answer must have been paid once");

        let steps_at_end = env.steps();
        let agent_at_end = env.state().agent;

        let err = env
            .step(GridAction::Done)
            .expect_err("a second Done must be refused, not paid a second success_reward");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status, terminal.status,
                "the error must carry the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }

        assert_eq!(
            env.steps(),
            steps_at_end,
            "a rejected step must not advance the step counter"
        );
        assert_eq!(
            (env.state().agent.x, env.state().agent.y),
            (agent_at_end.x, agent_at_end.y),
            "a rejected step must not move the agent"
        );
        assert_eq!(
            env.state().agent.direction,
            agent_at_end.direction,
            "a rejected step must not turn the agent"
        );
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Terminated,
            "a rejected step must not reopen the episode"
        );
    }

    /// `reset()` re-opens a terminated episode, so a latched guard cannot strand
    /// the environment for the rest of a run.
    #[test]
    fn test_memory_env_reset_reopens_terminated_episode() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = env_default();
        drive_to_commit(&mut env, false);
        assert!(
            env.step(GridAction::Done).is_err(),
            "the episode has ended; a step must be rejected before reset()"
        );

        env.reset().expect("reset must succeed after termination");
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );

        let snap = env
            .step(GridAction::Forward)
            .expect("reset() must re-open the environment for a new episode");
        assert!(
            !snap.is_done(),
            "the first step of a fresh episode must not be done"
        );
    }

    #[test]
    fn test_memory_env_larger_size_is_solvable() {
        let mut env =
            MemoryEnv::with_config(MemoryConfig::new(17, default_max_steps(17), 3), false)
                .expect("valid config");
        for _ in 0..8 {
            env.reset().expect("reset");
            drive_to_junction(&mut env);
            let turn = oracle_turn(&env);
            let reward = answer(&mut env, turn);
            assert!(reward > 0.0, "the oracle must win at size 17 too");
        }
    }
}
