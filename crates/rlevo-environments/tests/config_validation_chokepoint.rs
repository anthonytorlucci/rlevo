//! Conformance test for the ADR 0026 **consumption chokepoint** contract.
//!
//! # Why this file exists
//!
//! ADR 0026 places the config-validation obligation on the **consumer**, not on
//! config construction. `*Config` types in this crate legitimately expose `pub`
//! fields — struct-literal and `..Default::default()` construction of an invalid
//! config is *allowed by design*, and is meant to be caught by the constructor
//! that consumes it (`Env::with_config(cfg)` → `cfg.validate()?`).
//!
//! A prior review round argued that because all 32 config structs have `pub`
//! fields, struct-literal construction bypasses `validate()`, and read that
//! `pub` field as the bug. It is not: at the time of that review, every one of
//! this crate's 35 environment `with_config` constructors — covering 34
//! distinct `*Config` types, since `LunarLanderDiscrete` and
//! `LunarLanderContinuous` each have a `with_config` consuming the *same*
//! `LunarLanderConfig` — and all 8 RL agent constructors already called
//! `config.validate()?`. (The RL figure is the same 8/8 ADR 0055 reports:
//! `dqn`, `c51`, `qrdqn`, `ddpg`, `td3`, `sac`, `ppo`, `ppg`.) The real risk
//! that review was groping at is that **nothing pinned the convention** — a
//! new environment whose `with_config` forgot `validate()?` would ship silently,
//! and no test would fail.
//!
//! # The second failure mode: chokepoint present, predicate incomplete
//!
//! A second, distinct failure mode was found later: "forgot to call
//! `validate()`" is not the only way the contract breaks, and empirically not
//! the common one. The chokepoint can be present and running while its
//! `validate()` is simply **incomplete** — the call returns `Ok(())` and the
//! constructor then builds a broken (or panicking) env from a config the
//! contract was supposed to have refused. Two measured examples from the
//! pre-fix tree:
//!
//! - `EmptyEnv::with_config(EmptyConfig { size: 1, ..Default::default() }, false)`
//!   panicked at `empty.rs:277` with a `usize` underflow — *after* its
//!   `config.validate()` had returned `Ok(())`.
//! - `UnlockPickupEnv::with_config(UnlockPickupConfig { size: 3, ..Default::default() }, false)`
//!   returned `Ok`, having built a board whose cells were
//!   `[Wall, Wall, Wall, Wall, Key(Yellow), Box(Purple), Wall, Wall, Wall]` — no
//!   `Door` cell at all, the key having silently overwritten it.
//!
//! This file pins that mode too, and it is why the cases below assert a
//! *specific* `field` and `ConstraintKind` per constraint rather than merely that
//! some rejection happened: an incomplete predicate is invisible to any assertion
//! that only asks whether `validate()` ran.
//!
//! None of this retracts the refutation above of the `pub`-field theory — that
//! analysis was substantially right, and the mechanism it named remains
//! false. `pub` fields do not bypass anything, and `#[non_exhaustive]` would
//! not have prevented either construction: both were **same-crate**, where
//! the cross-crate restriction never applies, and more fundamentally the
//! attribute restricts *construction syntax*, not *predicate completeness* —
//! a `size = 1` arriving through a hand-written setter or builder hits the
//! same incomplete `validate()` and panics identically.
//!
//! This file is that pin. For each environment config with a non-vacuous
//! [`Validate`] impl, it hands a deliberately invalid config to the public
//! constructor and asserts the constructor **rejects it** with the exact
//! [`ConfigError`] — config name, field name, and [`ConstraintKind`] variant.
//! That claim is meant literally: every `rlevo-environments` config with a real
//! constraint is covered below, and the only omissions are the vacuous impls
//! listed at the end of this comment. Adding an environment means adding a case
//! here.
//!
//! # Scope: `rlevo-environments` only, by decision
//!
//! Configs in `rlevo-evolution`, `rlevo-reinforcement-learning`, and
//! `rlevo-benchmarks` are deliberately **out of scope for this file** — evolution
//! validates through a single generic bound at
//! `crates/rlevo-evolution/src/strategy.rs:505-517` (one chokepoint, not 30, so a
//! per-config sweep would assert the same line repeatedly), RL validates
//! per-agent inside each `new`, and `rlevo-benchmarks` has no [`Validate`] impl
//! at all. Those crates need their own conformance pins; a single-crate
//! integration test cannot reach them (`rules.md` §5 — cross-crate tests live in
//! `crates/rlevo/tests/`).
//!
//! # Refactor trap — do not weaken these assertions to `is_err()`
//!
//! Every case asserts the **structured** error, never a stringified message.
//! ADR 0026 made `ConfigError` allocation-free and `PartialEq` precisely so
//! kind-level assertions are cheap. A bare `assert!(result.is_err())` would keep
//! passing if a constructor started rejecting for an unrelated reason (a map
//! parse failure, a physics-init failure, a different field's guard firing
//! first), which is exactly the silent regression this file exists to catch.
//! Assert the `field` and the `ConstraintKind`, or the test proves nothing.
//!
//! # Which `ConstraintKind` a size floor gets
//!
//! Several grid configs reject a `size` (or a room count) that is too small, and
//! the *variant* they reject with is a deliberate distinction that these cases
//! pin:
//!
//! - [`ConstraintKind::TooSmall`] — via `config::at_least` — is the variant for a
//!   plain minimum-count floor. The nine `grids/` envs with a `MIN_SIZE`-style
//!   constant use it, including the ones whose floor is a *policy* choice rather
//!   than a hard geometric limit: measured on this checkout, `lava_gap` builds a
//!   clean board at `4` (floor `5`), `crossing` at `5` (floor `7`),
//!   `unlock_pickup` at `5` (floor `7`), `four_rooms` at `9` (floor `11`), and
//!   `multi_room` at `height = 3` (floor `5`). A machine-readable
//!   `TooSmall { min, got }` is the honest report for those — the caller can act
//!   on the number.
//! - [`ConstraintKind::Custom`] is reserved for a floor (or a shape) derived from
//!   a non-obvious invariant, where the prose payload carries information the
//!   `min` alone cannot: `memory.rs`'s Invariant M, `go_to_door.rs`'s
//!   four-distinct-doors requirement, and `four_rooms`' odd-`size` parity rule.
//!
//! So do not "upgrade" a `TooSmall` case here to `assert_rejected_custom`: that
//! is a strictly weaker assertion (it drops `min` and `got`) *and* it would
//! contradict the convention. Conversely, do not fabricate a `Custom` expectation
//! for one of the nine floors — none of them is geometry-derived.
//!
//! # Placement
//!
//! `rules.md §5` — purpose is *verification*, scope is **one crate's public
//! surface** (`rlevo_environments`'s `with_config` constructors), so this is a
//! single-crate integration test in `crates/rlevo-environments/tests/`.
//!
//! # Intentionally excluded: vacuous `Validate` impls
//!
//! `TaxiConfig` (`toy_text/taxi.rs`), `BlackjackConfig` (`toy_text/blackjack.rs`),
//! and `CliffWalkingConfig` (`toy_text/cliff_walking.rs`) return an unconditional
//! `Ok(())` — their configs carry no checkable invariant (fixed-layout grids plus
//! a seed). There is no invalid value to construct, so no conformance case is
//! written for them. Do **not** fabricate one; if any of those configs later
//! grows a real constraint, add its case here at the same time.

use rlevo_core::config::{ConfigError, ConstraintKind};

use rlevo_environments::classic::{
    Acrobot, AcrobotConfig, AdversarialBandit, AdversarialBanditConfig, BookDynamics, CartPole,
    CartPoleConfig, ContextualBandit, ContextualBanditConfig, KArmedBandit, KArmedBanditConfig,
    MountainCar, MountainCarConfig, MountainCarContinuous, MountainCarContinuousConfig,
    NonStationaryBandit, NonStationaryBanditConfig, Pendulum, PendulumConfig, SantaFeAnt,
    SantaFeAntConfig,
};
use rlevo_environments::grids::{
    CrossingConfig, CrossingEnv, DistShiftConfig, DistShiftEnv, DoorKeyConfig, DoorKeyEnv,
    DynamicObstaclesConfig, DynamicObstaclesEnv, EmptyConfig, EmptyEnv, FourRoomsConfig,
    FourRoomsEnv, GoToDoorConfig, GoToDoorEnv, LavaGapConfig, LavaGapEnv, MemoryConfig, MemoryEnv,
    MultiRoomConfig, MultiRoomEnv, UnlockConfig, UnlockEnv, UnlockPickupConfig, UnlockPickupEnv,
};
use rlevo_environments::pixel_grid::{PixelGridConfig, PixelGridEnv};
use rlevo_environments::toy_text::MapError;
use rlevo_environments::toy_text::frozen_lake::{FrozenLake, FrozenLakeConfig};

/// Asserts that a config-consuming constructor rejected its input with exactly
/// the expected [`ConfigError`].
///
/// Takes the outcome by value and matches rather than calling `unwrap_err`, so
/// the environment type is not required to implement `Debug`.
///
/// # Panics
///
/// Panics if the constructor returned `Ok`, or if any of the three structured
/// fields (`config`, `field`, `kind`) differs from the expectation.
#[track_caller]
fn assert_rejected<T>(
    outcome: Result<T, ConfigError>,
    config: &'static str,
    field: &'static str,
    kind: &ConstraintKind,
) {
    match outcome {
        Ok(_) => panic!(
            "{config}::with_config accepted an invalid `{field}` — the ADR 0026 \
             consumption chokepoint is missing a `config.validate()?` call"
        ),
        Err(err) => {
            assert_eq!(
                err.config, config,
                "rejection must name the offending config type"
            );
            assert_eq!(
                err.field, field,
                "rejection must name the offending field, not merely fail"
            );
            assert_eq!(
                &err.kind, kind,
                "rejection must report the violated constraint kind for `{field}`"
            );
        }
    }
}

/// Same as [`assert_rejected`], but only pins the `ConstraintKind` *variant* for
/// [`ConstraintKind::Custom`] cases, whose `&'static str` payload is a private
/// const in the owning module and therefore unnameable from an integration test.
///
/// # Panics
///
/// Panics if the constructor returned `Ok`, or if the error is not a
/// `Custom` rejection naming `field`.
#[track_caller]
fn assert_rejected_custom<T>(
    outcome: Result<T, ConfigError>,
    config: &'static str,
    field: &'static str,
) {
    match outcome {
        Ok(_) => panic!(
            "{config}::with_config accepted an invalid `{field}` — the ADR 0026 \
             consumption chokepoint is missing a `config.validate()?` call"
        ),
        Err(err) => {
            assert_eq!(err.config, config, "rejection must name the config type");
            assert_eq!(err.field, field, "rejection must name the offending field");
            assert!(
                matches!(err.kind, ConstraintKind::Custom(_)),
                "`{field}` must be rejected as ConstraintKind::Custom, got {:?}",
                err.kind
            );
        }
    }
}

// ---------------------------------------------------------------------------
// classic control
// ---------------------------------------------------------------------------

#[test]
fn test_cartpole_with_config_rejects_nonpositive_masscart() {
    let cfg = CartPoleConfig {
        masscart: 0.0,
        ..CartPoleConfig::default()
    };
    assert_rejected(
        CartPole::with_config(cfg),
        "CartPoleConfig",
        "masscart",
        &ConstraintKind::NotPositive { got: 0.0 },
    );
}

#[test]
fn test_acrobot_with_config_rejects_nonpositive_link_mass() {
    let cfg = AcrobotConfig {
        link_mass_1: -1.0,
        ..AcrobotConfig::default()
    };
    assert_rejected(
        Acrobot::<BookDynamics>::with_config(cfg),
        "AcrobotConfig",
        "link_mass_1",
        &ConstraintKind::NotPositive { got: -1.0 },
    );
}

#[test]
fn test_pendulum_with_config_rejects_nonpositive_dt() {
    let cfg = PendulumConfig {
        dt: 0.0,
        ..PendulumConfig::default()
    };
    assert_rejected(
        Pendulum::with_config(cfg),
        "PendulumConfig",
        "dt",
        &ConstraintKind::NotPositive { got: 0.0 },
    );
}

#[test]
fn test_mountain_car_with_config_rejects_nonpositive_max_speed() {
    let cfg = MountainCarConfig {
        max_speed: 0.0,
        ..MountainCarConfig::default()
    };
    assert_rejected(
        MountainCar::with_config(cfg),
        "MountainCarConfig",
        "max_speed",
        &ConstraintKind::NotPositive { got: 0.0 },
    );
}

#[test]
fn test_mountain_car_continuous_with_config_rejects_nonpositive_power() {
    let cfg = MountainCarContinuousConfig {
        power: 0.0,
        ..MountainCarContinuousConfig::default()
    };
    assert_rejected(
        MountainCarContinuous::with_config(cfg),
        "MountainCarContinuousConfig",
        "power",
        &ConstraintKind::NotPositive { got: 0.0 },
    );
}

#[test]
fn test_santa_fe_ant_with_config_rejects_zero_max_steps() {
    let cfg = SantaFeAntConfig {
        max_steps: 0,
        ..SantaFeAntConfig::default()
    };
    assert_rejected(
        SantaFeAnt::with_config(cfg),
        "SantaFeAntConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

// ---------------------------------------------------------------------------
// bandits
// ---------------------------------------------------------------------------

#[test]
fn test_k_armed_bandit_with_config_rejects_zero_max_steps() {
    let cfg = KArmedBanditConfig {
        max_steps: 0,
        ..KArmedBanditConfig::default()
    };
    assert_rejected(
        KArmedBandit::<10>::with_config(cfg),
        "KArmedBanditConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

/// `period` is checked *after* `max_steps`, so a valid `max_steps` is kept here
/// deliberately: this case proves the second guard is reached, not just the
/// first.
#[test]
fn test_adversarial_bandit_with_config_rejects_zero_period() {
    let cfg = AdversarialBanditConfig {
        period: 0,
        ..AdversarialBanditConfig::default()
    };
    assert_rejected(
        AdversarialBandit::<10>::with_config(cfg),
        "AdversarialBanditConfig",
        "period",
        &ConstraintKind::Zero,
    );
}

#[test]
fn test_non_stationary_bandit_with_config_rejects_negative_sigma_walk() {
    let cfg = NonStationaryBanditConfig {
        sigma_walk: -1.0,
        ..NonStationaryBanditConfig::default()
    };
    assert_rejected(
        NonStationaryBandit::<10>::with_config(cfg),
        "NonStationaryBanditConfig",
        "sigma_walk",
        &ConstraintKind::OutOfRange {
            lo: 0.0,
            hi: f64::INFINITY,
            got: -1.0,
        },
    );
}

#[test]
fn test_contextual_bandit_with_config_rejects_zero_max_steps() {
    let cfg = ContextualBanditConfig {
        max_steps: 0,
        ..ContextualBanditConfig::default()
    };
    assert_rejected(
        ContextualBandit::<4, 10>::with_config(cfg),
        "ContextualBanditConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

// ---------------------------------------------------------------------------
// grid worlds
// ---------------------------------------------------------------------------

#[test]
fn test_empty_grid_with_config_rejects_zero_size() {
    let cfg = EmptyConfig {
        size: 0,
        ..EmptyConfig::default()
    };
    assert_rejected(
        EmptyEnv::with_config(cfg, false),
        "EmptyConfig",
        "size",
        &ConstraintKind::TooSmall { min: 4, got: 0 },
    );
}

/// `EmptyConfig` floors `size` at `MIN_SIZE` (4). `3` is deliberately *not* `0`:
/// a zero would also be caught by any generic `nonzero` guard, so only a
/// non-zero sub-floor value proves the real floor is enforced.
#[test]
fn test_empty_grid_with_config_rejects_size_below_min() {
    let cfg = EmptyConfig {
        size: 3,
        ..EmptyConfig::default()
    };
    assert_rejected(
        EmptyEnv::with_config(cfg, false),
        "EmptyConfig",
        "size",
        &ConstraintKind::TooSmall { min: 4, got: 3 },
    );
}

#[test]
fn test_dist_shift_with_config_rejects_zero_max_steps() {
    let cfg = DistShiftConfig {
        max_steps: 0,
        ..DistShiftConfig::default()
    };
    assert_rejected(
        DistShiftEnv::with_config(cfg, false),
        "DistShiftConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

#[test]
fn test_multi_room_with_config_rejects_zero_num_rooms() {
    let cfg = MultiRoomConfig {
        num_rooms: 0,
        ..MultiRoomConfig::default()
    };
    assert_rejected(
        MultiRoomEnv::with_config(cfg, false),
        "MultiRoomConfig",
        "num_rooms",
        &ConstraintKind::TooSmall { min: 2, got: 0 },
    );
}

// `MultiRoomConfig` is the only grid config with *three* independent floors —
// `num_rooms` >= 2, `room_width` >= 3, `height` >= 5 — checked in that order.
// Each gets its own sub-floor case, with the other two left at their defaults,
// so a case cannot pass on the strength of an earlier guard firing.

/// `num_rooms = 1` is the value that actually motivated the floor: it built a
/// **single-room** env with no dividing wall and therefore no door to unlock.
#[test]
fn test_multi_room_with_config_rejects_num_rooms_below_min() {
    let cfg = MultiRoomConfig {
        num_rooms: 1,
        ..MultiRoomConfig::default()
    };
    assert_rejected(
        MultiRoomEnv::with_config(cfg, false),
        "MultiRoomConfig",
        "num_rooms",
        &ConstraintKind::TooSmall { min: 2, got: 1 },
    );
}

/// `room_width` floors at `MIN_ROOM_WIDTH` (3); `2` leaves each room a single
/// interior column.
#[test]
fn test_multi_room_with_config_rejects_room_width_below_min() {
    let cfg = MultiRoomConfig {
        room_width: 2,
        ..MultiRoomConfig::default()
    };
    assert_rejected(
        MultiRoomEnv::with_config(cfg, false),
        "MultiRoomConfig",
        "room_width",
        &ConstraintKind::TooSmall { min: 3, got: 2 },
    );
}

/// `height` floors at `MIN_HEIGHT` (5). `4` is a *policy* rejection, not a
/// geometric one — the board at `height = 3` still builds cleanly — which is
/// exactly why the floor reports `TooSmall` and not `Custom`.
#[test]
fn test_multi_room_with_config_rejects_height_below_min() {
    let cfg = MultiRoomConfig {
        height: 4,
        ..MultiRoomConfig::default()
    };
    assert_rejected(
        MultiRoomEnv::with_config(cfg, false),
        "MultiRoomConfig",
        "height",
        &ConstraintKind::TooSmall { min: 5, got: 4 },
    );
}

// The seven grid configs below share one `at_least(size, MIN_SIZE)` +
// `nonzero(max_steps)` shape, and the block pins it in three tiers:
//
//   1. `size = 0`      → `TooSmall { min: MIN_SIZE, got: 0 }`. Pins that the
//      floor, not a leftover `nonzero`, is what rejects a zero. Before the fix
//      these cases expected `Zero`, which is precisely the confusion the tier
//      structure now rules out.
//   2. `size = MIN - 1` → `TooSmall { min: MIN_SIZE, got: MIN - 1 }`. This is the
//      tier that actually pins the defect. A `size = 0` case alone cannot
//      distinguish a real floor from an incidental non-zero guard, because both
//      reject `0`; only a non-zero sub-floor value can. Each `MIN_SIZE` differs
//      per env (`empty` 4, `unlock` 4, `dynamic_obstacles` 5, `door_key` 5,
//      `lava_gap` 5, `crossing` 7, `unlock_pickup` 7, `four_rooms` 11), so the
//      expected `min` is per-env too — do not copy one env's number to another.
//   3. `max_steps = 0` → `Zero`. `max_steps` has no floor above 1, so it keeps
//      `nonzero`; it is the one field in these configs whose rejection is still
//      `ConstraintKind::Zero`.
//
// The broken field stays alternated across the block, as it always was, but for
// a restated reason: `size` is checked *before* `max_steps`, so a `size` case can
// only ever prove the first guard runs. The envs that zero `max_steps` (leaving
// `size` valid) are what prove the *second* guard is reached at all. Do not
// normalize the block onto `size` — that would drop every proof that validation
// continues past the first `?`. Every env is covered by tier 2; tiers 1 and 3 are
// split across the block on purpose.

#[test]
fn test_crossing_with_config_rejects_zero_size() {
    let cfg = CrossingConfig {
        size: 0,
        ..CrossingConfig::default()
    };
    assert_rejected(
        CrossingEnv::with_config(cfg, false),
        "CrossingConfig",
        "size",
        &ConstraintKind::TooSmall { min: 7, got: 0 },
    );
}

/// Tier 2 for `crossing`: `MIN_SIZE` is `7`, and `6` is non-zero, so only the
/// floor can reject it. Measured on this checkout, `size = 5` still builds a
/// clean board — the floor is policy, which is why `TooSmall` (with its
/// machine-readable `min`) is the right variant rather than a `Custom` prose
/// claim about geometry.
#[test]
fn test_crossing_with_config_rejects_size_below_min() {
    let cfg = CrossingConfig {
        size: 6,
        ..CrossingConfig::default()
    };
    assert_rejected(
        CrossingEnv::with_config(cfg, false),
        "CrossingConfig",
        "size",
        &ConstraintKind::TooSmall { min: 7, got: 6 },
    );
}

#[test]
fn test_door_key_with_config_rejects_zero_max_steps() {
    let cfg = DoorKeyConfig {
        max_steps: 0,
        ..DoorKeyConfig::default()
    };
    assert_rejected(
        DoorKeyEnv::with_config(cfg, false),
        "DoorKeyConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

/// Tier 2 for `door_key`: `MIN_SIZE` is `5`, so `4` is the largest rejected
/// `size`. Reported as `TooSmall`, unlike sibling `go_to_door`'s same-valued
/// floor, which stays `Custom` because it encodes the four-distinct-doors
/// invariant rather than a plain count.
#[test]
fn test_door_key_with_config_rejects_size_below_min() {
    let cfg = DoorKeyConfig {
        size: 4,
        ..DoorKeyConfig::default()
    };
    assert_rejected(
        DoorKeyEnv::with_config(cfg, false),
        "DoorKeyConfig",
        "size",
        &ConstraintKind::TooSmall { min: 5, got: 4 },
    );
}

#[test]
fn test_dynamic_obstacles_with_config_rejects_zero_size() {
    let cfg = DynamicObstaclesConfig {
        size: 0,
        ..DynamicObstaclesConfig::default()
    };
    assert_rejected(
        DynamicObstaclesEnv::with_config(cfg, false),
        "DynamicObstaclesConfig",
        "size",
        &ConstraintKind::TooSmall { min: 5, got: 0 },
    );
}

/// Tier 2 for `dynamic_obstacles`: `MIN_SIZE` is `5`, so `4` is non-zero and
/// still below the floor.
#[test]
fn test_dynamic_obstacles_with_config_rejects_size_below_min() {
    let cfg = DynamicObstaclesConfig {
        size: 4,
        ..DynamicObstaclesConfig::default()
    };
    assert_rejected(
        DynamicObstaclesEnv::with_config(cfg, false),
        "DynamicObstaclesConfig",
        "size",
        &ConstraintKind::TooSmall { min: 5, got: 4 },
    );
}

#[test]
fn test_four_rooms_with_config_rejects_zero_max_steps() {
    let cfg = FourRoomsConfig {
        max_steps: 0,
        ..FourRoomsConfig::default()
    };
    assert_rejected(
        FourRoomsEnv::with_config(cfg, false),
        "FourRoomsConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

/// Tier 2 for `four_rooms`: `MIN_SIZE` is `11`. `9` is chosen because it is
/// **odd** — it therefore satisfies the parity guard, isolating the floor from
/// it. (A board at `9` builds cleanly on this checkout; the floor is policy, so
/// `TooSmall` is the honest variant.) The parity guard is exercised separately
/// by `test_four_rooms_with_config_rejects_even_size` below.
#[test]
fn test_four_rooms_with_config_rejects_size_below_min() {
    let cfg = FourRoomsConfig {
        size: 9,
        ..FourRoomsConfig::default()
    };
    assert_rejected(
        FourRoomsEnv::with_config(cfg, false),
        "FourRoomsConfig",
        "size",
        &ConstraintKind::TooSmall { min: 11, got: 9 },
    );
}

/// `FourRoomsEnv` also requires an **odd** `size`, so the interior cross has a
/// single centre row and column. `14` is even but comfortably above `MIN_SIZE`
/// (11), which isolates the parity guard from the floor: the `at_least` check
/// runs first, so any value that is *both* sub-floor and even would report
/// `TooSmall` and prove nothing about parity.
///
/// `14` and not `12`, and the value must **not** be a multiple of 3: `12` is
/// divisible by both 2 and 3, so mutating the guard's modulus
/// (`size.is_multiple_of(2)` → `is_multiple_of(3)` in `four_rooms.rs`) left this
/// case, and the entire suite, green. `14 % 3 == 2`, so the modulus is now pinned
/// as well as the parity intent. Do not "simplify" this back to `12`.
///
/// This is the one `four_rooms` `size` case that is `Custom` rather than
/// `TooSmall` — parity has no `min` to report, so the prose payload is the whole
/// information content.
#[test]
fn test_four_rooms_with_config_rejects_even_size() {
    let cfg = FourRoomsConfig {
        size: 14,
        ..FourRoomsConfig::default()
    };
    assert_rejected_custom(
        FourRoomsEnv::with_config(cfg, false),
        "FourRoomsConfig",
        "size",
    );
}

#[test]
fn test_lava_gap_with_config_rejects_zero_size() {
    let cfg = LavaGapConfig {
        size: 0,
        ..LavaGapConfig::default()
    };
    assert_rejected(
        LavaGapEnv::with_config(cfg, false),
        "LavaGapConfig",
        "size",
        &ConstraintKind::TooSmall { min: 5, got: 0 },
    );
}

/// Tier 2 for `lava_gap`: `MIN_SIZE` is `5`. Measured on this checkout a board
/// at `4` still builds cleanly, so this rejection is a policy floor — asserted
/// as `TooSmall`, never as a `Custom` geometric claim.
#[test]
fn test_lava_gap_with_config_rejects_size_below_min() {
    let cfg = LavaGapConfig {
        size: 4,
        ..LavaGapConfig::default()
    };
    assert_rejected(
        LavaGapEnv::with_config(cfg, false),
        "LavaGapConfig",
        "size",
        &ConstraintKind::TooSmall { min: 5, got: 4 },
    );
}

#[test]
fn test_unlock_with_config_rejects_zero_size() {
    let cfg = UnlockConfig {
        size: 0,
        ..UnlockConfig::default()
    };
    assert_rejected(
        UnlockEnv::with_config(cfg, false),
        "UnlockConfig",
        "size",
        &ConstraintKind::TooSmall { min: 4, got: 0 },
    );
}

/// Tier 2 for `unlock`: `MIN_SIZE` is `4`, so `3` is the largest rejected size.
#[test]
fn test_unlock_with_config_rejects_size_below_min() {
    let cfg = UnlockConfig {
        size: 3,
        ..UnlockConfig::default()
    };
    assert_rejected(
        UnlockEnv::with_config(cfg, false),
        "UnlockConfig",
        "size",
        &ConstraintKind::TooSmall { min: 4, got: 3 },
    );
}

#[test]
fn test_unlock_pickup_with_config_rejects_zero_max_steps() {
    let cfg = UnlockPickupConfig {
        max_steps: 0,
        ..UnlockPickupConfig::default()
    };
    assert_rejected(
        UnlockPickupEnv::with_config(cfg, false),
        "UnlockPickupConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

/// Tier 2 for `unlock_pickup`: `MIN_SIZE` is `7`, twice the sibling `unlock`
/// floor (4) because this env must host a second room plus a box. `6` is
/// non-zero, so only the floor can reject it. Measured on this checkout a board
/// at `5` still builds cleanly — policy floor, hence `TooSmall`.
#[test]
fn test_unlock_pickup_with_config_rejects_size_below_min() {
    let cfg = UnlockPickupConfig {
        size: 6,
        ..UnlockPickupConfig::default()
    };
    assert_rejected(
        UnlockPickupEnv::with_config(cfg, false),
        "UnlockPickupConfig",
        "size",
        &ConstraintKind::TooSmall { min: 7, got: 6 },
    );
}

/// `GoToDoorEnv` needs `size >= MIN_SIZE` (5) to host four distinct doors; a
/// `size` of 1 is well below it. `1` is deliberately *not* `0` — `0` would also
/// trip a generic `nonzero` guard, so it could not distinguish the real
/// `MIN_SIZE` constraint from an incidental one.
#[test]
fn test_go_to_door_with_config_rejects_size_below_min() {
    let cfg = GoToDoorConfig {
        size: 1,
        ..GoToDoorConfig::default()
    };
    assert_rejected_custom(
        GoToDoorEnv::with_config(cfg, false),
        "GoToDoorConfig",
        "size",
    );
}

/// `MemoryEnv` requires an **odd** `size` (the corridor needs a centre column).
/// An even size that is otherwise large enough isolates the parity guard from
/// the minimum-size guard.
#[test]
fn test_memory_with_config_rejects_even_size() {
    let cfg = MemoryConfig {
        size: 8,
        ..MemoryConfig::default()
    };
    assert_rejected_custom(MemoryEnv::with_config(cfg, false), "MemoryConfig", "size");
}

// ---------------------------------------------------------------------------
// pixel grid (modality-changing env, ADR 0020)
// ---------------------------------------------------------------------------

#[test]
fn test_pixel_grid_with_config_rejects_zero_max_steps() {
    let cfg = PixelGridConfig {
        max_steps: 0,
        ..PixelGridConfig::default()
    };
    assert_rejected(
        PixelGridEnv::with_config(cfg, false),
        "PixelGridConfig",
        "max_steps",
        &ConstraintKind::Zero,
    );
}

// ---------------------------------------------------------------------------
// toy text
// ---------------------------------------------------------------------------

/// `FrozenLake::with_config` returns [`MapError`], not [`ConfigError`] — the map
/// itself can fail to parse. The chokepoint contract still holds: the config
/// rejection arrives *transparently wrapped* in `MapError::InvalidConfig`, and
/// this test unwraps to the same structured `ConfigError` every other case
/// asserts on. Do not relax this to `is_err()`: a `MapError::GoalUnreachable`
/// would satisfy that while proving `validate()` was never called.
#[test]
fn test_frozen_lake_with_config_rejects_out_of_range_success_rate() {
    let cfg = FrozenLakeConfig {
        success_rate: 1.5,
        ..FrozenLakeConfig::default()
    };
    match FrozenLake::with_config(cfg) {
        Ok(_) => panic!(
            "FrozenLake::with_config accepted success_rate = 1.5 — the ADR 0026 \
             consumption chokepoint is missing a `config.validate()?` call"
        ),
        Err(MapError::InvalidConfig(err)) => {
            assert_eq!(err.config, "FrozenLakeConfig");
            assert_eq!(err.field, "success_rate");
            assert_eq!(
                err.kind,
                ConstraintKind::OutOfRange {
                    lo: 0.0,
                    hi: 1.0,
                    got: 1.5
                },
                "success_rate must be rejected as out of [0, 1]"
            );
        }
        Err(other) => panic!(
            "an invalid success_rate must surface as MapError::InvalidConfig, got {other:?} — \
             this means the map path failed before `validate()` ran"
        ),
    }
}

// ---------------------------------------------------------------------------
// feature-gated physics environments
//
// Gated so this file also conforms under `--no-default-features`; without the
// gate a lean build would fail to compile rather than simply skipping the case.
// ---------------------------------------------------------------------------

#[cfg(feature = "locomotion")]
mod locomotion {
    use super::{ConstraintKind, assert_rejected};
    use rlevo_environments::locomotion::inverted_double_pendulum::{
        InvertedDoublePendulum, InvertedDoublePendulumConfig,
    };
    use rlevo_environments::locomotion::inverted_pendulum::{
        InvertedPendulum, InvertedPendulumConfig,
    };
    use rlevo_environments::locomotion::reacher::{Reacher, ReacherConfig};
    use rlevo_environments::locomotion::swimmer::{Swimmer, SwimmerConfig};

    #[test]
    fn test_inverted_pendulum_with_config_rejects_nonpositive_cart_mass() {
        let cfg = InvertedPendulumConfig {
            cart_mass: 0.0,
            ..InvertedPendulumConfig::default()
        };
        assert_rejected(
            InvertedPendulum::with_config(cfg),
            "InvertedPendulumConfig",
            "cart_mass",
            &ConstraintKind::NotPositive { got: 0.0 },
        );
    }

    /// Breaks `pole_mass` rather than `cart_mass` so this case cannot pass on
    /// the strength of the sibling `InvertedPendulum` test above — the two
    /// configs share a field vocabulary but are separate `Validate` impls behind
    /// separate constructors.
    #[test]
    fn test_inverted_double_pendulum_with_config_rejects_nonpositive_pole_mass() {
        let cfg = InvertedDoublePendulumConfig {
            pole_mass: 0.0,
            ..InvertedDoublePendulumConfig::default()
        };
        assert_rejected(
            InvertedDoublePendulum::with_config(cfg),
            "InvertedDoublePendulumConfig",
            "pole_mass",
            &ConstraintKind::NotPositive { got: 0.0 },
        );
    }

    #[test]
    fn test_swimmer_with_config_rejects_nonpositive_dt() {
        let cfg = SwimmerConfig {
            dt: 0.0,
            ..SwimmerConfig::default()
        };
        assert_rejected(
            Swimmer::with_config(cfg),
            "SwimmerConfig",
            "dt",
            &ConstraintKind::NotPositive { got: 0.0 },
        );
    }

    #[test]
    fn test_reacher_with_config_rejects_nonpositive_link_length() {
        let cfg = ReacherConfig {
            link1_length: -1.0,
            ..ReacherConfig::default()
        };
        assert_rejected(
            Reacher::with_config(cfg),
            "ReacherConfig",
            "link1_length",
            &ConstraintKind::NotPositive { got: -1.0 },
        );
    }
}

#[cfg(feature = "box2d")]
mod box2d {
    use super::{ConstraintKind, assert_rejected};
    use rlevo_environments::box2d::bipedal_walker::{BipedalWalker, BipedalWalkerConfig};
    use rlevo_environments::box2d::car_racing::{CarRacing, CarRacingConfig};
    use rlevo_environments::box2d::lunar_lander::{LunarLanderConfig, LunarLanderDiscrete};

    #[test]
    fn test_lunar_lander_with_config_rejects_nonpositive_main_engine_power() {
        let cfg = LunarLanderConfig {
            main_engine_power: 0.0,
            ..LunarLanderConfig::default()
        };
        assert_rejected(
            LunarLanderDiscrete::with_config(cfg),
            "LunarLanderConfig",
            "main_engine_power",
            &ConstraintKind::NotPositive { got: 0.0 },
        );
    }

    #[test]
    fn test_car_racing_with_config_rejects_zero_checkpoints() {
        let cfg = CarRacingConfig {
            track_n_checkpoints: 0,
            ..CarRacingConfig::default()
        };
        assert_rejected(
            CarRacing::with_config(cfg),
            "CarRacingConfig",
            "track_n_checkpoints",
            &ConstraintKind::Zero,
        );
    }

    #[test]
    fn test_bipedal_walker_with_config_rejects_nonpositive_motors_torque() {
        let cfg = BipedalWalkerConfig {
            motors_torque: 0.0,
            ..BipedalWalkerConfig::default()
        };
        assert_rejected(
            BipedalWalker::with_config(cfg),
            "BipedalWalkerConfig",
            "motors_torque",
            &ConstraintKind::NotPositive { got: 0.0 },
        );
    }
}

// ---------------------------------------------------------------------------
// The dormant `Deserialize` gap
// ---------------------------------------------------------------------------

/// **Characterization test — pins current, intentional behaviour. Do not
/// "fix" this into a failing deserialize.**
///
/// `serde`'s derived `Deserialize` has no hook into [`Validate`], so decoding an
/// invalid `GoToDoorConfig` (`size = 1`, below `MIN_SIZE = 5`) succeeds. Under
/// ADR 0026 that is *correct*: the validation obligation sits on the **consumer**
/// — "the loader calls `validate()` and propagates `Err`" — not on the
/// deserializer. The constructor is still the chokepoint, and it still rejects.
///
/// Nothing in the workspace deserializes a config today, so the gap is dormant.
/// It stops being dormant the moment someone adds a config loader: that loader
/// must call `validate()` itself, and this test is where the two halves of the
/// contract are stated side by side. If a future change makes deserialization
/// itself validating (e.g. `#[serde(try_from = ...)]`), update this test *and*
/// ADR 0026 together — do not silently flip the assertion.
#[test]
fn test_deserialize_admits_invalid_config_but_constructor_still_rejects() {
    let bincode_cfg = bincode::config::standard();

    // A config that `validate()` rejects, round-tripped through serde.
    let invalid = GoToDoorConfig {
        size: 1,
        ..GoToDoorConfig::default()
    };
    let bytes = bincode::serde::encode_to_vec(invalid, bincode_cfg).expect("encode");
    let (decoded, _): (GoToDoorConfig, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode_cfg)
            .expect("deserialization is NOT validating by design — see ADR 0026 §2");

    assert_eq!(
        decoded.size, 1,
        "the invalid `size` must survive the round trip verbatim; \
         a silently clamped value would hide the gap this test documents"
    );

    // ...and the consumption chokepoint is what catches it.
    assert_rejected_custom(
        GoToDoorEnv::with_config(decoded, false),
        "GoToDoorConfig",
        "size",
    );
}
