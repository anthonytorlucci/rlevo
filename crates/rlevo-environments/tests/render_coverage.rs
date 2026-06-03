//! Cross-family regression guard for `AsciiRenderable` coverage.
//!
//! Every public env in the five in-scope 2D families (classic, grids,
//! toy_text, landscapes, box2d) implements [`AsciiRenderable`]. This
//! integration test enumerates one representative env per family — and
//! one of every box2d variant since they share no parent type — and
//! asserts the rendering contract:
//!
//! 1. `render_ascii()` returns at least one non-empty line;
//! 2. every plain line fits within the 80-column budget;
//! 3. `render_styled().plain_text()` round-trips to the same characters as
//!    `render_ascii()` modulo trailing newlines.
//!
//! When a new env family lands, add a `pub fn check_<env_name>()` here.
//! When an existing env's render changes shape, this test catches the
//! drift without having to re-read every per-env unit test.
//!
//! Locomotion is intentionally absent — locomotion envs have no
//! library-tier ASCII renderer; the report tier owns them via
//! `FamilyPayload::Locomotion2D`.

use rlevo_core::environment::{ConstructableEnv, Environment};
use rlevo_environments::render::AsciiRenderable;

/// Run the three invariants against any [`AsciiRenderable`].
fn assert_render_invariants<R: AsciiRenderable>(env: &R, name: &str) {
    let plain = env.render_ascii();
    assert!(
        !plain.is_empty(),
        "{name}: render_ascii() returned an empty string"
    );

    for (idx, line) in plain.lines().enumerate() {
        assert!(
            line.chars().count() <= 80,
            "{name}: render_ascii line {idx} exceeds 80 cols ({} chars): {line:?}",
            line.chars().count()
        );
    }

    let styled = env.render_styled();
    assert!(
        !styled.is_empty(),
        "{name}: render_styled() returned an empty frame"
    );

    // Plain projection of the styled frame should equal render_ascii with
    // trailing newlines stripped (render_ascii may end with `\n`; styled
    // frames carry line breaks structurally rather than as glyphs).
    let plain_no_trailing: String = plain.lines().collect::<Vec<_>>().join("\n");
    assert_eq!(
        styled.plain_text(),
        plain_no_trailing,
        "{name}: styled.plain_text() must equal render_ascii() modulo trailing newlines"
    );
}

// ─── classic ──────────────────────────────────────────────────────────────────

#[test]
fn classic_mountain_car() {
    use rlevo_environments::classic::mountain_car::MountainCar;
    let mut env = MountainCar::new(false);
    env.reset().unwrap();
    assert_render_invariants(&env, "MountainCar");
}

#[test]
fn classic_cartpole() {
    use rlevo_environments::classic::cartpole::CartPole;
    let mut env = CartPole::new(false);
    env.reset().unwrap();
    assert_render_invariants(&env, "CartPole");
}

#[test]
fn classic_pendulum() {
    use rlevo_environments::classic::pendulum::Pendulum;
    let env = Pendulum::new(false);
    assert_render_invariants(&env, "Pendulum");
}

#[test]
fn classic_acrobot() {
    use rlevo_environments::classic::acrobot::Acrobot;
    let mut env = Acrobot::new(false);
    env.reset().unwrap();
    assert_render_invariants(&env, "Acrobot");
}

#[test]
fn classic_mountain_car_continuous() {
    use rlevo_environments::classic::mountain_car_continuous::MountainCarContinuous;
    let mut env = MountainCarContinuous::new(false);
    env.reset().unwrap();
    assert_render_invariants(&env, "MountainCarContinuous");
}

#[test]
fn classic_k_armed_bandit() {
    use rlevo_environments::classic::bandit::KArmedBandit;
    let env: KArmedBandit<10> = KArmedBandit::with_seed(7);
    assert_render_invariants(&env, "KArmedBandit");
}

#[test]
fn classic_contextual_bandit() {
    use rlevo_environments::classic::bandit::ContextualBandit;
    let env: ContextualBandit<10, 4> = ContextualBandit::with_seed(7);
    assert_render_invariants(&env, "ContextualBandit");
}

#[test]
fn classic_adversarial_bandit() {
    use rlevo_environments::classic::bandit::AdversarialBandit;
    let env: AdversarialBandit<10> = AdversarialBandit::with_seed(7);
    assert_render_invariants(&env, "AdversarialBandit");
}

#[test]
fn classic_non_stationary_bandit() {
    use rlevo_environments::classic::bandit::NonStationaryBandit;
    let env: NonStationaryBandit<10> = NonStationaryBandit::with_seed(7);
    assert_render_invariants(&env, "NonStationaryBandit");
}

// ─── grids ────────────────────────────────────────────────────────────────────

#[test]
fn grids_empty() {
    use rlevo_environments::grids::empty::{EmptyConfig, EmptyEnv};
    let mut env = EmptyEnv::with_config(EmptyConfig::default(), false);
    env.reset().unwrap();
    assert_render_invariants(&env, "EmptyEnv");
}

#[test]
fn grids_lava_gap() {
    use rlevo_environments::grids::lava_gap::{LavaGapConfig, LavaGapEnv};
    let mut env = LavaGapEnv::with_config(LavaGapConfig::default(), false);
    env.reset().unwrap();
    assert_render_invariants(&env, "LavaGapEnv");
}

#[test]
fn grids_door_key() {
    use rlevo_environments::grids::door_key::{DoorKeyConfig, DoorKeyEnv};
    let mut env = DoorKeyEnv::with_config(DoorKeyConfig::default(), false);
    env.reset().unwrap();
    assert_render_invariants(&env, "DoorKeyEnv");
}

// ─── toy_text ─────────────────────────────────────────────────────────────────

#[test]
fn toy_text_frozen_lake() {
    use rlevo_environments::toy_text::frozen_lake::{FrozenLake, FrozenLakeConfig};
    let mut env = FrozenLake::with_config(FrozenLakeConfig::default()).unwrap();
    env.reset().unwrap();
    assert_render_invariants(&env, "FrozenLake");
}

#[test]
fn toy_text_cliff_walking() {
    use rlevo_environments::toy_text::cliff_walking::{CliffWalking, CliffWalkingConfig};
    let mut env = CliffWalking::with_config(CliffWalkingConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "CliffWalking");
}

#[test]
fn toy_text_taxi() {
    use rlevo_environments::toy_text::taxi::{Taxi, TaxiConfig};
    let mut env = Taxi::with_config(TaxiConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "Taxi");
}

#[test]
fn toy_text_blackjack() {
    use rlevo_environments::toy_text::blackjack::{Blackjack, BlackjackConfig};
    let mut env = Blackjack::with_config(BlackjackConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "Blackjack");
}

// ─── landscapes ───────────────────────────────────────────────────────────────

#[test]
fn landscapes_sphere() {
    use rlevo_environments::landscapes::sphere::Sphere;
    let env = Sphere::new(2);
    assert_render_invariants(&env, "Sphere");
}

#[test]
fn landscapes_rastrigin() {
    use rlevo_environments::landscapes::rastrigin::Rastrigin;
    let env = Rastrigin::new(2);
    assert_render_invariants(&env, "Rastrigin");
}

#[test]
fn landscapes_ackley() {
    use rlevo_environments::landscapes::ackley::Ackley;
    let env = Ackley::new(2);
    assert_render_invariants(&env, "Ackley");
}

// ─── box2d ────────────────────────────────────────────────────────────────────

#[cfg(feature = "box2d")]
#[test]
fn box2d_lunar_lander_discrete() {
    use rlevo_environments::box2d::lunar_lander::{LunarLanderConfig, LunarLanderDiscrete};
    let mut env = LunarLanderDiscrete::with_config(LunarLanderConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "LunarLanderDiscrete");
}

#[cfg(feature = "box2d")]
#[test]
fn box2d_lunar_lander_continuous() {
    use rlevo_environments::box2d::lunar_lander::{LunarLanderConfig, LunarLanderContinuous};
    let mut env = LunarLanderContinuous::with_config(LunarLanderConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "LunarLanderContinuous");
}

#[cfg(feature = "box2d")]
#[test]
fn box2d_bipedal_walker() {
    use rlevo_environments::box2d::bipedal_walker::{BipedalWalker, BipedalWalkerConfig};
    let mut env = BipedalWalker::with_config(BipedalWalkerConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "BipedalWalker");
}

#[cfg(feature = "box2d")]
#[test]
fn box2d_car_racing() {
    use rlevo_environments::box2d::car_racing::{CarRacing, CarRacingConfig};
    let mut env = CarRacing::with_config(CarRacingConfig::default());
    env.reset().unwrap();
    assert_render_invariants(&env, "CarRacing");
}
