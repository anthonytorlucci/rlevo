//! [`RecordedEnvFamily`](rlevo_benchmarks::record::RecordedEnvFamily) impls for the built-in environments.
//!
//! These tie each concrete env type to the [`EnvFamily`] its recordings
//! belong to, so a driver can derive the family once from the env type
//! (`RecordingConfig::for_env::<E>` / `E::FAMILY`) instead of restating the
//! literal at every recording / TUI call site.
//!
//! Gated by the `record` feature — it is the env-side half of the harness's
//! `record` tier and, like the rest of [`crate::bench`], an opt-in coupling
//! to `rlevo-benchmarks` rather than a property of [`Environment`] itself
//! (ADR 0007).
//!
//! # `FAMILY` is a *render* family, not a module
//!
//! `FAMILY` selects the report-client adapter that decodes this env's frames,
//! so it follows the [`FamilyPayload`] variant the env emits — i.e. the
//! `*PayloadSource` trait it implements — not the module it happens to live
//! in. [`SantaFeAnt`] is the worked example: it lives in [`crate::classic`]
//! but implements `GridPayloadSource`, projects a `FamilyPayload::Grid`, and
//! therefore records as [`EnvFamily::Grids`]. An env with no payload source
//! (the bandits) stays on the ASCII path and takes the family whose adapter
//! it falls back through.
//!
//! # No env produces [`EnvFamily::Landscapes`]
//!
//! That variant has no producer on this side of the seam, and the gap is
//! deliberate: nothing in [`crate::landscapes`] implements [`Environment`].
//! Landscapes are fitness surfaces (they implement
//! [`Landscape`](rlevo_core::fitness::Landscape), consumed by
//! `rlevo-evolution`'s `FromLandscape`), not steppable environments, so there
//! is no env type to hang the impl on. `EnvFamily::Landscapes` is produced by
//! the evolution drivers, which pass the literal. Do not "fix" the absence by
//! inventing an env wrapper for it.
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`EnvFamily`]: rlevo_benchmarks::record::EnvFamily
//! [`EnvFamily::Grids`]: rlevo_benchmarks::record::EnvFamily::Grids
//! [`EnvFamily::Landscapes`]: rlevo_benchmarks::record::EnvFamily::Landscapes
//! [`FamilyPayload`]: rlevo_benchmarks::record::FamilyPayload
//! [`SantaFeAnt`]: crate::classic::SantaFeAnt

use rlevo_benchmarks::record::{EnvFamily, RecordedEnvFamily};

use crate::classic::acrobot::{Acrobot, AcrobotDynamicsFn};
use crate::classic::bandit::{
    AdversarialBandit, ContextualBandit, KArmedBandit, NonStationaryBandit,
};
use crate::classic::cartpole::CartPole;
use crate::classic::mountain_car::MountainCar;
use crate::classic::mountain_car_continuous::MountainCarContinuous;
use crate::classic::pendulum::Pendulum;
use crate::classic::santa_fe_ant::SantaFeAnt;
use crate::grids::{
    CrossingEnv, DistShiftEnv, DoorKeyEnv, DynamicObstaclesEnv, EmptyEnv, FourRoomsEnv,
    GoToDoorEnv, LavaGapEnv, MemoryEnv, MultiRoomEnv, UnlockEnv, UnlockPickupEnv,
};
use crate::toy_text::blackjack::Blackjack;
use crate::toy_text::cliff_walking::CliffWalking;
use crate::toy_text::frozen_lake::FrozenLake;
use crate::toy_text::taxi::Taxi;
use crate::wrappers::TimeLimit;

// ---------------------------------------------------------------------------
// Classic — `Classic2DPayloadSource` (structured 2-D line-art).
// ---------------------------------------------------------------------------

impl RecordedEnvFamily for CartPole {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

impl RecordedEnvFamily for Pendulum {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

impl RecordedEnvFamily for MountainCar {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

impl RecordedEnvFamily for MountainCarContinuous {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

// Bound is `AcrobotDynamicsFn` alone, not `+ Default`: a caller-supplied
// dynamics fn built through `with_config_and_dynamics` must still record. The
// impl is generic rather than one-per-instantiation because the struct's
// `D = BookDynamics` default makes bare `Acrobot` *be* `Acrobot<BookDynamics>`
// — a second impl for it would overlap (E0119).
impl<D: AcrobotDynamicsFn> RecordedEnvFamily for Acrobot<D> {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

// ---------------------------------------------------------------------------
// Classic — bandits.
//
// No payload source: a bandit has no spatial state to project, so it records
// `FamilyPayload::Ascii` and renders through the classic adapter's fallback
// path (which `report-client/src/adapters/classic.rs` documents as its
// expected home for them). Impls are generic over `K` (and `C`) — the
// `TenArmedBandit = KArmedBandit<10>` alias is transparent, so a dedicated
// impl for it would overlap this one (E0119).
// ---------------------------------------------------------------------------

impl<const K: usize> RecordedEnvFamily for KArmedBandit<K> {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

impl<const K: usize> RecordedEnvFamily for AdversarialBandit<K> {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

impl<const K: usize> RecordedEnvFamily for NonStationaryBandit<K> {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

impl<const C: usize, const K: usize> RecordedEnvFamily for ContextualBandit<C, K> {
    const FAMILY: EnvFamily = EnvFamily::Classic;
}

// ---------------------------------------------------------------------------
// Grids — `GridPayloadSource` (structured tile grid).
// ---------------------------------------------------------------------------

// The Santa Fe ant projects its 32×32 trail onto a `FamilyPayload::Grid`
// (it implements `GridPayloadSource`), so it records as the `Grids` family
// despite living in `classic/`.
impl RecordedEnvFamily for SantaFeAnt {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for EmptyEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for DoorKeyEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for MultiRoomEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for DynamicObstaclesEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for LavaGapEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for CrossingEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for MemoryEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for DistShiftEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for UnlockEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for UnlockPickupEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for FourRoomsEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

impl RecordedEnvFamily for GoToDoorEnv {
    const FAMILY: EnvFamily = EnvFamily::Grids;
}

// ---------------------------------------------------------------------------
// `ToyText` — `TabularPayloadSource` (tile grid or card table).
// ---------------------------------------------------------------------------

impl RecordedEnvFamily for FrozenLake {
    const FAMILY: EnvFamily = EnvFamily::ToyText;
}

impl RecordedEnvFamily for CliffWalking {
    const FAMILY: EnvFamily = EnvFamily::ToyText;
}

impl RecordedEnvFamily for Taxi {
    const FAMILY: EnvFamily = EnvFamily::ToyText;
}

impl RecordedEnvFamily for Blackjack {
    const FAMILY: EnvFamily = EnvFamily::ToyText;
}

// ---------------------------------------------------------------------------
// Box2D — behind the `box2d` feature (rapier2d).
// ---------------------------------------------------------------------------

#[cfg(feature = "box2d")]
impl RecordedEnvFamily for crate::box2d::lunar_lander::LunarLanderDiscrete {
    const FAMILY: EnvFamily = EnvFamily::Box2d;
}

#[cfg(feature = "box2d")]
impl RecordedEnvFamily for crate::box2d::lunar_lander::LunarLanderContinuous {
    const FAMILY: EnvFamily = EnvFamily::Box2d;
}

#[cfg(feature = "box2d")]
impl RecordedEnvFamily for crate::box2d::bipedal_walker::BipedalWalker {
    const FAMILY: EnvFamily = EnvFamily::Box2d;
}

#[cfg(feature = "box2d")]
impl RecordedEnvFamily for crate::box2d::car_racing::CarRacing {
    const FAMILY: EnvFamily = EnvFamily::Box2d;
}

// ---------------------------------------------------------------------------
// Locomotion — behind the `locomotion` feature (rapier3d).
//
// The env structs are generic over `B: LocomotionBackend` (defaulted to
// `Rapier3DBackend`), so the impls are generic too: that covers the
// `*Rapier` aliases and any future backend without a second, overlapping impl.
// ---------------------------------------------------------------------------

#[cfg(feature = "locomotion")]
impl<B: crate::locomotion::backend::LocomotionBackend> RecordedEnvFamily
    for crate::locomotion::inverted_pendulum::InvertedPendulum<B>
{
    const FAMILY: EnvFamily = EnvFamily::Locomotion;
}

#[cfg(feature = "locomotion")]
impl<B: crate::locomotion::backend::LocomotionBackend> RecordedEnvFamily
    for crate::locomotion::inverted_double_pendulum::InvertedDoublePendulum<B>
{
    const FAMILY: EnvFamily = EnvFamily::Locomotion;
}

#[cfg(feature = "locomotion")]
impl<B: crate::locomotion::backend::LocomotionBackend> RecordedEnvFamily
    for crate::locomotion::swimmer::Swimmer<B>
{
    const FAMILY: EnvFamily = EnvFamily::Locomotion;
}

#[cfg(feature = "locomotion")]
impl<B: crate::locomotion::backend::LocomotionBackend> RecordedEnvFamily
    for crate::locomotion::reacher::Reacher<B>
{
    const FAMILY: EnvFamily = EnvFamily::Locomotion;
}

// ---------------------------------------------------------------------------
// Wrappers.
// ---------------------------------------------------------------------------

/// Forward the family through the wrapper: a `TimeLimit` changes when an
/// episode ends, never how it renders. Mirrors the `Classic2DPayloadSource`
/// forwarding impl in `wrappers::time_limit`, so recording a wrapped env picks
/// the same adapter as recording the env directly.
impl<E: RecordedEnvFamily> RecordedEnvFamily for TimeLimit<E> {
    const FAMILY: EnvFamily = E::FAMILY;
}

#[cfg(test)]
mod tests {
    use rlevo_benchmarks::record::{Classic2DPayload, FamilyPayload, GridPayload, TabularPayload};
    use rlevo_core::environment::ConstructableEnv;
    use rlevo_core::render::payload::{
        Classic2DPayloadSource, GridPayloadSource, TabularPayloadSource,
    };

    use super::{
        Acrobot, AdversarialBandit, Blackjack, CartPole, CliffWalking, ContextualBandit,
        CrossingEnv, DistShiftEnv, DoorKeyEnv, DynamicObstaclesEnv, EmptyEnv, EnvFamily,
        FourRoomsEnv, FrozenLake, GoToDoorEnv, KArmedBandit, LavaGapEnv, MemoryEnv, MountainCar,
        MountainCarContinuous, MultiRoomEnv, NonStationaryBandit, Pendulum, RecordedEnvFamily,
        SantaFeAnt, Taxi, TimeLimit, UnlockEnv, UnlockPickupEnv,
    };
    use crate::classic::{NipsDynamics, TenArmedBandit};

    // -----------------------------------------------------------------------
    // Family ↔ payload consistency.
    //
    // `const FAMILY` and the env's `*PayloadSource` impl are two independent
    // hand-written facts with nothing holding them together: an env could emit
    // `FamilyPayload::Grid` while declaring `EnvFamily::Classic` and compile
    // clean, only to render through the classic adapter's fallback at report
    // time. The family cannot be *derived* from the payload trait (the blanket
    // impls would overlap), so this pairing is the only mechanism available.
    // -----------------------------------------------------------------------

    /// The single [`EnvFamily`] whose report adapter decodes each rich payload
    /// variant.
    fn family_of(payload: &FamilyPayload) -> EnvFamily {
        match payload {
            FamilyPayload::Classic2D(_) => EnvFamily::Classic,
            FamilyPayload::Grid(_) => EnvFamily::Grids,
            FamilyPayload::TabularText(_) => EnvFamily::ToyText,
            FamilyPayload::Box2dBodies(_) => EnvFamily::Box2d,
            FamilyPayload::Locomotion2D(_) => EnvFamily::Locomotion,
            FamilyPayload::Landscape2D(_) => EnvFamily::Landscapes,
            // `Ascii` is family-agnostic (any family may fall back to it) and
            // `FamilyPayload` is `#[non_exhaustive]`. Neither is reachable from
            // an env that has a payload source, which is all this maps.
            other => panic!("payload {other:?} does not identify a single family"),
        }
    }

    /// Assert `E`'s declared family agrees with the payload variant it emits.
    /// Each helper builds the payload exactly as `RecordingTap`'s per-family
    /// constructor does, so a drift between the two is a test failure rather
    /// than a wrong SVG.
    fn assert_classic2d_family<E>()
    where
        E: ConstructableEnv + Classic2DPayloadSource + RecordedEnvFamily,
    {
        let env = E::new(false);
        let emitted = FamilyPayload::Classic2D(Classic2DPayload::from(env.classic2d_snapshot()));
        assert_family(&emitted, E::FAMILY, std::any::type_name::<E>());
    }

    fn assert_grid_family<E>()
    where
        E: ConstructableEnv + GridPayloadSource + RecordedEnvFamily,
    {
        let env = E::new(false);
        let emitted = FamilyPayload::Grid(GridPayload::from(env.grid_snapshot()));
        assert_family(&emitted, E::FAMILY, std::any::type_name::<E>());
    }

    fn assert_tabular_family<E>()
    where
        E: ConstructableEnv + TabularPayloadSource + RecordedEnvFamily,
    {
        let env = E::new(false);
        let emitted = FamilyPayload::TabularText(TabularPayload::from(env.tabular_snapshot()));
        assert_family(&emitted, E::FAMILY, std::any::type_name::<E>());
    }

    #[cfg(feature = "box2d")]
    fn assert_box2d_family<E>()
    where
        E: ConstructableEnv + rlevo_core::render::payload::Box2dPayloadSource + RecordedEnvFamily,
    {
        use rlevo_benchmarks::record::Box2dPayload;

        let env = E::new(false);
        let emitted = FamilyPayload::Box2dBodies(Box2dPayload::from(env.box2d_snapshot()));
        assert_family(&emitted, E::FAMILY, std::any::type_name::<E>());
    }

    #[cfg(feature = "locomotion")]
    fn assert_locomotion_family<E>()
    where
        E: ConstructableEnv
            + rlevo_core::render::payload::Locomotion2DPayloadSource
            + RecordedEnvFamily,
    {
        use rlevo_benchmarks::record::Locomotion2DPayload;

        let env = E::new(false);
        let emitted =
            FamilyPayload::Locomotion2D(Locomotion2DPayload::from(env.locomotion2d_snapshot()));
        assert_family(&emitted, E::FAMILY, std::any::type_name::<E>());
    }

    fn assert_family(emitted: &FamilyPayload, declared: EnvFamily, env: &str) {
        assert_eq!(
            declared,
            family_of(emitted),
            "{env}: declared `RecordedEnvFamily::FAMILY` must match the family of the \
             `FamilyPayload` variant it records, or the report tier decodes it with the wrong \
             adapter"
        );
    }

    #[test]
    fn test_recorded_env_family_agrees_with_emitted_payload() {
        assert_classic2d_family::<CartPole>();
        assert_classic2d_family::<Pendulum>();
        assert_classic2d_family::<MountainCar>();
        assert_classic2d_family::<MountainCarContinuous>();
        assert_classic2d_family::<Acrobot>();

        assert_grid_family::<SantaFeAnt>();
        assert_grid_family::<EmptyEnv>();
        assert_grid_family::<DoorKeyEnv>();
        assert_grid_family::<MultiRoomEnv>();
        assert_grid_family::<DynamicObstaclesEnv>();
        assert_grid_family::<LavaGapEnv>();
        assert_grid_family::<CrossingEnv>();
        assert_grid_family::<MemoryEnv>();
        assert_grid_family::<DistShiftEnv>();
        assert_grid_family::<UnlockEnv>();
        assert_grid_family::<UnlockPickupEnv>();
        assert_grid_family::<FourRoomsEnv>();
        assert_grid_family::<GoToDoorEnv>();

        assert_tabular_family::<FrozenLake>();
        assert_tabular_family::<CliffWalking>();
        assert_tabular_family::<Taxi>();
        assert_tabular_family::<Blackjack>();

        // The remaining `box2d` / `locomotion` envs (`BipedalWalker`,
        // `CarRacing`, `InvertedDoublePendulum`, `Swimmer`, `Reacher`) have no
        // payload source yet, so there is no emitted variant to check them
        // against; their `FAMILY` is pinned below only.
        #[cfg(feature = "box2d")]
        {
            assert_box2d_family::<crate::box2d::lunar_lander::LunarLanderDiscrete>();
            assert_box2d_family::<crate::box2d::lunar_lander::LunarLanderContinuous>();
        }
        #[cfg(feature = "locomotion")]
        assert_locomotion_family::<crate::locomotion::inverted_pendulum::InvertedPendulumRapier>();
    }

    // -----------------------------------------------------------------------
    // Pinned assignments.
    // -----------------------------------------------------------------------

    #[test]
    fn test_recorded_env_family_pins_classic_envs() {
        assert_eq!(
            <CartPole as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "CartPole records as the classic family"
        );
        assert_eq!(
            <Pendulum as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "Pendulum records as the classic family"
        );
        assert_eq!(
            <MountainCar as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "MountainCar records as the classic family"
        );
        assert_eq!(
            <MountainCarContinuous as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "MountainCarContinuous records as the classic family"
        );
        assert_eq!(
            <Acrobot as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "Acrobot with the default dynamics records as the classic family"
        );
        assert_eq!(
            <Acrobot<NipsDynamics> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "the generic impl covers any dynamics fn, not just the default"
        );
    }

    #[test]
    fn test_recorded_env_family_pins_bandits() {
        assert_eq!(
            <KArmedBandit<4> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "a k-armed bandit falls back through the classic adapter"
        );
        assert_eq!(
            <AdversarialBandit<4> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "an adversarial bandit falls back through the classic adapter"
        );
        assert_eq!(
            <NonStationaryBandit<4> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "a non-stationary bandit falls back through the classic adapter"
        );
        assert_eq!(
            <ContextualBandit<3, 4> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "a contextual bandit falls back through the classic adapter"
        );
    }

    #[test]
    fn test_recorded_env_family_resolves_through_type_alias() {
        assert_eq!(
            <TenArmedBandit as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "the `TenArmedBandit = KArmedBandit<10>` alias resolves to the generic impl — \
             `bench::suites::ten_armed_bandit_suite` depends on it"
        );
    }

    #[test]
    fn test_recorded_env_family_pins_grid_envs() {
        for (env, family) in [
            ("SantaFeAnt", <SantaFeAnt as RecordedEnvFamily>::FAMILY),
            ("EmptyEnv", <EmptyEnv as RecordedEnvFamily>::FAMILY),
            ("DoorKeyEnv", <DoorKeyEnv as RecordedEnvFamily>::FAMILY),
            ("MultiRoomEnv", <MultiRoomEnv as RecordedEnvFamily>::FAMILY),
            (
                "DynamicObstaclesEnv",
                <DynamicObstaclesEnv as RecordedEnvFamily>::FAMILY,
            ),
            ("LavaGapEnv", <LavaGapEnv as RecordedEnvFamily>::FAMILY),
            ("CrossingEnv", <CrossingEnv as RecordedEnvFamily>::FAMILY),
            ("MemoryEnv", <MemoryEnv as RecordedEnvFamily>::FAMILY),
            ("DistShiftEnv", <DistShiftEnv as RecordedEnvFamily>::FAMILY),
            ("UnlockEnv", <UnlockEnv as RecordedEnvFamily>::FAMILY),
            (
                "UnlockPickupEnv",
                <UnlockPickupEnv as RecordedEnvFamily>::FAMILY,
            ),
            ("FourRoomsEnv", <FourRoomsEnv as RecordedEnvFamily>::FAMILY),
            ("GoToDoorEnv", <GoToDoorEnv as RecordedEnvFamily>::FAMILY),
        ] {
            assert_eq!(
                family,
                EnvFamily::Grids,
                "{env} projects a grid payload and must record as the grids family"
            );
        }
    }

    #[test]
    fn test_recorded_env_family_pins_toy_text_envs() {
        for (env, family) in [
            ("FrozenLake", <FrozenLake as RecordedEnvFamily>::FAMILY),
            ("CliffWalking", <CliffWalking as RecordedEnvFamily>::FAMILY),
            ("Taxi", <Taxi as RecordedEnvFamily>::FAMILY),
            ("Blackjack", <Blackjack as RecordedEnvFamily>::FAMILY),
        ] {
            assert_eq!(
                family,
                EnvFamily::ToyText,
                "{env} projects a tabular payload and must record as the toy-text family"
            );
        }
    }

    #[cfg(feature = "box2d")]
    #[test]
    fn test_recorded_env_family_pins_box2d_envs() {
        use crate::box2d::bipedal_walker::BipedalWalker;
        use crate::box2d::car_racing::CarRacing;
        use crate::box2d::lunar_lander::{LunarLanderContinuous, LunarLanderDiscrete};

        for (env, family) in [
            (
                "LunarLanderDiscrete",
                <LunarLanderDiscrete as RecordedEnvFamily>::FAMILY,
            ),
            (
                "LunarLanderContinuous",
                <LunarLanderContinuous as RecordedEnvFamily>::FAMILY,
            ),
            (
                "BipedalWalker",
                <BipedalWalker as RecordedEnvFamily>::FAMILY,
            ),
            ("CarRacing", <CarRacing as RecordedEnvFamily>::FAMILY),
        ] {
            assert_eq!(
                family,
                EnvFamily::Box2d,
                "{env} is a rigid-body env and must record as the box2d family"
            );
        }
    }

    #[cfg(feature = "locomotion")]
    #[test]
    fn test_recorded_env_family_pins_locomotion_envs() {
        use crate::locomotion::inverted_double_pendulum::InvertedDoublePendulumRapier;
        use crate::locomotion::inverted_pendulum::InvertedPendulumRapier;
        use crate::locomotion::reacher::ReacherRapier;
        use crate::locomotion::swimmer::SwimmerRapier;

        for (env, family) in [
            (
                "InvertedPendulum",
                <InvertedPendulumRapier as RecordedEnvFamily>::FAMILY,
            ),
            (
                "InvertedDoublePendulum",
                <InvertedDoublePendulumRapier as RecordedEnvFamily>::FAMILY,
            ),
            ("Swimmer", <SwimmerRapier as RecordedEnvFamily>::FAMILY),
            ("Reacher", <ReacherRapier as RecordedEnvFamily>::FAMILY),
        ] {
            assert_eq!(
                family,
                EnvFamily::Locomotion,
                "{env} is a locomotion env and must record as the locomotion family"
            );
        }
    }

    #[test]
    fn test_recorded_env_family_forwards_through_time_limit() {
        assert_eq!(
            <TimeLimit<CartPole> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Classic,
            "a `TimeLimit` changes when the episode ends, not how the env renders"
        );
        assert_eq!(
            <TimeLimit<EmptyEnv> as RecordedEnvFamily>::FAMILY,
            EnvFamily::Grids,
            "the forwarder carries the inner env's family, whatever it is"
        );
    }
}
