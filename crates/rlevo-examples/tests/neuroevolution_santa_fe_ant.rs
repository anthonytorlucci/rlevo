//! Cross-crate agent test for the Santa Fe neuroevolution example (issue #69).
//!
//! `#[path]`-includes the example support module so its `#[cfg(test)]` unit tests
//! (weight round-trip, rollout range, recurrence-carries-state) compile and run
//! here, and adds the headline `#[ignore]`'d agent test: a short GRU evolution run
//! must beat a uniform-random baseline **and** exceed the memoryless reflex
//! plateau — the constructive proof that memory is load-bearing on this POMDP
//! (the complement to #68's reflex-plateau doc-test, the negative control).

#[path = "../examples/evolution/santa_fe_ant_support.rs"]
mod support;

use burn::backend::Flex;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::classic::santa_fe_ant::{SantaFeAnt, SantaFeAntAction, SantaFeAntConfig};
use rlevo_test_support::baseline::{random_return, uniform_discrete};
use rlevo_test_support::flex::flex_guard;

use support::{MAX_STEPS, ScoreMode, quick_evolve_gru};

type B = Flex;

/// Mean pellets eaten by a uniform-random action policy over `episodes` episodes.
fn random_baseline(seed: u64, episodes: usize) -> f32 {
    let mut env = SantaFeAnt::with_config(SantaFeAntConfig {
        max_steps: MAX_STEPS,
        render: false,
    })
    .expect("valid config");
    let mut rng = StdRng::seed_from_u64(seed);
    random_return(
        &mut env,
        episodes,
        MAX_STEPS,
        &mut rng,
        uniform_discrete::<1, SantaFeAntAction>,
    )
}

/// Pellets eaten by the memoryless reflex policy (`food_ahead ? Move : TurnRight`).
/// This is the plateau a recurrent policy must beat to prove memory pays off.
fn reflex_plateau() -> f32 {
    let mut env = SantaFeAnt::with_config(SantaFeAntConfig {
        max_steps: MAX_STEPS,
        render: false,
    })
    .expect("valid config");
    let mut snap = env.reset().expect("reset");
    let mut eaten = 0.0_f32;
    loop {
        let action = if snap.observation().food_ahead {
            SantaFeAntAction::Move
        } else {
            SantaFeAntAction::TurnRight
        };
        snap = env.step(action).expect("step");
        eaten += f32::from(*snap.reward());
        if snap.is_done() {
            break;
        }
    }
    eaten
}

#[test]
#[ignore = "neuroevolution run (~2-3 min on CPU): an evolved recurrent GRU policy \
            on the Santa Fe Trail must beat a uniform-random baseline by >= 10 \
            pellets AND exceed the memoryless reflex plateau (observed: \
            random~12, reflex~11, evolved~58) — run with `cargo test -- --ignored`"]
fn neuroevolution_santa_fe_ant_improves_over_random() {
    let _guard = flex_guard();
    let seed = 42;

    let random = random_baseline(seed, 30);
    let reflex = reflex_plateau();
    let summary =
        quick_evolve_gru::<B>(ScoreMode::Deterministic, seed, 48, 60, &Default::default());
    let evolved = summary.best_pellets;

    eprintln!("random={random:.1}  reflex={reflex:.1}  evolved={evolved:.1}");

    assert!(
        evolved > random + 10.0,
        "evolved policy ({evolved:.1}) failed to beat random baseline ({random:.1}) by >= 10 pellets",
    );
    assert!(
        evolved > reflex,
        "evolved recurrent policy ({evolved:.1}) did not exceed the memoryless reflex plateau \
         ({reflex:.1}) — memory is not load-bearing in this run",
    );
}
