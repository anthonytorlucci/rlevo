//! Scripted rollout of [`DoorKeyEnv`] with an ASCII trace per step.
//!
//! This example walks through a canonical 8-action optimal solution for a 5×5
//! `DoorKey` layout and prints the ASCII render of the grid before and after
//! each step. It is meant as a tutorial for readers learning the grid env state
//! machine — the printed trace makes it obvious which action triggered which
//! state change.
//!
//! ## Why the seed is pinned
//!
//! `DoorKeyEnv` **samples its layout every episode** (ADR 0062): the split
//! column, the door's row, and the key, agent and goal placements are all drawn
//! fresh on each `reset`. A fixed action script is therefore not a property of
//! the *environment*, only of one particular board — call plain `reset()` here
//! and the script below solves roughly one board in forty.
//!
//! [`Environment::reset`] draws from the env's own stream, so it gives you a new
//! board each time; [`DoorKeyEnv::reset_with_seed`] is the replay hatch that
//! reproduces one exact episode. This example pins [`SEED`] because the *script*
//! is the lesson — the point is to watch `Locked → Closed → Open` unfold, not to
//! watch a search find a route. Any example that needs to survive an arbitrary
//! board must plan against the board it is handed instead (see
//! `rlevo-environments/tests/common`, the BFS the solvability oracles use).
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo --example grid_door_key_scripted
//! ```

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::grids::core::GridAction;
use rlevo_environments::grids::{DoorKeyConfig, DoorKeyEnv};

/// The episode [`SCRIPT`] was written against.
///
/// At `size = 5` this seed reproduces the layout `DoorKey` used to build
/// deterministically before it gained per-episode sampling: agent at `(1, 2)`
/// facing North, yellow key at `(1, 1)`, locked door at `(2, 2)`, goal at
/// `(3, 3)`.
///
/// ```text
/// # # # # #
/// # k # . #
/// # ^ * . #
/// # . # G #
/// # # # # #
/// ```
///
/// It is not the only seed the script clears — 100, 146, 171, 180, 214, 313,
/// 329 and 333 also work at this size — but it is the one whose board matches
/// the per-action comments below.
const SEED: u64 = 99;

/// Canonical 8-action solution for the 5×5 `DoorKey` board at [`SEED`].
const SCRIPT: [GridAction; 8] = [
    GridAction::Pickup,    // grab yellow key at (1, 1)
    GridAction::TurnRight, // face east toward the locked door
    GridAction::Toggle,    // unlock door at (2, 2) (Locked → Closed)
    GridAction::Toggle,    // open door (Closed → Open)
    GridAction::Forward,   // step onto the door cell
    GridAction::Forward,   // enter the right room at (3, 2)
    GridAction::TurnRight, // face south toward the goal
    GridAction::Forward,   // step onto the goal at (3, 3)
];

fn main() {
    let cfg = DoorKeyConfig::new(5, 100, 0);
    println!(
        "DoorKeyEnv scripted rollout — size={} seed={SEED} steps={}",
        cfg.size,
        SCRIPT.len()
    );

    let mut env = DoorKeyEnv::with_config(cfg, false).expect("valid config");
    // Not `reset()`: the layout is sampled per episode, and this script only
    // solves the board `SEED` produces.
    env.reset_with_seed(SEED).expect("reset");

    println!("\ninitial state:");
    println!("{}", env.ascii());

    for (i, action) in SCRIPT.iter().enumerate() {
        let snap = env.step(*action).expect("step");
        println!("step {:>2} action={action:?}", i + 1);
        println!("{}", env.ascii());
        if snap.is_done() {
            let reward = f32::from(*snap.reward());
            // Reaching the goal pays a positive, step-count-discounted reward;
            // walking into lava or running out of budget also terminates, but
            // pays 0.0. A tutorial that cannot tell those apart teaches nothing.
            assert!(
                reward > 0.0,
                "step {} ended the episode with reward {reward}: the script \
                 failed on the board seed {SEED} produced",
                i + 1
            );
            println!("terminated after {} steps with reward = {reward:.4}", i + 1);
            return;
        }
    }

    // Not a `println!`: an example that reports its own breakage on stdout and
    // then exits 0 is invisible to CI and to anyone running it in a script.
    panic!(
        "the {}-action script did not terminate the episode on seed {SEED}; \
         the layout DoorKey samples for this seed has changed",
        SCRIPT.len()
    );
}
