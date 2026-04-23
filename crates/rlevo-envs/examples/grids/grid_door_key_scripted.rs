//! Scripted rollout of [`DoorKeyEnv`] with an ASCII trace per step.
//!
//! This example walks through the canonical 8-action optimal solution
//! for the 5×5 `DoorKey` layout and prints the ASCII render of the grid
//! before and after each step. It is meant as a tutorial for readers
//! learning the grid env state machine — the printed trace makes it
//! obvious which action triggered which state change.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo-envs --example grid_door_key_scripted
//! ```

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_envs::grids::core::GridAction;
use rlevo_envs::grids::{DoorKeyConfig, DoorKeyEnv};

/// Canonical 8-action solution for the default 5×5 `DoorKey` layout.
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
        "DoorKeyEnv scripted rollout — size={} steps={}",
        cfg.size,
        SCRIPT.len()
    );

    let mut env = DoorKeyEnv::with_config(cfg, false);
    env.reset().expect("reset");

    println!("\ninitial state:");
    println!("{}", env.ascii());

    for (i, action) in SCRIPT.iter().enumerate() {
        let snap = env.step(*action).expect("step");
        println!("step {:>2} action={action:?}", i + 1);
        println!("{}", env.ascii());
        if snap.is_done() {
            let reward = f32::from(*snap.reward());
            println!("terminated after {} steps with reward = {reward:.4}", i + 1);
            return;
        }
    }

    println!("script exhausted without terminating (this should not happen).");
}
