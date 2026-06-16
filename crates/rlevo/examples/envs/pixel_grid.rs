//! Scripted rollout of [`PixelGridEnv`] showing the modality change.
//!
//! The environment's true latent is a compact rank-1 pair of cell indices
//! `(agent, goal)`, but the observation the agent receives is a rank-3
//! `20×20×3` RGB image rendered by `Observable::project`. This example walks the
//! optimal 8-step path to the goal and, at each step, prints:
//!
//! - the latent `(agent, goal)` indices (the rank-1 state),
//! - the projected image's rank and shape (the rank-3 observation), and
//! - a glyph-redundant ASCII frame (`@` agent, `*` goal) — legible without
//!   color, so the agent/goal distinction never relies on hue alone.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p rlevo --example pixel_grid
//! ```

use rlevo_core::base::Observation;
use rlevo_core::environment::{Environment, Snapshot};
use rlevo_environments::pixel_grid::{
    PixelGridAction, PixelGridConfig, PixelGridEnv, PixelObservation,
};

/// Optimal path for the default fixed layout: agent at cell 0 (top-left),
/// goal at cell 24 (bottom-right) — four Downs then four Rights.
const SCRIPT: [PixelGridAction; 8] = [
    PixelGridAction::Down,
    PixelGridAction::Down,
    PixelGridAction::Down,
    PixelGridAction::Down,
    PixelGridAction::Right,
    PixelGridAction::Right,
    PixelGridAction::Right,
    PixelGridAction::Right,
];

fn main() {
    let shape = <PixelObservation as Observation<3>>::shape();
    println!(
        "PixelGridEnv scripted rollout — observation is rank {} {:?} over a rank-1 [2] state",
        <PixelObservation as Observation<3>>::RANK,
        shape,
    );

    let mut env = PixelGridEnv::with_config(PixelGridConfig::new(100, 0, false), false);
    let snap = env.reset().expect("reset");

    println!("\ninitial state — latent (agent={}, goal={})", env.state().agent(), env.state().goal());
    println!("projected image: {} pixels (= {:?})", snap.observation().pixels().len(), shape);
    println!("{}", env.state().render_ascii());

    for (i, action) in SCRIPT.iter().enumerate() {
        let snap = env.step(*action).expect("step");
        println!(
            "step {:>2} action={action:?} — latent (agent={}, goal={}), image {} pixels",
            i + 1,
            env.state().agent(),
            env.state().goal(),
            snap.observation().pixels().len(),
        );
        println!("{}", env.state().render_ascii());
        if snap.is_done() {
            let reward = f32::from(*snap.reward());
            println!("terminated after {} steps with reward = {reward:.4}", i + 1);
            return;
        }
    }

    println!("script exhausted without terminating (this should not happen).");
}
