//! Widgets the render thread composes into the live dashboard.
//!
//! Each panel is a stateless [`Widget`](ratatui::widgets::Widget) that
//! borrows a snapshot of [`AppState`](crate::tui::state::AppState) at
//! construction time. The render thread builds the panel, hands it to
//! `Frame::render_widget`, and discards it — no per-panel state lives
//! across ticks. State lives in `AppState`; presentation lives here.
//!
//! Panel ordering and layout (the rectangles each panel renders into)
//! belong to the runner, not to individual panels. This keeps the
//! widgets reusable when M3 adds the gradient-norm / fitness / log
//! panels and the layout changes shape.

pub mod env_panel;
pub mod locomotion_placeholder;
pub mod reward_sparkline;

pub use env_panel::EnvPanel;
pub use locomotion_placeholder::LocomotionPlaceholder;
pub use reward_sparkline::RewardSparkline;
