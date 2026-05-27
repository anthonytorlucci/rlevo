//! Plain-text rendering surface for environments.
//!
//! The [`AsciiRenderable`] trait lives in `rlevo-core` so that
//! `rlevo-benchmarks` (the host of the live `ratatui` TUI) can bound on it
//! without depending on `rlevo-environments` — a circular package dep that
//! Cargo rejects. Concrete env impls still live in `rlevo-environments` and
//! are unaffected (the trait being in core just removes a coupling).

use super::{Renderer, StyledFrame};

/// An environment that can render itself as an ASCII string.
///
/// Implement this for each environment that wants text output. The
/// [`AsciiRenderer`] delegates to [`render_ascii`](Self::render_ascii) and
/// returns the `String` as its `Frame`.
///
/// # Two projections
///
/// Two methods, two consumers:
///
/// - [`render_ascii`](Self::render_ascii) returns a plain `String` for logs,
///   snapshot tests, grep-friendly output, and `EpisodeRecord.ascii`. Every
///   implementor must provide it.
/// - [`render_styled`](Self::render_styled) returns a
///   [`StyledFrame`] carrying colour and modifier hints
///   for the live `ratatui` TUI and the static-HTML report tier. The default
///   impl wraps the plain text as a single unstyled span, so existing
///   implementors continue to compile without changes.
///
/// Override `render_styled` only when the env benefits from colour cues.
/// Use the project palette constants in [`super::palette`] rather than raw
/// [`super::Color`] values so that the accessibility contract (hue-redundant
/// signalling for hazard/goal semantics) is preserved.
pub trait AsciiRenderable {
    /// Produce a text representation of the current environment state.
    fn render_ascii(&self) -> String;

    /// Produce a styled projection of the current environment state.
    ///
    /// The default implementation wraps `render_ascii` as a single unstyled
    /// span per line — sufficient for envs that ship only the plain
    /// projection. Envs that want colour override this method directly.
    fn render_styled(&self) -> StyledFrame {
        StyledFrame::unstyled(self.render_ascii())
    }
}

/// A renderer that produces ASCII `String` frames.
///
/// # Example
///
/// ```no_run,ignore
/// use rlevo_core::render::{AsciiRenderable, AsciiRenderer, Renderer};
///
/// let renderer = AsciiRenderer;
/// let frame: String = renderer.render(&my_env);
/// println!("{frame}");
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AsciiRenderer;

impl<E: AsciiRenderable> Renderer<E> for AsciiRenderer {
    type Frame = String;

    fn render(&self, env: &E) -> String {
        env.render_ascii()
    }
}
