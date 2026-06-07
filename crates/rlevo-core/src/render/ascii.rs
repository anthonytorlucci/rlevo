//! Optional plain-text rendering surface for environments.
//!
//! [`AsciiRenderable`] is an **optional debug helper**, not a library
//! invariant (ADR-0013). The two visualisation products do not depend on it:
//! the live TUI is metrics-only and renders no env, and the post-run report
//! renders env playback from the structured `FamilyPayload` carried in each
//! `EpisodeRecord` frame. An env implements this trait only when a quick
//! grep-friendly text dump is useful — for logs, snapshot tests, or a
//! user-built env that wants the report's legacy `<pre>` fallback without
//! writing a structured payload adapter.
//!
//! The trait lives in `rlevo-core` (rather than `rlevo-environments`) so that
//! `rlevo-benchmarks` can still bound on it without a circular package dep.
//! It is not a supertrait of `Environment`.

use super::{Renderer, StyledFrame};

/// An environment that can render itself as an ASCII string.
///
/// Optional (ADR-0013): implement it only for envs that want a text dump.
/// The [`AsciiRenderer`] delegates to [`render_ascii`](Self::render_ascii)
/// and returns the `String` as its `Frame`.
///
/// # Two projections
///
/// Two methods, two consumers:
///
/// - [`render_ascii`](Self::render_ascii) returns a plain `String` for logs,
///   snapshot tests, grep-friendly output, and the optional
///   `FrameRecord.ascii` slot. Every implementor must provide it.
/// - [`render_styled`](Self::render_styled) returns a
///   [`StyledFrame`] carrying colour and modifier hints for the report's
///   legacy `<pre>`/CSS-span fallback. The default impl wraps the plain text
///   as a single unstyled span, so implementors that only want the plain
///   projection compile without changes.
///
/// Override `render_styled` only when the text dump benefits from colour
/// cues. Use the project palette constants in [`super::palette`] rather than
/// raw [`super::Color`] values so that the accessibility contract
/// (hue-redundant signalling for hazard/goal semantics) is preserved.
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
/// Implements [`Renderer<E>`](super::Renderer) for any `E: AsciiRenderable`,
/// delegating to [`AsciiRenderable::render_ascii`].  Pair with
/// [`NullRenderer`](crate::render::NullRenderer) when rendering is
/// disabled so the call compiles away entirely.
///
/// # Example
///
/// ```no_run
/// use rlevo_core::render::{AsciiRenderable, AsciiRenderer, Renderer};
///
/// struct GridEnv { width: usize, height: usize }
///
/// impl AsciiRenderable for GridEnv {
///     fn render_ascii(&self) -> String {
///         (0..self.height)
///             .map(|_| ".".repeat(self.width))
///             .collect::<Vec<_>>()
///             .join("\n")
///     }
/// }
///
/// let env = GridEnv { width: 5, height: 3 };
/// let renderer = AsciiRenderer;
/// let frame: String = renderer.render(&env);
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
