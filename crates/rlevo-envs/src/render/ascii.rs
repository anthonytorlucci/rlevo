use rlevo_core::render::Renderer;

/// An environment that can render itself as an ASCII string.
///
/// Implement this for each environment that wants text output. The
/// [`AsciiRenderer`] delegates to this method and returns the `String`
/// as its `Frame`.
pub trait AsciiRenderable {
    /// Produce a text representation of the current environment state.
    fn render_ascii(&self) -> String;
}

/// A renderer that produces ASCII `String` frames.
///
/// # Example
///
/// ```rust,ignore
/// use evorl_envs::render::AsciiRenderer;
/// use evorl_core::render::Renderer;
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
