//! Gene Expression Programming (GEP).
//!
//! GEP (Ferreira, 2001) evolves programs encoded as **fixed-length linear
//! chromosomes** that decode into expression trees. Each chromosome is a `head`
//! (whose loci may hold any symbol) followed by a `tail` (terminals only),
//! sized so that *every* chromosome respecting that split decodes to a complete
//! tree — no repair pass is ever needed. This makes the genetic operators
//! simple, position-aligned array edits, which is GEP's headline advantage over
//! tree-based GP.
//!
//! # Module map
//!
//! - [`config`] — [`GepConfig`]: runtime head/tail/population parameters; the
//!   tail length is derived and the head/tail constraint asserted at construction.
//! - [`alphabet`] — [`Alphabet`] and [`SymbolKind`]: the function set plus the
//!   variable/constant terminal layer, with the contiguous id-space and the
//!   tail [`terminal_range`](Alphabet::terminal_range).
//! - [`tree`] — [`ExpressionTree`]: the level-order decoded phenotype with
//!   `eval`/`depth`/`node_count`.
//! - [`decode`] — [`GenotypePhenotypeMap`] and [`GepDecoder`]: the deterministic
//!   ORF-scan + BFS decoder.
//! - [`operators`] — point mutation (locus-class aware), IS/RIS transposition,
//!   and one-/two-point crossover, all valid by construction.
//! - [`strategy`] — [`GepStrategy`], [`GepState`], and the [`GepSymRegression`]
//!   fitness function.
//!
//! Genotype storage is a `Tensor<B, 2, Int>` of shape `(pop_size, head_len +
//! tail_len)`; decoding and evaluation run host-side, per chromosome (3e-R1 §5).
//!
//! # Reference
//!
//! - Ferreira (2001), *Gene Expression Programming: a New Adaptive Algorithm
//!   for Solving Problems*, Complex Systems 13(2).

pub mod alphabet;
pub mod config;
pub mod decode;
pub mod operators;
pub mod strategy;
pub mod tree;

pub use alphabet::{Alphabet, SymbolKind};
pub use config::GepConfig;
pub use decode::{GenotypePhenotypeMap, GepDecoder};
pub use strategy::{GepState, GepStrategy, GepSymRegression};
pub use tree::ExpressionTree;
