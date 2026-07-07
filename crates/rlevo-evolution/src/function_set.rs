//! Shared opcode contract for tree- and graph-structured genetic programming.
//!
//! Both Cartesian Genetic Programming ([`crate::algorithms::gp_cgp`]) and
//! Gene Expression Programming ([`crate::algorithms::gep`]) evaluate programs
//! built from the same primitive set: a small table of arithmetic and
//! transcendental functions plus a constant. [`FunctionSet`] is the minimal
//! contract those families share — it describes the **function opcodes only**.
//!
//! # What is *not* here
//!
//! Terminals (variables drawn from an input row, problem constants) are
//! deliberately absent from this trait. CGP wires its inputs through the graph
//! connection genes, and GEP resolves them in its tree evaluator
//! ([`crate::algorithms::gep::ExpressionTree::eval`]); neither needs the
//! function set to carry terminal state. Keeping terminals out of
//! [`FunctionSet`] means the CGP retrofit adds no dead methods — the GEP-only
//! terminal layer lives in [`crate::algorithms::gep::Alphabet`].
//!
//! # Symbol id-space
//!
//! A [`Symbol`] is an `i32` opcode id. Function ids occupy `0..num_functions()`.
//! GEP extends this with variable and constant ids above the function range
//! (see [`Alphabet`](crate::algorithms::gep::Alphabet)); CGP uses only the
//! function ids. `i32` is mandatory because populations are stored as Burn
//! integer tensors (`Tensor<B, 2, Int>`), whose element type is `i32`.

use std::fmt::{self, Debug};

/// A program symbol: an `i32` opcode id.
///
/// Within a [`FunctionSet`], ids `0..num_functions()` name function opcodes.
/// GEP alphabets assign higher ids to variable and constant terminals; see
/// [`Alphabet`](crate::algorithms::gep::Alphabet). The newtype keeps symbol
/// ids from being confused with the many other `i32` indices in the genome
/// (node indices, input indices, gene positions).
///
/// The inner id is private so a `Symbol` cannot be silently confused with a
/// bare `i32`. Read it with [`value`](Symbol::value) (or `i32::from(symbol)`);
/// construct one from a raw genome id with the crate-internal `from_raw`. Ids
/// are not range-checked at construction — out-of-range ids classify as inert
/// (arity `0`, `apply` → `0.0`), matching the evaluator's tolerance for
/// mutated-but-unrepaired genotypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(i32);

impl Symbol {
    /// The raw opcode id.
    ///
    /// Exposed for tensor serialization and table indexing; it is not a
    /// constructor, so it cannot be used to fabricate an unchecked symbol from
    /// outside the crate.
    #[must_use]
    #[inline]
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Constructs a symbol from a raw `i32` id without range-checking it.
    ///
    /// For the GP evaluators that read ids straight from genome tensors, where
    /// the id may legitimately fall outside the function range (an unrepaired
    /// mutation). Out-of-range ids are inert at evaluation, so no validation is
    /// needed here.
    #[must_use]
    #[inline]
    pub(crate) const fn from_raw(id: i32) -> Self {
        Self(id)
    }
}

impl From<Symbol> for i32 {
    #[inline]
    fn from(symbol: Symbol) -> Self {
        symbol.0
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sym{}", self.0)
    }
}

/// Collapses a non-finite phenotype value to the inert `0.0`.
///
/// This is the **node-level** numerical-stability guard: intermediate node
/// values in the CGP ([`evaluate_cgp_with`](crate::algorithms::gp_cgp)) and GEP
/// ([`ExpressionTree::eval`](crate::algorithms::gep::ExpressionTree::eval))
/// evaluators are fed back into downstream node arithmetic, so a non-finite
/// intermediate (overflow `±inf`, `0/0` `NaN`) must be neutralized *in place*
/// with a finite value rather than allowed to poison the rest of the tree.
///
/// This is **not** the fitness `NaN → −inf` convention from `rules.md §3`: that
/// rule applies to the final aggregated fitness and is handled separately by
/// [`sanitize_fitness`](crate::fitness::sanitize_fitness). `0.0` is correct
/// here precisely because the value is an operand, not a fitness score.
#[must_use]
#[inline]
pub(crate) fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// The function-opcode contract shared by CGP and GEP.
///
/// Implementors describe a fixed table of functions. Each function has an
/// arity (number of arguments) and an [`apply`](FunctionSet::apply)
/// implementation that combines already-evaluated arguments into a result.
///
/// The trait is intentionally tiny: it says nothing about terminals, genome
/// layout, or evaluation order. Callers thread a concrete `&F` (never
/// `&dyn FunctionSet`) through their evaluation hot loops so the opcode
/// [`apply`](FunctionSet::apply) inlines.
pub trait FunctionSet: Send + Sync + Debug {
    /// Number of function opcodes. Their ids are `0..num_functions()`.
    fn num_functions(&self) -> usize;

    /// Arity (argument count) of a function opcode.
    ///
    /// Only meaningful for function symbols (`symbol.0` in
    /// `0..num_functions()`). Out-of-range ids return `0` so callers that
    /// evaluate mutated-but-unrepaired genotypes do not panic.
    fn arity(&self, symbol: Symbol) -> usize;

    /// Largest arity among all functions.
    ///
    /// Drives the GEP tail-length constraint
    /// (`tail_len >= head_len * (max_arity - 1) + 1`); see
    /// [`GepConfig::new`](crate::algorithms::gep::GepConfig::new).
    fn max_arity(&self) -> usize;

    /// Applies a function opcode to its already-evaluated arguments.
    ///
    /// Callers pass `args.len() == self.arity(symbol)`. For zero-arity
    /// functions (constants such as `1.0`), `args` is empty. Variable and
    /// constant *terminals* are never passed here — they are resolved by the
    /// caller's evaluator before `apply` is reached.
    ///
    /// A shorter-than-arity slice does **not** panic: missing arguments read as
    /// `0.0`. This keeps a malformed or unrepaired genotype (runtime data) from
    /// aborting a training run, per `rules.md §4`.
    ///
    /// # Numerical
    ///
    /// `apply` performs raw IEEE-754 `f32` arithmetic and **may return a
    /// non-finite value** — overflow yields `±inf`, `0.0 / 0.0` yields `NaN`.
    /// It does **not** sanitize its result. Finiteness is the caller's
    /// responsibility at two distinct layers: intermediate node values are
    /// collapsed to `0.0` for phenotype-evaluation stability (see the
    /// `finite_or_zero` helper), and the final fitness is mapped to `−inf` by
    /// the crate's `sanitize_fitness` per `rules.md §3`.
    fn apply(&self, symbol: Symbol, args: &[f32]) -> f32;
}

/// The canonical v1 arithmetic function set shared by CGP and GEP.
///
/// Eight opcodes, matching the historical inline CGP table exactly:
///
/// | id | op | arity | formula |
/// |----|----|-------|---------|
/// | 0 | add | 2 | `a + b` |
/// | 1 | sub | 2 | `a - b` |
/// | 2 | mul | 2 | `a * b` |
/// | 3 | `protected_div` | 2 | `a / b`, or `a` if `|b| < 1e-6` |
/// | 4 | sin | 1 | `sin(a)` |
/// | 5 | cos | 1 | `cos(a)` |
/// | 6 | tanh | 1 | `tanh(a)` |
/// | 7 | const | 0 | `1.0` |
///
/// The single zero-arity opcode (`const 1.0`) is the **last** function id by
/// construction. GEP relies on this ordering so that all terminal-valued
/// symbols (zero-arity functions, then variables, then constants) form one
/// contiguous id range usable as a tail column mask — see
/// [`Alphabet::terminal_range`](crate::algorithms::gep::Alphabet::terminal_range).
#[derive(Clone, Copy, Debug, Default)]
pub struct ArithmeticFunctionSet;

impl ArithmeticFunctionSet {
    /// Arity of each opcode, indexed by id.
    ///
    /// Mirrors [`crate::algorithms::gp_cgp::FUNCTION_ARITIES`]; kept as an
    /// associated constant so [`FunctionSet::arity`] is a simple table lookup.
    pub const ARITIES: [usize; 8] = [2, 2, 2, 2, 1, 1, 1, 0];
}

impl FunctionSet for ArithmeticFunctionSet {
    fn num_functions(&self) -> usize {
        Self::ARITIES.len()
    }

    fn arity(&self, symbol: Symbol) -> usize {
        usize::try_from(symbol.value())
            .ok()
            .and_then(|i| Self::ARITIES.get(i).copied())
            .unwrap_or(0)
    }

    fn max_arity(&self) -> usize {
        2
    }

    fn apply(&self, symbol: Symbol, args: &[f32]) -> f32 {
        // Arms 0..=3 use two args, 4..=6 use one, 7 uses none. Missing
        // arguments (a shorter-than-arity slice) read as 0.0 rather than
        // panicking. Unknown ids collapse to 0.0, matching the historical
        // `_ => 0.0` arm of the inline CGP match.
        let a = args.first().copied().unwrap_or(0.0);
        let b = args.get(1).copied().unwrap_or(0.0);
        match symbol.value() {
            0 => a + b,
            1 => a - b,
            2 => a * b,
            3 => {
                if b.abs() < 1e-6 {
                    a
                } else {
                    a / b
                }
            }
            4 => a.sin(),
            5 => a.cos(),
            6 => a.tanh(),
            7 => 1.0,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `apply` reproduces the historical inline CGP opcode match across all
    /// eight opcodes for a representative `(a, b)` pair.
    #[test]
    fn arithmetic_apply_matches_inline_cgp_match() {
        let fs = ArithmeticFunctionSet;
        let (a, b) = (0.7_f32, 0.25_f32);
        let inline = |func: usize, a: f32, b: f32| -> f32 {
            match func {
                0 => a + b,
                1 => a - b,
                2 => a * b,
                3 => {
                    if b.abs() < 1e-6 {
                        a
                    } else {
                        a / b
                    }
                }
                4 => a.sin(),
                5 => a.cos(),
                6 => a.tanh(),
                7 => 1.0,
                _ => 0.0,
            }
        };
        for func in 0..8 {
            // `func` is in `0..8`, so neither truncation nor wrap can occur.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let sym = Symbol::from_raw(func as i32);
            let arity = fs.arity(sym);
            let arg_buf = [a, b];
            let got = fs.apply(sym, &arg_buf[..arity]);
            let want = inline(func, a, b);
            approx::assert_relative_eq!(got, want, epsilon = 1e-7);
        }
    }

    /// A shorter-than-arity slice must not panic; missing arguments read as
    /// `0.0` (`rules.md §4`: never panic on runtime data).
    #[test]
    fn apply_handles_short_args_without_panic() {
        let fs = ArithmeticFunctionSet;
        // arity(add) == 2, but the slice is empty: 0.0 + 0.0 == 0.0.
        approx::assert_relative_eq!(fs.apply(Symbol::from_raw(0), &[]), 0.0, epsilon = 1e-7);
        // arity(sub) == 2 with a single arg: 5.0 - 0.0 == 5.0.
        approx::assert_relative_eq!(fs.apply(Symbol::from_raw(1), &[5.0]), 5.0, epsilon = 1e-7);
    }

    /// `apply` does not sanitize: a `NaN` argument propagates through raw
    /// arithmetic (finiteness is the caller's responsibility, see
    /// [`finite_or_zero`]).
    #[test]
    fn apply_propagates_non_finite_arguments() {
        let fs = ArithmeticFunctionSet;
        assert!(fs.apply(Symbol::from_raw(0), &[f32::NAN, 1.0]).is_nan());
        assert!(
            fs.apply(Symbol::from_raw(2), &[f32::INFINITY, 2.0])
                .is_infinite()
        );
    }

    /// Protected division returns the numerator when the denominator is near
    /// zero (no `inf`).
    #[test]
    fn protected_div_guards_small_denominator() {
        let fs = ArithmeticFunctionSet;
        let got = fs.apply(Symbol::from_raw(3), &[3.0, 1e-9]);
        approx::assert_relative_eq!(got, 3.0, epsilon = 1e-7);
    }

    /// Out-of-range opcodes report arity 0 and evaluate to 0.0 rather than
    /// panicking, matching the inline match's `_ => 0.0`.
    #[test]
    fn out_of_range_opcode_is_inert() {
        let fs = ArithmeticFunctionSet;
        assert_eq!(fs.arity(Symbol::from_raw(99)), 0);
        assert_eq!(fs.arity(Symbol::from_raw(-1)), 0);
        approx::assert_relative_eq!(fs.apply(Symbol::from_raw(99), &[]), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn arities_and_counts() {
        let fs = ArithmeticFunctionSet;
        assert_eq!(fs.num_functions(), 8);
        assert_eq!(fs.max_arity(), 2);
        assert_eq!(fs.arity(Symbol::from_raw(0)), 2);
        assert_eq!(fs.arity(Symbol::from_raw(6)), 1);
        assert_eq!(fs.arity(Symbol::from_raw(7)), 0);
    }
}
