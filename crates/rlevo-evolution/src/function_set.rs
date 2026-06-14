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

use std::fmt::Debug;

/// A program symbol: an `i32` opcode id.
///
/// Within a [`FunctionSet`], ids `0..num_functions()` name function opcodes.
/// GEP alphabets assign higher ids to variable and constant terminals; see
/// [`Alphabet`](crate::algorithms::gep::Alphabet). The newtype keeps symbol
/// ids from being confused with the many other `i32` indices in the genome
/// (node indices, input indices, gene positions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(pub i32);

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
    /// # Preconditions
    ///
    /// `args.len() == self.arity(symbol)`. For zero-arity functions
    /// (constants such as `1.0`), `args` is empty. Variable and constant
    /// *terminals* are never passed here — they are resolved by the caller's
    /// evaluator before `apply` is reached.
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
        usize::try_from(symbol.0)
            .ok()
            .and_then(|i| Self::ARITIES.get(i).copied())
            .unwrap_or(0)
    }

    fn max_arity(&self) -> usize {
        2
    }

    fn apply(&self, symbol: Symbol, args: &[f32]) -> f32 {
        // Arms 0..=3 read two args, 4..=6 read one, 7 reads none. The caller
        // slices `args` to the opcode's arity, so the indexing below is in
        // bounds. Unknown ids collapse to 0.0, matching the historical
        // `_ => 0.0` arm of the inline CGP match.
        match symbol.0 {
            0 => args[0] + args[1],
            1 => args[0] - args[1],
            2 => args[0] * args[1],
            3 => {
                let (a, b) = (args[0], args[1]);
                if b.abs() < 1e-6 { a } else { a / b }
            }
            4 => args[0].sin(),
            5 => args[0].cos(),
            6 => args[0].tanh(),
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
                    if b.abs() < 1e-6 { a } else { a / b }
                }
                4 => a.sin(),
                5 => a.cos(),
                6 => a.tanh(),
                7 => 1.0,
                _ => 0.0,
            }
        };
        for func in 0..8 {
            let sym = Symbol(i32::try_from(func).unwrap());
            let arity = fs.arity(sym);
            let arg_buf = [a, b];
            let got = fs.apply(sym, &arg_buf[..arity]);
            let want = inline(func, a, b);
            approx::assert_relative_eq!(got, want, epsilon = 1e-7);
        }
    }

    /// Protected division returns the numerator when the denominator is near
    /// zero (no `inf`).
    #[test]
    fn protected_div_guards_small_denominator() {
        let fs = ArithmeticFunctionSet;
        let got = fs.apply(Symbol(3), &[3.0, 1e-9]);
        approx::assert_relative_eq!(got, 3.0, epsilon = 1e-7);
    }

    /// Out-of-range opcodes report arity 0 and evaluate to 0.0 rather than
    /// panicking, matching the inline match's `_ => 0.0`.
    #[test]
    fn out_of_range_opcode_is_inert() {
        let fs = ArithmeticFunctionSet;
        assert_eq!(fs.arity(Symbol(99)), 0);
        assert_eq!(fs.arity(Symbol(-1)), 0);
        approx::assert_relative_eq!(fs.apply(Symbol(99), &[]), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn arities_and_counts() {
        let fs = ArithmeticFunctionSet;
        assert_eq!(fs.num_functions(), 8);
        assert_eq!(fs.max_arity(), 2);
        assert_eq!(fs.arity(Symbol(0)), 2);
        assert_eq!(fs.arity(Symbol(6)), 1);
        assert_eq!(fs.arity(Symbol(7)), 0);
    }
}
