//! The GEP symbol alphabet: functions plus a terminal layer.
//!
//! [`Alphabet`] wraps a shared [`FunctionSet`] with the GEP-only terminal
//! machinery (input variables and problem constants) that CGP does not need.
//! It assigns every symbol a contiguous, non-negative `i32` id so the head and
//! tail sampling ranges are simple integer intervals (usable directly as tensor
//! column masks).

use std::ops::Range;

use rand::{Rng, RngExt};

use crate::function_set::{FunctionSet, Symbol};

/// Builds a function-block [`Symbol`] from a `usize` id.
///
/// Function counts are tiny, so the `usize -> i32` conversion never truncates;
/// the lint allowance documents that the cast is bounded by `num_functions()`.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn function_symbol(i: usize) -> Symbol {
    Symbol::from_raw(i as i32)
}

/// The semantic class of a decoded symbol.
///
/// Produced by [`Alphabet::classify`]; consumed by the tree evaluator
/// ([`ExpressionTree::eval`](super::ExpressionTree::eval)) to decide how a node
/// produces its value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolKind {
    /// A function opcode of the given arity. Zero-arity functions (e.g. a
    /// constant `1.0`) are leaves evaluated via
    /// [`FunctionSet::apply`] with an
    /// empty argument slice.
    Function {
        /// Number of children the function consumes.
        arity: usize,
    },
    /// An input variable; its value is `inputs[input_index]` at evaluation.
    Variable {
        /// Index into the input row.
        input_index: usize,
    },
    /// A problem constant with a fixed scalar value.
    Constant {
        /// The constant's scalar value.
        value: f32,
    },
}

/// A GEP alphabet over a function set `F`, plus variables and constants.
///
/// # Id-space
///
/// Ids are contiguous and non-negative, laid out in three blocks:
///
/// | block | id range | meaning |
/// |-------|----------|---------|
/// | functions | `0 .. n_func` | opcodes from `F` |
/// | variables | `n_func .. n_func + n_vars` | `input_index = id - n_func` |
/// | constants | `n_func + n_vars .. len()` | `constants[id - n_func - n_vars]` |
///
/// # Terminal range
///
/// A *terminal* is any arity-0 symbol. [`Alphabet::terminal_range`] returns the
/// half-open id interval covering all of them — the arity-0 functions, the
/// variables, and the constants. This is contiguous **only if the function
/// set's arity-0 functions are its trailing ids** (true for
/// [`ArithmeticFunctionSet`](crate::function_set::ArithmeticFunctionSet), whose
/// only zero-arity opcode is the last). The invariant is checked with a
/// `debug_assert!` in [`Alphabet::new`].
#[derive(Debug, Clone)]
pub struct Alphabet<F: FunctionSet> {
    /// The shared function-opcode set.
    pub functions: F,
    /// Number of input variables.
    pub n_vars: usize,
    /// Problem constants, addressed by the high id block.
    pub constants: Vec<f32>,
}

impl<F: FunctionSet> Alphabet<F> {
    /// Builds an alphabet from a function set, a variable count, and constants.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if any arity-0 function id precedes a function
    /// with arity ≥ 1 — that ordering would make [`terminal_range`](Self::terminal_range) include a
    /// non-terminal. (Release builds skip the check; well-formed function sets
    /// such as [`ArithmeticFunctionSet`](crate::function_set::ArithmeticFunctionSet)
    /// always satisfy it.)
    #[must_use]
    pub fn new(functions: F, n_vars: usize, constants: Vec<f32>) -> Self {
        let alphabet = Self {
            functions,
            n_vars,
            constants,
        };
        debug_assert!(
            alphabet.zero_arity_functions_are_trailing(),
            "Alphabet: arity-0 functions must be the trailing function ids so \
             terminal_range() is contiguous"
        );
        alphabet
    }

    /// Number of function opcodes.
    #[must_use]
    pub fn n_func(&self) -> usize {
        self.functions.num_functions()
    }

    /// Total alphabet size: `n_func + n_vars + n_const`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_func() + self.n_vars + self.constants.len()
    }

    /// Whether the alphabet is empty (never true for a usable GEP alphabet).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Largest function arity (delegates to the function set).
    #[must_use]
    pub fn max_arity(&self) -> usize {
        self.functions.max_arity()
    }

    /// Arity of a symbol: the function arity for function ids, else `0`
    /// (variables and constants are leaves).
    #[must_use]
    pub fn arity(&self, symbol: Symbol) -> usize {
        match usize::try_from(symbol.value()) {
            Ok(id) if id < self.n_func() => self.functions.arity(symbol),
            _ => 0,
        }
    }

    /// Classifies a symbol into [`SymbolKind`] by its id block.
    ///
    /// Out-of-range ids (negative, or `>= len()`) classify as an inert
    /// zero-arity `Function`, so a stray symbol evaluates to the function
    /// set's out-of-range value rather than panicking.
    #[must_use]
    pub fn classify(&self, symbol: Symbol) -> SymbolKind {
        let n_func = self.n_func();
        let n_vars = self.n_vars;
        let n_const = self.constants.len();
        let Ok(id_u) = usize::try_from(symbol.value()) else {
            return SymbolKind::Function { arity: 0 };
        };
        if id_u < n_func {
            SymbolKind::Function {
                arity: self.functions.arity(symbol),
            }
        } else if id_u < n_func + n_vars {
            SymbolKind::Variable {
                input_index: id_u - n_func,
            }
        } else if id_u < n_func + n_vars + n_const {
            SymbolKind::Constant {
                value: self.constants[id_u - n_func - n_vars],
            }
        } else {
            SymbolKind::Function { arity: 0 }
        }
    }

    /// Half-open id range `[start, len())` of all terminal symbols.
    ///
    /// `start` is the first arity-0 function id (or `n_func` if the function
    /// set has no constants of its own). Head loci sample from `0..len()`; tail
    /// loci sample from this range.
    #[must_use]
    pub fn terminal_range(&self) -> Range<i32> {
        let n_func = self.n_func();
        let first_terminal = (0..n_func).find(|&i| self.functions.arity(function_symbol(i)) == 0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let start = first_terminal.unwrap_or(n_func) as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let end = self.len() as i32;
        start..end
    }

    /// Samples a uniformly-random head symbol from `0..len()`.
    pub fn sample_head_symbol(&self, rng: &mut dyn Rng) -> Symbol {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let upper = self.len() as i32;
        Symbol::from_raw(rng.random_range(0..upper))
    }

    /// Samples a uniformly-random terminal symbol from [`terminal_range`](Self::terminal_range).
    pub fn sample_tail_symbol(&self, rng: &mut dyn Rng) -> Symbol {
        let range = self.terminal_range();
        Symbol::from_raw(rng.random_range(range))
    }

    /// True iff every arity-0 function id is `>=` every arity-≥1 function id.
    fn zero_arity_functions_are_trailing(&self) -> bool {
        let n_func = self.n_func();
        let mut seen_zero = false;
        for i in 0..n_func {
            let arity = self.functions.arity(function_symbol(i));
            if arity == 0 {
                seen_zero = true;
            } else if seen_zero {
                // A non-leaf function appeared after a leaf function.
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_set::ArithmeticFunctionSet;
    use crate::rng::{SeedPurpose, seed_stream};

    fn alphabet(n_vars: usize, constants: Vec<f32>) -> Alphabet<ArithmeticFunctionSet> {
        Alphabet::new(ArithmeticFunctionSet, n_vars, constants)
    }

    /// A deliberately ill-formed function set: an arity-0 opcode (id 0) precedes
    /// an arity-2 opcode (id 1), violating the "zero-arity functions trailing"
    /// layout precondition that keeps [`Alphabet::terminal_range`] contiguous.
    #[derive(Debug)]
    struct BadLayoutFunctionSet;

    impl FunctionSet for BadLayoutFunctionSet {
        fn num_functions(&self) -> usize {
            2
        }

        fn arity(&self, symbol: Symbol) -> usize {
            match symbol.value() {
                1 => 2,
                _ => 0,
            }
        }

        fn max_arity(&self) -> usize {
            2
        }

        fn apply(&self, _symbol: Symbol, _args: &[f32]) -> f32 {
            0.0
        }
    }

    #[test]
    fn id_space_layout() {
        // n_func = 8, n_vars = 2, n_const = 1 -> len = 11.
        let a = alphabet(2, vec![2.5]);
        assert_eq!(a.n_func(), 8);
        assert_eq!(a.len(), 11);
        // function id 0
        assert_eq!(
            a.classify(Symbol::from_raw(0)),
            SymbolKind::Function { arity: 2 }
        );
        // arity-0 function (const 1.0) id 7
        assert_eq!(
            a.classify(Symbol::from_raw(7)),
            SymbolKind::Function { arity: 0 }
        );
        // variables ids 8, 9
        assert_eq!(
            a.classify(Symbol::from_raw(8)),
            SymbolKind::Variable { input_index: 0 }
        );
        assert_eq!(
            a.classify(Symbol::from_raw(9)),
            SymbolKind::Variable { input_index: 1 }
        );
        // constant id 10
        assert_eq!(
            a.classify(Symbol::from_raw(10)),
            SymbolKind::Constant { value: 2.5 }
        );
    }

    #[test]
    fn terminal_range_starts_at_first_zero_arity_function() {
        // Arity-0 function is id 7; terminals = {7} ∪ variables ∪ constants.
        let a = alphabet(2, vec![1.0]);
        assert_eq!(a.terminal_range(), 7..11);
    }

    #[test]
    fn terminal_range_without_constants() {
        let a = alphabet(1, vec![]);
        // len = 8 + 1 = 9; first arity-0 function id 7.
        assert_eq!(a.terminal_range(), 7..9);
    }

    #[test]
    fn arity_is_zero_for_terminals() {
        let a = alphabet(2, vec![1.0]);
        assert_eq!(a.arity(Symbol::from_raw(0)), 2);
        assert_eq!(a.arity(Symbol::from_raw(7)), 0);
        assert_eq!(a.arity(Symbol::from_raw(8)), 0); // variable
        assert_eq!(a.arity(Symbol::from_raw(10)), 0); // constant
    }

    #[test]
    fn out_of_range_classifies_inert() {
        let a = alphabet(1, vec![]);
        assert_eq!(
            a.classify(Symbol::from_raw(-1)),
            SymbolKind::Function { arity: 0 }
        );
        assert_eq!(
            a.classify(Symbol::from_raw(999)),
            SymbolKind::Function { arity: 0 }
        );
    }

    /// A function set whose arity-0 opcode precedes an arity-≥1 opcode breaks the
    /// contiguity precondition; [`Alphabet::new`] catches it with a
    /// `debug_assert!`.
    #[test]
    #[should_panic(expected = "arity-0 functions must be the trailing")]
    fn new_panics_when_zero_arity_function_precedes_a_function() {
        let _ = Alphabet::new(BadLayoutFunctionSet, 1, vec![]);
    }

    /// `sample_tail_symbol` only ever returns terminals (arity 0) inside the
    /// terminal id range, over many draws.
    #[test]
    fn sample_tail_symbol_is_always_a_terminal() {
        let a: Alphabet<ArithmeticFunctionSet> = alphabet(2, vec![1.0, 2.0]);
        let range: Range<i32> = a.terminal_range();
        let mut rng = seed_stream(1, 0, SeedPurpose::Mutation);
        for _ in 0..1000 {
            let s: Symbol = a.sample_tail_symbol(&mut rng);
            assert_eq!(a.arity(s), 0, "tail symbol {s} is not a terminal");
            assert!(range.contains(&s.value()), "tail symbol {s} outside range");
        }
    }

    /// `sample_head_symbol` stays within the full id space `0..len()`.
    #[test]
    fn sample_head_symbol_stays_in_id_space() {
        let a: Alphabet<ArithmeticFunctionSet> = alphabet(2, vec![1.0]);
        let len: i32 = i32::try_from(a.len()).unwrap();
        let mut rng = seed_stream(2, 0, SeedPurpose::Init);
        for _ in 0..1000 {
            let s: Symbol = a.sample_head_symbol(&mut rng);
            assert!(
                (0..len).contains(&s.value()),
                "head symbol {s} outside 0..{len}"
            );
        }
    }

    /// A non-finite problem constant is still a well-formed terminal: it
    /// classifies as `Constant`, reports arity 0, and lies in the terminal
    /// range. (Finiteness of the *value* is the evaluator's concern, not the
    /// alphabet's.)
    #[test]
    fn non_finite_constant_is_a_terminal() {
        // ids: functions 0..8, variable 8, constants 9 (NaN), 10 (+Inf).
        let a: Alphabet<ArithmeticFunctionSet> = alphabet(1, vec![f32::NAN, f32::INFINITY]);
        let range: Range<i32> = a.terminal_range();

        assert_eq!(a.arity(Symbol::from_raw(9)), 0);
        assert_eq!(a.arity(Symbol::from_raw(10)), 0);
        assert!(range.contains(&9) && range.contains(&10));

        match a.classify(Symbol::from_raw(9)) {
            SymbolKind::Constant { value } => assert!(value.is_nan()),
            other => panic!("expected Constant, got {other:?}"),
        }
        match a.classify(Symbol::from_raw(10)) {
            SymbolKind::Constant { value } => assert!(value.is_infinite()),
            other => panic!("expected Constant, got {other:?}"),
        }
    }
}
