//! GEP genetic operators, all valid by construction (no repair pass).
//!
//! Every operator preserves the head/tail position-class invariant and the
//! fixed chromosome length, so any offspring still decodes to a complete
//! expression tree. Operators act on host-side `&mut [Symbol]`
//! chromosomes: the population tensor is pulled to host once per generation
//! (alongside the host-side roulette selection it shares the round-trip with),
//! the operators run, and the result is re-uploaded.
//!
//! - **Point mutation** is locus-class aware — the single operator that needs a
//!   sampling constraint. Head loci draw any symbol; tail loci draw terminals
//!   only, keeping the tail invariant intact.
//! - **IS / RIS transposition** rewrite only the head (fixed length, tail
//!   untouched); any symbol is legal in the head, so they cannot break validity.
//! - **One-/two-point crossover** swap class-aligned segments between two
//!   equal-layout parents, so a tail locus always exchanges with a tail locus.

use rand::{Rng, RngExt};

use crate::function_set::{FunctionSet, Symbol};

use super::alphabet::Alphabet;

/// Maximum transposed-sequence length (Ferreira uses short IS/RIS elements).
const MAX_TRANSPOSON_LEN: usize = 3;

/// Applies per-gene point mutation in place, respecting locus classes.
///
/// Each locus mutates independently with probability `rate`. A head locus
/// (index `< head_len`) is replaced by any symbol; a tail locus is replaced by
/// a terminal drawn from [`Alphabet::terminal_range`]. The tail invariant is
/// therefore preserved without any repair.
pub fn point_mutation<F: FunctionSet>(
    chromosome: &mut [Symbol],
    head_len: usize,
    alphabet: &Alphabet<F>,
    rate: f32,
    rng: &mut dyn Rng,
) {
    for (i, locus) in chromosome.iter_mut().enumerate() {
        // `< rate` reads as "mutate with probability `rate`" and is robust to a
        // stray non-finite `rate` (`x < NaN` is `false` ⇒ no mutation). Callers
        // pass a validated `Probability::get()`, so this is defense-in-depth.
        if rng.random::<f32>() < rate {
            *locus = if i < head_len {
                alphabet.sample_head_symbol(rng)
            } else {
                alphabet.sample_tail_symbol(rng)
            };
        }
    }
}

/// Insertion-Sequence (IS) transposition: copies a short sequence and inserts
/// it at a non-root head locus, shifting the rest of the head right and dropping
/// the overflow off the head end. The tail is untouched.
pub fn is_transposition(chromosome: &mut [Symbol], head_len: usize, rng: &mut dyn Rng) {
    let genome_len = chromosome.len();
    if head_len < 2 || genome_len == 0 {
        return; // no valid (non-root) insertion site
    }
    let max_len = MAX_TRANSPOSON_LEN.min(genome_len);
    let len = rng.random_range(1..=max_len);
    let source_start = rng.random_range(0..=(genome_len - len));
    let insert = rng.random_range(1..head_len); // != 0: keep the root

    let seq: Vec<Symbol> = chromosome[source_start..source_start + len].to_vec();
    let mut new_head: Vec<Symbol> = Vec::with_capacity(head_len + len);
    new_head.extend_from_slice(&chromosome[0..insert]);
    new_head.extend_from_slice(&seq);
    new_head.extend_from_slice(&chromosome[insert..head_len]);
    new_head.truncate(head_len);
    chromosome[0..head_len].copy_from_slice(&new_head);
}

/// Root-Insertion-Sequence (RIS) transposition: finds a function symbol in the
/// head, copies the sequence starting there to head position 0, shifts right,
/// and drops the overflow. Guarantees the root becomes a function. The tail is
/// untouched.
pub fn ris_transposition<F: FunctionSet>(
    chromosome: &mut [Symbol],
    head_len: usize,
    alphabet: &Alphabet<F>,
    rng: &mut dyn Rng,
) {
    if head_len == 0 {
        return;
    }
    // Scan the head from a random offset for the first function (arity >= 1).
    let offset = rng.random_range(0..head_len);
    let func_pos = (0..head_len)
        .map(|k| (offset + k) % head_len)
        .find(|&i| alphabet.arity(chromosome[i]) >= 1);
    let Some(func_pos) = func_pos else {
        return; // no function in the head; nothing to root
    };

    let max_len = MAX_TRANSPOSON_LEN.min(head_len - func_pos);
    let len = rng.random_range(1..=max_len);
    let seq: Vec<Symbol> = chromosome[func_pos..func_pos + len].to_vec();

    let mut new_head: Vec<Symbol> = Vec::with_capacity(head_len + len);
    new_head.extend_from_slice(&seq);
    new_head.extend_from_slice(&chromosome[0..head_len]);
    new_head.truncate(head_len);
    chromosome[0..head_len].copy_from_slice(&new_head);
}

/// One-point crossover: swaps the suffixes of two equal-length parents at a
/// random cut. Class-aligned, so the tail stays terminal-only.
pub fn one_point_crossover(a: &mut [Symbol], b: &mut [Symbol], rng: &mut dyn Rng) {
    let n = a.len();
    debug_assert_eq!(n, b.len(), "crossover parents must share genome length");
    if n < 2 {
        return;
    }
    let cut = rng.random_range(1..n);
    a[cut..].swap_with_slice(&mut b[cut..]);
}

/// Two-point crossover: swaps the middle segment between two random cuts.
/// Class-aligned, so the tail stays terminal-only.
pub fn two_point_crossover(a: &mut [Symbol], b: &mut [Symbol], rng: &mut dyn Rng) {
    let n = a.len();
    debug_assert_eq!(n, b.len(), "crossover parents must share genome length");
    if n < 2 {
        return;
    }
    let mut c1 = rng.random_range(0..n);
    let mut c2 = rng.random_range(1..=n);
    if c1 > c2 {
        std::mem::swap(&mut c1, &mut c2);
    }
    if c1 == c2 {
        return;
    }
    a[c1..c2].swap_with_slice(&mut b[c1..c2]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::gep::decode::GenotypePhenotypeMap;
    use crate::algorithms::gep::{GepConfig, GepDecoder};
    use crate::function_set::ArithmeticFunctionSet;
    use crate::rng::{SeedPurpose, seed_stream};

    type Fs = ArithmeticFunctionSet;

    fn alphabet() -> Alphabet<Fs> {
        Alphabet::new(ArithmeticFunctionSet, 1, vec![])
    }

    /// Samples a fresh valid chromosome (head = any, tail = terminals).
    fn sample_valid(alphabet: &Alphabet<Fs>, cfg: &GepConfig, rng: &mut dyn Rng) -> Vec<Symbol> {
        let mut g = Vec::with_capacity(cfg.genome_len());
        for _ in 0..cfg.head_len {
            g.push(alphabet.sample_head_symbol(rng));
        }
        for _ in 0..cfg.tail_len {
            g.push(alphabet.sample_tail_symbol(rng));
        }
        g
    }

    /// True iff every tail locus holds a terminal (arity 0).
    fn tail_all_terminals(g: &[Symbol], head_len: usize, a: &Alphabet<Fs>) -> bool {
        g[head_len..].iter().all(|&s| a.arity(s) == 0)
    }

    /// True iff the decoded tree is a complete expression tree (the ORF closed
    /// inside the chromosome — the "no repair needed" guarantee).
    fn decodes_complete(g: &[Symbol], a: &Alphabet<Fs>) -> bool {
        let tree = GepDecoder.decode(a, g);
        let mut needed: i64 = 1;
        for &s in tree.nodes() {
            needed += i64::try_from(a.arity(s)).unwrap() - 1;
        }
        needed == 0 && tree.node_count() >= 1
    }

    /// 1000 point mutations never violate the tail invariant.
    #[test]
    fn point_mutation_preserves_tail_invariant() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 100).unwrap();
        let mut rng = seed_stream(1, 0, SeedPurpose::Mutation);
        for _ in 0..1000 {
            let mut g = sample_valid(&a, &cfg, &mut rng);
            // Force a high rate to exercise many loci.
            point_mutation(&mut g, cfg.head_len, &a, 0.5, &mut rng);
            assert!(
                tail_all_terminals(&g, cfg.head_len, &a),
                "tail invariant violated: {g:?}"
            );
            assert!(decodes_complete(&g, &a));
        }
    }

    /// `rate == 0.0` never mutates; `rate == 1.0` mutates every locus. Pins the
    /// `< rate` gate semantics (finding operators.rs §1.1).
    #[test]
    fn test_point_mutation_rate_bounds_are_none_and_all() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 100).unwrap();
        let mut rng = seed_stream(9, 0, SeedPurpose::Mutation);

        // rate 0 -> identity on every locus.
        let original = sample_valid(&a, &cfg, &mut rng);
        let mut g = original.clone();
        point_mutation(&mut g, cfg.head_len, &a, 0.0, &mut rng);
        assert_eq!(g, original, "rate 0.0 must leave the chromosome unchanged");

        // rate 1 -> every locus is resampled (still a valid symbol for its class).
        let mut g = sample_valid(&a, &cfg, &mut rng);
        point_mutation(&mut g, cfg.head_len, &a, 1.0, &mut rng);
        assert!(
            tail_all_terminals(&g, cfg.head_len, &a),
            "rate 1.0 must preserve the tail invariant: {g:?}"
        );
        assert!(
            decodes_complete(&g, &a),
            "rate 1.0 offspring must still decode completely"
        );
    }

    /// 500 trials of each operator yield offspring that decode completely.
    #[test]
    fn all_operators_yield_decodable_offspring() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 100).unwrap();
        let mut rng = seed_stream(2, 0, SeedPurpose::Crossover);

        for _ in 0..500 {
            // IS
            let mut g = sample_valid(&a, &cfg, &mut rng);
            is_transposition(&mut g, cfg.head_len, &mut rng);
            assert!(tail_all_terminals(&g, cfg.head_len, &a));
            assert!(decodes_complete(&g, &a));

            // RIS
            let mut g = sample_valid(&a, &cfg, &mut rng);
            ris_transposition(&mut g, cfg.head_len, &a, &mut rng);
            assert!(tail_all_terminals(&g, cfg.head_len, &a));
            assert!(decodes_complete(&g, &a));

            // 1-point crossover
            let mut p1 = sample_valid(&a, &cfg, &mut rng);
            let mut p2 = sample_valid(&a, &cfg, &mut rng);
            one_point_crossover(&mut p1, &mut p2, &mut rng);
            assert!(tail_all_terminals(&p1, cfg.head_len, &a));
            assert!(tail_all_terminals(&p2, cfg.head_len, &a));
            assert!(decodes_complete(&p1, &a));
            assert!(decodes_complete(&p2, &a));

            // 2-point crossover
            let mut p1 = sample_valid(&a, &cfg, &mut rng);
            let mut p2 = sample_valid(&a, &cfg, &mut rng);
            two_point_crossover(&mut p1, &mut p2, &mut rng);
            assert!(tail_all_terminals(&p1, cfg.head_len, &a));
            assert!(tail_all_terminals(&p2, cfg.head_len, &a));
            assert!(decodes_complete(&p1, &a));
            assert!(decodes_complete(&p2, &a));
        }
    }

    /// RIS makes the root a function (when the head holds at least one).
    #[test]
    fn ris_roots_a_function() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 10).unwrap();
        let mut rng = seed_stream(3, 0, SeedPurpose::Crossover);
        let mut rooted = 0;
        for _ in 0..200 {
            let mut g = sample_valid(&a, &cfg, &mut rng);
            // Guarantee at least one function in the head.
            g[0] = Symbol::from_raw(0);
            ris_transposition(&mut g, cfg.head_len, &a, &mut rng);
            if a.arity(g[0]) >= 1 {
                rooted += 1;
            }
        }
        assert_eq!(rooted, 200, "RIS should always root a function");
    }

    /// Transposition leaves the tail bytes untouched.
    #[test]
    #[allow(clippy::similar_names)]
    fn transposition_does_not_touch_tail() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 10).unwrap();
        let mut rng = seed_stream(4, 0, SeedPurpose::Crossover);
        let g = sample_valid(&a, &cfg, &mut rng);
        let tail_before = g[cfg.head_len..].to_vec();
        let mut g_is = g.clone();
        is_transposition(&mut g_is, cfg.head_len, &mut rng);
        assert_eq!(&g_is[cfg.head_len..], &tail_before[..]);
        let mut g_ris = g.clone();
        ris_transposition(&mut g_ris, cfg.head_len, &a, &mut rng);
        assert_eq!(&g_ris[cfg.head_len..], &tail_before[..]);
    }

    /// A `NaN` rate is a no-op: `x < NaN` is `false`, so no locus mutates
    /// (defense-in-depth against a stray non-finite rate, operators.rs §1).
    #[test]
    fn point_mutation_nan_rate_is_no_op() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 10).unwrap();
        let mut rng = seed_stream(21, 0, SeedPurpose::Mutation);
        let original = sample_valid(&a, &cfg, &mut rng);
        let mut g = original.clone();
        point_mutation(&mut g, cfg.head_len, &a, f32::NAN, &mut rng);
        assert_eq!(g, original, "NaN rate must leave the chromosome unchanged");
    }

    /// An out-of-range rate (`> 1`) mutates every locus — `x < 2.0` is always
    /// true — yet the tail invariant and the decode guarantee still hold.
    #[test]
    fn point_mutation_out_of_range_rate_resamples_all_but_stays_valid() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 10).unwrap();
        let mut rng = seed_stream(22, 0, SeedPurpose::Mutation);
        let mut g = sample_valid(&a, &cfg, &mut rng);
        point_mutation(&mut g, cfg.head_len, &a, 2.0, &mut rng);
        assert!(tail_all_terminals(&g, cfg.head_len, &a));
        assert!(decodes_complete(&g, &a));
    }

    /// No operator panics on an empty chromosome / empty parent pair.
    #[test]
    fn operators_do_not_panic_on_empty_slices() {
        let a = alphabet();
        let mut rng = seed_stream(23, 0, SeedPurpose::Crossover);

        let mut empty: Vec<Symbol> = Vec::new();
        point_mutation(&mut empty, 0, &a, 1.0, &mut rng);
        is_transposition(&mut empty, 0, &mut rng);
        ris_transposition(&mut empty, 0, &a, &mut rng);
        assert!(empty.is_empty());

        let mut lhs: Vec<Symbol> = Vec::new();
        let mut rhs: Vec<Symbol> = Vec::new();
        one_point_crossover(&mut lhs, &mut rhs, &mut rng);
        two_point_crossover(&mut lhs, &mut rhs, &mut rng);
        assert!(lhs.is_empty() && rhs.is_empty());
    }

    /// With `head_len == 1` both transpositions are safe: IS has no non-root
    /// insertion site (no-op) and RIS keeps the single root, preserving length,
    /// the tail invariant, and a complete decode.
    #[test]
    fn transposition_with_head_len_one_is_safe() {
        let a = alphabet();
        let cfg = GepConfig::new(1, 2, 1, 10).unwrap();
        assert_eq!(cfg.head_len, 1);
        let mut rng = seed_stream(24, 0, SeedPurpose::Transposition);
        for _ in 0..200 {
            let mut g = sample_valid(&a, &cfg, &mut rng);
            // Guarantee a function at the single head locus for RIS.
            g[0] = Symbol::from_raw(0);
            let len_before = g.len();
            is_transposition(&mut g, cfg.head_len, &mut rng);
            ris_transposition(&mut g, cfg.head_len, &a, &mut rng);
            assert_eq!(g.len(), len_before);
            assert!(tail_all_terminals(&g, cfg.head_len, &a));
            assert!(decodes_complete(&g, &a));
        }
    }

    /// Single-locus parents (`n == 1`) leave both crossovers as no-ops.
    #[test]
    fn crossover_with_single_locus_is_no_op() {
        let mut rng = seed_stream(25, 0, SeedPurpose::Crossover);
        let a0: Vec<Symbol> = vec![Symbol::from_raw(8)];
        let b0: Vec<Symbol> = vec![Symbol::from_raw(0)];

        let mut a1 = a0.clone();
        let mut b1 = b0.clone();
        one_point_crossover(&mut a1, &mut b1, &mut rng);
        two_point_crossover(&mut a1, &mut b1, &mut rng);
        assert_eq!(a1, a0);
        assert_eq!(b1, b0);
    }

    /// One-point crossover documents equal parent lengths as a precondition
    /// (`debug_assert_eq!`); a mismatch panics in debug builds.
    #[test]
    #[should_panic(expected = "crossover parents must share genome length")]
    fn one_point_crossover_panics_on_mismatched_lengths() {
        let mut rng = seed_stream(26, 0, SeedPurpose::Crossover);
        let mut a = vec![Symbol::from_raw(8); 4];
        let mut b = vec![Symbol::from_raw(8); 5];
        one_point_crossover(&mut a, &mut b, &mut rng);
    }

    /// Two-point crossover shares the equal-length precondition.
    #[test]
    #[should_panic(expected = "crossover parents must share genome length")]
    fn two_point_crossover_panics_on_mismatched_lengths() {
        let mut rng = seed_stream(27, 0, SeedPurpose::Crossover);
        let mut a = vec![Symbol::from_raw(8); 4];
        let mut b = vec![Symbol::from_raw(8); 5];
        two_point_crossover(&mut a, &mut b, &mut rng);
    }

    /// RIS is a no-op when the head holds no function symbol — there is nothing
    /// to root, so the chromosome is left untouched (§7.4).
    #[test]
    fn ris_no_op_when_head_has_no_functions() {
        let a = alphabet();
        let cfg = GepConfig::new(7, 2, 1, 10).unwrap();
        let mut rng = seed_stream(28, 0, SeedPurpose::Transposition);
        for _ in 0..200 {
            let mut g = sample_valid(&a, &cfg, &mut rng);
            // Force the whole head to a terminal (variable id 8 = n_func).
            for locus in &mut g[..cfg.head_len] {
                *locus = Symbol::from_raw(8);
            }
            let before = g.clone();
            ris_transposition(&mut g, cfg.head_len, &a, &mut rng);
            assert_eq!(g, before, "RIS must not alter an all-terminal head");
        }
    }
}
