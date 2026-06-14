//! Runtime configuration for a Gene Expression Programming run.

/// Static parameters for a [`GepStrategy`](super::GepStrategy) run.
///
/// GEP chromosomes are fixed-length: a `head` of `head_len` loci (which may
/// hold any symbol) followed by a `tail` of `tail_len` loci (terminals only).
/// The tail length is **not** a free parameter — it is derived from the head
/// length and the function set's maximum arity so that every chromosome
/// decodes to a complete expression tree without repair (see
/// [`GepConfig::new`]).
///
/// Unlike the const-generic experiments considered during design, the genome
/// dimensions are ordinary runtime fields (matching the shipped
/// [`CgpConfig`](crate::algorithms::gp_cgp::CgpConfig) precedent); no
/// `generic_const_exprs` is required.
#[derive(Debug, Clone)]
pub struct GepConfig {
    /// Head length: number of leading loci that may hold any symbol.
    pub head_len: usize,
    /// Tail length: number of trailing loci that may hold terminals only.
    ///
    /// Derived in [`GepConfig::new`] as `head_len * (max_arity - 1) + 1`.
    pub tail_len: usize,
    /// Number of individuals in the population.
    pub pop_size: usize,
    /// Number of input variables the program sees.
    pub n_vars: usize,
    /// Per-gene point-mutation probability.
    pub mutation_rate: f32,
    /// Per-individual probability of applying one IS transposition.
    pub is_transpose_rate: f32,
    /// Per-individual probability of applying one RIS transposition.
    pub ris_transpose_rate: f32,
    /// Per-pair probability of one-point crossover.
    pub crossover_1p_rate: f32,
    /// Per-pair probability of two-point crossover.
    pub crossover_2p_rate: f32,
}

impl GepConfig {
    /// Builds a config, deriving and validating the tail length.
    ///
    /// The tail length is set to `head_len * (max_arity - 1) + 1`, the minimum
    /// that guarantees any head/tail-respecting chromosome decodes to a
    /// complete tree (3e-R1 §3). `max_arity` is the function set's largest
    /// arity ([`FunctionSet::max_arity`](crate::function_set::FunctionSet::max_arity)).
    ///
    /// Operator rates default to canonical Ferreira (2001) values; mutate the
    /// public fields afterwards to override. The point-mutation rate defaults
    /// to `2 / genome_len` (≈ two genes per chromosome).
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message if `head_len`, `max_arity`, `n_vars`,
    /// or `pop_size` is zero — each would make the genome layout or the tree
    /// decode degenerate.
    #[must_use]
    pub fn new(head_len: usize, max_arity: usize, n_vars: usize, pop_size: usize) -> Self {
        assert!(head_len >= 1, "GepConfig: head_len must be >= 1");
        assert!(
            max_arity >= 1,
            "GepConfig: max_arity must be >= 1 (function set has no multi-ary functions)"
        );
        assert!(n_vars >= 1, "GepConfig: n_vars must be >= 1");
        assert!(pop_size >= 1, "GepConfig: pop_size must be >= 1");

        // Tail sized to the worst case: a head of all-max-arity functions
        // demands exactly `head_len * (max_arity - 1) + 1` terminals (3e-R1 §3),
        // so this is the minimum tail that guarantees a repair-free decode.
        let tail_len = head_len * (max_arity - 1) + 1;
        let genome_len = head_len + tail_len;
        #[allow(clippy::cast_precision_loss)]
        let mutation_rate = 2.0 / genome_len as f32;

        Self {
            head_len,
            tail_len,
            pop_size,
            n_vars,
            mutation_rate,
            is_transpose_rate: 0.1,
            ris_transpose_rate: 0.1,
            crossover_1p_rate: 0.3,
            crossover_2p_rate: 0.3,
        }
    }

    /// Total chromosome length (`head_len + tail_len`).
    #[must_use]
    pub fn genome_len(&self) -> usize {
        self.head_len + self.tail_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_tail_for_max_arity_two() {
        // head 7, max_arity 2 -> tail = 7 * 1 + 1 = 8, genome = 15.
        let cfg = GepConfig::new(7, 2, 1, 100);
        assert_eq!(cfg.tail_len, 8);
        assert_eq!(cfg.genome_len(), 15);
    }

    #[test]
    fn derives_tail_for_max_arity_three() {
        // head 5, max_arity 3 -> tail = 5 * 2 + 1 = 11.
        let cfg = GepConfig::new(5, 3, 2, 50);
        assert_eq!(cfg.tail_len, 11);
        assert_eq!(cfg.genome_len(), 16);
    }

    #[test]
    #[should_panic(expected = "head_len must be >= 1")]
    fn rejects_zero_head() {
        let _ = GepConfig::new(0, 2, 1, 10);
    }

    #[test]
    #[should_panic(expected = "n_vars must be >= 1")]
    fn rejects_zero_vars() {
        let _ = GepConfig::new(4, 2, 0, 10);
    }

    #[test]
    #[should_panic(expected = "max_arity must be >= 1")]
    fn rejects_zero_arity() {
        let _ = GepConfig::new(4, 0, 1, 10);
    }
}
