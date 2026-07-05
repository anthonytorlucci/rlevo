//! Runtime configuration for a Gene Expression Programming run.

use rlevo_core::config::{self, ConfigError, Validate};

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
    /// complete tree. `max_arity` is the function set's largest
    /// arity ([`FunctionSet::max_arity`](crate::function_set::FunctionSet::max_arity)).
    ///
    /// Operator rates default to canonical Ferreira (2001) values; mutate the
    /// public fields afterwards to override. The point-mutation rate defaults
    /// to `2 / genome_len` (≈ two genes per chromosome).
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `max_arity`, `head_len`, `n_vars`, or
    /// `pop_size` is zero — each would make the genome layout or the tree
    /// decode degenerate. `max_arity` is checked here (it is consumed to derive
    /// `tail_len` rather than stored); the remaining field invariants are
    /// checked via [`Validate`].
    pub fn new(
        head_len: usize,
        max_arity: usize,
        n_vars: usize,
        pop_size: usize,
    ) -> Result<Self, ConfigError> {
        // `max_arity` is not a stored field (it is consumed below to derive
        // `tail_len`), so it cannot be re-checked by `validate`; guard it here.
        config::at_least("GepConfig", "max_arity", max_arity, 1)?;

        // Tail sized to the worst case: a head of all-max-arity functions
        // demands exactly `head_len * (max_arity - 1) + 1` terminals, so this is
        // the minimum tail that guarantees a repair-free decode.
        let tail_len = head_len * (max_arity - 1) + 1;
        let genome_len = head_len + tail_len;
        #[allow(clippy::cast_precision_loss)]
        let mutation_rate = 2.0 / genome_len as f32;

        let config = Self {
            head_len,
            tail_len,
            pop_size,
            n_vars,
            mutation_rate,
            is_transpose_rate: 0.1,
            ris_transpose_rate: 0.1,
            crossover_1p_rate: 0.3,
            crossover_2p_rate: 0.3,
        };
        config.validate()?;
        Ok(config)
    }

    /// Total chromosome length (`head_len + tail_len`).
    #[must_use]
    pub fn genome_len(&self) -> usize {
        self.head_len + self.tail_len
    }
}

impl Validate for GepConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "GepConfig";
        config::at_least(C, "head_len", self.head_len, 1)?;
        config::at_least(C, "tail_len", self.tail_len, 1)?;
        config::at_least(C, "n_vars", self.n_vars, 1)?;
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        config::in_range(C, "mutation_rate", 0.0, 1.0, f64::from(self.mutation_rate))?;
        config::in_range(C, "is_transpose_rate", 0.0, 1.0, f64::from(self.is_transpose_rate))?;
        config::in_range(C, "ris_transpose_rate", 0.0, 1.0, f64::from(self.ris_transpose_rate))?;
        config::in_range(C, "crossover_1p_rate", 0.0, 1.0, f64::from(self.crossover_1p_rate))?;
        config::in_range(C, "crossover_2p_rate", 0.0, 1.0, f64::from(self.crossover_2p_rate))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_tail_for_max_arity_two() {
        // head 7, max_arity 2 -> tail = 7 * 1 + 1 = 8, genome = 15.
        let cfg = GepConfig::new(7, 2, 1, 100).unwrap();
        assert_eq!(cfg.tail_len, 8);
        assert_eq!(cfg.genome_len(), 15);
    }

    #[test]
    fn derives_tail_for_max_arity_three() {
        // head 5, max_arity 3 -> tail = 5 * 2 + 1 = 11.
        let cfg = GepConfig::new(5, 3, 2, 50).unwrap();
        assert_eq!(cfg.tail_len, 11);
        assert_eq!(cfg.genome_len(), 16);
    }

    #[test]
    fn rejects_zero_head() {
        let err = GepConfig::new(0, 2, 1, 10).unwrap_err();
        assert_eq!(err.field, "head_len");
    }

    #[test]
    fn rejects_zero_vars() {
        let err = GepConfig::new(4, 2, 0, 10).unwrap_err();
        assert_eq!(err.field, "n_vars");
    }

    #[test]
    fn rejects_zero_arity() {
        let err = GepConfig::new(4, 0, 1, 10).unwrap_err();
        assert_eq!(err.field, "max_arity");
    }

    #[test]
    fn accepts_valid_config() {
        assert!(GepConfig::new(6, 2, 3, 64).unwrap().validate().is_ok());
    }
}
