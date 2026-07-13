//! Runtime configuration for a Gene Expression Programming run.

use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::probability::Probability;

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
    /// Per-gene point-mutation probability. Valid by construction (`[0, 1]`).
    pub mutation_rate: Probability,
    /// Per-individual probability of applying one IS transposition. Valid by
    /// construction (`[0, 1]`).
    pub is_transpose_rate: Probability,
    /// Per-individual probability of applying one RIS transposition. Valid by
    /// construction (`[0, 1]`).
    pub ris_transpose_rate: Probability,
    /// Per-pair probability of one-point crossover. Valid by construction
    /// (`[0, 1]`).
    pub crossover_1p_rate: Probability,
    /// Per-pair probability of two-point crossover. Valid by construction
    /// (`[0, 1]`).
    pub crossover_2p_rate: Probability,
}

impl GepConfig {
    /// Builds a config, deriving and validating the tail length.
    ///
    /// The tail length is set to `head_len * (max_arity - 1) + 1`, the minimum
    /// that guarantees any head/tail-respecting chromosome decodes to a
    /// complete tree. `max_arity` is the function set's largest
    /// arity ([`FunctionSet::max_arity`](crate::function_set::FunctionSet::max_arity)).
    ///
    /// The IS/RIS transposition rates default to Ferreira's (2001) canonical
    /// `0.1`; the crossover rates (`0.3`/`0.3`) are rlevo's own choice, not
    /// Ferreira's commonly cited `0.2`/`0.5`. Assign a validated
    /// [`Probability`] to the public fields afterwards to override.
    /// The point-mutation rate defaults to `2 / genome_len` (≈ two genes per
    /// chromosome). Because the rates are [`Probability`], a `NaN`/`Inf`/
    /// out-of-`[0, 1]` rate is unrepresentable — the silent operator
    /// degeneracy of a bare `f32` rate cannot occur.
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
        // Guard `head_len` before deriving `mutation_rate`: `head_len == 0`
        // yields `genome_len == 1` and `2 / genome_len == 2.0`, which would
        // panic `Probability::new` below (an invalid config that `validate`
        // should reject, not crash on). Re-checked in `validate` too.
        config::at_least("GepConfig", "head_len", head_len, 1)?;

        // Tail sized to the worst case: a head of all-max-arity functions
        // demands exactly `head_len * (max_arity - 1) + 1` terminals, so this is
        // the minimum tail that guarantees a repair-free decode.
        let tail_len = head_len * (max_arity - 1) + 1;
        let genome_len = head_len + tail_len;
        // `genome_len >= 2` (head_len >= 1, tail_len >= 1), so `2 / genome_len`
        // lies in `(0, 1]` — provably a valid `Probability`, hence `new`.
        #[allow(clippy::cast_precision_loss)]
        let mutation_rate = Probability::new(2.0 / genome_len as f32);

        let config = Self {
            head_len,
            tail_len,
            pop_size,
            n_vars,
            mutation_rate,
            is_transpose_rate: Probability::new(0.1),
            ris_transpose_rate: Probability::new(0.1),
            crossover_1p_rate: Probability::new(0.3),
            crossover_2p_rate: Probability::new(0.3),
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
        // The five operator rates are `Probability`: valid by construction, so
        // no `in_range` checks here — see ADR 0031.
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

    #[test]
    fn rejects_zero_pop() {
        let err = GepConfig::new(4, 2, 1, 0).unwrap_err();
        assert_eq!(err.field, "pop_size");
    }

    /// A unary function set (`max_arity == 1`) collapses the Ferreira tail to a
    /// single locus, `t = h·(1−1)+1 = 1`, for every head size.
    #[test]
    fn unary_max_arity_derives_unit_tail() {
        for head in [1usize, 2, 5, 7, 20] {
            let cfg = GepConfig::new(head, 1, 1, 10).unwrap();
            assert_eq!(cfg.tail_len, 1, "unary tail must be 1 for head {head}");
            assert_eq!(cfg.genome_len(), head + 1);
            // The derived layout still satisfies the config invariants.
            assert!(cfg.validate().is_ok());
        }
    }

    /// The operator rates are [`Probability`], so a `NaN`, negative, or `> 1`
    /// rate is unrepresentable: it cannot be built to assign to a rate field,
    /// while a valid rate assigns cleanly.
    #[test]
    fn rate_fields_reject_nan_negative_and_above_one() {
        assert!(Probability::try_new(f32::NAN).is_err());
        assert!(Probability::try_new(-0.1).is_err());
        assert!(Probability::try_new(1.5).is_err());

        let mut cfg: GepConfig = GepConfig::new(4, 2, 1, 10).unwrap();
        cfg.mutation_rate = Probability::try_new(0.05).unwrap();
        cfg.crossover_1p_rate = Probability::try_new(0.9).unwrap();
        // `Probability`'s derived `PartialEq` compares the wrapped value without
        // tripping `clippy::float_cmp`; both were assigned exact literals.
        assert_eq!(cfg.mutation_rate, Probability::try_new(0.05).unwrap());
        assert_eq!(cfg.crossover_1p_rate, Probability::try_new(0.9).unwrap());
    }
}
