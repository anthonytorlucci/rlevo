//! Speciation — compatibility distance, the representative-assignment pass,
//! fitness sharing, stagnation, and offspring apportionment.
//!
//! Species protect structural innovations from immediate global competition long
//! enough to optimize their new weights. A genome joins the first species whose
//! frozen representative is within `compat_threshold` of it
//! ([`speciate`]), each species' size-adjusted fitness drives how many offspring
//! it is allotted ([`allocate_offspring`]), and species that stop improving are
//! pruned ([`remove_stagnant`]).
//!
//! # Orientation
//!
//! NEAT is **maximization** (higher fitness is better) — matching the crate-wide
//! maximise convention. `best_fitness` tracks a maximum and fitness sharing
//! assumes **non-negative** raw fitness (a NEAT-specific precondition orthogonal
//! to objective sense); cost objectives are reconciled into canonical space by
//! the harness/adapter chokepoint, not by hand here.

use rand::RngExt;
use rand::rngs::StdRng;

use super::topology::{InnovationId, TopologyGenome};

/// Identifier for a species. Monotone within a run.
///
/// An **opaque newtype** over `u64` (not a bare alias), so a `SpeciesId` cannot
/// be confused with a node id, an innovation, or a population index. It has no
/// invariant — every `u64` is a legal id — so [`new`](SpeciesId::new) is
/// infallible. Construct with `new`, read with [`get`](SpeciesId::get); the
/// crate-internal `succ` is the only arithmetic, used solely
/// by the [`speciate`] id counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeciesId(u64);

impl SpeciesId {
    /// Wrap a raw id. Infallible — a species id has no invariant.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The underlying id.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The next id in sequence. Crate-internal because only [`speciate`]
    /// allocates fresh species ids.
    #[must_use]
    pub(crate) const fn succ(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Number of top-fitness species shielded from stagnation removal, so a
/// population-wide plateau cannot wipe every species at once.
pub const STAGNATION_PROTECT_TOP_K: usize = 2;

/// A cluster of compatible genomes.
///
/// `members` holds **indices** into the population `Vec`, not genomes, so the
/// population stays the single owner of genome data and the partition is a cheap
/// view. `representative` is a *cloned* genome (not an index) so it survives
/// population turnover between generations.
///
/// Fields are `pub(crate)`: the NEAT speciation, reproduction, and stagnation
/// operators (spread across `species.rs` and `neat.rs`) mutate them in place,
/// but no code outside this crate constructs or reads a `Species` field, so
/// crate-visibility blocks external struct-literal construction of an
/// inconsistent species without imposing accessor churn on the operators.
#[derive(Clone, Debug)]
pub struct Species {
    /// Stable species id.
    pub(crate) id: SpeciesId,
    /// Frozen comparison anchor for this generation's assignment pass — a
    /// randomly chosen member of the previous generation (canonical NEAT).
    pub(crate) representative: TopologyGenome,
    /// Indices into the population `Vec` assigned to this species this generation.
    pub(crate) members: Vec<usize>,
    /// Best (maximization-oriented) fitness ever seen in this species.
    pub(crate) best_fitness: f32,
    /// Generation at which `best_fitness` last improved — drives stagnation.
    pub(crate) last_improved_generation: u64,
    /// Sum of size-adjusted fitness over members (= mean raw fitness); drives
    /// offspring allocation.
    pub(crate) adjusted_fitness_sum: f32,
}

impl Species {
    /// This species' stable identifier.
    #[must_use]
    pub fn id(&self) -> SpeciesId {
        self.id
    }
}

/// Compatibility distance `$\delta = c_1 \cdot E/N + c_2 \cdot D/N + c_3 \cdot \bar{W}$` between two genomes
/// (Stanley & Miikkulainen, 2002).
///
/// `E` is the excess gene count, `D` the disjoint gene count, `$\bar{W}$` the mean
/// absolute weight difference over matching genes, and `N` the connection-gene
/// count of the larger genome — or `1` when both genomes are small, to avoid
/// over-penalizing tiny networks (Stanley & Miikkulainen 2002 footnote; the
/// paper does not itself state a numeric "small" threshold — the `< 20 genes`
/// cutoff used here is rlevo's own choice).
///
/// Runs in `O(n)` by merging the two innovation-sorted connection lists.
// `a`/`b` are the two genomes and `i`/`j`/`n` the merge indices — the canonical
// names for the NEAT distance formula; renaming them would only obscure it.
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn compatibility_distance(
    a: &TopologyGenome,
    b: &TopologyGenome,
    c1: f32,
    c2: f32,
    c3: f32,
) -> f32 {
    let ca = &a.connections;
    let cb = &b.connections;
    // When one genome is empty, every gene of the other is excess relative to it
    // (it lies entirely outside the empty genome's innovation range). Current v1
    // operators never produce an empty genome, but classifying this correctly
    // keeps the formula sound if a delete operator is ever added.
    if ca.is_empty() || cb.is_empty() {
        let excess = ca.len().max(cb.len());
        #[allow(clippy::cast_precision_loss)]
        let n = if excess < 20 { 1.0 } else { excess as f32 };
        #[allow(clippy::cast_precision_loss)]
        return c1 * excess as f32 / n;
    }
    let max_a = ca.last().map_or(InnovationId::new(0), |c| c.innovation);
    let max_b = cb.last().map_or(InnovationId::new(0), |c| c.innovation);

    let (mut i, mut j) = (0usize, 0usize);
    let (mut excess, mut disjoint, mut matching) = (0u32, 0u32, 0u32);
    let mut weight_diff_sum = 0.0_f32;

    while i < ca.len() && j < cb.len() {
        match ca[i].innovation.cmp(&cb[j].innovation) {
            std::cmp::Ordering::Equal => {
                weight_diff_sum += (ca[i].weight - cb[j].weight).abs();
                matching += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                if ca[i].innovation > max_b {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if cb[j].innovation > max_a {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
                j += 1;
            }
        }
    }
    while i < ca.len() {
        if ca[i].innovation > max_b {
            excess += 1;
        } else {
            disjoint += 1;
        }
        i += 1;
    }
    while j < cb.len() {
        if cb[j].innovation > max_a {
            excess += 1;
        } else {
            disjoint += 1;
        }
        j += 1;
    }

    let n_genes = ca.len().max(cb.len());
    // Casts: gene / E / D / matching counts are small, well within f32's exact
    // integer range for any realistic genome.
    #[allow(clippy::cast_precision_loss)]
    {
        let n = if n_genes < 20 { 1.0 } else { n_genes as f32 };
        let w_bar = if matching > 0 {
            weight_diff_sum / matching as f32
        } else {
            0.0
        };
        c1 * excess as f32 / n + c2 * disjoint as f32 / n + c3 * w_bar
    }
}

/// Representative-assignment speciation pass — `$O(N \times \text{num\_species})$`.
///
/// Carries forward each existing species' (previously chosen) representative,
/// clears its membership, assigns every genome to the first species within
/// `compat_threshold` (creating a new species when none matches), drops empty
/// species, recomputes per-species best/stagnation and size-adjusted fitness,
/// then picks each survivor's next representative at random (seeded).
///
/// Runs inside `tell`: that is the only point where the new population, its
/// `fitness`, and the prior species' cloned representatives all coexist
/// consistently.
///
/// # Fitness hygiene
///
/// Each `fitness[i]` is passed through
/// `sanitize_fitness` before it feeds a
/// per-species best or `adjusted_fitness_sum` reduction (ADR 0034 correctness
/// floor). This guards direct (non-harness) callers — `NeatStrategy::tell`
/// sanitizes too, so the guard is idempotent on that path — and stops a single
/// `NaN` member from poisoning offspring apportionment for the whole run.
///
/// # Panics
///
/// Panics if `fitness.len()` differs from `population.len()`.
// The speciation pass intrinsically needs the population, its fitness, the
// species partition, the three distance coefficients + threshold, the id
// counter, the generation, and the RNG; bundling them only adds indirection.
#[allow(clippy::too_many_arguments)]
pub fn speciate(
    population: &[TopologyGenome],
    fitness: &[f32],
    species: &mut Vec<Species>,
    c1: f32,
    c2: f32,
    c3: f32,
    compat_threshold: f32,
    next_species_id: &mut SpeciesId,
    generation: u64,
    rng: &mut StdRng,
) {
    assert_eq!(
        population.len(),
        fitness.len(),
        "population and fitness must have equal length"
    );

    // 1. Carry forward representatives; reset per-generation membership.
    for s in species.iter_mut() {
        s.members.clear();
        s.adjusted_fitness_sum = 0.0;
    }

    // 2. Assign each genome to the first compatible species, else a new one.
    for (i, genome) in population.iter().enumerate() {
        let mut placed = false;
        for s in species.iter_mut() {
            if compatibility_distance(genome, &s.representative, c1, c2, c3) < compat_threshold {
                s.members.push(i);
                placed = true;
                break;
            }
        }
        if !placed {
            let id = *next_species_id;
            *next_species_id = id.succ();
            species.push(Species {
                id,
                representative: genome.clone(),
                members: vec![i],
                best_fitness: f32::NEG_INFINITY,
                last_improved_generation: generation,
                adjusted_fitness_sum: 0.0,
            });
        }
    }

    // 3. Drop empty species.
    species.retain(|s| !s.members.is_empty());

    // 4. Update best/stagnation and size-adjusted fitness (maximization).
    //
    // Sanitize NaN → −inf / +∞ → f32::MAX (ADR 0034) before *every* reduction: a
    // raw NaN member would otherwise poison `adjusted_fitness_sum` (a NaN sum),
    // corrupting offspring apportionment for *all* species for the rest of the
    // run. `speciate` is `pub` and directly callable/tested, so it guards here
    // itself (correctness floor, rules.md §3) rather than trusting the caller;
    // `NeatStrategy::tell` also sanitizes, which makes this idempotent on the
    // harness-analogue path. `remove_stagnant` already sanitizes — this keeps
    // the module consistent.
    for s in species.iter_mut() {
        let species_best = s
            .members
            .iter()
            .map(|&i| crate::fitness::sanitize_fitness(fitness[i]))
            .fold(f32::NEG_INFINITY, f32::max);
        if species_best > s.best_fitness {
            s.best_fitness = species_best;
            s.last_improved_generation = generation;
        }
        // `adjusted_fitness_sum = Σ raw/|species| = mean raw fitness`. The
        // reduction runs through `sanitized_mean`, which sanitizes each member,
        // accumulates in `f64`, and narrows once after the division (ADR 0069
        // §Decision 1).
        //
        // The accumulator width — not the ADR 0034 clamp — is what keeps the mean
        // finite: a sanitized `+∞` arrives as the *finite* `f32::MAX` and so joins
        // the sum, and *two* such members saturate an `f32` accumulator
        // (`f32::MAX + f32::MAX == f32::INFINITY`), handing `allocate_offspring`
        // an infinite `total` that erases fitness-proportional apportionment for
        // the whole population. The rationale lives on the primitive; only the
        // site-specific consequences are noted here.
        //
        // Site-specific: a broken (sanitized-to-`−∞`) member still drives this
        // species' mean to `−∞`, which the downstream `.max(0.0)` in
        // `allocate_offspring` floors to a zero share, preserving the non-negative
        // fitness-sharing precondition. `sanitized_mean`'s empty-input case
        // (`−∞`) is unreachable here — step 3 above dropped every empty species.
        s.adjusted_fitness_sum =
            crate::fitness::sanitized_mean(s.members.iter().map(|&i| fitness[i]));
    }

    // 5. Pick each survivor's next representative at random (canonical, seeded).
    for s in species.iter_mut() {
        let pick = rng.random_range(0..s.members.len());
        s.representative = population[s.members[pick]].clone();
    }
}

/// Remove species whose best fitness has not improved for `stagnation_limit`
/// generations, protecting the top [`STAGNATION_PROTECT_TOP_K`] by best fitness
/// and never emptying the population.
pub fn remove_stagnant(species: &mut Vec<Species>, generation: u64, stagnation_limit: u64) {
    if species.len() <= 1 {
        return;
    }
    // Rank by best fitness (descending) to find the protected set. Sanitize
    // NaN → −inf (worst) so a NaN-fitness species can never be protected.
    let mut order: Vec<usize> = (0..species.len()).collect();
    let sane: Vec<f32> = species
        .iter()
        .map(|s| crate::fitness::sanitize_fitness(s.best_fitness))
        .collect();
    order.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));
    let protected_count = STAGNATION_PROTECT_TOP_K.min(species.len());
    let mut keep = vec![false; species.len()];
    for &idx in order.iter().take(protected_count) {
        keep[idx] = true;
    }
    for (idx, s) in species.iter().enumerate() {
        if !keep[idx] {
            let stagnant =
                generation.saturating_sub(s.last_improved_generation) >= stagnation_limit;
            keep[idx] = !stagnant;
        }
    }
    // Never wipe the whole population: keep the single best if nothing survived.
    if !keep.iter().any(|&k| k) {
        keep[order[0]] = true;
    }
    let mut idx = 0usize;
    species.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Allocate exactly `pop_size` offspring across species, proportional to each
/// species' size-adjusted fitness, using the **largest-remainder (Hamilton)**
/// method.
///
/// Floor each species' real-valued share, then award the leftover seats to the
/// species with the largest fractional parts (ties broken by best fitness). The
/// returned counts sum to `pop_size` exactly (H3). When the total adjusted
/// fitness is non-positive (e.g. all-zero fitness — H4), seats are split as
/// evenly as possible instead.
#[must_use]
pub fn allocate_offspring(species: &[Species], pop_size: usize) -> Vec<usize> {
    let n = species.len();
    if n == 0 {
        return Vec::new();
    }
    // `total` is accumulated in `f64` by `sanitized_sum` and **kept** there (ADR
    // 0069 §Decision 1/2): it legitimately exceeds `f32` range, being a sum across
    // species of values each bounded by `f32::MAX` (ADR 0034 clamps a raw `+∞` to
    // that finite value, so it joins the sum rather than being excluded). Two such
    // species overflow an `f32` accumulator to `+∞` even though every term is
    // finite — a second, independent overflow site distinct from `speciate`'s
    // per-species mean above, and one that fires even when `speciate` gave every
    // species an individually finite mean.
    //
    // What an infinite `total` costs: it slips past the `total <= 0.0` guard
    // below, every share then evaluates to `x / ∞ == 0.0`, every `base` floors to
    // `0`, and all `pop_size` seats fall through to the round-robin leftover pass
    // — a uniform `[10, 10, 10]` split where the fitness-proportional answer was
    // `[15, 15, 0]`. Narrowing `total` back to `f32` after a correct `f64`
    // accumulation reproduces that collapse by exactly that route, because the
    // narrowing itself overflows: `[f32::MAX, f32::MAX, 1.0]` sums to `≈ 6.8e38`,
    // and `6.8e38_f64 as f32` is `+∞`. So `total` stays wide.
    //
    // No `NaN` arises on either path: the share's numerator is computed in `f64`,
    // where `pop_size × f32::MAX` stays finite for any population that can
    // physically exist (`f64::MAX / f32::MAX ≈ 5.3e269`), so a saturated species
    // divides `finite / ∞ == 0.0`. (`∞ / ∞ == NaN` — which sorts *first*
    // under the `total_cmp` tiebreak below, since `NaN` is positive in Rust — was
    // the failure mode of the fully-`f32` arithmetic this replaced, where the
    // `pop_size as f32 * f32::MAX` numerator overflowed on its own.)
    //
    // `share_term` is the single spelling of a species' contribution: floored at
    // zero (fitness sharing requires a non-negative share) and sanitized. `total`
    // and every numerator below must use *the same* value, because the
    // apportionment's `Σ share == pop_size` identity is exactly "each numerator is
    // one of `total`'s terms". `sanitized_sum` sanitizes what it accumulates, so
    // an unsanitized numerator would break that identity: a `+∞` term would divide
    // a *finite* (clamped) total to an infinite share, whose `as usize` cast
    // saturates to `usize::MAX` and sends the overshoot-reclaim loop below on
    // `usize::MAX − pop_size` iterations. `speciate` cannot produce a `+∞`
    // `adjusted_fitness_sum`, so this is a floor, not a live path — but the field
    // is `pub(crate)` and writable by any future in-crate operator.
    let share_term =
        |s: &Species| crate::fitness::sanitize_fitness(s.adjusted_fitness_sum.max(0.0));
    let total: f64 = crate::fitness::sanitized_sum(species.iter().map(share_term));

    if total <= 0.0 {
        let base = pop_size / n;
        let mut counts = vec![base; n];
        let mut leftover = pop_size - base * n;
        let mut k = 0usize;
        while leftover > 0 {
            counts[k % n] += 1;
            k += 1;
            leftover -= 1;
        }
        return counts;
    }

    let mut counts = vec![0usize; n];
    // Fractional remainders stay `f64` alongside `total`, so the largest-remainder
    // tiebreak keeps the precision of the shares it ranks.
    let mut fracs: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut assigned = 0usize;
    for (i, s) in species.iter().enumerate() {
        // Cast: `pop_size` is exact in `f64` at any realistic population size. The
        // share is computed in `f64` because the numerator reaches
        // `pop_size × f32::MAX`; the quotient is bounded above by `pop_size`
        // (this species' `share_term` is one of `total`'s terms), so the
        // `base as usize` cast below stays sound.
        #[allow(clippy::cast_precision_loss)]
        let share = pop_size as f64 * f64::from(share_term(s)) / total;
        let base = share.floor();
        // Casts: `base` is a non-negative floored share <= pop_size.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let base_usize = base as usize;
        counts[i] = base_usize;
        assigned += base_usize;
        fracs.push((i, share - base));
    }

    // Primary key: fractional remainder (finite by construction). Secondary
    // tiebreak by best fitness, sanitizing NaN → −inf (worst).
    fracs.sort_by(|a, b| {
        b.1.total_cmp(&a.1).then_with(|| {
            let (fa, fb) = (
                crate::fitness::sanitize_fitness(species[a.0].best_fitness),
                crate::fitness::sanitize_fitness(species[b.0].best_fitness),
            );
            fb.total_cmp(&fa)
        })
    });
    // Reconcile to an exact total: independently-floored f32 shares can under-
    // or (rarely) over-shoot `pop_size`. Award leftover seats to the largest
    // fractional remainders; reclaim any overshoot from the smallest.
    match assigned.cmp(&pop_size) {
        std::cmp::Ordering::Less => {
            let mut leftover = pop_size - assigned;
            let mut k = 0usize;
            while leftover > 0 {
                counts[fracs[k % n].0] += 1;
                k += 1;
                leftover -= 1;
            }
        }
        std::cmp::Ordering::Greater => {
            let mut excess = assigned - pop_size;
            let mut k = 0usize;
            while excess > 0 {
                let idx = fracs[n - 1 - (k % n)].0;
                if counts[idx] > 0 {
                    counts[idx] -= 1;
                    excess -= 1;
                }
                k += 1;
            }
        }
        std::cmp::Ordering::Equal => {}
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuroevolution::topology::{
        ActivationFn, ConnectionGene, NodeGene, NodeId, NodeKind,
    };
    use rand::SeedableRng;

    fn conn(innovation: u64, weight: f32) -> ConnectionGene {
        ConnectionGene {
            innovation: InnovationId::new(innovation),
            source: NodeId::new(0),
            target: NodeId::new(1),
            weight,
            enabled: true,
        }
    }

    fn genome_with(conns: Vec<ConnectionGene>) -> TopologyGenome {
        let nodes = vec![
            NodeGene {
                id: NodeId::new(0),
                kind: NodeKind::Input,
                activation: ActivationFn::Linear,
                bias: 0.0,
            },
            NodeGene {
                id: NodeId::new(1),
                kind: NodeKind::Output,
                activation: ActivationFn::Sigmoid,
                bias: 0.0,
            },
        ];
        TopologyGenome::new(nodes, conns)
    }

    #[test]
    fn test_compatibility_distance_identical_is_zero() {
        let g = genome_with(vec![conn(0, 1.0), conn(1, -0.5)]);
        approx::assert_relative_eq!(
            compatibility_distance(&g, &g, 1.0, 1.0, 0.4),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_compatibility_distance_matching_weight_term() {
        // Same innovations, weights differ by 1.0 and 2.0 → W̄ = 1.5, N=1 (<20).
        let a = genome_with(vec![conn(0, 0.0), conn(1, 0.0)]);
        let b = genome_with(vec![conn(0, 1.0), conn(1, 2.0)]);
        approx::assert_relative_eq!(
            compatibility_distance(&a, &b, 1.0, 1.0, 1.0),
            1.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_compatibility_distance_disjoint_and_excess() {
        // a: {0,1,2}, b: {0,3}. max_a=2, max_b=3.
        // innov 1,2 in a only and <= max_b → disjoint (2). innov 3 in b only and
        // > max_a → excess (1). innov 0 matches.
        let a = genome_with(vec![conn(0, 0.0), conn(1, 0.0), conn(2, 0.0)]);
        let b = genome_with(vec![conn(0, 0.0), conn(3, 0.0)]);
        // c1·E/N + c2·D/N + c3·W̄ = 1·1/1 + 1·2/1 + 0 = 3.
        approx::assert_relative_eq!(
            compatibility_distance(&a, &b, 1.0, 1.0, 0.4),
            3.0,
            epsilon = 1e-6
        );
    }

    fn species_with(id: u64, adjusted: f32, best: f32, last_improved: u64) -> Species {
        Species {
            id: SpeciesId::new(id),
            representative: genome_with(vec![conn(0, 0.0)]),
            members: vec![0],
            best_fitness: best,
            last_improved_generation: last_improved,
            adjusted_fitness_sum: adjusted,
        }
    }

    #[test]
    fn test_compatibility_distance_empty_genome_branch() {
        // Early-return branch (one genome has no connections): every gene of the
        // other is excess. `excess = max(len_a, len_b)`, `N = 1` when `excess <
        // 20` else `excess`, and δ = c1·excess/N. With 25 genes (≥ 20) N = excess,
        // so δ = c1·excess/excess = c1 exactly.
        let full = genome_with((0..25).map(|k| conn(k, 0.0)).collect());
        let empty = genome_with(vec![]);

        // empty-vs-full: excess = 25, N = 25 → δ = c1 = 1.0.
        approx::assert_relative_eq!(
            compatibility_distance(&empty, &full, 1.0, 1.0, 0.4),
            1.0,
            epsilon = 1e-6
        );
        // full-vs-empty: symmetric — same δ = 1.0.
        approx::assert_relative_eq!(
            compatibility_distance(&full, &empty, 1.0, 1.0, 0.4),
            1.0,
            epsilon = 1e-6
        );
        // empty-vs-empty: excess = 0, N = 1 → δ = c1·0/1 = 0.0.
        approx::assert_relative_eq!(
            compatibility_distance(&empty, &empty, 1.0, 1.0, 0.4),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "population and fitness must have equal length")]
    fn test_speciate_panics_on_length_mismatch() {
        let mut rng = StdRng::seed_from_u64(0);
        let population = vec![
            genome_with(vec![conn(0, 0.0)]),
            genome_with(vec![conn(0, 1.0)]),
        ];
        // Deliberately shorter than the population to trip the `assert_eq!`.
        let fitness = vec![1.0];
        let mut species: Vec<Species> = Vec::new();
        let mut next_id = SpeciesId::new(0);
        speciate(
            &population,
            &fitness,
            &mut species,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut next_id,
            0,
            &mut rng,
        );
    }

    #[test]
    fn test_allocate_offspring_single_species_gets_all() {
        // n == 1 with positive adjusted fitness: share == pop_size, so the lone
        // species receives every seat.
        let species = vec![species_with(0, 3.0, 3.0, 0)];
        let counts = allocate_offspring(&species, 32);
        assert_eq!(counts, vec![32]);
    }

    #[test]
    fn test_allocate_offspring_empty_species_returns_empty() {
        let species: Vec<Species> = Vec::new();
        let counts = allocate_offspring(&species, 32);
        assert!(counts.is_empty(), "no species → no offspring counts");
    }

    #[test]
    fn test_allocate_offspring_counts_len_equals_species_len() {
        let species = vec![
            species_with(0, 1.0, 1.0, 0),
            species_with(1, 2.0, 1.0, 0),
            species_with(2, 3.0, 1.0, 0),
        ];
        let counts = allocate_offspring(&species, 20);
        assert_eq!(
            counts.len(),
            species.len(),
            "one count per species, positionally aligned"
        );
    }

    #[test]
    fn test_remove_stagnant_keeps_best_when_all_stagnant() {
        // All three species are stagnant (last improved gen 0; gen 30, limit 15).
        // The top-K=2 protection shields the two highest-fitness species, so the
        // population is never wiped: species 0 (best 9) and species 1 (best 5)
        // survive, species 2 (best 1) is removed. (The lower `keep single best`
        // fallback is unreachable here — top-K already keeps 2.)
        let mut species = vec![
            species_with(0, 1.0, 9.0, 0),
            species_with(1, 1.0, 5.0, 0),
            species_with(2, 1.0, 1.0, 0),
        ];
        remove_stagnant(&mut species, 30, 15);
        assert_eq!(
            species.len(),
            2,
            "top-K protection keeps the population alive"
        );
        let ids: Vec<u64> = species.iter().map(|s| s.id().get()).collect();
        assert!(ids.contains(&0), "highest-fitness species survives");
        assert!(ids.contains(&1), "second-highest-fitness species survives");
        assert!(
            !ids.contains(&2),
            "lowest-fitness stagnant species is removed"
        );
        assert!(!species.is_empty(), "removal never empties the population");
    }

    #[test]
    fn test_remove_stagnant_single_species_is_noop() {
        // `species.len() <= 1` returns early — even a long-stagnant lone species
        // is retained (never empty the population).
        let mut species = vec![species_with(0, 1.0, 1.0, 0)];
        remove_stagnant(&mut species, 1000, 15);
        assert_eq!(species.len(), 1, "a single species is a no-op");
        assert_eq!(species[0].id().get(), 0);
    }

    #[test]
    fn test_allocate_offspring_sums_to_pop_size() {
        // Largest-remainder over shares 1.0, 2.0, 3.0 of 64 seats.
        let species = vec![
            species_with(0, 1.0, 1.0, 0),
            species_with(1, 2.0, 1.0, 0),
            species_with(2, 3.0, 1.0, 0),
        ];
        let counts = allocate_offspring(&species, 64);
        assert_eq!(
            counts.iter().sum::<usize>(),
            64,
            "apportionment must sum exactly"
        );
        // Roughly proportional: species 2 (share 3) gets the most.
        assert!(counts[2] >= counts[1] && counts[1] >= counts[0]);
    }

    #[test]
    fn test_allocate_offspring_zero_total_splits_evenly() {
        let species = vec![
            species_with(0, 0.0, 0.0, 0),
            species_with(1, 0.0, 0.0, 0),
            species_with(2, 0.0, 0.0, 0),
        ];
        let counts = allocate_offspring(&species, 10);
        assert_eq!(counts.iter().sum::<usize>(), 10);
        // 10 / 3 → [4, 3, 3].
        assert_eq!(counts, vec![4, 3, 3]);
    }

    #[test]
    fn test_remove_stagnant_protects_top_k() {
        // gen 30; limit 15. species 0 (best 9, stagnant since gen 0) is protected
        // as a top-K; species 2 (best 1, stagnant since gen 0) is removed;
        // species 1 (improved gen 25) survives.
        let mut species = vec![
            species_with(0, 1.0, 9.0, 0),
            species_with(1, 1.0, 5.0, 25),
            species_with(2, 1.0, 1.0, 0),
        ];
        remove_stagnant(&mut species, 30, 15);
        let ids: Vec<u64> = species.iter().map(|s| s.id().get()).collect();
        assert!(
            ids.contains(&0),
            "top-fitness stagnant species is protected"
        );
        assert!(ids.contains(&1), "recently-improved species survives");
        assert!(!ids.contains(&2), "low-fitness stagnant species is removed");
    }

    #[test]
    fn test_speciate_assigns_to_first_compatible_representative() {
        let mut rng = StdRng::seed_from_u64(0);
        // Two clearly distinct genomes (very different weights) + a near-clone of
        // the first → 2 species, first genome and its clone together.
        let g0 = genome_with(vec![conn(0, 0.0)]);
        let g1 = genome_with(vec![conn(0, 10.0)]);
        let g0_clone = genome_with(vec![conn(0, 0.05)]);
        let population = vec![g0, g1, g0_clone];
        let fitness = vec![1.0, 1.0, 1.0];
        let mut species: Vec<Species> = Vec::new();
        let mut next_id = SpeciesId::new(0);
        // c3=1.0, threshold 1.0: |0-10|=10 > 1 (split); |0-0.05|=0.05 < 1 (join).
        speciate(
            &population,
            &fitness,
            &mut species,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut next_id,
            0,
            &mut rng,
        );
        assert_eq!(species.len(), 2, "distinct genome forms its own species");
        let sizes: Vec<usize> = species.iter().map(|s| s.members.len()).collect();
        assert!(
            sizes.contains(&2) && sizes.contains(&1),
            "g0 and its clone share a species"
        );
    }

    /// Regression (ADR 0034): `speciate` sanitizes each member's
    /// fitness before the per-species best / `adjusted_fitness_sum` reductions.
    /// A raw `NaN` must not become a species best nor poison the size-adjusted
    /// mean (which would corrupt offspring apportionment for the whole run); a
    /// `$+\infty$` member ranks top but as a **finite** value (`f32::MAX`).
    #[test]
    fn test_speciate_sanitizes_nan_and_inf_fitness() {
        let mut rng = StdRng::seed_from_u64(0);
        // Three near-clones → a single species, so all three feed one reduction.
        let population = vec![
            genome_with(vec![conn(0, 0.0)]),
            genome_with(vec![conn(0, 0.01)]),
            genome_with(vec![conn(0, 0.02)]),
        ];
        // Member 0 NaN (must NOT become best / poison the sum), member 1 +∞
        // (ranks top but finite), member 2 a plain 2.0.
        let fitness = vec![f32::NAN, f32::INFINITY, 2.0];
        let mut species: Vec<Species> = Vec::new();
        let mut next_id = SpeciesId::new(0);
        speciate(
            &population,
            &fitness,
            &mut species,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut next_id,
            0,
            &mut rng,
        );
        assert_eq!(species.len(), 1, "near-clones form a single species");

        let s = &species[0];
        // best_fitness is the sanitized +∞ = f32::MAX — finite, never NaN.
        assert!(
            !s.best_fitness.is_nan(),
            "a NaN member never becomes a species best"
        );
        approx::assert_relative_eq!(s.best_fitness, f32::MAX);
        // The size-adjusted mean is not NaN: sanitizing the NaN to −∞ makes the
        // mean −∞ (not NaN), which the downstream `.max(0.0)` floors — the key
        // regression is that it can no longer poison apportionment to NaN.
        assert!(
            !s.adjusted_fitness_sum.is_nan(),
            "a NaN member never poisons adjusted_fitness_sum"
        );

        // Apportionment stays well-defined (sums exactly) despite the broken member.
        let counts = allocate_offspring(&species, 8);
        assert_eq!(
            counts.iter().sum::<usize>(),
            8,
            "offspring apportionment sums exactly, uncorrupted by a NaN member"
        );
    }

    /// Build a population of `weights.len()` single-connection genomes, one per
    /// weight, and speciate it. Weights within `1.0` of each other share a
    /// species (c3 = 1.0, threshold = 1.0).
    fn speciate_weights(weights: &[f32], fitness: &[f32]) -> Vec<Species> {
        let mut rng = StdRng::seed_from_u64(0);
        let population: Vec<TopologyGenome> = weights
            .iter()
            .map(|&w| genome_with(vec![conn(0, w)]))
            .collect();
        let mut species: Vec<Species> = Vec::new();
        let mut next_id = SpeciesId::new(0);
        speciate(
            &population,
            fitness,
            &mut species,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut next_id,
            0,
            &mut rng,
        );
        species
    }

    /// Regression (ADR 0069): **two** `$+\infty$` members in one species must not
    /// overflow `adjusted_fitness_sum` to `$+\infty$`.
    ///
    /// ADR 0034 sanitizes `$+\infty$` to `f32::MAX`, which is *finite* and therefore
    /// joins the sum — and `f32::MAX + f32::MAX == f32::INFINITY`. A single `$+\infty$`
    /// member (the case
    /// [`test_speciate_sanitizes_nan_and_inf_fitness`](test_speciate_sanitizes_nan_and_inf_fitness)
    /// covers) is precisely the one that does *not* overflow, so this needs a
    /// second one.
    #[test]
    fn test_speciate_two_inf_members_do_not_overflow_adjusted_sum() {
        // All-`+∞` species: the mean of `n` copies of `f32::MAX` is `f32::MAX`.
        let species = speciate_weights(&[0.0, 0.01], &[f32::INFINITY, f32::INFINITY]);
        assert_eq!(species.len(), 1, "near-clones form a single species");
        assert!(
            species[0].adjusted_fitness_sum.is_finite(),
            "two sanitized `+∞` members must not saturate the accumulator to `+∞`"
        );
        approx::assert_relative_eq!(species[0].adjusted_fitness_sum, f32::MAX);

        // Mixed species: mean of (f32::MAX, f32::MAX, 2.0), taken in `f64`.
        let species = speciate_weights(&[0.0, 0.01, 0.02], &[f32::INFINITY, f32::INFINITY, 2.0]);
        assert_eq!(species.len(), 1, "near-clones form a single species");
        assert!(
            species[0].adjusted_fitness_sum.is_finite(),
            "two sanitized `+∞` members must not saturate the accumulator to `+∞`"
        );
        #[allow(clippy::cast_possible_truncation)]
        let expected = ((2.0 * f64::from(f32::MAX) + 2.0) / 3.0) as f32;
        approx::assert_relative_eq!(species[0].adjusted_fitness_sum, expected);
    }

    /// Regression (ADR 0069): a species poisoned by two `$+\infty$` members must not
    /// erase fitness-proportional apportionment for the **whole** population.
    ///
    /// This is the **end-to-end** assertion: `speciate` must not hand
    /// `allocate_offspring` an infinite `adjusted_fitness_sum`, and
    /// `allocate_offspring` must not turn one into an infinite `total`. In the
    /// original all-`f32` arithmetic both happened, the `total <= 0.0` guard did
    /// not fire (`$+\infty > 0$`), healthy species computed `$\text{finite} / \infty = 0$`, the
    /// poisoned one `$\infty / \infty = \text{NaN}$` (which floors to `0`), and every seat fell
    /// through to the round-robin leftover pass — `[10, 10, 10]` instead of a
    /// proportional split.
    ///
    /// It is deliberately **not** the discriminating guard for either accumulator
    /// on its own: `allocate_offspring` now sanitizes its own terms
    /// (`share_term`), so a `speciate` that regressed to an `f32` accumulator is
    /// caught here only in combination. The single-site guards are
    /// [`test_speciate_two_inf_members_do_not_overflow_adjusted_sum`] and
    /// [`test_allocate_offspring_total_does_not_overflow_across_species`].
    #[test]
    fn test_allocate_offspring_poisoned_species_keeps_proportionality() {
        // Three species (weights 10 apart), two members each.
        let weights = [0.0, 0.01, 10.0, 10.01, 20.0, 20.01];

        // Baseline control: means 100 / 10 / 1 over 30 seats.
        let species = speciate_weights(&weights, &[100.0, 100.0, 10.0, 10.0, 1.0, 1.0]);
        assert_eq!(species.len(), 3, "three well-separated species");
        assert_eq!(
            allocate_offspring(&species, 30),
            vec![27, 3, 0],
            "healthy apportionment is fitness-proportional (largest remainder)"
        );

        // Same population, but the top species' two members both score `+∞`.
        let species = speciate_weights(
            &weights,
            &[f32::INFINITY, f32::INFINITY, 10.0, 10.0, 1.0, 1.0],
        );
        assert_eq!(species.len(), 3, "three well-separated species");
        let counts = allocate_offspring(&species, 30);
        assert_eq!(
            counts,
            vec![30, 0, 0],
            "the top-fitness species keeps its share instead of collapsing to a \
             round-robin [10, 10, 10] split"
        );
        assert!(
            counts[0] > counts[1] && counts[0] > counts[2],
            "the highest-fitness species still dominates apportionment"
        );
    }

    /// Regression (ADR 0069): `allocate_offspring`'s `total` overflows
    /// **independently** of `speciate`. Each species' `adjusted_fitness_sum` here
    /// is individually finite, yet their sum saturates an `f32` accumulator — so
    /// `total` must be accumulated *and kept* in `f64`.
    #[test]
    fn test_allocate_offspring_total_does_not_overflow_across_species() {
        let species = vec![
            species_with(0, f32::MAX, f32::MAX, 0),
            species_with(1, f32::MAX, f32::MAX, 0),
            species_with(2, 1.0, 1.0, 0),
        ];
        assert!(
            species.iter().all(|s| s.adjusted_fitness_sum.is_finite()),
            "each species' adjusted fitness is finite on its own"
        );
        assert_eq!(
            allocate_offspring(&species, 30),
            vec![15, 15, 0],
            "the two `f32::MAX` species split the seats evenly rather than \
             collapsing to a round-robin [10, 10, 10] split"
        );
    }

    /// `total` and each share numerator must read a species' contribution through
    /// the **same** `share_term`, so `$\Sigma\,\text{share} = \text{pop\_size}$` stays an identity.
    ///
    /// `speciate` cannot write a `$+\infty$` `adjusted_fitness_sum` (that is what
    /// [`test_speciate_two_inf_members_do_not_overflow_adjusted_sum`] pins), but
    /// the field is `pub(crate)` and any in-crate operator can. `sanitized_sum`
    /// clamps such a term to `f32::MAX` inside `total`; a numerator that skipped
    /// the clamp would divide `$\infty$` by a *finite* total, and `∞.floor() as usize`
    /// saturates to `usize::MAX` — so the overshoot-reclaim loop would run
    /// `usize::MAX − pop_size` times. Note the failure mode this test guards is a
    /// **divergence**, not a wrong answer: if it ever regresses it will hang here
    /// rather than fail.
    ///
    /// The expected apportionment is the one a `f32::MAX` term would get, which is
    /// what the clamp means.
    #[test]
    fn test_allocate_offspring_infinite_term_is_clamped_like_the_total() {
        let poisoned = vec![
            species_with(0, f32::INFINITY, f32::INFINITY, 0),
            species_with(1, 10.0, 10.0, 0),
            species_with(2, 1.0, 1.0, 0),
        ];
        let clamped = vec![
            species_with(0, f32::MAX, f32::MAX, 0),
            species_with(1, 10.0, 10.0, 0),
            species_with(2, 1.0, 1.0, 0),
        ];
        let counts = allocate_offspring(&poisoned, 30);
        assert_eq!(
            counts.iter().sum::<usize>(),
            30,
            "apportionment still sums to `pop_size` exactly (H3)"
        );
        assert_eq!(
            counts,
            allocate_offspring(&clamped, 30),
            "a `+∞` term apportions exactly as its sanitized `f32::MAX` does"
        );
        assert_eq!(
            counts,
            vec![30, 0, 0],
            "and the saturated species takes all"
        );
    }

    // ------------------------------------------------------------------
    // ADR 0069 §Decision 5 — behavioural rescale-invariance property.
    // ------------------------------------------------------------------

    use crate::rng::{SeedPurpose, seed_stream};
    use proptest::collection::vec as prop_vec;
    // Explicit list, not the glob prelude: the prelude re-exports a `Rng` that
    // would collide with the `rand::RngExt`/`StdRng` pulled in via `use super::*`.
    use proptest::prelude::{ProptestConfig, any, prop_assert_eq, proptest};

    /// Speciate a generated population, drawing representatives from an ADR 0029
    /// `seed_stream` substream — the same `SeedPurpose::Representative` stream
    /// `NeatStrategy::tell` uses in production. proptest supplies only the `u64`
    /// seed; it never drives the algorithm's RNG itself (ADR 0036 §3).
    fn speciate_seeded(weights: &[f32], fitness: &[f32], seed: u64) -> Vec<Species> {
        let mut rng = seed_stream(seed, 0, SeedPurpose::Representative);
        let population: Vec<TopologyGenome> = weights
            .iter()
            .map(|&w| genome_with(vec![conn(0, w)]))
            .collect();
        let mut species: Vec<Species> = Vec::new();
        let mut next_id = SpeciesId::new(0);
        speciate(
            &population,
            fitness,
            &mut species,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut next_id,
            0,
            &mut rng,
        );
        species
    }

    /// `$\text{allocate\_offspring} \circ \text{speciate}$` with every fitness value multiplied by `c`.
    fn apportion_at_scale(
        weights: &[f32],
        fitness: &[f32],
        c: f32,
        pop_size: usize,
        seed: u64,
    ) -> Vec<usize> {
        let scaled: Vec<f32> = fitness.iter().map(|&f| f * c).collect();
        allocate_offspring(&speciate_seeded(weights, &scaled, seed), pop_size)
    }

    /// Largest `k` with `$2^k \cdot \text{max\_fitness}$` still finite in `f32` — the top of
    /// the rescale range the property is valid over (see the property's docs).
    fn max_exact_scale_exponent(max_fitness: f32) -> i32 {
        #[allow(clippy::cast_possible_truncation)]
        let mut k = (f64::from(f32::MAX) / f64::from(max_fitness))
            .log2()
            .floor() as i32;
        // Defensive: `log2().floor()` is one ULP away from an off-by-one at the
        // very top of the range, and an overflow here would silently move the
        // test into the clamped regime the property excludes.
        while !(2.0_f32.powi(k) * max_fitness).is_finite() {
            k -= 1;
        }
        k
    }

    proptest! {
        // Pure host arithmetic — no backend, no tensors — but each case runs
        // three full speciate+apportion passes over up to 12 cloned genomes.
        // 64 cases is the ADR 0036 §5 "cheap structural" tier.
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// **ADR 0069 §Decision 5, property 1.** Offspring apportionment is
        /// invariant to a positive rescale of the whole fitness vector:
        /// `allocate_offspring(speciate(c·f)) == allocate_offspring(speciate(f))`.
        ///
        /// This is the *behavioural* net §Decision 5 chooses over a source-text
        /// guard. It keys on the answer, not on the spelling: it fails if
        /// `speciate`'s per-species mean **or** `allocate_offspring`'s `total`
        /// accumulates in `f32`, even though the latter site contains no lexical
        /// trace of fitness hygiene at all.
        ///
        /// # The bound on `c`, and what "saturated" can mean here
        ///
        /// `c` ranges over powers of two up to `2^k_max`, the largest scale at
        /// which `$c \cdot \max(f)$` is still **representable** in `f32`. That upper end
        /// is the regime the defect lives in — per-species means within a factor
        /// of two of `f32::MAX`, so both an `f32` member sum inside a species and
        /// an `f32` sum of species means overflow to `$+\infty$`.
        ///
        /// It deliberately stops there. Past `k_max` the members themselves
        /// overflow and ADR 0034 clamps them to `f32::MAX`, which makes *every*
        /// saturated member equal: proportionality is destroyed by the clamp, by
        /// design, and no accumulator width can restore it. §Decision 5's phrase
        /// "including the case where `$c \cdot f$` saturates to `f32::MAX`" is therefore
        /// only true of *reaching* `f32::MAX`, not of *clamping* to it — the
        /// clamped case is an example test
        /// ([`test_allocate_offspring_poisoned_species_keeps_proportionality`]),
        /// where the expected answer is the one a tie yields, not the rescaled one.
        ///
        /// Powers of two make the invariance **exact**: `$c \cdot f$` is error-free, and
        /// scaling by a power of two commutes with round-to-nearest, so every
        /// intermediate (the `f64` member sum, the `f32` narrowed mean, the `f64`
        /// species total, the `f64` share) is exactly `c` times its unscaled
        /// counterpart and the share quotients are bit-identical. The assertion is
        /// therefore `==` on the counts, with no tolerance to hide a drift.
        ///
        /// There is **no lower** bound to state here, unlike the `z_score`
        /// property: apportionment compares each species' share against a total
        /// that shrinks with it, so it has no absolute-scale floor to trip over.
        /// Generated fitness stays at or above `1`, so a small `c` cannot reach
        /// the subnormal range either.
        ///
        /// Every case checks `k_max` explicitly in addition to the generated
        /// scale, so the saturating regime is exercised on *every* case rather
        /// than only when `headroom` happens to shrink to zero.
        #[test]
        fn prop_apportionment_is_invariant_to_positive_rescale(
            // (per-species fitness level, per-species member count).
            clusters in prop_vec((1u32..=1000, 1usize..=3), 2..=4),
            pop_size in 1usize..=64,
            headroom in 0u32..=120,
            seed in any::<u64>(),
        ) {
            // Cluster `k`'s genomes carry connection weights near `10·k`. With
            // `c3 = 1` and `compat_threshold = 1`, the compatibility distance is
            // the absolute weight difference: members of a cluster (≤ 0.02 apart)
            // group, distinct clusters (≥ 9.98 apart) split.
            let mut weights: Vec<f32> = Vec::new();
            let mut fitness: Vec<f32> = Vec::new();
            for (k, &(level, size)) in clusters.iter().enumerate() {
                for j in 0..size {
                    #[allow(clippy::cast_precision_loss)]
                    weights.push(10.0 * k as f32 + 0.01 * j as f32);
                    #[allow(clippy::cast_precision_loss)]
                    fitness.push(level as f32 + j as f32);
                }
            }

            let max_fitness = fitness.iter().copied().fold(0.0_f32, f32::max);
            let k_max = max_exact_scale_exponent(max_fitness);
            #[allow(clippy::cast_possible_wrap)]
            let k = k_max - headroom as i32;

            let baseline = apportion_at_scale(&weights, &fitness, 1.0, pop_size, seed);
            prop_assert_eq!(
                baseline.iter().sum::<usize>(),
                pop_size,
                "apportionment must sum to pop_size exactly (H3)"
            );

            // The saturating end of the valid range, on every case.
            prop_assert_eq!(
                apportion_at_scale(&weights, &fitness, 2.0_f32.powi(k_max), pop_size, seed),
                baseline.clone(),
                "apportionment changed at the maximal representable scale 2^{}",
                k_max
            );
            // A generated point inside the range.
            prop_assert_eq!(
                apportion_at_scale(&weights, &fitness, 2.0_f32.powi(k), pop_size, seed),
                baseline,
                "apportionment changed at scale 2^{}",
                k
            );
        }
    }
}
