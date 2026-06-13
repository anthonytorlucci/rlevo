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
//! NEAT is **maximization** (higher fitness is better) — opposite the crate-wide
//! minimize convention. `best_fitness` tracks a maximum and fitness sharing
//! assumes **non-negative** raw fitness; a task whose native objective is a cost
//! must supply `−cost` (the deferred `rlevo-hybrid` consumer's concern).

use rand::RngExt;
use rand::rngs::StdRng;

use super::topology::TopologyGenome;

/// Identifier for a species. Monotone within a run.
pub type SpeciesId = u64;

/// Number of top-fitness species shielded from stagnation removal, so a
/// population-wide plateau cannot wipe every species at once.
pub const STAGNATION_PROTECT_TOP_K: usize = 2;

/// A cluster of compatible genomes.
///
/// `members` holds **indices** into the population `Vec`, not genomes, so the
/// population stays the single owner of genome data and the partition is a cheap
/// view. `representative` is a *cloned* genome (not an index) so it survives
/// population turnover between generations.
#[derive(Clone, Debug)]
pub struct Species {
    /// Stable species id.
    pub id: SpeciesId,
    /// Frozen comparison anchor for this generation's assignment pass — a
    /// randomly chosen member of the previous generation (R1 §5).
    pub representative: TopologyGenome,
    /// Indices into the population `Vec` assigned to this species this generation.
    pub members: Vec<usize>,
    /// Best (maximization-oriented) fitness ever seen in this species.
    pub best_fitness: f32,
    /// Generation at which `best_fitness` last improved — drives stagnation.
    pub last_improved_generation: u64,
    /// Sum of size-adjusted fitness over members (= mean raw fitness); drives
    /// offspring allocation.
    pub adjusted_fitness_sum: f32,
}

/// Compatibility distance `δ = c1·E/N + c2·D/N + c3·W̄` between two genomes
/// (R1 §5).
///
/// `E` is the excess gene count, `D` the disjoint gene count, `W̄` the mean
/// absolute weight difference over matching genes, and `N` the connection-gene
/// count of the larger genome — or `1` when both genomes are small (< 20 genes),
/// to avoid over-penalizing tiny networks (Stanley 2002 footnote).
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
    let max_a = ca.last().map_or(0, |c| c.innovation);
    let max_b = cb.last().map_or(0, |c| c.innovation);

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

/// Representative-assignment speciation pass — `O(N × num_species)`.
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
            *next_species_id += 1;
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
    for s in species.iter_mut() {
        let species_best = s
            .members
            .iter()
            .map(|&i| fitness[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if species_best > s.best_fitness {
            s.best_fitness = species_best;
            s.last_improved_generation = generation;
        }
        let sum: f32 = s.members.iter().map(|&i| fitness[i]).sum();
        // `adjusted_fitness_sum = Σ raw/|species| = mean raw fitness`.
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / s.members.len() as f32;
        s.adjusted_fitness_sum = mean;
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
    // Rank by best fitness (descending) to find the protected set.
    let mut order: Vec<usize> = (0..species.len()).collect();
    order.sort_by(|&a, &b| {
        species[b]
            .best_fitness
            .partial_cmp(&species[a].best_fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
    let total: f32 = species.iter().map(|s| s.adjusted_fitness_sum.max(0.0)).sum();

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
    let mut fracs: Vec<(usize, f32)> = Vec::with_capacity(n);
    let mut assigned = 0usize;
    for (i, s) in species.iter().enumerate() {
        // Casts: pop_size and shares are small positive magnitudes.
        #[allow(clippy::cast_precision_loss)]
        let share = pop_size as f32 * s.adjusted_fitness_sum.max(0.0) / total;
        let base = share.floor();
        // Casts: `base` is a non-negative floored share <= pop_size.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let base_usize = base as usize;
        counts[i] = base_usize;
        assigned += base_usize;
        fracs.push((i, share - base));
    }

    fracs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                species[b.0]
                    .best_fitness
                    .partial_cmp(&species[a.0].best_fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
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
    use crate::neuroevolution::topology::{ActivationFn, ConnectionGene, NodeGene, NodeKind};
    use rand::SeedableRng;

    fn conn(innovation: u64, weight: f32) -> ConnectionGene {
        ConnectionGene {
            innovation,
            source: 0,
            target: 1,
            weight,
            enabled: true,
        }
    }

    fn genome_with(conns: Vec<ConnectionGene>) -> TopologyGenome {
        let nodes = vec![
            NodeGene { id: 0, kind: NodeKind::Input, activation: ActivationFn::Linear, bias: 0.0 },
            NodeGene { id: 1, kind: NodeKind::Output, activation: ActivationFn::Sigmoid, bias: 0.0 },
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

    fn species_with(id: SpeciesId, adjusted: f32, best: f32, last_improved: u64) -> Species {
        Species {
            id,
            representative: genome_with(vec![conn(0, 0.0)]),
            members: vec![0],
            best_fitness: best,
            last_improved_generation: last_improved,
            adjusted_fitness_sum: adjusted,
        }
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
        assert_eq!(counts.iter().sum::<usize>(), 64, "apportionment must sum exactly");
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
        let ids: Vec<SpeciesId> = species.iter().map(|s| s.id).collect();
        assert!(ids.contains(&0), "top-fitness stagnant species is protected");
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
        let mut next_id = 0u64;
        // c3=1.0, threshold 1.0: |0-10|=10 > 1 (split); |0-0.05|=0.05 < 1 (join).
        speciate(&population, &fitness, &mut species, 1.0, 1.0, 1.0, 1.0, &mut next_id, 0, &mut rng);
        assert_eq!(species.len(), 2, "distinct genome forms its own species");
        let sizes: Vec<usize> = species.iter().map(|s| s.members.len()).collect();
        assert!(sizes.contains(&2) && sizes.contains(&1), "g0 and its clone share a species");
    }
}
