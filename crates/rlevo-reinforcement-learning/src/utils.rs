//! Shared utility functions for reinforcement learning.
//!
//! Provides stateless helper functions used across multiple RL algorithms,
//! such as Bellman target computation and Polyak averaging.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorPrimitive};

/// Computes Bellman backup target Q-values for a mini-batch.
///
/// Applies the standard one-step TD target:
/// `target = reward + γ · max_next_Q · (1 − terminated)`.
///
/// # Arguments
///
/// - `terminated` — per-sample mask in `{0.0, 1.0}` that is `1.0` **only** for
///   an *environmental termination* (the MDP itself reached an absorbing
///   state). It must **not** be set for a *truncation* (a time-limit cutoff).
///   A truncated transition still has a well-defined continuation value, so the
///   bootstrap term must survive; zeroing it there biases every Q-value
///   downward on any time-limited environment. See Pardo et al., "Time Limits
///   in Reinforcement Learning", ICML 2018, Eq. 6 (partial-episode
///   bootstrapping), and Gymnasium's
///   `Q(s,a) = r + γ · ¬terminated · max_a' Q(s′,a')`.
///
/// # Non-finite hardening
///
/// The bootstrap term is **masked**, not scaled by `(1 − terminated)`. The two
/// agree exactly for finite inputs, but scaling propagates poison: IEEE-754
/// gives `NaN · 0.0 == NaN` and `inf · 0.0 == NaN`, so a single non-finite
/// entry in `next_q_max` would survive a terminal transition and contaminate
/// the target — the one place the algorithm has a guaranteed-correct value
/// (the reward alone). Selecting on the terminal mask forces the bootstrap to
/// be exactly `0` wherever `terminated == 1.0`, whatever `next_q_max` holds
/// there. This is defensive only; it does not alter any target computed from
/// finite inputs.
pub fn compute_target_q_values<B: Backend>(
    rewards: Tensor<B, 1>,
    next_q_max: Tensor<B, 1>,
    terminated: Tensor<B, 1>,
    gamma: f32,
) -> Tensor<B, 1> {
    let is_terminal = terminated.equal_elem(1.0);
    let bootstrap = next_q_max.mul_scalar(gamma).mask_fill(is_terminal, 0.0);
    rewards + bootstrap
}

// ---------------------------------------------------------------------------
// Polyak averaging
// ---------------------------------------------------------------------------

/// Error returned by [`polyak_update`] when the `active` and `target` networks
/// have mismatched [`ParamId`] topologies.
///
/// Both variants signal that `target` was not derived from `active` (a fresh
/// `init` mints new [`ParamId`]s, so two independently built modules never
/// match). Building `target` by cloning `active` makes both impossible by
/// construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PolyakError {
    /// A `target` parameter has no counterpart in `active` — the modules were
    /// built independently (issue #341 defect 1). Names the offending id.
    #[error(
        "polyak_update: target parameter {0:?} has no counterpart in the active network; \
         the modules were built independently — build target by cloning active"
    )]
    MissingActive(ParamId),
    /// An `active` parameter was never consumed by any `target` field — `target`
    /// is a strict subset of `active` (issue #341 defect 3). Applying only the
    /// overlap would be a silent partial update, so this is surfaced instead.
    #[error(
        "polyak_update: active parameter {0:?} has no counterpart in target — target is a \
         strict subset of active; a partial update would be silent"
    )]
    MissingTarget(ParamId),
}

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorPrimitive<B>>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        // Store the rank-erased on-device primitive handle, not a host
        // `TensorData` readback: keeping the value on-device avoids a
        // gratuitous device→host→device round-trip on every soft update
        // (issue #322). Both networks live on the same backend/device.
        self.tensors.insert(param.id, param.val().into_primitive());
    }
}

struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorPrimitive<B>>,
    seen: HashSet<ParamId>,
    tau: f32,
    error: Option<PolyakError>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        // `.get().cloned()`, not `.remove()`: a single `ParamId` may back
        // several target fields (tied weights — two module fields holding
        // clones of one `Param` share an id), so each id must remain lookable
        // more than once.
        //
        // On a lookup miss the mapper cannot fail out of `map_float` (the trait
        // method is infallible), so it records the *first* miss and returns the
        // parameter untouched; `polyak_update` reports the recorded error after
        // the walk completes.
        let Some(active) = self.active.get(&id).cloned() else {
            self.error.get_or_insert(PolyakError::MissingActive(id));
            return param;
        };
        self.seen.insert(id);
        let tau = self.tau;
        param.map(move |target_tensor| {
            // Rewrap the stored on-device primitive as a rank-`D` tensor; `D`
            // is inferred from this `map_float<D>` call site and the primitive
            // carries the real shape. No host upload occurs (issue #322).
            let active_tensor = Tensor::<B, D>::from_primitive(active);
            target_tensor.mul_scalar(1.0 - tau) + active_tensor.mul_scalar(tau)
        })
    }
}

/// Polyak-averages `active` into `target`: `target ← (1 − τ)·target + τ·active`.
///
/// Used by every off-policy algorithm that maintains a target network (DQN,
/// C51, QR-DQN, DDPG, TD3, SAC). Pass `tau = 1.0` for a hard copy.
///
/// Parameters are paired by [`ParamId`], not by position or name, so `target`
/// must carry the *same* `ParamId`s as `active`.
///
/// # Errors
///
/// Returns [`PolyakError::MissingActive`] if `target` holds a parameter whose
/// [`ParamId`] is absent from `active` — that is, if the two modules were built
/// independently rather than `target` being derived from `active`. The error
/// names the first offending [`ParamId`].
///
/// Returns [`PolyakError::MissingTarget`] if `active` holds a parameter whose
/// [`ParamId`] is absent from `target` — a strict-subset topology in which some
/// active parameters have no counterpart to update. Rather than silently
/// applying a partial update, the update reports the smallest leftover
/// [`ParamId`]. On either error the returned network is discarded and the
/// caller's `target` retains its prior weights.
///
/// Tied weights are supported: several `target` fields may share one
/// [`ParamId`] (module fields holding clones of a single [`Param`]), and each
/// is blended from the one matching active entry.
///
/// A fresh `init` mints new `ParamId`s, so two separately initialised networks
/// never match even when their architecture and weights are identical. Build
/// the target network by cloning the active one (`let target = active.clone();`)
/// and both errors are impossible by construction; every in-tree agent does this.
pub fn polyak_update<B: Backend, M: Module<B>>(
    active: &M,
    target: M,
    tau: f32,
) -> Result<M, PolyakError> {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        seen: HashSet::new(),
        tau,
        error: None,
        _marker: PhantomData,
    };
    let updated = target.map(&mut mapper);
    // A `target` parameter with no counterpart in `active` was recorded during
    // the walk; surface the first such miss.
    if let Some(e) = mapper.error {
        return Err(e);
    }
    // Every active parameter must have been consumed by some target field. Any
    // leftover means `target` is a strict subset of `active`; applying only the
    // overlap would be a silent partial update, so report the smallest leftover
    // id (`.min()` makes the report deterministic).
    if let Some(id) = mapper
        .active
        .keys()
        .filter(|id| !mapper.seen.contains(id))
        .min()
        .copied()
    {
        return Err(PolyakError::MissingTarget(id));
    }
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use burn::backend::Flex;
    use burn::tensor::backend::BackendTypes;

    /// CPU backend for these tests. No RNG is involved — every weight below is
    /// hand-set — so no backend seeding or `rayon` pinning is required.
    type TestBackend = Flex;
    type TestDevice = <TestBackend as BackendTypes>::Device;

    /// Tolerance for the exact-arithmetic assertions. Every expected value in
    /// this module is representable in binary floating point (the taus are
    /// dyadic), so this only absorbs backend reduction noise.
    const EPS: f32 = 1e-6;

    // -----------------------------------------------------------------------
    // compute_target_q_values
    // -----------------------------------------------------------------------

    /// Builds a rank-1 `f32` tensor on the default test device.
    fn floats(values: &[f32]) -> Tensor<TestBackend, 1> {
        Tensor::from_floats(values, &TestDevice::default())
    }

    /// Reads a rank-1 tensor back to host floats.
    fn host(tensor: &Tensor<TestBackend, 1>) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("target tensor is f32 by construction")
    }

    /// The pre-hardening formula, kept verbatim as the reference oracle for
    /// the "finite inputs are unchanged" test.
    fn scaled_reference(
        rewards: Tensor<TestBackend, 1>,
        next_q_max: Tensor<TestBackend, 1>,
        terminated: Tensor<TestBackend, 1>,
        gamma: f32,
    ) -> Tensor<TestBackend, 1> {
        rewards + gamma * next_q_max * (1.0 - terminated)
    }

    #[test]
    fn test_compute_target_q_values_masks_non_finite_bootstrap_on_terminal() {
        // Sample 0 is terminal with a NaN continuation value, sample 1 is
        // terminal with +inf, sample 2 with -inf. Under the old
        // `* (1.0 - terminated)` scaling all three produced NaN, because
        // `NaN * 0.0` and `inf * 0.0` are both NaN.
        let rewards = floats(&[1.5, -2.0, 0.25]);
        let next_q_max = floats(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let terminated = floats(&[1.0, 1.0, 1.0]);

        let got = host(&compute_target_q_values(
            rewards, next_q_max, terminated, 0.99,
        ));
        let want = [1.5_f32, -2.0, 0.25];

        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                g.is_finite(),
                "a terminal transition must never inherit a non-finite bootstrap; \
                 sample {i} got {g}"
            );
            assert_abs_diff_eq!(g, w, epsilon = EPS);
        }
    }

    #[test]
    fn test_compute_target_q_values_bootstraps_when_not_terminated() {
        const REWARDS: [f32; 3] = [1.0, -0.5, 0.0];
        // 1.0 + 0.9*2.0 = 2.8 ; -0.5 + 0.9*4.0 = 3.1 ; 0.0 + 0.9*(-3.0) = -2.7
        const WANT: [f32; 3] = [2.8, 3.1, -2.7];

        // Masking must not disturb the non-terminal path: every sample here
        // keeps the full `reward + gamma * next_q_max` target.
        let gamma = 0.9_f32;

        let got = host(&compute_target_q_values(
            floats(&REWARDS),
            floats(&[2.0, 4.0, -3.0]),
            floats(&[0.0, 0.0, 0.0]),
            gamma,
        ));

        for (i, (&g, &w)) in got.iter().zip(WANT.iter()).enumerate() {
            assert_abs_diff_eq!(g, w, epsilon = EPS);
            assert!(
                (g - REWARDS[i]).abs() > EPS,
                "sample {i} must actually bootstrap, not collapse to the reward; got {g}"
            );
        }
    }

    #[test]
    fn test_compute_target_q_values_matches_scaled_formula_on_finite_inputs() {
        // Mixed terminal / non-terminal batch, all finite: the masked form must
        // reproduce the old `* (1.0 - terminated)` result exactly.
        let gamma = 0.97_f32;
        let rewards = [0.0_f32, 1.25, -3.5, 2.0, -0.75, 10.0];
        let next_q_max = [5.0_f32, -5.0, 0.0, 123.5, -0.125, 1.0];
        let terminated = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0];

        let want = host(&scaled_reference(
            floats(&rewards),
            floats(&next_q_max),
            floats(&terminated),
            gamma,
        ));
        let got = host(&compute_target_q_values(
            floats(&rewards),
            floats(&next_q_max),
            floats(&terminated),
            gamma,
        ));

        assert_eq!(
            got.len(),
            want.len(),
            "masking must preserve the batch length"
        );
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_abs_diff_eq!(g, w, epsilon = EPS);
            assert!(
                g.is_finite(),
                "finite inputs must give a finite target; sample {i} got {g}"
            );
        }
    }

    // Hand-set constant weights. `active` and `target` differ in *every*
    // element, which is what lets the tau = 0 and tau = 0.25 tests distinguish
    // "did nothing" from "blended correctly" (see issue #336).
    const ACTIVE_WEIGHT: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    const ACTIVE_BIAS: [f32; 2] = [0.5, -0.5];
    const TARGET_WEIGHT: [[f32; 3]; 2] = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]];
    const TARGET_BIAS: [f32; 2] = [1.0, 3.0];

    /// Flattened `active` parameters in visit order: weight then bias.
    const ACTIVE_FLAT: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -0.5];
    /// Flattened `target` parameters in visit order: weight then bias.
    const TARGET_FLAT: [f32; 8] = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 1.0, 3.0];

    /// Minimal two-parameter [`Module`] with hand-set weights.
    ///
    /// Deliberately not a `Linear` — it carries no initialisation logic, so the
    /// only thing a test can observe is what [`polyak_update`] wrote.
    #[derive(Module, Debug)]
    struct TestNet<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> TestNet<B> {
        /// Builds a net with fresh [`ParamId`]s from the given constants.
        fn new(device: &B::Device, weight: [[f32; 3]; 2], bias: [f32; 2]) -> Self {
            Self {
                weight: Param::from_data(weight, device),
                bias: Param::from_data(bias, device),
            }
        }

        /// Builds a net that **shares `self`'s [`ParamId`]s** but holds
        /// different values.
        ///
        /// [`polyak_update`]'s mapper looks parameters up by `ParamId`, so a
        /// target network built independently of the active one would panic
        /// rather than blend. Real agents get this for free by cloning the
        /// policy net; these tests reproduce it explicitly.
        fn relabelled(&self, weight: [[f32; 3]; 2], bias: [f32; 2]) -> Self {
            let device = self.weight.val().device();
            Self {
                weight: Param::initialized(self.weight.id, Tensor::from_data(weight, &device)),
                bias: Param::initialized(self.bias.id, Tensor::from_data(bias, &device)),
            }
        }

        /// Flattens every parameter to host floats in visit order.
        fn flat(&self) -> Vec<f32> {
            let mut out = self
                .weight
                .val()
                .to_data()
                .to_vec::<f32>()
                .expect("weight is f32 by construction");
            out.extend(
                self.bias
                    .val()
                    .to_data()
                    .to_vec::<f32>()
                    .expect("bias is f32 by construction"),
            );
            out
        }

        /// Per-parameter shapes in visit order.
        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![
                self.weight.val().shape().dims::<2>().to_vec(),
                self.bias.val().shape().dims::<1>().to_vec(),
            ]
        }
    }

    /// Returns the `(active, target)` fixture pair, sharing `ParamId`s and
    /// differing in every element.
    fn fixture() -> (TestNet<TestBackend>, TestNet<TestBackend>) {
        let device = TestDevice::default();
        let active = TestNet::<TestBackend>::new(&device, ACTIVE_WEIGHT, ACTIVE_BIAS);
        let target = active.relabelled(TARGET_WEIGHT, TARGET_BIAS);
        (active, target)
    }

    /// Asserts the precondition every non-vacuous test below depends on: the
    /// two networks genuinely differ *before* the update.
    ///
    /// Without this, "return `target` untouched" and "return `active`
    /// outright" both pass a tau = 0 / tau = 1 / tau = 0.25 suite built on
    /// equal inputs. That vacuity is how issue #182 survived.
    fn assert_nets_differ(active: &TestNet<TestBackend>, target: &TestNet<TestBackend>) {
        let (a, t) = (active.flat(), target.flat());
        assert_eq!(
            a.len(),
            t.len(),
            "fixture nets must have equal param counts"
        );
        for (i, (av, tv)) in a.iter().zip(t.iter()).enumerate() {
            // Smallest gap in the fixture is bias[0]: |0.5 - 1.0| = 0.5.
            const MIN_SEPARATION: f32 = 0.4;
            assert!(
                (av - tv).abs() > MIN_SEPARATION,
                "precondition: active[{i}] ({av}) must differ from target[{i}] ({tv}), \
                 otherwise the test cannot distinguish a blend from a no-op"
            );
        }
    }

    #[test]
    fn test_polyak_update_returns_target_unchanged_when_tau_is_zero() {
        let (active, target) = fixture();
        assert_nets_differ(&active, &target);

        let updated =
            polyak_update::<TestBackend, _>(&active, target, 0.0).expect("ids match by fixture");

        for (i, (got, want)) in updated.flat().iter().zip(TARGET_FLAT.iter()).enumerate() {
            assert_abs_diff_eq!(got, want, epsilon = EPS);
            assert!(
                (got - ACTIVE_FLAT[i]).abs() > EPS,
                "tau = 0 must leave param {i} at the target value {want}, not pull it \
                 toward active ({}); got {got}",
                ACTIVE_FLAT[i]
            );
        }
    }

    #[test]
    fn test_polyak_update_copies_active_exactly_when_tau_is_one() {
        let (active, target) = fixture();
        assert_nets_differ(&active, &target);

        let updated =
            polyak_update::<TestBackend, _>(&active, target, 1.0).expect("ids match by fixture");

        for (i, (got, want)) in updated.flat().iter().zip(ACTIVE_FLAT.iter()).enumerate() {
            assert_abs_diff_eq!(got, want, epsilon = EPS);
            assert!(
                (got - TARGET_FLAT[i]).abs() > EPS,
                "tau = 1 is documented as a hard copy: param {i} must become the active \
                 value {want}, not stay at the target value {}; got {got}",
                TARGET_FLAT[i]
            );
        }
    }

    #[test]
    fn test_polyak_update_blends_exactly_when_tau_is_fractional() {
        // tau = 0.25 => 0.75 * target + 0.25 * active, hand-computed per param.
        // e.g. weight[0]: 0.75 * (-1.0) + 0.25 * 1.0 = -0.5
        //      bias[0]:   0.75 * ( 1.0) + 0.25 * 0.5 = 0.875
        const EXPECTED: [f32; 8] = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, 0.875, 2.125];

        let (active, target) = fixture();
        assert_nets_differ(&active, &target);

        let updated =
            polyak_update::<TestBackend, _>(&active, target, 0.25).expect("ids match by fixture");

        for (i, (got, want)) in updated.flat().iter().zip(EXPECTED.iter()).enumerate() {
            assert_abs_diff_eq!(got, want, epsilon = EPS);
            assert!(
                (got - TARGET_FLAT[i]).abs() > EPS && (got - ACTIVE_FLAT[i]).abs() > EPS,
                "param {i} must be a strict convex combination, distinct from both \
                 endpoints (target {}, active {}); got {got}",
                TARGET_FLAT[i],
                ACTIVE_FLAT[i]
            );
        }
    }

    #[test]
    fn test_polyak_update_preserves_shapes_and_param_count() {
        let (active, target) = fixture();
        let before_shapes = target.shapes();
        let before_len = target.flat().len();

        let updated =
            polyak_update::<TestBackend, _>(&active, target, 0.3).expect("ids match by fixture");

        assert_eq!(
            updated.shapes(),
            before_shapes,
            "polyak_update must preserve every parameter's shape"
        );
        assert_eq!(
            updated.shapes(),
            active.shapes(),
            "the updated target must keep the same shapes as the active net"
        );
        assert_eq!(
            updated.flat().len(),
            before_len,
            "polyak_update must neither add nor drop parameters"
        );
    }

    #[test]
    fn test_polyak_update_converges_monotonically_toward_active() {
        const TAU: f32 = 0.3;
        const STEPS: usize = 12;

        let (active, mut target) = fixture();
        assert_nets_differ(&active, &target);

        let mut prev = target.flat();
        for step in 0..STEPS {
            target = polyak_update::<TestBackend, _>(&active, target, TAU)
                .expect("ids match by fixture");
            let now = target.flat();

            for (i, (&before, &after)) in prev.iter().zip(now.iter()).enumerate() {
                let goal = ACTIVE_FLAT[i];
                let gap_before = (goal - before).abs();
                let gap_after = (goal - after).abs();

                assert!(
                    gap_after < gap_before,
                    "step {step}: param {i} must move strictly toward active ({goal}); \
                     gap went {gap_before} -> {gap_after}"
                );
                // No overshoot: the parameter stays on its original side of the
                // active value, i.e. within the closed interval [before, goal].
                assert!(
                    (after - before).signum() == (goal - before).signum()
                        && gap_after <= gap_before,
                    "step {step}: param {i} overshot active ({goal}): {before} -> {after}"
                );
            }
            prev = now;
        }

        // After 12 steps of tau = 0.3 the remaining gap is 0.7^12 ≈ 0.0138 of
        // the original, so every parameter is close to — but not yet equal to —
        // the active value.
        for (i, (&got, &goal)) in prev.iter().zip(ACTIVE_FLAT.iter()).enumerate() {
            let initial_gap = (goal - TARGET_FLAT[i]).abs();
            assert!(
                (goal - got).abs() < 0.02 * initial_gap,
                "param {i} should have converged close to active ({goal}) after \
                 {STEPS} steps; got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ParamId topology: tied weights, foreign target params, strict subsets
    // (issue #341)
    // -----------------------------------------------------------------------

    /// Two same-typed [`Param`] fields, so a fixture can make them share one
    /// [`ParamId`] (tied weights) or carry two distinct ids.
    #[derive(Module, Debug)]
    struct TiedNet<B: Backend> {
        a: Param<Tensor<B, 1>>,
        b: Param<Tensor<B, 1>>,
    }

    /// Three same-typed [`Param`] fields, so a fixture can leave **two**
    /// distinct active ids unconsumed by the target — enough to distinguish
    /// "reports the true `.min()` leftover" from "reports whatever id `HashMap`
    /// iteration happened to surface first".
    #[derive(Module, Debug)]
    struct TripleNet<B: Backend> {
        a: Param<Tensor<B, 1>>,
        b: Param<Tensor<B, 1>>,
        c: Param<Tensor<B, 1>>,
    }

    /// Reads a rank-1 param to host floats.
    fn param_vec<B: Backend>(param: &Param<Tensor<B, 1>>) -> Vec<f32> {
        param
            .val()
            .to_data()
            .to_vec::<f32>()
            .expect("param is f32 by construction")
    }

    #[test]
    fn test_polyak_update_blends_tied_weights_without_panic() {
        // Regression for the `.remove()` double-consume: `a` and `b` are clones
        // of one `Param`, so they share a single `ParamId`. The active network
        // therefore contributes exactly one collector entry, and *both* target
        // fields must be blendable from it. Under the old `.remove()` the second
        // field's lookup found the id already consumed and panicked.

        // Target values, and the tau = 0.25 blend 0.75 * target + 0.25 * active.
        const TARGET: [f32; 3] = [-1.0, -2.0, -3.0];
        const EXPECTED: [f32; 3] = [-0.5, -1.0, -1.5];

        let device = TestDevice::default();

        // Active: both fields are clones of one param -> one shared id, one value.
        let active_shared = Param::from_data([1.0_f32, 2.0, 3.0], &device);
        let shared_id = active_shared.id;
        let active = TiedNet::<TestBackend> {
            a: active_shared.clone(),
            b: active_shared.clone(),
        };

        // Target: both fields share that same id but hold a different value.
        let target_shared = Param::initialized(
            shared_id,
            Tensor::from_data([-1.0_f32, -2.0, -3.0], &device),
        );
        let target = TiedNet::<TestBackend> {
            a: target_shared.clone(),
            b: target_shared.clone(),
        };

        let updated = polyak_update::<TestBackend, _>(&active, target, 0.25)
            .expect("tied weights share one active entry; the update must succeed");

        for (field, values) in [("a", param_vec(&updated.a)), ("b", param_vec(&updated.b))] {
            for (i, &got) in values.iter().enumerate() {
                assert_abs_diff_eq!(got, EXPECTED[i], epsilon = EPS);
                assert!(
                    (got - TARGET[i]).abs() > EPS,
                    "tied field {field}[{i}] must be blended from the shared active entry, \
                     not left at its target value; got {got}"
                );
            }
        }
    }

    #[test]
    fn test_polyak_update_errors_on_foreign_target_param() {
        // Both fields of `target` carry a fresh id (`from_data` mints new ones),
        // so neither is present in `active`. The update must fail with
        // `MissingActive` naming the offending id rather than silently doing
        // nothing.
        let device = TestDevice::default();

        let active_shared = Param::from_data([1.0_f32], &device);
        let active = TiedNet::<TestBackend> {
            a: active_shared.clone(),
            b: active_shared.clone(),
        };

        // Independently minted ids -> no counterpart in active. Both fields
        // share `foreign.id`, so the first-recorded miss is that id.
        let foreign = Param::from_data([2.0_f32], &device);
        let foreign_id = foreign.id;
        let target = TiedNet::<TestBackend> {
            a: foreign.clone(),
            b: foreign.clone(),
        };

        let result = polyak_update::<TestBackend, _>(&active, target, 0.5);

        assert_eq!(result.err(), Some(PolyakError::MissingActive(foreign_id)));
    }

    #[test]
    fn test_polyak_update_errors_on_strict_subset_target() {
        // Active carries two distinct ids (A, B). Target's two fields both share
        // id A, so its param set {A} is a strict subset of active's {A, B}. The
        // B entry is never consumed; the old code silently dropped it, the new
        // code must report `MissingTarget(B)`.
        let device = TestDevice::default();

        let param_a = Param::from_data([1.0_f32], &device);
        let param_b = Param::from_data([2.0_f32], &device);
        let missing_id = param_b.id;
        let active = TiedNet::<TestBackend> {
            a: param_a.clone(),
            b: param_b.clone(),
        };

        // Target: both fields share A -> B has no counterpart in target.
        let target_a = Param::initialized(param_a.id, Tensor::from_data([-1.0_f32], &device));
        let target = TiedNet::<TestBackend> {
            a: target_a.clone(),
            b: target_a.clone(),
        };

        let result = polyak_update::<TestBackend, _>(&active, target, 0.5);

        assert_eq!(result.err(), Some(PolyakError::MissingTarget(missing_id)));
    }

    #[test]
    fn test_polyak_update_reports_min_missing_target_id() {
        // Active carries three distinct ids (A, B, C). The target's three fields
        // all share id A, so active's set {A, B, C} strictly contains target's
        // {A}: *two* entries — B and C — are never consumed. `MissingTarget` is
        // documented to name the smallest leftover id (`.min()`), so a single
        // leftover cannot distinguish "reports the true min" from "reports the
        // first id `HashMap` iteration yielded". Two leftovers can.
        let device = TestDevice::default();

        let param_a = Param::from_data([1.0_f32], &device);
        let param_b = Param::from_data([2.0_f32], &device);
        let param_c = Param::from_data([3.0_f32], &device);

        // Do not assume the minting order of ids; compute the expected minimum
        // leftover explicitly from the two ids the target fails to consume.
        let expected_min = param_b.id.min(param_c.id);
        // Both leftovers are distinct, so the assertion below is non-vacuous.
        assert_ne!(
            param_b.id, param_c.id,
            "the two leftover ids must differ for the min to be meaningful"
        );

        let active = TripleNet::<TestBackend> {
            a: param_a.clone(),
            b: param_b.clone(),
            c: param_c.clone(),
        };

        // Target: all three fields share A -> B and C have no counterpart.
        let target_a = Param::initialized(param_a.id, Tensor::from_data([-1.0_f32], &device));
        let target = TripleNet::<TestBackend> {
            a: target_a.clone(),
            b: target_a.clone(),
            c: target_a.clone(),
        };

        let result = polyak_update::<TestBackend, _>(&active, target, 0.5);

        assert_eq!(
            result.err(),
            Some(PolyakError::MissingTarget(expected_min)),
            "MissingTarget must report the smallest leftover id ({expected_min:?}), \
             not whichever of {:?} / {:?} HashMap order surfaced first",
            param_b.id,
            param_c.id
        );
    }

    // -----------------------------------------------------------------------
    // On-device transport (issue #322)
    // -----------------------------------------------------------------------

    #[test]
    fn test_polyak_update_feeds_reconstructed_target_back_in() {
        // Regression for the on-device transport swap: `active` is now carried
        // between the collector and the mapper as a rank-erased
        // `TensorPrimitive` handle, and each target parameter is rebuilt with
        // `Tensor::from_primitive`. This test pins two things at the behaviour
        // level.
        //
        // First, a single blend over a genuinely multi-rank net (rank-2 weight
        // + rank-1 bias) must still produce the exact 0.75·target + 0.25·active
        // combination — the primitive round-trip is numerically a no-op.
        //
        // Second, and the point of this test, the returned target is fed
        // *straight back* into a second `polyak_update`. A tensor rebuilt from
        // `from_primitive` must be a fully valid tensor you can read, blend, and
        // reconstruct again — not a degenerate handle. The second step must
        // continue converging toward active (every parameter's gap shrinks),
        // which it cannot do if the reconstructed tensor were somehow inert.
        //
        // The fixtures use `Param::from_data` (already materialised), so no
        // lazy first-read re-materialisation can masquerade as divergence.
        const TAU: f32 = 0.25;
        // First step: 0.75·target + 0.25·active, hand-computed per param, in
        // visit order (weight row-major, then bias). Matches the tau = 0.25
        // fractional-blend oracle above.
        const AFTER_ONE: [f32; 8] = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, 0.875, 2.125];

        let (active, target) = fixture();
        assert_nets_differ(&active, &target);

        let once =
            polyak_update::<TestBackend, _>(&active, target, TAU).expect("ids match by fixture");
        let after_one = once.flat();
        for (i, (&got, &want)) in after_one.iter().zip(AFTER_ONE.iter()).enumerate() {
            assert_abs_diff_eq!(got, want, epsilon = EPS);
            assert!(
                (got - ACTIVE_FLAT[i]).abs() > EPS,
                "after one step param {i} must be a strict blend, not yet equal to \
                 active ({}); got {got}",
                ACTIVE_FLAT[i]
            );
        }

        // Feed the reconstructed-from-primitive target back in unchanged.
        let twice = polyak_update::<TestBackend, _>(&active, once, TAU)
            .expect("the reconstructed target must remain a valid, updatable module");
        let after_two = twice.flat();

        for (i, (&one, &two)) in after_one.iter().zip(after_two.iter()).enumerate() {
            let goal = ACTIVE_FLAT[i];
            let gap_before = (goal - one).abs();
            let gap_after = (goal - two).abs();
            assert!(
                gap_after < gap_before,
                "second consecutive step must keep converging param {i} toward active \
                 ({goal}); gap went {gap_before} -> {gap_after}"
            );
        }
    }
}
