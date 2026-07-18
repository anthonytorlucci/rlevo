//! Shared utility functions for reinforcement learning.
//!
//! Provides stateless helper functions used across multiple RL algorithms,
//! such as Bellman target computation and Polyak averaging.

use std::collections::HashMap;
use std::marker::PhantomData;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

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

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}

struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        let active = self
            .active
            .remove(&id)
            .expect("param not collected from active network");
        let tau = self.tau;
        param.map(move |target_tensor| {
            let device = target_tensor.device();
            let active_tensor = Tensor::<B, D>::from_data(active, &device);
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
/// # Panics
///
/// Panics if `target` holds a parameter whose [`ParamId`] is absent from
/// `active` — that is, if the two modules were built independently rather than
/// `target` being derived from `active`.
///
/// A fresh `init` mints new `ParamId`s, so two separately initialised networks
/// never match even when their architecture and weights are identical. Build
/// the target network by cloning the active one (`let target = active.clone();`)
/// and it holds by construction; every in-tree agent does this.
pub fn polyak_update<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        tau,
        _marker: PhantomData,
    };
    target.map(&mut mapper)
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
    fn host(tensor: Tensor<TestBackend, 1>) -> Vec<f32> {
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

        let got = host(compute_target_q_values(
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
        // Masking must not disturb the non-terminal path: every sample here
        // keeps the full `reward + gamma * next_q_max` target.
        let gamma = 0.9_f32;
        const REWARDS: [f32; 3] = [1.0, -0.5, 0.0];
        // 1.0 + 0.9*2.0 = 2.8 ; -0.5 + 0.9*4.0 = 3.1 ; 0.0 + 0.9*(-3.0) = -2.7
        const WANT: [f32; 3] = [2.8, 3.1, -2.7];

        let got = host(compute_target_q_values(
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

        let want = host(scaled_reference(
            floats(&rewards),
            floats(&next_q_max),
            floats(&terminated),
            gamma,
        ));
        let got = host(compute_target_q_values(
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

        let updated = polyak_update::<TestBackend, _>(&active, target, 0.0);

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

        let updated = polyak_update::<TestBackend, _>(&active, target, 1.0);

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
        let (active, target) = fixture();
        assert_nets_differ(&active, &target);

        // tau = 0.25 => 0.75 * target + 0.25 * active, hand-computed per param.
        // e.g. weight[0]: 0.75 * (-1.0) + 0.25 * 1.0 = -0.5
        //      bias[0]:   0.75 * ( 1.0) + 0.25 * 0.5 = 0.875
        const EXPECTED: [f32; 8] = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, 0.875, 2.125];

        let updated = polyak_update::<TestBackend, _>(&active, target, 0.25);

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

        let updated = polyak_update::<TestBackend, _>(&active, target, 0.3);

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
        let (active, mut target) = fixture();
        assert_nets_differ(&active, &target);

        const TAU: f32 = 0.3;
        const STEPS: usize = 12;

        let mut prev = target.flat();
        for step in 0..STEPS {
            target = polyak_update::<TestBackend, _>(&active, target, TAU);
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
}
