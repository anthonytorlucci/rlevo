//! Bridge between a Burn [`Module`] and a flat parameter vector.
//!
//! Weight-only neuroevolution evolves a `Tensor<B, 2>` population of shape
//! `(pop_size, num_params)` and must, per population member, splat one flat
//! parameter row back into a concrete network to score it. The
//! [`ParamReshaper`] trait captures that bidirectional bridge:
//!
//! - [`flatten`](ParamReshaper::flatten) walks a module's float leaves in a
//!   deterministic order and concatenates them into a 1-D tensor.
//! - [`unflatten`](ParamReshaper::unflatten) clones a template module and
//!   replaces each float leaf with the matching slice of a flat tensor, in the
//!   *same* order.
//!
//! [`ModuleReshaper`] is the concrete implementation. It relies on the fact
//! that Burn's `#[derive(Module)]` generates `visit`/`map` traversals that
//! visit fields in declaration order, recursively — so `flatten` (a
//! [`ModuleVisitor`]) and `unflatten` (a [`ModuleMapper`]) agree leaf-for-leaf.
//!
//! # Non-trainable module state
//!
//! Burn's `visit`/`map` traversal touches every float leaf reachable through
//! the `Module` tree, **including** non-`Param` running statistics such as a
//! [`burn::nn::BatchNorm`] layer's running mean/variance (they are wrapped in a
//! `RunningState`, which is itself a `Module` that forwards to `visit_float` /
//! `map_float`). The proof-of-concept in this module's test submodule verifies this
//! empirically. The practical consequence: if an evolved network contains
//! `BatchNorm`, its running statistics are flattened, perturbed by evolution,
//! and re-splatted like any weight. For fixed-topology MLP policies (the v1
//! target) this is moot — there are no running buffers. Callers that evolve
//! batch-normalized networks should reset running statistics after
//! [`unflatten`](ParamReshaper::unflatten).
//!
//! # Gradient isolation
//!
//! This module is generic over `B: Backend`, **not** `AutodiffBackend`.
//! Tensors produced by [`unflatten`](ParamReshaper::unflatten) do not require
//! gradients. Callers holding an autodiff module call `.valid()` before
//! constructing a [`ModuleReshaper`], so the constraint is enforced at the
//! type level rather than by convention.

use std::marker::PhantomData;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Tensor, backend::Backend};

/// Bridges a Burn [`Module`] and a flat `Tensor<B, 1>` parameter vector.
///
/// Lives entirely in `rlevo-evolution` and depends only on `burn` — no
/// `rlevo-core` coupling.
///
/// # Invariants
///
/// - [`flatten`](Self::flatten) and [`unflatten`](Self::unflatten) must visit
///   float leaves in the *same* deterministic order, so that
///   `unflatten(flatten(m))` reconstructs `m` leaf-for-leaf.
/// - [`num_params`](Self::num_params) equals the total element count produced
///   by [`flatten`](Self::flatten).
///
/// Implementors are `Send + Sync` so a single reshaper can be shared across
/// parallel fitness evaluations.
pub trait ParamReshaper<B: Backend>: Send + Sync {
    /// The Burn module type this reshaper flattens and reconstructs.
    type Module: Module<B>;

    /// Total number of trainable float parameters (the flat-vector length).
    fn num_params(&self) -> usize;

    /// Flatten all float `Param` leaves of `module` into a 1-D tensor.
    ///
    /// The returned tensor is moved onto `device` and has length
    /// [`num_params`](Self::num_params). Leaf visitation order is
    /// deterministic and matches [`unflatten`](Self::unflatten).
    ///
    /// # Panics
    ///
    /// Panics if `module` has no float leaves (the underlying tensor
    /// concatenation requires at least one part).
    fn flatten(&self, module: &Self::Module, device: &B::Device) -> Tensor<B, 1>;

    /// Clone the template module and replace its float leaves with slices of
    /// `flat`, in the same order as [`flatten`](Self::flatten).
    ///
    /// # Panics
    ///
    /// Panics if `flat.dims()[0] != self.num_params()`.
    fn unflatten(&self, flat: Tensor<B, 1>) -> Self::Module;
}

/// A [`ParamReshaper`] backed by a cloned template module.
///
/// Construction clones the supplied module once and counts its float leaves.
/// Each [`unflatten`](ParamReshaper::unflatten) call clones that template and
/// maps the flat buffer into the clone's leaves; each
/// [`flatten`](ParamReshaper::flatten) call visits a module's leaves and
/// concatenates them.
///
/// # Example
///
/// ```ignore
/// use rlevo_evolution::param_reshaper::{ModuleReshaper, ParamReshaper};
///
/// let template = MyMlp::<B>::new(&device);
/// let reshaper = ModuleReshaper::new(template.clone());
/// let flat = reshaper.flatten(&template, &device);
/// let restored = reshaper.unflatten(flat);
/// ```
pub struct ModuleReshaper<B: Backend, M: Module<B>> {
    template: M,
    num_params: usize,
    // `fn() -> B` keeps the marker `Send + Sync` for any `B` and encodes that
    // `B` is produced, never consumed — mirroring the crate's other markers.
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend, M: Module<B>> std::fmt::Debug for ModuleReshaper<B, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleReshaper")
            .field("num_params", &self.num_params)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, M: Module<B>> ModuleReshaper<B, M> {
    /// Build a reshaper from a template module.
    ///
    /// The template is cloned and retained; its float-leaf count is computed
    /// once and cached as [`num_params`](ParamReshaper::num_params).
    #[must_use]
    pub fn new(template: M) -> Self {
        let mut counter = CountVisitor { count: 0 };
        template.visit(&mut counter);
        Self {
            template,
            num_params: counter.count,
            _backend: PhantomData,
        }
    }

    /// Borrow the retained template module.
    #[must_use]
    pub fn template(&self) -> &M {
        &self.template
    }

    /// Number of float parameters (flat-vector length).
    ///
    /// Inherent mirror of [`ParamReshaper::num_params`] so callers can read the
    /// width without the `M: Sync` bound the trait requires.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.num_params
    }
}

impl<B, M> ParamReshaper<B> for ModuleReshaper<B, M>
where
    B: Backend,
    // `Sync` is required by the `ParamReshaper` supertrait so the reshaper can
    // be shared across parallel evaluations; Burn modules built from
    // `Param<Tensor>` leaves satisfy it.
    M: Module<B> + Sync,
{
    type Module = M;

    fn num_params(&self) -> usize {
        self.num_params
    }

    fn flatten(&self, module: &M, device: &B::Device) -> Tensor<B, 1> {
        let mut visitor: FlattenVisitor<B> = FlattenVisitor { parts: Vec::new() };
        module.visit(&mut visitor);
        assert!(
            !visitor.parts.is_empty(),
            "module has no float parameters to flatten"
        );
        Tensor::cat(visitor.parts, 0).to_device(device)
    }

    fn unflatten(&self, flat: Tensor<B, 1>) -> M {
        let len = flat.dims()[0];
        assert_eq!(
            len, self.num_params,
            "flat length {len} does not match num_params {}",
            self.num_params
        );
        let mut mapper: SlicingMapper<B> = SlicingMapper { flat, cursor: 0 };
        self.template.clone().map(&mut mapper)
    }
}

/// Counts the total number of float-leaf elements in a module.
struct CountVisitor {
    count: usize,
}

impl<B: Backend> ModuleVisitor<B> for CountVisitor {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.count += param.dims().iter().product::<usize>();
    }
}

/// Collects each float leaf, reshaped to 1-D, in visitation order.
struct FlattenVisitor<B: Backend> {
    parts: Vec<Tensor<B, 1>>,
}

impl<B: Backend> ModuleVisitor<B> for FlattenVisitor<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let value: Tensor<B, D> = param.val();
        let n: usize = value.dims().iter().product();
        self.parts.push(value.reshape([n]));
    }
}

/// Replaces each float leaf with the next `n` elements of `flat`, reshaped to
/// the leaf's original shape, advancing a cursor in visitation order.
struct SlicingMapper<B: Backend> {
    flat: Tensor<B, 1>,
    cursor: usize,
}

impl<B: Backend> ModuleMapper<B> for SlicingMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let dims: [usize; D] = param.dims();
        let n: usize = dims.iter().product();
        let start = self.cursor;
        self.cursor += n;
        let flat = self.flat.clone();
        // `Param::map` preserves the parameter id and any load/save mapper while
        // swapping the inner tensor; the new tensor does not require grad
        // (gradient isolation — see module docs).
        param.map(move |_old| {
            #[allow(clippy::single_range_in_vec_init)]
            let slice = flat.slice([start..start + n]);
            slice.reshape(dims)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
    };
    use burn::tensor::TensorData;

    type TestBackend = Flex;

    /// 2-layer MLP: `Linear(3 -> 4) -> ReLU -> Linear(4 -> 2)`.
    ///
    /// Float-leaf count: `3*4 + 4` (l1 weight + bias) `+ 4*2 + 2`
    /// (l2 weight + bias) = `26`.
    #[derive(Module, Debug)]
    struct Mlp<B: Backend> {
        l1: Linear<B>,
        act: Relu,
        l2: Linear<B>,
    }

    impl<B: Backend> Mlp<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                l1: LinearConfig::new(3, 4).init(device),
                act: Relu::new(),
                l2: LinearConfig::new(4, 2).init(device),
            }
        }
    }

    fn approx_eq(a: &Tensor<TestBackend, 1>, b: &Tensor<TestBackend, 1>) {
        let av = a.clone().into_data().into_vec::<f32>().unwrap();
        let bv = b.clone().into_data().into_vec::<f32>().unwrap();
        assert_eq!(av.len(), bv.len(), "length mismatch");
        for (x, y) in av.iter().zip(bv.iter()) {
            approx::assert_relative_eq!(x, y, epsilon = 1e-6);
        }
    }

    #[test]
    fn num_params_matches_expected() {
        let device = Default::default();
        let mlp = Mlp::<TestBackend>::new(&device);
        let reshaper = ModuleReshaper::new(mlp);
        assert_eq!(reshaper.num_params(), 26);
    }

    /// AC #2: `unflatten(flatten(m)) ≈ m`. We compare via re-flatten, which is
    /// element-wise injective over the deterministic leaf order, so equality of
    /// the flat vectors is equivalent to equality of the modules' float leaves.
    #[test]
    fn round_trip_mlp() {
        let device = Default::default();
        let mlp = Mlp::<TestBackend>::new(&device);
        let reshaper = ModuleReshaper::new(mlp.clone());

        let flat = reshaper.flatten(&mlp, &device);
        assert_eq!(flat.dims(), [26]);

        let restored = reshaper.unflatten(flat.clone());
        let flat2 = reshaper.flatten(&restored, &device);
        approx_eq(&flat, &flat2);
    }

    /// Property test catching leaf-ordering bugs: a known flat buffer survives
    /// `unflatten -> flatten` unchanged.
    #[test]
    fn round_trip_arbitrary_flat() {
        let device = Default::default();
        let mlp = Mlp::<TestBackend>::new(&device);
        let reshaper = ModuleReshaper::new(mlp);

        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = (0..26).map(|i| i as f32 * 0.1 - 1.3).collect();
        let flat =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(values, [26]), &device);

        let module = reshaper.unflatten(flat.clone());
        let flat2 = reshaper.flatten(&module, &device);
        approx_eq(&flat, &flat2);
    }

    /// Closes 3d1-R2 OPEN item: does Burn's traversal touch non-trainable
    /// `BatchNorm` running statistics? Empirically yes — `RunningState` is a
    /// `Module` and forwards to `visit_float` / `map_float`. A `BatchNorm`
    /// over `d` features therefore exposes `4*d` float leaves: `gamma`,
    /// `beta`, `running_mean`, `running_var`.
    #[test]
    fn batchnorm_running_stats_are_traversed() {
        let device = Default::default();
        let d = 5;
        let bn: BatchNorm<TestBackend> = BatchNormConfig::new(d).init(&device);
        let reshaper = ModuleReshaper::new(bn.clone());
        // 4 * d if running stats are traversed; 2 * d if only gamma/beta are.
        assert_eq!(
            reshaper.num_params(),
            4 * d,
            "expected BatchNorm running stats to be traversed as float leaves"
        );
        // And the round-trip must still hold over all traversed leaves.
        let flat = reshaper.flatten(&bn, &device);
        let restored = reshaper.unflatten(flat.clone());
        approx_eq(&flat, &reshaper.flatten(&restored, &device));
    }

    /// A non-trivial module with a conv layer also round-trips, confirming the
    /// reshaper is not MLP-specific.
    #[test]
    fn round_trip_conv() {
        let device = Default::default();
        let conv: Conv2d<TestBackend> = Conv2dConfig::new([2, 3], [3, 3]).init(&device);
        let reshaper = ModuleReshaper::new(conv.clone());
        let flat = reshaper.flatten(&conv, &device);
        let restored = reshaper.unflatten(flat.clone());
        approx_eq(&flat, &reshaper.flatten(&restored, &device));
    }

    // --- Bounded-NAS enum-derive probe (issue #42, Step 1) -------------------
    //
    // Question: does Burn 0.21 `#[derive(Module)]` work on a Rust *enum* whose
    // arms hold heterogeneous concrete `Module` variants? The bounded-NAS
    // design (closure-erased `VariantEvaluator` registry) does NOT depend on
    // the answer, but the finding is recorded in spec §3.D for the audit trail.
    //
    // If `#[derive(Module)]` below fails to compile, the whole crate fails to
    // build and this probe never runs — a build failure IS the negative result.

    /// Minimal one-hidden-layer MLP variant for the enum-derive probe.
    #[derive(Module, Debug)]
    struct SmallMlp<B: Backend> {
        l1: Linear<B>,
        l2: Linear<B>,
    }

    impl<B: Backend> SmallMlp<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                l1: LinearConfig::new(2, 4).init(device),
                l2: LinearConfig::new(4, 1).init(device),
            }
        }
    }

    /// Minimal two-hidden-layer MLP variant for the enum-derive probe.
    #[derive(Module, Debug)]
    struct LargeMlp<B: Backend> {
        l1: Linear<B>,
        l2: Linear<B>,
        l3: Linear<B>,
    }

    impl<B: Backend> LargeMlp<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                l1: LinearConfig::new(2, 8).init(device),
                l2: LinearConfig::new(8, 4).init(device),
                l3: LinearConfig::new(4, 1).init(device),
            }
        }
    }

    /// Two-arm enum with heterogeneous `Module` variants. The backend generic
    /// must be named literally `B` for `#[derive(Module)]` to succeed.
    // This is a derive-capability probe, not a runtime data structure — the
    // size disparity between arms is irrelevant here.
    #[allow(clippy::large_enum_variant)]
    #[derive(Module, Debug)]
    enum TestArch<B: Backend> {
        Shallow(SmallMlp<B>),
        Deep(LargeMlp<B>),
    }

    /// Probe: confirm `#[derive(Module)]` on a heterogeneous-arm enum compiles
    /// and that the enum can be visited as a `Module` (flattened) — i.e. the
    /// derive emits real `visit`/`map` traversals, not just a stub.
    #[test]
    fn burn_enum_derive_probe() {
        let device = Default::default();

        let shallow = TestArch::<TestBackend>::Shallow(SmallMlp::new(&device));
        let deep = TestArch::<TestBackend>::Deep(LargeMlp::new(&device));

        // SmallMlp: 2*4 + 4 + 4*1 + 1 = 17 ; LargeMlp: 2*8+8 + 8*4+4 + 4*1+1 = 65.
        let shallow_reshaper = ModuleReshaper::new(shallow);
        let deep_reshaper = ModuleReshaper::new(deep);

        println!(
            "burn_enum_derive_probe: #[derive(Module)] on enum COMPILES; \
             enum is a Module. Shallow arm flattens to {} params, \
             Deep arm flattens to {} params.",
            shallow_reshaper.num_params(),
            deep_reshaper.num_params(),
        );

        // The enum derive visits the active arm's leaves.
        assert_eq!(shallow_reshaper.num_params(), 17);
        assert_eq!(deep_reshaper.num_params(), 65);
    }
}
