//! Cross-landscape regression net for issue #110 — an unguarded `dim` on an
//! n-dimensional landscape silently produces `NaN` or a wrong optimum
//! (`Sphere::new(0)` evaluates the empty sum `0`, i.e. reports itself solved on
//! every call; `Ackley::new(0)` divides by zero).
//!
//! Every landscape carries its own `new_rejects_*` unit test, but per-file tests
//! do nothing to stop the *next* landscape from landing unguarded. This file is
//! the single place that enumerates all of them and — critically — checks that
//! enumeration against the crate source tree itself:
//! [`table_covers_every_landscape_module`] reads `src/landscapes/` from disk at
//! test time and fails unless every module there is classified into exactly one
//! of
//!
//! - [`DIM_GUARDS`] — n-D landscapes with a `new(dim)` constructor, whose guard
//!   is then exercised by the other tests in this file;
//! - [`MULTI_FACTOR_LANDSCAPES`] — n-D landscapes whose dimension is a product of
//!   several constructor arguments, covered by a bespoke test
//!   ([`concatenated_trap_rejects_zero_factors`]);
//! - [`FIXED_2D_LANDSCAPES`] — landscapes defined only on 2-D inputs, which take
//!   no `dim` at all and so have nothing to guard;
//! - [`NON_LANDSCAPE_MODULES`] — support modules that are not landscapes.
//!
//! **Adding a landscape? You must classify it in one of those lists** — a new
//! `src/landscapes/*.rs` file that appears in none of them fails the test with
//! instructions. The check also runs in reverse: a listed module that no longer
//! exists on disk is a stale row and fails too.

use std::fs;
use std::path::Path;

use rlevo_core::config::ConfigError;

use rlevo_environments::landscapes::{
    ackley::Ackley, alpine1::Alpine1, concatenated_trap::ConcatenatedTrap, deb1::Deb1,
    eggholder::Eggholder, griewank::Griewank, lunacek_bi_rastrigin::LunacekBiRastrigin,
    michalewicz::Michalewicz, needle_eye::Needle, penalized1::Penalized1, rastrigin::Rastrigin,
    rosenbrock::Rosenbrock, rosenbrock_flat::RosenbrockFlat, schwefel::Schwefel, sphere::Sphere,
};

/// The landscape module directory, resolved from the crate root at compile time.
/// Cargo sets `CARGO_MANIFEST_DIR` for integration tests, so this points at the
/// real source tree the test binary was built from.
const LANDSCAPE_SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/landscapes");

/// One landscape's dimension guard, type-erased so landscapes with different
/// concrete types share a table.
///
/// `ctor` runs the real `new(dim)` and, on success, projects through the
/// `dim()` accessor — so the row exercises the constructor *and* asserts the
/// accessor round-trips the value that was accepted.
struct DimGuard {
    /// Landscape type name, for assertion messages.
    name: &'static str,
    /// The `landscapes::<module>` this row covers, cross-checked against disk by
    /// [`table_covers_every_landscape_module`].
    module: &'static str,
    /// `new(dim)` mapped to the constructed instance's reported `dim()`.
    ctor: fn(usize) -> Result<usize, ConfigError>,
    /// Smallest `dim` the landscape is well-defined at.
    min_dim: usize,
}

/// Builds a [`DimGuard`] row per `module: Type => min_dim` triple.
///
/// The module is spelled out rather than derived from the type name: the two do
/// not always agree (`Needle` lives in `needle_eye`), and a wrong guess would
/// weaken the disk cross-check into a rubber stamp.
macro_rules! dim_guards {
    ($($module:ident: $ty:ty => $min_dim:expr),+ $(,)?) => {
        &[$(DimGuard {
            name: stringify!($ty),
            module: stringify!($module),
            ctor: |dim| <$ty>::new(dim).map(|l| l.dim()),
            min_dim: $min_dim,
        }),+]
    };
}

/// Every n-D landscape with a `new(dim)` constructor, paired with the smallest
/// dimension at which its objective is well-defined.
///
/// `min_dim == 2` for the landscapes whose sum runs over *adjacent pairs*
/// (`Rosenbrock`, `RosenbrockFlat`) or is defined on 2-D coordinate pairs
/// (`Eggholder`), and for `LunacekBiRastrigin` whose depth term degenerates in
/// one dimension. Everything else is a separable per-coordinate sum and is
/// meaningful from `dim == 1`.
static DIM_GUARDS: &[DimGuard] = dim_guards![
    sphere: Sphere => 1,
    rastrigin: Rastrigin => 1,
    ackley: Ackley => 1,
    griewank: Griewank => 1,
    schwefel: Schwefel => 1,
    alpine1: Alpine1 => 1,
    deb1: Deb1 => 1,
    needle_eye: Needle => 1,
    michalewicz: Michalewicz => 1,
    penalized1: Penalized1 => 1,
    rosenbrock: Rosenbrock => 2,
    rosenbrock_flat: RosenbrockFlat => 2,
    eggholder: Eggholder => 2,
    lunacek_bi_rastrigin: LunacekBiRastrigin => 2,
];

/// n-D landscapes whose dimension is a *product* of several constructor
/// arguments, so they do not fit the `new(dim)` shape of [`DIM_GUARDS`] and get
/// a bespoke test instead — see [`concatenated_trap_rejects_zero_factors`].
static MULTI_FACTOR_LANDSCAPES: &[&str] = &["concatenated_trap"];

/// Landscapes defined *only* on 2-D inputs: their `new()` takes no arguments,
/// there is no `dim` to validate, and issue #110 cannot apply to them.
///
/// A landscape belongs here **only if it has no dimension parameter at all**. If
/// it takes a `dim`, it needs a [`DIM_GUARDS`] row instead.
static FIXED_2D_LANDSCAPES: &[&str] = &[
    "branin",
    "bukin6",
    "cross_in_tray",
    "easom",
    "goldstein_price",
    "himmelblau",
    "six_hump_camel",
    "trefethen",
];

/// Modules under `src/landscapes/` that do not define a landscape.
static NON_LANDSCAPE_MODULES: &[&str] = &["render"];

/// The heart of #110: `dim == 0` is never a legal landscape.
#[test]
fn every_landscape_rejects_zero_dim() {
    for guard in DIM_GUARDS {
        assert!(
            (guard.ctor)(0).is_err(),
            "{}::new(0) must be rejected (issue #110)",
            guard.name,
        );
    }
}

/// `dim == 1` is the boundary case: legal for the separable landscapes,
/// rejected by the four that need at least a coordinate pair.
#[test]
fn one_dim_is_accepted_exactly_by_the_separable_landscapes() {
    for guard in DIM_GUARDS {
        let got = (guard.ctor)(1);
        if guard.min_dim >= 2 {
            assert!(
                got.is_err(),
                "{} needs dim >= {} — new(1) must be rejected",
                guard.name,
                guard.min_dim,
            );
        } else {
            assert_eq!(
                got,
                Ok(1),
                "{} is well-defined at dim == 1 — new(1) must be accepted",
                guard.name,
            );
        }
    }
}

/// Each landscape accepts its own declared minimum, and `dim()` round-trips the
/// accepted value (guarding against a private-field/accessor mismatch).
#[test]
fn every_landscape_accepts_its_minimum_and_reports_it() {
    for guard in DIM_GUARDS {
        assert_eq!(
            (guard.ctor)(guard.min_dim),
            Ok(guard.min_dim),
            "{}::new({}) must be accepted and report dim() == {}",
            guard.name,
            guard.min_dim,
            guard.min_dim,
        );
        assert_eq!(
            (guard.ctor)(10),
            Ok(10),
            "{}::new(10) must be accepted and report dim() == 10",
            guard.name,
        );
    }
}

/// [`ConcatenatedTrap`] is the one landscape whose dimension is a *product*, so
/// either factor going to zero yields a zero-length genome.
#[test]
fn concatenated_trap_rejects_zero_factors() {
    assert!(
        ConcatenatedTrap::new(0, 5).is_err(),
        "ConcatenatedTrap::new(0, 5) must be rejected: num_blocks == 0 (issue #110)",
    );
    assert!(
        ConcatenatedTrap::new(4, 0).is_err(),
        "ConcatenatedTrap::new(4, 0) must be rejected: block_size == 0 (issue #110)",
    );
    let trap = ConcatenatedTrap::new(4, 5).expect("num_blocks >= 1 && block_size >= 1");
    assert_eq!(trap.dim(), 20, "dim() must be num_blocks * block_size");
}

/// Tripwire: every module in `src/landscapes/` must be classified, and every
/// classification must name a module that exists.
///
/// This reads the crate's own source directory, so it bites on a landscape that
/// is *added* without a guard — which no table-internal count can do.
#[test]
fn table_covers_every_landscape_module() {
    let on_disk = landscape_modules_on_disk();
    let registered = registered_modules();

    let unregistered: Vec<&String> = on_disk
        .iter()
        .filter(|module| !registered.contains(&module.as_str()))
        .collect();
    assert!(
        unregistered.is_empty(),
        "unclassified landscape module(s) in src/landscapes/: {unregistered:?}\n\
         Every landscape must be accounted for by this regression net (issue #110). \
         Classify each one in tests/landscape_dim_guards.rs:\n\
         - takes a `dim`: add a DIM_GUARDS row (`module: Type => min_dim`);\n\
         - dimension is a product of several args: add it to MULTI_FACTOR_LANDSCAPES \
         and cover it like concatenated_trap_rejects_zero_factors;\n\
         - genuinely 2-D with a no-argument `new()`: add it to FIXED_2D_LANDSCAPES;\n\
         - not a landscape at all: add it to NON_LANDSCAPE_MODULES.\n\
         Do not delete this assertion.",
    );

    let stale: Vec<&&str> = registered
        .iter()
        .filter(|module| !on_disk.iter().any(|found| found == *module))
        .collect();
    assert!(
        stale.is_empty(),
        "stale row(s) in tests/landscape_dim_guards.rs: {stale:?} — \
         no such module in src/landscapes/. Was it renamed or removed?",
    );
}

/// Every module name this file claims to cover, across all four lists.
///
/// Panics if a module is classified twice: overlapping lists would make the
/// coverage check ambiguous (and one of the two classifications is necessarily
/// wrong).
fn registered_modules() -> Vec<&'static str> {
    let mut modules: Vec<&'static str> = DIM_GUARDS
        .iter()
        .map(|guard| guard.module)
        .chain(MULTI_FACTOR_LANDSCAPES.iter().copied())
        .chain(FIXED_2D_LANDSCAPES.iter().copied())
        .chain(NON_LANDSCAPE_MODULES.iter().copied())
        .collect();
    modules.sort_unstable();

    let mut deduped = modules.clone();
    deduped.dedup();
    assert_eq!(
        modules, deduped,
        "a landscape module is classified in more than one list — each belongs to \
         exactly one of DIM_GUARDS / MULTI_FACTOR_LANDSCAPES / FIXED_2D_LANDSCAPES / \
         NON_LANDSCAPE_MODULES",
    );

    modules
}

/// The landscape modules that actually exist in the crate source tree.
///
/// Both module layouts are picked up: `foo.rs` and `foo/mod.rs`. `mod.rs` itself
/// only declares a module tree, so it is not a landscape module and is skipped;
/// non-Rust files (editor scratch, etc.) are ignored.
fn landscape_modules_on_disk() -> Vec<String> {
    let dir = Path::new(LANDSCAPE_SRC_DIR);
    let entries =
        fs::read_dir(dir).unwrap_or_else(|err| panic!("cannot read {}: {err}", dir.display()));

    let mut modules: Vec<String> = entries
        .map(|entry| entry.expect("readable directory entry").path())
        .filter_map(|path| {
            if path.is_dir() {
                return path.file_name()?.to_str().map(String::from);
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                return None;
            }
            let stem = path.file_stem()?.to_str()?;
            (stem != "mod").then(|| stem.to_owned())
        })
        .collect();
    modules.sort();

    assert!(
        !modules.is_empty(),
        "found no landscape modules under {} — the path this test walks is wrong, \
         which would silently disarm the coverage check",
        dir.display(),
    );
    modules
}
