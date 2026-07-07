//! Host-side dense linear algebra for covariance-matrix strategies.
//!
//! CMA-ES and CMSA-ES need a symmetric eigendecomposition (for the sampling
//! transform `B·diag(√Λ)` and the conditioning matrix `C^{-1/2}`) and a
//! Cholesky factor (for CMSA-ES sampling). Burn 0.21 ships **no** Cholesky or
//! eigendecomposition primitive, and the workspace deliberately avoids a
//! `nalgebra` dependency (ADR 0021 §3 / research note
//! `cma-es-sampling-and-numerics` §L4: the logged `nalgebra` 4×4 symmetric-eigen
//! bug and its non-portable LAPACK path do not justify the dependency for the
//! `D ≤ 30` regime these strategies target). Both routines therefore run on host
//! `Vec<f32>` buffers — covariance matrices are tiny, so the device round-trip
//! would dominate any on-device kernel anyway.
//!
//! All matrices are **row-major** `n × n`: entry `(i, j)` lives at index
//! `i * n + j`.

/// Hard cap on Jacobi sweeps; convergence is quadratic, so for the small `n`
/// the covariance-matrix strategies use this is never reached in practice.
const MAX_SWEEPS: usize = 100;

/// Symmetric eigendecomposition `A = V · diag(Λ) · Vᵀ`.
///
/// Returned by [`jacobi_eigen`]. Packaging the two buffers in a named struct
/// removes the positional ambiguity of a `(Vec<f32>, Vec<f32>)` pair — a caller
/// can no longer transpose eigenvalues and eigenvectors at the destructuring
/// site — and carries the column-layout invariant on the fields where it is
/// used.
#[derive(Debug, Clone)]
pub struct SymEigen {
    /// Eigenvalues `Λ` (unsorted), length `n`.
    pub values: Vec<f32>,
    /// Row-major `n × n` eigenvector matrix `V` whose **column** `k` is the
    /// eigenvector for `values[k]`: component `i` lives at `vectors[i * n + k]`.
    pub vectors: Vec<f32>,
}

/// Symmetric eigendecomposition via the cyclic Jacobi method.
///
/// `a` is an `n × n` **symmetric** matrix in row-major order. Returns a
/// [`SymEigen`] carrying the eigenvalues (unsorted) and the row-major
/// eigenvector matrix; see the [`SymEigen`] field docs for the column-layout
/// invariant (`vectors` column `k` is the eigenvector for `values[k]`).
///
/// The eigenvector columns are orthonormal, so the input is reconstructed as
/// `V · diag(Λ) · Vᵀ`. The classic numerically stable rotation (Golub & Van
/// Loan, *Matrix Computations*, §8.4) is used; sweeps stop once the
/// off-diagonal Frobenius mass is negligible or after [`MAX_SWEEPS`].
///
/// Jacobi is the eigensolver `pycma` itself uses; it is slower than tridiagonal
/// QR but more accurate on the small eigenvalues that govern an ill-conditioned
/// covariance (Demmel & Veselić, 1992) — exactly the regime CMA-ES drives `C`
/// into late in a run.
///
/// # Panics
///
/// Panics if `a.len() != n * n`.
// Jacobi rotation uses the conventional single-letter math names (p, q for the
// pivot pair; c, s, t for cos/sin/tan of the rotation angle).
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn jacobi_eigen(a: &[f32], n: usize) -> SymEigen {
    assert_eq!(a.len(), n * n, "matrix buffer must be n*n");
    let mut work: Vec<f32> = a.to_vec();
    let mut vecs: Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        vecs[i * n + i] = 1.0;
    }
    if n <= 1 {
        return SymEigen {
            values: work,
            vectors: vecs,
        };
    }

    // Off-diagonal mass below this (sum of squares) counts as converged.
    let tol: f32 = 1e-14;

    for _ in 0..MAX_SWEEPS {
        let mut off: f32 = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += work[p * n + q] * work[p * n + q];
            }
        }
        if off <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq: f32 = work[p * n + q];
                if apq.abs() <= f32::EPSILON {
                    continue;
                }
                let app: f32 = work[p * n + p];
                let aqq: f32 = work[q * n + q];
                // Symmetric Schur: choose the rotation that annihilates (p, q).
                let theta: f32 = (aqq - app) / (2.0 * apq);
                let t: f32 = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c: f32 = 1.0 / (1.0 + t * t).sqrt();
                let s: f32 = t * c;
                // A ← Jᵀ A J, applied as a column update then a row update.
                for r in 0..n {
                    let arp: f32 = work[r * n + p];
                    let arq: f32 = work[r * n + q];
                    work[r * n + p] = c * arp - s * arq;
                    work[r * n + q] = s * arp + c * arq;
                }
                for r in 0..n {
                    let apr: f32 = work[p * n + r];
                    let aqr: f32 = work[q * n + r];
                    work[p * n + r] = c * apr - s * aqr;
                    work[q * n + r] = s * apr + c * aqr;
                }
                // Pin the annihilated off-diagonal to exact zero / symmetry.
                work[p * n + q] = 0.0;
                work[q * n + p] = 0.0;
                // Accumulate the eigenvector basis: V ← V J.
                for r in 0..n {
                    let vrp: f32 = vecs[r * n + p];
                    let vrq: f32 = vecs[r * n + q];
                    vecs[r * n + p] = c * vrp - s * vrq;
                    vecs[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let eigvals: Vec<f32> = (0..n).map(|i| work[i * n + i]).collect();
    SymEigen {
        values: eigvals,
        vectors: vecs,
    }
}

/// Lower-triangular Cholesky factor `L` with `L · Lᵀ = a`.
///
/// `a` is an `n × n` **symmetric positive-definite** matrix in row-major order.
/// Returns the lower-triangular `L` (row-major `n × n`, zeros above the
/// diagonal) or `None` if a non-positive **or non-finite** pivot is
/// encountered. Callers recover by jittering the diagonal and retrying.
///
/// The pivot guard rejects any NaN-bearing (or infinite) input matrix, not just
/// a directly-NaN diagonal: a NaN anywhere in `a` — including strictly
/// off-diagonal — propagates into a later diagonal pivot through the
/// `sum -= l[i*n+k] * l[j*n+k]` accumulation, so by the time a diagonal `sum` is
/// tested it is itself NaN. Testing `sum.is_finite()` before `sum <= 0.0` is
/// essential because `NaN <= 0.0` is `false`; without the finiteness check a NaN
/// pivot would slip through `sqrt` and poison every entry of the returned
/// factor, silently defeating the jitter-retry recovery in
/// `cmsa_es::cholesky_with_jitter` (which only retries on `None`).
///
/// # Panics
///
/// Panics if `a.len() != n * n`.
#[must_use]
pub fn cholesky(a: &[f32], n: usize) -> Option<Vec<f32>> {
    assert_eq!(a.len(), n * n, "matrix buffer must be n*n");
    let mut l: Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum: f32 = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Matrix–vector product `y = M · x` for a row-major `n × n` matrix `M`.
///
/// # Panics
///
/// Panics if `m.len() != n * n` or `x.len() != n`.
#[must_use]
pub fn matvec(m: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(m.len(), n * n, "matrix buffer must be n*n");
    assert_eq!(x.len(), n, "vector length must be n");
    let mut y: Vec<f32> = vec![0.0; n];
    for i in 0..n {
        let mut acc: f32 = 0.0;
        for j in 0..n {
            acc += m[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Forces the row-major `n × n` matrix `m` to be exactly symmetric in place.
///
/// For every `j < i`, both `(i, j)` and `(j, i)` are set to the average
/// `0.5 * (m[i*n+j] + m[j*n+i])`; the diagonal is untouched.
///
/// This is **not** a fix for round-off drift in the strategy loop. The CMA-ES /
/// CMSA-ES in-loop covariance updates preserve bit-exact symmetry on their own:
/// IEEE-754 multiplication is commutative, and the two triangle entries `C[i,j]`
/// and `C[j,i]` accumulate the identical rank-1 / rank-μ terms in the identical
/// order, so they stay bit-for-bit equal without help. The helper exists as a
/// **construction-boundary normalization** for caller-supplied covariance
/// matrices — a state constructor handed an externally-built or deserialized
/// `C` whose triangles may not agree — and as cheap defense-in-depth. It mirrors
/// `pycma`, which likewise keeps `C` exactly symmetric rather than trusting the
/// update to stay symmetric.
///
/// # Panics
///
/// Panics if `m.len() != n * n`.
pub fn symmetrize(m: &mut [f32], n: usize) {
    assert_eq!(m.len(), n * n, "matrix buffer must be n*n");
    for i in 0..n {
        for j in 0..i {
            let avg: f32 = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = avg;
            m[j * n + i] = avg;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct `V · diag(Λ) · Vᵀ` from an eigendecomposition.
    fn reconstruct(eigvals: &[f32], eigvecs: &[f32], n: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc: f32 = 0.0;
                for k in 0..n {
                    acc += eigvecs[i * n + k] * eigvals[k] * eigvecs[j * n + k];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn assert_matrix_close(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            approx::assert_relative_eq!(x, y, epsilon = eps);
        }
    }

    #[test]
    fn eigen_diagonal_matrix() {
        // diag(3, 5, 7): eigenvalues are the diagonal, eigenvectors the axes.
        let a: Vec<f32> = vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
        let SymEigen { values, vectors } = jacobi_eigen(&a, 3);
        let recon = reconstruct(&values, &vectors, 3);
        assert_matrix_close(&a, &recon, 1e-5);
    }

    #[test]
    fn eigen_known_2x2() {
        // [[2,1],[1,2]] has eigenvalues {1, 3}.
        let a: Vec<f32> = vec![2.0, 1.0, 1.0, 2.0];
        let SymEigen { values, vectors } = jacobi_eigen(&a, 2);
        let mut sorted: Vec<f32> = values.clone();
        sorted.sort_by(f32::total_cmp);
        approx::assert_relative_eq!(sorted[0], 1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(sorted[1], 3.0, epsilon = 1e-5);
        let recon = reconstruct(&values, &vectors, 2);
        assert_matrix_close(&a, &recon, 1e-5);
    }

    #[test]
    fn eigen_3x3_reconstructs_and_is_orthonormal() {
        // Symmetric, non-trivially coupled.
        let a: Vec<f32> = vec![4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let SymEigen { values, vectors } = jacobi_eigen(&a, 3);
        let recon = reconstruct(&values, &vectors, 3);
        assert_matrix_close(&a, &recon, 1e-4);
        // Columns orthonormal: VᵀV ≈ I.
        for p in 0..3 {
            for q in 0..3 {
                let mut dot: f32 = 0.0;
                for i in 0..3 {
                    dot += vectors[i * 3 + p] * vectors[i * 3 + q];
                }
                let expected: f32 = if p == q { 1.0 } else { 0.0 };
                approx::assert_relative_eq!(dot, expected, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn eigen_identity_is_fixed_point() {
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let SymEigen { values, vectors } = jacobi_eigen(&a, 2);
        for v in &values {
            approx::assert_relative_eq!(v, &1.0, epsilon = 1e-6);
        }
        // Identity input: no rotation, basis stays the identity.
        assert_matrix_close(&vectors, &[1.0, 0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn cholesky_known_2x2() {
        // [[4,2],[2,3]] = L Lᵀ with L = [[2,0],[1,√2]].
        let a: Vec<f32> = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).expect("matrix is positive-definite");
        approx::assert_relative_eq!(l[0], 2.0, epsilon = 1e-6);
        approx::assert_relative_eq!(l[1], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(l[2], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(l[3], 2.0_f32.sqrt(), epsilon = 1e-6);
        // Round-trip: L Lᵀ ≈ A.
        let mut recon: Vec<f32> = vec![0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc: f32 = 0.0;
                for k in 0..2 {
                    acc += l[i * 2 + k] * l[j * 2 + k];
                }
                recon[i * 2 + j] = acc;
            }
        }
        assert_matrix_close(&a, &recon, 1e-6);
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        // [[1,2],[2,1]] has eigenvalues {-1, 3}: indefinite.
        let a: Vec<f32> = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    #[test]
    fn cholesky_rejects_nan_on_diagonal() {
        // A NaN diagonal pivot: `NaN <= 0.0` is false, so the finiteness guard
        // is what rejects it (not the sign test).
        let a: Vec<f32> = vec![f32::NAN, 0.0, 0.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    #[test]
    fn cholesky_rejects_off_diagonal_only_nan() {
        // The ONLY NaN is off-diagonal; the diagonal is finite and positive.
        // It reaches the pivot at (1, 1) via the `sum -= l[i]·l[j]`
        // accumulation, exercising the propagation-to-pivot path.
        let a: Vec<f32> = vec![1.0, f32::NAN, f32::NAN, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    #[test]
    fn cholesky_rejects_infinity() {
        // An infinite entry is likewise non-finite; the pivot becomes
        // non-finite and is rejected.
        let a: Vec<f32> = vec![f32::INFINITY, 0.0, 0.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    #[test]
    fn symmetrize_averages_asymmetric_and_is_idempotent() {
        // Off-diagonal (0,1)=2, (1,0)=4 → both become 3; diagonal untouched.
        let mut m: Vec<f32> = vec![1.0, 2.0, 4.0, 5.0];
        symmetrize(&mut m, 2);
        assert_matrix_close(&m, &[1.0, 3.0, 3.0, 5.0], 1e-6);
        // Idempotent: a second pass over the now-symmetric matrix is a no-op.
        let once: Vec<f32> = m.clone();
        symmetrize(&mut m, 2);
        assert_matrix_close(&m, &once, 1e-6);
    }

    #[test]
    fn symmetrize_leaves_symmetric_unchanged() {
        let mut m: Vec<f32> = vec![4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let before: Vec<f32> = m.clone();
        symmetrize(&mut m, 3);
        assert_matrix_close(&m, &before, 1e-6);
    }

    #[test]
    fn symmetrize_handles_scalar_and_identity() {
        // 1×1: nothing to average, value preserved.
        let mut one: Vec<f32> = vec![7.0];
        symmetrize(&mut one, 1);
        assert_eq!(one, vec![7.0]);
        // Identity is already symmetric.
        let mut id: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        symmetrize(&mut id, 2);
        assert_matrix_close(&id, &[1.0, 0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn matvec_identity_and_general() {
        let id: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let x: Vec<f32> = vec![3.0, -2.0];
        assert_eq!(matvec(&id, &x, 2), vec![3.0, -2.0]);
        let m: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        // [1 2; 3 4] · [1; 1] = [3; 7].
        assert_eq!(matvec(&m, &[1.0, 1.0], 2), vec![3.0, 7.0]);
    }
}
