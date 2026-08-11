use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn pairwise_distances_impl(points: ArrayView2<f64>) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    let data = points.as_slice().expect("contiguous input required");
    let mut out = vec![0.0f64; n * n];

    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let pi = &data[i * d..(i + 1) * d];
        for (j, cell) in row.iter_mut().enumerate() {
            let pj = &data[j * d..(j + 1) * d];
            let mut acc = 0.0;
            for k in 0..d {
                let diff = pi[k] - pj[k];
                acc += diff * diff;
            }
            *cell = acc.sqrt();
        }
    });

    Array2::from_shape_vec((n, n), out).expect("shape mismatch")
}

/// Gravitational accelerations by direct summation (O(N^2), rayon-parallel),
/// with Plummer softening to avoid singularities at close encounters.
fn gravity_accel_impl(pos: ArrayView2<f64>, masses: &[f64], softening: f64) -> Array2<f64> {
    let n = pos.nrows();
    let data = pos.as_slice().expect("contiguous input required");
    let eps2 = softening * softening;
    let mut acc = vec![0.0f64; n * 3];

    acc.par_chunks_mut(3).enumerate().for_each(|(i, ai)| {
        let pi = &data[i * 3..i * 3 + 3];
        for j in 0..n {
            if i == j {
                continue;
            }
            let pj = &data[j * 3..j * 3 + 3];
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let dz = pj[2] - pi[2];
            let r2 = dx * dx + dy * dy + dz * dz + eps2;
            let inv_r3 = masses[j] / (r2 * r2.sqrt());
            ai[0] += dx * inv_r3;
            ai[1] += dy * inv_r3;
            ai[2] += dz * inv_r3;
        }
    });

    Array2::from_shape_vec((n, 3), acc).expect("shape mismatch")
}

/// Pairwise Euclidean distance matrix for an (n, d) point cloud.
#[pyfunction]
fn pairwise_distances<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = points.as_array();
    let result = py.detach(|| pairwise_distances_impl(arr));
    Ok(result.into_pyarray(py))
}

/// Gravitational accelerations (G=1 units) for positions (n, 3) and masses (n,).
#[pyfunction]
#[pyo3(signature = (positions, masses, softening=0.05))]
fn gravity_accelerations<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    masses: Vec<f64>,
    softening: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pos = positions.as_array();
    if pos.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions must have shape (n, 3)",
        ));
    }
    if masses.len() != pos.nrows() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "masses length must match number of positions",
        ));
    }
    let result = py.detach(|| gravity_accel_impl(pos, &masses, softening));
    Ok(result.into_pyarray(py))
}

#[pymodule]
fn _numerics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairwise_distances, m)?)?;
    m.add_function(wrap_pyfunction!(gravity_accelerations, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn distances_unit_triangle() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let d = pairwise_distances_impl(pts.view());
        assert!((d[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((d[[0, 2]] - 1.0).abs() < 1e-12);
        assert!((d[[1, 2]] - 2.0f64.sqrt()).abs() < 1e-12);
        assert_eq!(d[[0, 0]], 0.0);
    }

    #[test]
    fn gravity_two_body_symmetry() {
        let pos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let masses = vec![1.0, 1.0];
        let acc = gravity_accel_impl(pos.view(), &masses, 0.0);
        // equal masses: accelerations equal and opposite along x
        assert!((acc[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((acc[[1, 0]] + 1.0).abs() < 1e-12);
        assert!(acc[[0, 1]].abs() < 1e-12);
    }
}
