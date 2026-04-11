use csv::ReaderBuilder;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::HashMap;

#[pyfunction]
fn matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_view = a.as_array();
    let b_view = b.as_array();

    let (m, k_left) = a_view.dim();
    let (k_right, n) = b_view.dim();
    if k_left != k_right {
        return Err(PyValueError::new_err(format!(
            "shape mismatch for matmul: left is ({m},{k_left}), right is ({k_right},{n})"
        )));
    }

    let mut out = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for p in 0..k_left {
            let a_ip = a_view[(i, p)];
            for j in 0..n {
                out[(i, j)] += a_ip * b_view[(p, j)];
            }
        }
    }

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn aggregate_ratings_csv<'py>(
    py: Python<'py>,
    csv_path: &str,
    max_movie_id: usize,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f64>>, u32)> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .map_err(|e| PyValueError::new_err(format!("failed to open CSV '{csv_path}': {e}")))?;

    let mut movie_count = vec![0_u64; max_movie_id + 1];
    let mut movie_sum = vec![0_f64; max_movie_id + 1];
    let mut max_user_id: u32 = 0;

    for rec in rdr.records() {
        let row = rec.map_err(|e| PyValueError::new_err(format!("failed reading CSV row: {e}")))?;

        let user_id: u32 = row
            .get(0)
            .ok_or_else(|| PyValueError::new_err("missing userId column"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid userId: {e}")))?;
        let movie_id: usize = row
            .get(1)
            .ok_or_else(|| PyValueError::new_err("missing movieId column"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid movieId: {e}")))?;
        let rating: f64 = row
            .get(2)
            .ok_or_else(|| PyValueError::new_err("missing rating column"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid rating: {e}")))?;

        if user_id > max_user_id {
            max_user_id = user_id;
        }

        if movie_id <= max_movie_id {
            movie_count[movie_id] += 1;
            movie_sum[movie_id] += rating;
        }
    }

    Ok((
        movie_count.into_pyarray_bound(py),
        movie_sum.into_pyarray_bound(py),
        max_user_id,
    ))
}

#[pyfunction]
fn build_user_preferences_from_csv<'py>(
    py: Python<'py>,
    ratings_csv_path: &str,
    genome_scores_csv_path: &str,
    max_active_users: usize,
    rating_scale: f32,
    num_movies: usize,
    num_tags: usize,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<f32>>)> {
    if max_active_users == 0 {
        return Err(PyValueError::new_err("max_active_users must be > 0"));
    }

    // Pass 1: count user activity.
    let mut user_counts: HashMap<u32, u32> = HashMap::new();
    let mut rdr_counts = ReaderBuilder::new()
        .has_headers(true)
        .from_path(ratings_csv_path)
        .map_err(|e| PyValueError::new_err(format!("failed to open ratings CSV: {e}")))?;

    for rec in rdr_counts.records() {
        let row = rec.map_err(|e| PyValueError::new_err(format!("failed reading ratings row: {e}")))?;
        let user_id: u32 = row
            .get(0)
            .ok_or_else(|| PyValueError::new_err("missing userId column in ratings"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid ratings userId: {e}")))?;
        *user_counts.entry(user_id).or_insert(0) += 1;
    }

    if user_counts.is_empty() {
        return Err(PyValueError::new_err("ratings CSV has no rows"));
    }

    let mut users: Vec<(u32, u32)> = user_counts.into_iter().collect();
    users.sort_by_key(|(uid, cnt)| (Reverse(*cnt), *uid));
    users.truncate(max_active_users.min(users.len()));
    users.sort_by_key(|(uid, _)| *uid);

    let user_index_to_id: Vec<i32> = users.iter().map(|(uid, _)| *uid as i32).collect();
    let num_active_users = user_index_to_id.len();

    let mut user_id_to_index: HashMap<u32, usize> = HashMap::with_capacity(num_active_users);
    for (idx, uid) in user_index_to_id.iter().enumerate() {
        user_id_to_index.insert(*uid as u32, idx);
    }

    // Build sparse movie->[(tag,relevance)] from genome scores.
    let mut movie_tags: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_movies + 1];
    let mut rdr_genome = ReaderBuilder::new()
        .has_headers(true)
        .from_path(genome_scores_csv_path)
        .map_err(|e| PyValueError::new_err(format!("failed to open genome scores CSV: {e}")))?;

    for rec in rdr_genome.records() {
        let row = rec.map_err(|e| PyValueError::new_err(format!("failed reading genome row: {e}")))?;
        let movie_id: usize = row
            .get(0)
            .ok_or_else(|| PyValueError::new_err("missing movieId column in genome scores"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid genome movieId: {e}")))?;
        let tag_id: usize = row
            .get(1)
            .ok_or_else(|| PyValueError::new_err("missing tagId column in genome scores"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid genome tagId: {e}")))?;
        let relevance: f32 = row
            .get(2)
            .ok_or_else(|| PyValueError::new_err("missing relevance column in genome scores"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid genome relevance: {e}")))?;

        if movie_id <= num_movies && tag_id <= num_tags {
            movie_tags[movie_id].push((tag_id, relevance));
        }
    }

    let total_size = num_active_users * (num_tags + 1);
    let mut numer = vec![0.0_f32; total_size];
    let mut denom = vec![0.0_f32; total_size];

    // Pass 2: accumulate weighted and denominator sums.
    let mut rdr_ratings = ReaderBuilder::new()
        .has_headers(true)
        .from_path(ratings_csv_path)
        .map_err(|e| PyValueError::new_err(format!("failed to reopen ratings CSV: {e}")))?;

    for rec in rdr_ratings.records() {
        let row = rec.map_err(|e| PyValueError::new_err(format!("failed reading ratings row: {e}")))?;
        let user_id: u32 = row
            .get(0)
            .ok_or_else(|| PyValueError::new_err("missing userId column in ratings"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid ratings userId: {e}")))?;

        let Some(&u_idx) = user_id_to_index.get(&user_id) else {
            continue;
        };

        let movie_id: usize = row
            .get(1)
            .ok_or_else(|| PyValueError::new_err("missing movieId column in ratings"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid ratings movieId: {e}")))?;
        if movie_id > num_movies {
            continue;
        }

        let rating: f32 = row
            .get(2)
            .ok_or_else(|| PyValueError::new_err("missing rating column in ratings"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid ratings rating: {e}")))?;
        let scaled = rating * rating_scale;

        let base = u_idx * (num_tags + 1);
        for (tag_id, rel) in &movie_tags[movie_id] {
            let idx = base + *tag_id;
            numer[idx] += scaled * *rel;
            denom[idx] += *rel;
        }
    }

    // Finalize preferences in-place in numer.
    for i in 0..total_size {
        if denom[i] != 0.0 {
            numer[i] /= denom[i];
        }
    }

    let out = Array2::from_shape_vec((num_active_users, num_tags + 1), numer)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output matrix: {e}")))?;

    Ok((
        user_index_to_id.into_pyarray_bound(py),
        out.into_pyarray_bound(py),
    ))
}

#[pymodule]
fn hybrid_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_ratings_csv, m)?)?;
    m.add_function(wrap_pyfunction!(build_user_preferences_from_csv, m)?)?;
    Ok(())
}
