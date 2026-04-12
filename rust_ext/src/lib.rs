use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct GenomeScoreRow {
    #[serde(rename = "movieId")]
    movie_id: u32,
    #[serde(rename = "tagId")]
    tag_id: u32,
    relevance: f32,
}

#[derive(Deserialize)]
struct RatingRow {
    #[serde(rename = "userId")]
    user_id: u32,
    #[serde(rename = "movieId")]
    movie_id: u32,
    rating: f32,
}

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

#[pyfunction]
fn evaluate_model_wrapper(script_path: String, args: Vec<String>) -> PyResult<(i32, String, String)> {
    let script = Path::new(&script_path);
    if !script.exists() {
        return Err(py_value_error(format!(
            "evaluate script was not found at path: {}",
            script_path
        )));
    }

    let output = Command::new("python3")
        .arg(script)
        .args(args)
        .output()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to execute evaluator: {e}")))?;

    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Ok((code, stdout, stderr))
}

#[pyfunction]
fn populate_user_preferences_matrix(
    ratings_csv_path: String,
    genome_scores_csv_path: String,
    max_active_users: usize,
    rating_scale: f32,
    num_movies: usize,
    num_tags: usize,
) -> PyResult<(Vec<u32>, Vec<Vec<f32>>)> {
    if max_active_users == 0 {
        return Err(py_value_error("max_active_users must be greater than 0"));
    }
    if rating_scale <= 0.0 {
        return Err(py_value_error("rating_scale must be greater than 0"));
    }

    let ratings_path = Path::new(&ratings_csv_path);
    if !ratings_path.exists() {
        return Err(py_value_error(format!(
            "ratings csv was not found at path: {}",
            ratings_csv_path
        )));
    }

    let genome_path = Path::new(&genome_scores_csv_path);
    if !genome_path.exists() {
        return Err(py_value_error(format!(
            "genome scores csv was not found at path: {}",
            genome_scores_csv_path
        )));
    }

    let n_tags = num_tags + 1;

    let mut movie_tag_relevance: HashMap<u32, Vec<(usize, f32)>> = HashMap::new();
    let mut genome_reader = csv::Reader::from_path(genome_path)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read genome scores csv: {e}")))?;

    for row in genome_reader.deserialize::<GenomeScoreRow>() {
        let record = row
            .map_err(|e| PyRuntimeError::new_err(format!("invalid genome scores row: {e}")))?;

        if record.movie_id as usize > num_movies {
            continue;
        }
        if record.tag_id as usize > num_tags {
            continue;
        }

        movie_tag_relevance
            .entry(record.movie_id)
            .or_default()
            .push((record.tag_id as usize, record.relevance));
    }

    let mut user_id_to_index: HashMap<u32, usize> = HashMap::new();
    let mut user_index_to_id: Vec<u32> = Vec::with_capacity(max_active_users);
    let mut ratings_by_user: Vec<Vec<(u32, f32)>> = Vec::with_capacity(max_active_users);

    let mut ratings_reader = csv::Reader::from_path(ratings_path)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read ratings csv: {e}")))?;

    for row in ratings_reader.deserialize::<RatingRow>() {
        let record =
            row.map_err(|e| PyRuntimeError::new_err(format!("invalid ratings row: {e}")))?;

        let user_index = if let Some(&idx) = user_id_to_index.get(&record.user_id) {
            idx
        } else {
            if user_index_to_id.len() >= max_active_users {
                continue;
            }
            let next_idx = user_index_to_id.len();
            user_id_to_index.insert(record.user_id, next_idx);
            user_index_to_id.push(record.user_id);
            ratings_by_user.push(Vec::new());
            next_idx
        };

        ratings_by_user[user_index].push((record.movie_id, record.rating));
    }

    // Compute each user's preference vector in parallel — no shared mutable state.
    let preferences: Vec<Vec<f32>> = ratings_by_user
        .par_iter()
        .map(|user_ratings| {
            let mut numerator = vec![0.0f32; n_tags];
            let mut denominator = vec![0.0f32; n_tags];

            for (movie_id, rating) in user_ratings {
                let normalized = rating / rating_scale;
                if let Some(tag_rows) = movie_tag_relevance.get(movie_id) {
                    for (tag_id, relevance) in tag_rows {
                        numerator[*tag_id] += normalized * relevance;
                        denominator[*tag_id] += relevance;
                    }
                }
            }

            numerator
                .iter_mut()
                .zip(denominator.iter())
                .for_each(|(n, d)| {
                    if *d > 0.0 {
                        *n /= *d;
                    }
                });

            numerator
        })
        .collect();

    Ok((user_index_to_id, preferences))
}

#[pymodule]
fn hybrid_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(evaluate_model_wrapper, module)?)?;
    module.add_function(wrap_pyfunction!(populate_user_preferences_matrix, module)?)?;
    Ok(())
}
