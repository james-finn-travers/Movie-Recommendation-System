# Hybrid Inference Engine

Hybrid Inference Engine project built on the MovieLens latest dataset.
The system combines content-based features (genome tag relevance) with user-user collaborative filtering to generate personalized recommendations.

## Overview

The pipeline has three main stages:

1. Build a sparse movie-tag relevance matrix from MovieLens genome scores.
2. Build user preference vectors from historical ratings and movie-tag relevance.
3. Find nearest-neighbor users and rank candidate movies from similar users.

The notebook provides an interactive interface (ipywidgets) where you can add ratings and request recommendations.

## System Architecture

```mermaid
flowchart LR
	A[Ingestion Layer<br/>MovieLens CSVs] --> B[Compute Layer<br/>Sparse Matrix + Memmap Preferences]
	B --> C[Serving Layer<br/>Hybrid KNN-Collaborative Filtering]
	C --> D[Top-N Recommendations]
```

- **Ingestion Layer:** Raw MovieLens CSVs processed via chunked pandas readers to minimize memory footprint.
- **Compute Layer:** Sparse matrix construction using scipy.sparse and numpy memmaps for high-performance, out-of-core preference vector calculation.
- **Serving Layer:** Hybrid KNN-Collaborative Filtering engine providing sub-150ms inference.

## Project Structure

- `Movie_Recommendation_System.ipynb`: Main notebook for loading data, preprocessing, building matrices, and running the interactive recommender UI.
- `user_preferences_script.py`: Script to build:
	- `users_preferences.dat` (memmap user preference matrix)
	- `user_index_to_id.npy` (mapping from matrix row index to original userId)
- `evaluate_model.py`: Offline evaluator for the recommender in ultra-low-memory mode.
- `benchmark_report.py`: Generates markdown benchmark summary with run-to-run deltas and pass/fail quality/latency gates.
- `run_benchmark_pipeline.py`: One-command runner that executes benchmark evaluation and report generation.
- `ml-latest/`: MovieLens dataset folder (`ratings.csv`, `movies.csv`, `genome-scores.csv`, etc.).
- `movie_tags.npz`: Sparse movie-tag matrix artifact used by notebook/scripts.
- `users_preferences.dat`: User preference memmap artifact.
- `user_index_to_id.npy`: User-id index artifact.

## Requirements

- Python 3.10+
- Core Python packages:
	- numpy
	- pandas
	- scipy
	- scikit-learn
	- matplotlib
	- ipywidgets
	- jupyter
	- pyarrow (recommended for parquet cache in notebook)

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib ipywidgets jupyter pyarrow
```

If needed, place or unzip MovieLens data so the folder exists at:

- `./ml-latest/`

## How To Run

### 1. Run the Notebook

Open `Movie_Recommendation_System.ipynb` and run cells from top to bottom.

The notebook includes:

- Data loading with optional low-memory controls.
- Title cleaning + TF-IDF search for matching user-entered movie names.
- Sparse movie-tag matrix construction and optional save to `movie_tags.npz`.
- Interactive widget dashboard for adding ratings and requesting recommendations.

### 2. Build User Preference Artifacts (if missing)

If `users_preferences.dat` and `user_index_to_id.npy` are not present, run:

```bash
python3 user_preferences_script.py
```

Useful options:

```bash
python3 user_preferences_script.py \
	--data-dir ./ml-latest \
	--movie-tags ./movie_tags.npz \
	--output-dir . \
	--min-user-ratings 15 \
	--chunk-size 500000
```

### 3. Evaluate Model Quality (optional)

```bash
python3 evaluate_model.py
```

Useful options:

```bash
python3 evaluate_model.py \
	--data-dir ./ml-latest \
	--movie-tags ./movie_tags.npz \
	--prefs ./users_preferences.dat \
	--user-index ./user_index_to_id.npy \
	--sample-users 500 \
	--k-neighbors 25
```

### 4. Run Repeatable Benchmark Checkpoints (every 150 users)

This mode records ranking quality and latency percentiles at each user-count checkpoint.

```bash
python3 evaluate_model.py \
	--data-dir ./ml-latest \
	--movie-tags ./movie_tags.npz \
	--prefs ./users_preferences.dat \
	--user-index ./user_index_to_id.npy \
	--sample-users 600 \
	--benchmark-step 150 \
	--k-neighbors 25 \
	--results-path ./benchmark_results.csv
```

Recorded at each checkpoint (`150, 300, 450, ...`):

- Precision@5/10/20
- Recall@5/10/20
- NDCG@5/10/20
- Latency percentiles (`p50`, `p95`, `p99`) in milliseconds

### 5. Generate Benchmark Report (latest vs previous run)

Create a markdown report with:

- Latest run summary
- Delta vs previous run
- Pass/fail gates (latency and quality regressions)

```bash
python3 benchmark_report.py \
	--results-path ./benchmark_results.csv \
	--output-md ./benchmark_report.md \
	--max-p95-ms 150 \
	--max-p99-ms 250 \
	--max-recall10-drop-pct 5 \
	--max-ndcg10-drop-pct 5
```

### 6. One Command: Benchmark + Report

```bash
python3 run_benchmark_pipeline.py \
	--sample-users 600 \
	--benchmark-step 150 \
	--results-path ./benchmark_results.csv \
	--report-path ./benchmark_report.md
```

## Notes On Performance

- The notebook has low-memory switches for constrained machines.
- Parquet caching can speed repeated runs but may temporarily increase memory usage while writing.
- The scripts are chunked and designed to avoid loading the full ratings table in memory.

## Outputs

Primary generated artifacts:

- `movie_tags.npz`
- `users_preferences.dat`
- `user_index_to_id.npy`
- `benchmark_results.csv`
- `benchmark_report.md`

These files are reused by both the notebook and evaluation script.
