from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import hybrid_rust as _hybrid_rust
except ImportError:
    _hybrid_rust = None


CHUNK = 500_000


def has_rust_extension() -> bool:
    return _hybrid_rust is not None


def matmul_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32 = np.ascontiguousarray(a, dtype=np.float32)
    b32 = np.ascontiguousarray(b, dtype=np.float32)
    if _hybrid_rust is None:
        return a32 @ b32
    return _hybrid_rust.matmul_f32(a32, b32)


def aggregate_ratings_csv(csv_path: Path, max_movie_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    if _hybrid_rust is not None:
        cnt, sm, max_uid = _hybrid_rust.aggregate_ratings_csv(str(csv_path), int(max_movie_id))
        return np.asarray(cnt, dtype=np.int64), np.asarray(sm, dtype=np.float64), int(max_uid)

    movie_sum = np.zeros(max_movie_id + 1, dtype=np.float64)
    movie_cnt = np.zeros(max_movie_id + 1, dtype=np.int64)
    max_uid = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "uint32", "movieId": "uint32", "rating": "float32"},
        chunksize=CHUNK,
    ):
        uids = chunk["userId"].to_numpy(dtype=np.int32, copy=False)
        mids = chunk["movieId"].to_numpy(dtype=np.int32, copy=False)
        rats = chunk["rating"].to_numpy(dtype=np.float64, copy=False)

        cmax = int(uids.max()) if len(uids) else 0
        if cmax > max_uid:
            max_uid = cmax

        valid = mids <= max_movie_id
        if not np.all(valid):
            mids = mids[valid]
            rats = rats[valid]

        movie_sum += np.bincount(mids, weights=rats, minlength=max_movie_id + 1)
        movie_cnt += np.bincount(mids, minlength=max_movie_id + 1)

    return movie_cnt, movie_sum, max_uid


def build_user_preferences_from_csv(
    ratings_csv_path: Path,
    genome_scores_csv_path: Path,
    max_active_users: int,
    rating_scale: float,
    num_movies: int,
    num_tags: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _hybrid_rust is None:
        raise RuntimeError(
            "Rust extension not available. Install with: pip install -e ./rust_ext"
        )

    user_index_to_id, users_preferences = _hybrid_rust.build_user_preferences_from_csv(
        str(ratings_csv_path),
        str(genome_scores_csv_path),
        int(max_active_users),
        float(rating_scale),
        int(num_movies),
        int(num_tags),
    )

    return (
        np.asarray(user_index_to_id, dtype=np.int32),
        np.asarray(users_preferences, dtype=np.float32),
    )
