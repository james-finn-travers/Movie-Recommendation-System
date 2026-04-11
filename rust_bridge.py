from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import hybrid_rust
except ImportError as exc:  # pragma: no cover
    hybrid_rust = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_rust_module() -> Any:
    if hybrid_rust is None:
        raise ImportError(
            "hybrid_rust extension is not installed. Build it with maturin from rust_ext/."
        ) from _IMPORT_ERROR
    return hybrid_rust


def build_user_preferences_from_csv(
    ratings_csv_path: str | Path,
    genome_scores_csv_path: str | Path,
    max_active_users: int,
    rating_scale: float,
    num_movies: int,
    num_tags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build user preference vectors with Rust and return NumPy arrays.

    Returns:
        tuple: (user_index_to_id, users_preferences)
            - user_index_to_id: np.ndarray[uint32] with shape (n_users,)
            - users_preferences: np.ndarray[float32] with shape (n_users, num_tags + 1)
    """
    ext = _require_rust_module()
    if hasattr(ext, "populate_user_preferences_matrix"):
        user_ids, preferences = ext.populate_user_preferences_matrix(
            str(ratings_csv_path),
            str(genome_scores_csv_path),
            int(max_active_users),
            float(rating_scale),
            int(num_movies),
            int(num_tags),
        )
    else:
        user_ids, preferences = _build_user_preferences_python_fallback(
            ratings_csv_path=ratings_csv_path,
            genome_scores_csv_path=genome_scores_csv_path,
            max_active_users=max_active_users,
            rating_scale=rating_scale,
            num_movies=num_movies,
            num_tags=num_tags,
        )

    user_index_to_id = np.asarray(user_ids, dtype=np.uint32)
    users_preferences = np.asarray(preferences, dtype=np.float32)
    return user_index_to_id, users_preferences


def evaluate_model_with_rust_wrapper(
    script_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run evaluate_model.py through the Rust subprocess wrapper.

    Keyword arguments are converted into CLI flags.
    Example: sample_users=500 -> --sample-users 500
    """
    ext = _require_rust_module()

    args: list[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        args.append(flag)
        args.append(str(value))

    code, stdout, stderr = ext.evaluate_model_wrapper(str(script_path), args)
    result = {
        "return_code": int(code),
        "stdout": stdout,
        "stderr": stderr,
        "args": args,
    }
    if code != 0:
        raise RuntimeError(
            f"Rust evaluation wrapper failed with exit code {code}. stderr:\n{stderr.strip()}"
        )
    return result


def _build_user_preferences_python_fallback(
    ratings_csv_path: str | Path,
    genome_scores_csv_path: str | Path,
    max_active_users: int,
    rating_scale: float,
    num_movies: int,
    num_tags: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_active_users <= 0:
        raise ValueError("max_active_users must be greater than 0")
    if rating_scale <= 0:
        raise ValueError("rating_scale must be greater than 0")

    n_tags = int(num_tags) + 1
    movie_to_tags: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for chunk in pd.read_csv(
        genome_scores_csv_path,
        usecols=["movieId", "tagId", "relevance"],
        dtype={"movieId": "uint32", "tagId": "uint32", "relevance": "float32"},
        chunksize=1_000_000,
    ):
        mids = chunk["movieId"].to_numpy(dtype=np.int32, copy=False)
        tids = chunk["tagId"].to_numpy(dtype=np.int32, copy=False)
        rel = chunk["relevance"].to_numpy(dtype=np.float32, copy=False)

        valid = (mids <= int(num_movies)) & (tids <= int(num_tags))
        mids = mids[valid]
        tids = tids[valid]
        rel = rel[valid]

        if len(mids) == 0:
            continue

        order = np.argsort(mids, kind="mergesort")
        mids_s = mids[order]
        tids_s = tids[order]
        rel_s = rel[order]

        split_idx = np.flatnonzero(np.diff(mids_s)) + 1
        mid_parts = np.split(mids_s, split_idx)
        tid_parts = np.split(tids_s, split_idx)
        rel_parts = np.split(rel_s, split_idx)

        for m_part, t_part, r_part in zip(mid_parts, tid_parts, rel_parts):
            mid = int(m_part[0])
            prev = movie_to_tags.get(mid)
            if prev is None:
                movie_to_tags[mid] = (t_part.astype(np.int32, copy=True), r_part.astype(np.float32, copy=True))
            else:
                movie_to_tags[mid] = (
                    np.concatenate([prev[0], t_part.astype(np.int32, copy=False)]),
                    np.concatenate([prev[1], r_part.astype(np.float32, copy=False)]),
                )

    user_id_to_idx: dict[int, int] = {}
    user_ids: list[int] = []
    numerators: list[np.ndarray] = []
    denominators: list[np.ndarray] = []

    for chunk in pd.read_csv(
        ratings_csv_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "uint32", "movieId": "uint32", "rating": "float32"},
        chunksize=1_000_000,
    ):
        uids = chunk["userId"].to_numpy(dtype=np.int32, copy=False)
        mids = chunk["movieId"].to_numpy(dtype=np.int32, copy=False)
        rats = chunk["rating"].to_numpy(dtype=np.float32, copy=False) / float(rating_scale)

        for uid, mid, rat in zip(uids, mids, rats):
            tags = movie_to_tags.get(int(mid))
            if tags is None:
                continue

            user_idx = user_id_to_idx.get(int(uid))
            if user_idx is None:
                if len(user_ids) >= int(max_active_users):
                    continue
                user_idx = len(user_ids)
                user_id_to_idx[int(uid)] = user_idx
                user_ids.append(int(uid))
                numerators.append(np.zeros(n_tags, dtype=np.float32))
                denominators.append(np.zeros(n_tags, dtype=np.float32))

            tag_ids, relevance = tags
            numerators[user_idx][tag_ids] += rat * relevance
            denominators[user_idx][tag_ids] += relevance

    n_users = len(user_ids)
    preferences = np.zeros((n_users, n_tags), dtype=np.float32)
    for i in range(n_users):
        np.divide(
            numerators[i],
            np.where(denominators[i] != 0, denominators[i], 1.0),
            out=preferences[i],
        )

    return np.asarray(user_ids, dtype=np.uint32), preferences
