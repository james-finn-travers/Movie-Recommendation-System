"""
Evaluate the User-User CF recommendation model (ultra-low-memory).

Designed to run with < 1.5 GB.  All CSV processing uses tiny chunks
with numpy bincount instead of pandas groupby.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/home/james/Movie-Recommendation-System")
CHUNK = 500_000  # small to keep peak memory low


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "ml-latest")
    p.add_argument("--movie-tags", type=Path, default=ROOT / "movie_tags.npz")
    p.add_argument("--prefs", type=Path, default=ROOT / "users_preferences.dat")
    p.add_argument("--user-index", type=Path, default=ROOT / "user_index_to_id.npy")
    p.add_argument("--sample-users", type=int, default=500)
    p.add_argument("--benchmark-step", type=int, default=150)
    p.add_argument("--min-ratings", type=int, default=20)
    p.add_argument("--min-pos-rating", type=float, default=3.5)
    p.add_argument("--k-neighbors", type=int, default=25)
    p.add_argument("--min-neighbor-votes", type=int, default=2)
    p.add_argument("--results-path", type=Path, default=ROOT / "benchmark_results.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    csv_path = args.data_dir / "ratings.csv"

    print("Loading movie_tags...", flush=True)
    movie_tags = sparse.load_npz(args.movie_tags).tocsr()
    user_index_to_id = np.load(args.user_index)

    # ── Pass 1: stats via numpy only ──────────────────────────────────────────
    print("Pass 1: stats (numpy bincount, no groupby)...", flush=True)
    t1 = time.time()

    # Pre-allocate: max movieId = movie_tags.shape[0]-1, max userId ~ 330K
    max_mid = movie_tags.shape[0]  # 288984
    movie_rating_sum = np.zeros(max_mid + 1, dtype=np.float64)
    movie_rating_cnt = np.zeros(max_mid + 1, dtype=np.int32)

    # First mini-pass: find max userId
    max_uid = 0
    for chunk in pd.read_csv(csv_path, usecols=["userId"],
                             dtype={"userId": "uint32"}, chunksize=CHUNK):
        cmax = int(chunk["userId"].max())
        if cmax > max_uid:
            max_uid = cmax
        del chunk
    gc.collect()

    user_count = np.zeros(max_uid + 1, dtype=np.int32)
    user_last_ts = np.zeros(max_uid + 1, dtype=np.uint32)
    user_last_mid = np.zeros(max_uid + 1, dtype=np.int32)
    user_last_rat = np.zeros(max_uid + 1, dtype=np.float32)

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK,
                             dtype={"userId": "uint32", "movieId": "uint32",
                                    "rating": "float32", "timestamp": "uint32"}):
        uids = chunk["userId"].to_numpy(dtype=np.int32, copy=False)
        mids = chunk["movieId"].to_numpy(dtype=np.int32, copy=False)
        rats = chunk["rating"].to_numpy(dtype=np.float64, copy=False)
        ts = chunk["timestamp"].to_numpy(dtype=np.uint32, copy=False)
        del chunk
        gc.collect()

        # Per-movie: accumulate with bincount
        valid_mid = mids <= max_mid
        if not np.all(valid_mid):
            v_mids = mids[valid_mid]
            v_rats = rats[valid_mid]
        else:
            v_mids = mids
            v_rats = rats

        movie_rating_sum[:len(movie_rating_sum)] += np.bincount(
            v_mids, weights=v_rats, minlength=max_mid + 1
        )
        movie_rating_cnt[:len(movie_rating_cnt)] += np.bincount(
            v_mids, minlength=max_mid + 1
        ).astype(np.int32)

        # Per-user: count
        np.add.at(user_count, uids, 1)

        # Per-user: track latest timestamp
        newer = ts > user_last_ts[uids]
        if np.any(newer):
            idx = np.where(newer)[0]
            u_new = uids[idx]
            user_last_ts[u_new] = ts[idx]
            user_last_mid[u_new] = mids[idx]
            user_last_rat[u_new] = rats[idx].astype(np.float32)

    # Build eligible set
    with np.errstate(divide="ignore", invalid="ignore"):
        movie_avg = np.where(movie_rating_cnt > 0,
                             movie_rating_sum / movie_rating_cnt, 0.0)
    eligible_mask = (movie_rating_cnt >= 100) & (movie_avg >= 3.0)
    eligible_mids = np.flatnonzero(eligible_mask)
    eligible = set(int(m) for m in eligible_mids)
    print(f"  Eligible movies: {len(eligible):,}", flush=True)

    del movie_rating_sum, movie_rating_cnt, movie_avg, eligible_mask, eligible_mids
    gc.collect()

    print(f"  Pass 1 done in {time.time() - t1:.1f}s", flush=True)

    # ── Select test users ─────────────────────────────────────────────────────
    test_mask = (user_count >= args.min_ratings) & (user_last_rat >= args.min_pos_rating)
    candidate_uids = np.flatnonzero(test_mask)
    # Filter: last movie in eligible
    elig_last = np.array([int(user_last_mid[u]) in eligible for u in candidate_uids])
    candidate_uids = candidate_uids[elig_last]

    n_sample = min(args.sample_users, len(candidate_uids))
    sampled_uids = rng.choice(candidate_uids, size=n_sample, replace=False)
    sampled_set = set(int(u) for u in sampled_uids)
    held_out = {int(u): int(user_last_mid[u]) for u in sampled_uids}
    held_out_ts_map = {int(u): int(user_last_ts[u]) for u in sampled_uids}
    print(f"Test candidates: {len(candidate_uids):,} | sampled: {n_sample}", flush=True)

    del user_count, user_last_ts, user_last_mid, user_last_rat
    del test_mask, candidate_uids
    gc.collect()

    # ── Pass 2: Load ONLY sampled users' histories ────────────────────────────
    print("Pass 2: sampled users' histories...", flush=True)
    t2 = time.time()
    user_hist: dict[int, list[tuple[int, float]]] = {u: [] for u in sampled_set}

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK,
                             dtype={"userId": "uint32", "movieId": "uint32",
                                    "rating": "float32", "timestamp": "uint32"}):
        uids = chunk["userId"].to_numpy(dtype=np.int32, copy=False)
        mask = np.isin(uids, list(sampled_set))
        if not np.any(mask):
            del chunk
            gc.collect()
            continue
        sub = chunk[mask]
        for row in sub.itertuples(index=False):
            uid = int(row.userId)
            ts = int(row.timestamp)
            mid = int(row.movieId)
            # Skip the held-out rating
            if ts >= held_out_ts_map[uid] and mid == held_out[uid]:
                continue
            user_hist[uid].append((mid, float(row.rating) / 5.0))
        del chunk, sub
        gc.collect()

    del held_out_ts_map
    print(f"  Done in {time.time()-t2:.1f}s", flush=True)

    # ── Build preference vectors ──────────────────────────────────────────────
    print("Building preference vectors...", flush=True)
    t3 = time.time()
    n_tags = movie_tags.shape[1]
    test_prefs = np.zeros((n_sample, n_tags), dtype=np.float32)
    test_seen: list[set[int]] = []
    uid_order = [int(u) for u in sampled_uids]

    for j, uid in enumerate(uid_order):
        hist = user_hist[uid]
        if not hist:
            test_seen.append(set())
            continue
        mids_a = np.array([h[0] for h in hist], dtype=np.int32)
        rats_a = np.array([h[1] for h in hist], dtype=np.float32)
        test_seen.append(set(int(m) for m in mids_a))

        valid = mids_a < movie_tags.shape[0]
        m = mids_a[valid]
        r = rats_a[valid]
        if len(m) == 0:
            continue
        r_sp = sparse.csr_matrix(
            (r, (np.zeros(len(m), dtype=np.int32), m)),
            shape=(1, movie_tags.shape[0]), dtype=np.float32,
        )
        num = (r_sp @ movie_tags).toarray().ravel()
        r_bin = r_sp.copy()
        r_bin.data[:] = 1.0
        den = (r_bin @ movie_tags).toarray().ravel()
        msk = den != 0
        np.divide(num, den, out=test_prefs[j], where=msk)

    del user_hist
    gc.collect()
    print(f"  Built {n_sample} vectors in {time.time()-t3:.1f}s", flush=True)

    # ── Batch KNN ─────────────────────────────────────────────────────────────
    print("Batch KNN query...", flush=True)
    t4 = time.time()
    uprefs = np.memmap(args.prefs, dtype=np.float32, mode="r",
                       shape=(len(user_index_to_id), movie_tags.shape[1]))
    nn = NearestNeighbors(n_neighbors=args.k_neighbors, metric="cosine", algorithm="brute")
    nn.fit(uprefs)
    all_dist, all_idx = nn.kneighbors(test_prefs)
    del uprefs, nn, test_prefs
    gc.collect()
    print(f"  KNN done in {time.time()-t4:.1f}s", flush=True)

    # Which neighbor userIds?
    needed = set()
    for j in range(n_sample):
        for k in all_idx[j]:
            needed.add(int(user_index_to_id[k]))
    print(f"  Unique neighbors: {len(needed):,}", flush=True)

    # ── Pass 3: Load neighbor ratings ─────────────────────────────────────────
    print("Pass 3: neighbor ratings...", flush=True)
    t5 = time.time()
    nb_mids: dict[int, list[int]] = {u: [] for u in needed}
    nb_rats: dict[int, list[float]] = {u: [] for u in needed}

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK,
                             usecols=["userId", "movieId", "rating"],
                             dtype={"userId": "uint32", "movieId": "uint32",
                                    "rating": "float32"}):
        uids = chunk["userId"].to_numpy(dtype=np.int32, copy=False)
        mask = np.isin(uids, list(needed))
        if not np.any(mask):
            del chunk
            gc.collect()
            continue
        sub_uids = uids[mask]
        sub_mids = chunk["movieId"].to_numpy(dtype=np.int32, copy=False)[mask]
        sub_rats = chunk["rating"].to_numpy(dtype=np.float32, copy=False)[mask] / 5.0
        for idx in range(len(sub_uids)):
            u = int(sub_uids[idx])
            nb_mids[u].append(int(sub_mids[idx]))
            nb_rats[u].append(float(sub_rats[idx]))
        del chunk
        gc.collect()

    nb_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for u in needed:
        if nb_mids[u]:
            nb_data[u] = (np.array(nb_mids[u], dtype=np.int32),
                          np.array(nb_rats[u], dtype=np.float32))
    del nb_mids, nb_rats
    gc.collect()
    print(f"  Loaded {len(nb_data):,} neighbors in {time.time()-t5:.1f}s", flush=True)

    # ── Score & evaluate ──────────────────────────────────────────────────────
    print("Scoring and benchmarking...", flush=True)
    t6 = time.time()
    Ns = [5, 10, 20]
    precision_at_k = {n: [] for n in Ns}
    recall_at_k = {n: [] for n in Ns}
    ndcg_at_k = {n: [] for n in Ns}
    latency_ms: list[float] = []

    for j in range(n_sample):
        t_infer = time.perf_counter()
        uid = uid_order[j]
        gt = held_out[uid]
        seen = test_seen[j]

        sims = np.maximum(1.0 - all_dist[j], 0.0)
        n_uids = [int(user_index_to_id[k]) for k in all_idx[j]]

        ch_m, ch_r, ch_s, ch_u = [], [], [], []
        for nu, sim in zip(n_uids, sims):
            nd = nb_data.get(nu)
            if nd is None:
                continue
            nm, nr = nd
            n = len(nm)
            ch_m.append(nm); ch_r.append(nr)
            ch_s.append(np.full(n, sim, dtype=np.float32))
            ch_u.append(np.full(n, nu, dtype=np.int32))

        if ch_m:
            cat_m = np.concatenate(ch_m)
            cat_r = np.concatenate(ch_r)
            cat_s = np.concatenate(ch_s)
            cat_u = np.concatenate(ch_u)

            w = cat_r * cat_s
            uniq, inv = np.unique(cat_m, return_inverse=True)
            nu_ = len(uniq)
            ws = np.bincount(inv, weights=w, minlength=nu_).astype(np.float32)
            wt = np.bincount(inv, weights=cat_s, minlength=nu_).astype(np.float32)

            pairs = inv.astype(np.int64) * 1_000_000 + cat_u.astype(np.int64)
            up = np.unique(pairs)
            vm = (up // 1_000_000).astype(np.int32)
            nv = np.bincount(vm, minlength=nu_)

            sc = np.where(wt > 0, ws / wt, 0.0)
            ok = ((nv >= args.min_neighbor_votes)
                  & np.array([int(m) in eligible for m in uniq])
                  & ~np.array([int(m) in seen for m in uniq]))
            oi = np.flatnonzero(ok)
            if len(oi) > 0:
                order = np.argsort(sc[oi])[::-1]
                knn_r = [int(uniq[oi[k]]) for k in order]
            else:
                knn_r = []
        else:
            knn_r = []

        latency_ms.append((time.perf_counter() - t_infer) * 1000.0)

        for n in Ns:
            kt = knn_r[:n]
            hit = float(gt in kt)
            precision_at_k[n].append(hit / float(n))
            # One held-out relevant item per user => Recall@K equals hit rate.
            recall_at_k[n].append(hit)
            ndcg_at_k[n].append(1.0 / np.log2(kt.index(gt) + 2) if gt in kt else 0.0)

        if (j + 1) % 100 == 0:
            print(f"  [{j+1}/{n_sample}] {(j+1)/(time.time()-t6):.1f} users/sec",
                  flush=True)

    total = time.time() - t0
    print(f"\nDone in {total:.1f}s | {n_sample} users\n")

    step = max(1, args.benchmark_step)
    checkpoints = list(range(step, n_sample + 1, step))
    if not checkpoints or checkpoints[-1] != n_sample:
        checkpoints.append(n_sample)

    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for users_eval in checkpoints:
        lat_prefix = np.array(latency_ms[:users_eval], dtype=np.float64)
        p50, p95, p99 = np.percentile(lat_prefix, [50, 95, 99])

        row = {
            "run_timestamp": run_ts,
            "seed": int(args.seed),
            "users_evaluated": int(users_eval),
            "benchmark_step": int(step),
            "k_neighbors": int(args.k_neighbors),
            "min_neighbor_votes": int(args.min_neighbor_votes),
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "latency_p99_ms": float(p99),
            "total_runtime_sec": float(total),
        }
        for n in Ns:
            row[f"precision_at_{n}"] = float(np.mean(precision_at_k[n][:users_eval]))
            row[f"recall_at_{n}"] = float(np.mean(recall_at_k[n][:users_eval]))
            row[f"ndcg_at_{n}"] = float(np.mean(ndcg_at_k[n][:users_eval]))
        rows.append(row)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.results_path.exists()
    fieldnames = [
        "run_timestamp",
        "seed",
        "users_evaluated",
        "benchmark_step",
        "k_neighbors",
        "min_neighbor_votes",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "precision_at_20",
        "recall_at_20",
        "ndcg_at_20",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "total_runtime_sec",
    ]
    with args.results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    hdr = (
        f"{'Users':>6} | {'P@10':>7} | {'R@10':>7} | {'NDCG@10':>8} | "
        f"{'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print(
            f"{row['users_evaluated']:>6} | "
            f"{row['precision_at_10']:>7.4f} | "
            f"{row['recall_at_10']:>7.4f} | "
            f"{row['ndcg_at_10']:>8.4f} | "
            f"{row['latency_p50_ms']:>8.2f} | "
            f"{row['latency_p95_ms']:>8.2f} | "
            f"{row['latency_p99_ms']:>8.2f}"
        )
    print(sep)
    print(f"Saved benchmark rows to: {args.results_path}")
    print(f"({n_sample} users max, step={step}, seed={args.seed})")


if __name__ == "__main__":
    main()
