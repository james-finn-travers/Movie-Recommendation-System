# Benchmark Report

Latest run timestamp: 2026-04-11T03:45:07
Users evaluated: 500
Seed: 42

## Latest Metrics

| Metric | Value |
|---|---:|
| Precision@10 | 0.0014 |
| Recall@10 | 0.0140 |
| NDCG@10 | 0.0080 |
| Latency p50 (ms) | 2.28 |
| Latency p95 (ms) | 7.50 |
| Latency p99 (ms) | 12.65 |
| Total runtime (s) | 45.00 |

## Delta vs Previous Run

| Metric | Previous | Latest | Delta |
|---|---:|---:|---:|
| precision_at_10 | 0.0014 | 0.0014 | +0.00% |
| recall_at_10 | 0.0140 | 0.0140 | +0.00% |
| ndcg_at_10 | 0.0080 | 0.0080 | +0.00% |
| latency_p50_ms | 2.6643 | 2.2752 | -14.60% |
| latency_p95_ms | 6.7205 | 7.5039 | +11.66% |
| latency_p99_ms | 9.7961 | 12.6473 | +29.11% |

## Gate Checks

| Gate | Status | Details |
|---|---|---|
| Latency p95 | PASS | 7.50 ms <= 150.00 ms |
| Latency p99 | PASS | 12.65 ms <= 250.00 ms |
| Recall@10 regression | PASS | drop=0.00% <= 5.00% |
| NDCG@10 regression | PASS | drop=0.00% <= 5.00% |

