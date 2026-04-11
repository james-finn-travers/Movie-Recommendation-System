# Benchmark Report

Latest run timestamp: 2026-04-11T02:20:41
Users evaluated: 500
Seed: 42

## Latest Metrics

| Metric | Value |
|---|---:|
| Precision@10 | 0.0004 |
| Recall@10 | 0.0040 |
| NDCG@10 | 0.0012 |
| Latency p50 (ms) | 2.71 |
| Latency p95 (ms) | 5.31 |
| Latency p99 (ms) | 7.67 |
| Total runtime (s) | 39.24 |

## Delta vs Previous Run

| Metric | Previous | Latest | Delta |
|---|---:|---:|---:|
| precision_at_10 | 0.0004 | 0.0004 | +0.00% |
| recall_at_10 | 0.0040 | 0.0040 | +0.00% |
| ndcg_at_10 | 0.0012 | 0.0012 | +0.00% |
| latency_p50_ms | 3.5511 | 2.7082 | -23.73% |
| latency_p95_ms | 9.2398 | 5.3134 | -42.49% |
| latency_p99_ms | 17.6219 | 7.6699 | -56.47% |

## Gate Checks

| Gate | Status | Details |
|---|---|---|
| Latency p95 | PASS | 5.31 ms <= 150.00 ms |
| Latency p99 | PASS | 7.67 ms <= 250.00 ms |
| Recall@10 regression | PASS | drop=0.00% <= 5.00% |
| NDCG@10 regression | PASS | drop=0.00% <= 5.00% |

