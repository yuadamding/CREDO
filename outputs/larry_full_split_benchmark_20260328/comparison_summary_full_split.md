# Full-LARRY Fixed-Split Benchmark Summary

Output root: `/home/yding1995/opscc_sc/CAPE/outputs/larry_full_split_benchmark_20260328`

| Metric | CAPE | scDiffeq |
| --- | ---: | ---: |
| Train time (s) | 10.9 | 2553.1 |
| Eval time (s) | 0.0 | 0.2 |
| Peak GPU mem (MB) | 99.5 | n/a |
| Sinkhorn t=4 (test PCA) | 59.4272 | 87.4851 |
| Sinkhorn t=6 (test PCA) | 43.6487 | 123.3869 |
| t6 label TV distance | 0.2166 | 0.3495 |

CAPE training speedup vs scDiffeq: 234.23x
