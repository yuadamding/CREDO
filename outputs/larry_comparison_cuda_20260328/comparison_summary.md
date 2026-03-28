# LARRY Benchmark Summary

Output directory: `/home/yding1995/opscc_sc/CAPE/outputs/larry_comparison_cuda_20260328`

| Metric | CAPE | scDiffeq |
| --- | ---: | ---: |
| Train time (s) | 23.1 | 555.9 |
| Eval time (s) | 0.1 | 0.1 |
| Fate sim time (s) | 0.2 | 0.1 |
| Peak GPU mem (MB) | 81.7 | 2227.6 |
| Sinkhorn t=4 (PCA) | 23.6652 | 51.7195 |
| Sinkhorn t=6 (PCA) | 21.6135 | 69.5642 |
| Fate mono frac | 0.0000 | 0.0000 |

CAPE training speedup vs scDiffeq: 24.06x
