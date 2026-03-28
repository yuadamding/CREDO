# CAPE HNSCC P4/P60 Run

Output dir: `/home/yding1995/opscc_sc/CAPE/outputs/hnscc_cape_full_wta_vram10gb_20260328_181653`
Data path: `/home/yding1995/opscc_sc/scDiffeq/hnscc/GSE235325_P4P60_scdiffeq_compatible.h5ad`

- Guide-confident only: `True`
- WTA column: `Library`
- Train WTAs: `wta13, wta14, wta15, wta16, wta17, wta18, wta4, wta5, wta6, wta7, wta9`
- Test WTAs: `wta10, wta11, wta12, wta8`
- Train particles / steps: `128` / `16`
- Eval particles / steps: `384` / `24`
- Eval target atoms per perturbation: `768`
- Weak-form test functions: `12`
- Weak loss weight: `0.1`
- Max train target atoms per perturbation: `768`
- Supported perturbations: `121`
- Control ids: `ctrl`
- Train time (s): `470.6`
- Train peak GPU allocated / reserved (MB): `7547.0` / `8282.0`
- Eval peak GPU allocated / reserved (MB): `515.7` / `780.0`

## Train Endpoint Summary

- Mean UOT: `140.6668`
- Median UOT: `140.0587`
- Mean mass rel error: `0.3543`

## Test Endpoint Summary

- Mean UOT: `213.2774`
- Median UOT: `211.0581`
- Mean mass rel error: `0.5608`
