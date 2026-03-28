# CAPE HNSCC P4/P60 Run

Output dir: `/home/yding1995/opscc_sc/CAPE/outputs/hnscc_cape_full_wta_calibration_default_20260328`
Data path: `/home/yding1995/opscc_sc/scDiffeq/hnscc/GSE235325_P4P60_scdiffeq_compatible.h5ad`

- Guide-confident only: `True`
- WTA column: `Library`
- Train WTAs: `wta13, wta14, wta15, wta16, wta17, wta18, wta4, wta5, wta6, wta7, wta9`
- Test WTAs: `wta10, wta11, wta12, wta8`
- Train particles / steps: `64` / `12`
- Eval particles / steps: `256` / `24`
- Eval target atoms per perturbation: `512`
- Weak-form test functions: `8`
- Weak loss weight: `0.1`
- Max train target atoms per perturbation: `512`
- Supported perturbations: `121`
- Control ids: `ctrl`
- Train time (s): `2.5`
- Train peak GPU allocated / reserved (MB): `2376.8` / `2438.0`
- Eval peak GPU allocated / reserved (MB): `352.4` / `420.0`

## Train Endpoint Summary

- Mean UOT: `455.9013`
- Median UOT: `455.2589`
- Mean mass rel error: `0.4614`

## Test Endpoint Summary

- Mean UOT: `456.5908`
- Median UOT: `458.9383`
- Mean mass rel error: `7.8849`
