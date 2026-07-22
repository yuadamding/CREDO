# Archived transformer-v2 LPS replay

This adapter imports the four preserved donor-fold checkpoints into the stable
CREDO runtime and recreates one standardized 268-row OOF metric table. It uses
the preserved VAE latent cache; it never refits the representation.

```bash
credo replay \
  --bundle-root ../trained_models/LPS_checkpoints/lps_oof_tx_ind32_refine500_vae40/oof_inference_bundle \
  --study-source ../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad \
  --output ../trained_models/LPS_checkpoints/lps_oof_tx_ind32_refine500_vae40/credo_replay \
  --device cuda
```

Each fold writes a portable imported bundle, common `metrics.parquet`, and
`comparison.json`. The root writes `oof_metrics.parquet` and
`replay_manifest.json`. Source checkpoints are read and hash-verified but never
modified.

The generated complete-shape legacy fixture runs in normal locked CPU CI. The
full 268-row BF16 replay requires the local archived bundle and CUDA study input;
it is recorded evidence rather than a publicly downloadable scheduled job.

The shared source-only VAE is transductive with respect to donor identity.
These outputs support donor-held-out dynamics claims conditional on that
representation, not fully nested donor generalization or exact retraining.
