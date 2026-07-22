# Archived transformer-v2 LPS replay

This adapter imports the four preserved donor-fold checkpoints into the stable
CREDO runtime and recreates one standardized 268-row OOF metric table. It uses
the preserved VAE latent cache; it never refits the representation.

```bash
python examples/lps_v2_replay/replay.py \
  --bundle-root ../trained_models/LPS_checkpoints/lps_oof_tx_ind32_refine500_vae40/oof_inference_bundle \
  --study-source ../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad \
  --output ../trained_models/LPS_checkpoints/lps_oof_tx_ind32_refine500_vae40/credo_replay \
  --device cuda
```

Each fold writes an inference-only `envelope.json`, common `metrics.parquet`,
and `comparison.json`. The root writes `oof_metrics.parquet` and
`replay_manifest.json`. Source checkpoints are read and hash-verified but never
modified.

The shared source-only VAE is transductive with respect to donor identity.
These outputs support donor-held-out dynamics claims conditional on that
representation, not fully nested donor generalization or exact retraining.
