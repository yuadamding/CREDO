# HNSCC P4/P60 adapter

The adapter pools confident guide-assigned cells by guide and checkpoint,
maps guides to shared target-gene embedding identities, shares one exact
control reference, and caps only geometric support. Mass remains the uncapped
captured-cell count.

Captured-count mass supports capture-scale expansion and depletion diagnostics;
it is not an absolute tumor-population size and must not be reported as
absolute growth.

```bash
python examples/hnscc_p4_p60/prepare.py
credo validate examples/hnscc_p4_p60/config.yaml
credo train examples/hnscc_p4_p60/config.yaml
```
