# Synthetic example

This fixture has two donors, two non-targeting guides, two target genes with
two guides each, three physical checkpoints, complete donor-time count blocks,
and one deliberately missing downstream guide-time measure.

```bash
python examples/synthetic/generate.py
credo validate examples/synthetic/config.yaml
credo run examples/synthetic/config.yaml --device cpu
```

It is a numerical contract test, not a biological benchmark.
