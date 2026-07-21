# GSE314342 adapter

This adapter consumes the downloaded primary human CD4+ T-cell Perturb-seq
support release in the repository parent's `inputs/GSE314342` directory. It
emits opaque donor-guide `measure_id` values, target-gene `embedding_id` values,
donor context groups,
relative-within-donor mass, and complete donor-time count blocks. No
GSE-specific branch enters `src/credo`.

The modeled physical interval starts at the already perturbed resting state;
it does not identify the unobserved perturbation-to-resting transition. The
late checkpoint is labeled 48 hours according to the processed author
metadata. Relative mass supports within-donor expansion comparisons, not
absolute population-growth claims.

```bash
python examples/gse314342/prepare.py --pilot
credo validate examples/gse314342/config.yaml
credo run examples/gse314342/config.yaml
```

Remove `--pilot` and point the config to `canonical/` for the full cohort.
The D3-only pilot cannot support a leakage-free donor holdout while using
compositional counts, so its manifest correctly reports `train_self_eval`;
the full four-donor cohort supports complete-donor validation.
The complete conversion retains 91,848 measures with resting source support and
excludes 5,412 downstream-only IDs without renormalizing their original
whole-library donor-time mass denominator.
