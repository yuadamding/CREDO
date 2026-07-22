# GSE314342 adapter

This example consumes the quality-tiered primary human CD4+ T-cell Perturb-seq
release in the repository parent's `inputs/GSE314342/canonical_gold`
directory. It uses opaque donor-guide `measure_id` values, shared target-gene
embeddings, relative-within-donor mass, and conditionally complete donor-time
count blocks. No GSE-specific branch enters `src/credo`.

The modeled physical interval starts at the already perturbed resting state;
it does not identify the unobserved perturbation-to-resting transition. The
late checkpoint is labeled 48 hours according to the processed author
metadata. Relative mass supports within-donor expansion comparisons, not
absolute population-growth claims.

```bash
credo validate examples/gse314342/config.yaml
credo run examples/gse314342/config.yaml
```

The workspace builder `../scripts/prepare_gse314342_credo.py` derives the gold,
early, late, silver, and null catalogs from the base conversion. The primary
config keeps ecological context disabled, uses the audited `growth_max: 9.2`,
and target-balances optimization batches. The four-donor cohort supports
explicit complete-donor validation; `validation.strategy: checkpoint` can hold
out 8 hours for a non-strict benchmark. A strict benchmark additionally needs
the separate Rest-plus-48-hour encoder described in the input audit.
