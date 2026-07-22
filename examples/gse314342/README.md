# GSE314342 adapter

This example consumes the authors-aligned primary human CD4+ T-cell
Perturb-seq input in the repository parent's
`inputs/GSE314342/canonical_authors` directory. All 12 cell objects are
rescanned with the authors' post-QC unique-guide filter, their pseudobulk and
DE eligibility rules are replayed, and passing Rest rows define finite
donor-guide measures. No GSE-specific branch enters `src/credo`.

The modeled physical interval starts at the already perturbed resting state;
it does not identify the unobserved perturbation-to-resting transition. The
late checkpoint is labeled 48 hours according to the processed author
metadata. Relative mass supports within-donor expansion comparisons, not
absolute population-growth claims.

```bash
credo validate examples/gse314342/config.yaml
credo train examples/gse314342/config.yaml
```

The [workspace preprocessor](../../../scripts/preprocess_gse314342_authors.py)
recomputes the authors' post-QC pseudobulks and final `keep_for_DE` decisions.
The [workspace builder](../../../scripts/prepare_gse314342_credo.py) converts
that population into the canonical authors input and also derives the local
gold, early, late, silver, and null catalogs. Every
`dataset.json` is schema 2 and binds the support file to the preserved VAE
encoder, decoder, gene set, normalization, fitting cohort, and artifact hashes.
CREDO verifies that contract and reads the dense support lazily through a
bounded cache.

The five local quality catalogs are CREDO-derived efficacy cohorts: they use
released condition-specific guide or target expression statistics and do not
reproduce the authors' `keep_for_DE` filter. They are appropriate for
confirmed-perturbation sensitivity analyses, not unbiased discovery or strict
checkpoint holdouts. The workspace's
[authors-processing audit](../../../inputs/GSE314342/provenance/authors_processing_audit.md)
records the exact upstream crosswalk and count reconciliation.

The primary config keeps ecological context disabled, uses the audited
`growth_max: 9.4`, and uses target round-robin optimization batches. Complete
donors can be held out from dynamics fitting, but the current VAE used all four
donors, so those folds are shared-representation, transductive evaluations.
Likewise, `validation.strategy: checkpoint` can hold out 8 hours only as a
non-strict benchmark because the VAE used all three checkpoints. Strict donor
or checkpoint claims require a separately fitted nested representation.
