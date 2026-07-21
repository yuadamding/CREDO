# Downloaded GSE314342 data

The downloaded cohort is stored outside this Git repository at
`/home/yding1995/opscc_sc/inputs/GSE314342`. It contains the processed primary human CD4+ T-cell
Perturb-seq release and the compact latent support previously built from it.

Key local artifacts include:

- `gse314342_credo_support.h5ad`: 11,434,670 latent support atoms with
  `obsm["X_credo"]` and a 64-atom cap per donor-guide-time measure;
- `canonical/`: the complete current CREDO contract with 11,407,072 retained
  atoms, 91,848 source-supported donor-guide measures, 12,637 embeddings, 3,635
  controls, 12 donor-time denominators, and eight downstream count blocks;
- `canonical_pilot/`: the D3 current contract with 15,417 atoms, 122 measures,
  65 embeddings, 16 controls, three denominators, and two count blocks;
- `measure_manifest.csv`: donor, guide, target-gene embedding, control, and QC
  identity for each finite-measure view;
- `guide_counts_and_masses.csv`: full eligible-cell counts and smoothed
  within-donor/time guide frequencies;
- `guide_count_blocks.csv`: complete donor-time source-exposure count blocks;
- `gse314342_credo_pilot_d3.h5ad` and matching `pilot_d3_*` tables: a compact
  D3 numerical smoke cohort;
- build, scan, encoder, selection, and checksum provenance.

The cohort has four donors and the ordered checkpoints `Rest`, `Stim8hr`, and
`Stim48hr`. GEO lane titles and processed metadata disagree on the final label;
[`late_time_resolution.json`](late_time_resolution.json) records the reviewed
decision to use 48 hours.

The CREDO adapter writes the canonical five-file data contract without
duplicating cohort logic in the model package:

```bash
python examples/gse314342/prepare.py --pilot
credo validate examples/gse314342/config.yaml
```

Remove `--pilot` to rebuild `canonical/` from the full support. The full adapter
excludes 5,412 downstream-only donor-guide IDs because canonical measures must
exist at `Rest`; it does not renormalize the retained whole-library masses.
Donor-guide `measure_id` values
remain separate from shared target-gene `embedding_id` values. All
non-targeting guides share the exact control reference. Mass is
`relative_within_group`, so effects describe relative donor-specific guide
representation after the already perturbed resting state, not absolute T-cell
population growth or the unobserved perturbation-to-rest transition.

Primary records: [GEO GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)
and Zhu, Dann, et al., [*Genome-scale perturb-seq in primary human CD4+ T cells
maps context-specific regulators of T cell programs and human immune traits*](https://doi.org/10.64898/2025.12.23.696273).
