# LPS adapter

This adapter replaces the deleted LPS-specific runner. It converts the existing
donor/cell-state cohort into the same canonical files used by every CREDO run.
The physical trajectory begins at 90 minutes and continues through 6 and 10
hours; it does not estimate the unobserved stimulation-to-90-minute transition.

Static baseline controls and stimulated cell-state measures occupy separate
donor-specific context groups, so their independently normalized masses are not
mixed into one artificial ecosystem. Donors share learned cell-state
embeddings, while all baseline controls share the exact soft reference. Mass is
relative within each declared donor/time/scope denominator.

If `obsm["X_credo"]` is absent, preparation computes a deterministic 32-component
SVD representation from library-normalized, log-transformed counts on the
selected support. This preprocessing remains in the adapter, outside the model
package.

```bash
python examples/lps/prepare.py
credo validate examples/lps/config.yaml
credo run examples/lps/config.yaml
```
