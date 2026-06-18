# CREDO Python Package

This is the installable `credo` package for control-referenced ecological
dynamics over finite-measure Perturb-seq endpoints and trajectories.

The package includes typed configuration schemas, data contracts, endpoint and
multi-time losses, model components, training adapters, evaluation gates, and the
optional `credo.search` setting-search layer. Dataset runners, analysis scripts,
tests, and environment helpers live in the repository bundle outside this Python
package directory.

Install from the repository root with:

```bash
python -m pip install -e package
```

The command-line data validator is exposed as:

```bash
credo-validate-data --data-path /path/to/input.h5ad --schema trajectory --strict
```
