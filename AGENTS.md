# CREDO count-SDE v4 contributor rules

- Never modify or write into the sibling `../CREDO/` checkout.
- Keep biological data and cohort-named adapters outside this repository.
  Explicit user-approved exception (2026-10-10): `studies/full_ddr_20261006/`
  preserves an isolated source-only snapshot of the full-DDR run. Do not import
  it into v4, overwrite frozen compatibility artifacts, add biological data,
  or treat publication as authority to launch its operational scripts.
- All persisted scientific objects are strict, hash-bound, and versioned.
- New channels require positive, null, adversarial, and ablation tests.
- Do not change a checkpoint schema in the same change that adds a model channel.
- Run `python -m pytest -q` and `python -m ruff check .` before handoff.
