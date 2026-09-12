# Release policy

`4.0.0.dev40` is engineering-only. It cannot be relabeled as alpha, beta,
release-candidate, stable, or biological evidence.

The intended ladder is `dev1 → dev2 → a1 → a2 → b1 → rc1 → 4.0.0`. Every
milestone is built from fresh exact bytes. Stable `4.0.0` additionally requires
the scheduled CUDA matrix, declared device envelope, deterministic-build
receipt, SBOM/provenance archive, family-specific legacy replay receipt, stable
quoted recipe discovery, and a fully verified sealed audit.

Development intentionally omits the `credo.recipes` entry point. The canonical
loader remains `credo-v4 open-run`. A stable discovery descriptor is added only
after the bridge and legacy compatibility tests pass.

See [implementation status](implementation-status.md) for the exact boundary of
the current checkout. Scientific channels additionally require the ordered
component receipts in [component qualification](component-qualification.md);
a successful GPU job or monolithic training loss is never a release gate.
