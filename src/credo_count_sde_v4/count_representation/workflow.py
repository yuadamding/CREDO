"""Calibration on held-out fitting-donor cells, then fresh all-fitting-row refit."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..canonical import contract_id, sha256_file
from ..errors import ContractError
from ..forecast.artifacts import write_json
from ..persistence.artifacts import load_tensor_file, save_tensor_file
from .artifacts import output_check, publish, runtime_check, verify
from .contracts import CountRepresentationSpec, RepresentationManifest
from .network import SparseCountAutoencoder, count_loss, initialize
from .stream import Exposure, Partition, halves, iter_counts, partition_audit


def fit_priors(
    root: Path, spec: CountRepresentationSpec, partition: Partition
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    sums = np.zeros((2, len(spec.rna_positions)), dtype=np.float64)
    exposure = Exposure()
    for counts, cells in iter_counts(root, spec, partition=partition, split="train"):
        exposure.update(cells)
        for index, condition in enumerate(("source", "destination")):
            selected = cells.condition_role.eq(condition).to_numpy()
            sums[index] += np.asarray(counts[selected].sum(axis=0), dtype=np.float64).ravel()
    if np.any(sums.sum(axis=1) <= 0):
        raise ContractError("Each fitting condition needs positive RNA for its count baseline.")
    priors = sums + 0.5
    priors /= priors.sum(axis=1, keepdims=True)
    return priors, exposure.record()


def train_epoch(
    root: Path,
    spec: CountRepresentationSpec,
    models: dict[str, SparseCountAutoencoder],
    optimizers: dict[str, Any],
    *,
    epoch: int,
    partition: Partition | None = None,
) -> dict[str, Any]:
    exposure = Exposure()
    updates: dict[str, int] = defaultdict(int)
    skipped = 0
    for counts, cells in iter_counts(
        root, spec, partition=partition, split="all" if partition is None else "train", epoch=epoch
    ):
        exposure.update(cells)
        seed = int(contract_id(["fit-thinning", spec.rules.seed, epoch])[:16], 16)
        first, second = halves(counts, cells, seed)
        if not counts.sum():
            skipped += len(cells)
            continue
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            losses = []
            for observed, target in ((first, second), (second, first)):
                score, depth = count_loss(model.log_probabilities(observed), target)
                if bool((depth > 0).any()):
                    losses.append(score[depth > 0].sum())
                else:
                    losses.append(score.sum() * 0)
            denominator = int(np.count_nonzero(np.asarray(first.sum(axis=1)))) + int(
                np.count_nonzero(np.asarray(second.sum(axis=1)))
            )
            loss = torch.stack(losses).sum() / denominator
            if not bool(torch.isfinite(loss)):
                raise ContractError("Nonfinite count training loss.")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), spec.rules.gradient_clip, error_if_nonfinite=True
            )
            optimizers[name].step()
            updates[name] += 1
    return dict(
        epoch=epoch + 1,
        exposure=exposure.record(),
        updates=dict(updates),
        zero_RNA_rows_without_optimizer_loss=skipped,
        objective="equal_scored_cell_direction_symmetric_A_B_conditional_CE",
    )


def audit_models(
    root: Path,
    spec: CountRepresentationSpec,
    partition: Partition,
    models: dict[str, SparseCountAutoencoder],
    priors: np.ndarray[Any, Any],
) -> dict[str, Any]:
    names = [*models, "condition_composition", "autoencoder_latent_ablated"]
    sums = {name: np.zeros(4, dtype=np.float64) for name in names}
    by_guide: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    exposure = Exposure()
    zero_input = zero_target = 0
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        for counts, cells in iter_counts(root, spec, partition=partition, split="audit"):
            exposure.update(cells)
            # Fixed molecule split for these entirely held-out cells at every candidate.
            first, second = halves(counts, cells, spec.rules.seed)
            for observed, target in ((first, second), (second, first)):
                zero_input += int(np.count_nonzero(np.asarray(observed.sum(axis=1)) == 0))
                target_depth = np.asarray(target.sum(axis=1), dtype=np.int64).ravel()
                valid = target_depth > 0
                zero_target += int((~valid).sum())
                for name in names:
                    if name == "condition_composition":
                        indices = cells.condition_role.eq("destination").to_numpy().astype(int)
                        logits = torch.as_tensor(
                            np.log(priors[indices]), dtype=torch.float32, device=spec.rules.device
                        )
                    elif name == "autoencoder_latent_ablated":
                        logits = models["autoencoder"].log_probabilities(observed, ablate=True)
                    else:
                        logits = models[name].log_probabilities(observed)
                    score, depth = count_loss(logits, target)
                    values = score.detach().cpu().numpy().astype(np.float64)
                    if not np.isfinite(values[valid]).all():
                        raise ContractError("Nonfinite held-out count score.")
                    sums[name] += [
                        values[valid].sum(),
                        valid.sum(),
                        np.dot(values[valid], target_depth[valid]),
                        target_depth[valid].sum(),
                    ]
                    for i in np.flatnonzero(valid):
                        key = (
                            str(cells.condition_role.iloc[i]),
                            int(cells.guide_index.iloc[i]),
                            name,
                        )
                        by_guide[key][0] += float(values[i])
                        by_guide[key][1] += 1
    report: dict[str, Any] = dict(
        exposure=exposure.record(),
        zero_input_half_directions=zero_input,
        zero_target_half_directions=zero_target,
        models={},
    )
    for name, (ce, n, nll, umis) in sums.items():
        report["models"][name] = dict(
            mean_cell_direction_CE=float(ce / n) if n else None,
            RNA_UMI_weighted_CE=float(nll / umis) if umis else None,
            scored_cell_directions=int(n),
            scored_RNA_UMIs=int(umis),
        )
    report["guide_condition_scores"] = [
        dict(
            condition=c,
            guide_index=g,
            family=f,
            mean_cell_direction_CE=total / n,
            scored_cell_directions=int(n),
        )
        for (c, g, f), (total, n) in sorted(by_guide.items())
    ]
    report["interpretation"] = (
        "cell_held_out_and_molecule_split; selection_set_not_independent_confirmation"
    )
    return report


def calibrate_representation(
    root: Path, spec: CountRepresentationSpec, destination: Path
) -> RepresentationManifest:
    runtime_check(spec)
    output_check(root, destination)
    partition, partition_report = partition_audit(root, spec)
    priors, prior_exposure = fit_priors(root, spec, partition)
    models = {
        "autoencoder": initialize(spec.rules, len(spec.rna_positions)),
        "count_factor": initialize(spec.rules, len(spec.rna_positions), factor=True),
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=spec.rules.learning_rate, weight_decay=spec.rules.weight_decay
        )
        for name, model in models.items()
    }
    history, candidates = [], []
    best_scores = {name: float("inf") for name in models}
    best_weights: dict[str, dict[str, torch.Tensor]] = {}
    for epoch in range(max(spec.rules.candidate_epochs)):
        history.append(
            train_epoch(root, spec, models, optimizers, epoch=epoch, partition=partition)
        )
        if epoch + 1 in spec.rules.candidate_epochs:
            measured = dict(epoch=epoch + 1, **audit_models(root, spec, partition, models, priors))
            candidates.append(measured)
            for name, model in models.items():
                score = measured["models"][name]["mean_cell_direction_CE"]
                if score is not None and score < best_scores[name]:
                    best_scores[name] = score
                    best_weights[name] = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
    selected = {}
    for family in models:
        finite = [
            r for r in candidates if r["models"][family]["mean_cell_direction_CE"] is not None
        ]
        if not finite:
            raise ContractError("No defined held-out count score for exposure selection.")
        selected[family] = min(
            finite, key=lambda r: (r["models"][family]["mean_cell_direction_CE"], r["epoch"])
        )["epoch"]
    best = next(r for r in candidates if r["epoch"] == selected["autoencoder"])
    factor = next(r for r in candidates if r["epoch"] == selected["count_factor"])
    ae_score = best["models"]["autoencoder"]["mean_cell_direction_CE"]
    competitors = [
        best["models"]["condition_composition"]["mean_cell_direction_CE"],
        factor["models"]["count_factor"]["mean_cell_direction_CE"],
        best["models"]["autoencoder_latent_ablated"]["mean_cell_direction_CE"],
    ]
    count_gate = all(
        s is not None and ae_score < s - spec.rules.minimum_count_improvement for s in competitors
    )
    result = dict(
        partition=partition_report,
        prior_exposure=prior_exposure,
        epochs=history,
        candidates=candidates,
        selected_epochs=selected,
        heldout_count_gate=bool(count_gate),
        perturbation_preservation_qualified=False,
        representation_qualified=False,
        scientific_promotion=False,
        query_cells_used=0,
        selection_scope="fitting_donor_audit_only; independent_qualification_required",
    )

    def writer(path: Path) -> dict[str, Any]:
        write_json(path / "audit_rows.json", partition)
        write_json(path / "calibration.json", result)
        save_tensor_file(
            path / "condition_priors.safetensors", {"priors": torch.from_numpy(priors)}
        )
        for name, state in best_weights.items():
            save_tensor_file(path / f"calibration_{name}.safetensors", state)
        return dict(
            selected_epochs=selected,
            heldout_count_gate=bool(count_gate),
            representation_qualified=False,
            query_cells_used=0,
            calibration_training_rows=partition_report["calibration_training_rows"],
            audit_rows=partition_report["audit_rows"],
        )

    runtime_check(spec)
    return publish(
        destination,
        spec,
        stage="calibration",
        parents={"fitting": spec.fitting.identity()},
        writer=writer,
    )


def fit_scaling(
    root: Path, spec: CountRepresentationSpec, model: SparseCountAutoencoder
) -> dict[str, Any]:
    model.eval().requires_grad_(False)
    n = 0
    mean = np.zeros(spec.rules.latent_dim, dtype=np.float64)
    m2 = mean.copy()
    exposure = Exposure()
    with torch.inference_mode():
        for counts, cells in iter_counts(root, spec):
            exposure.update(cells)
            valid = cells.RNA_UMIs.to_numpy() > 0
            z = model.raw_encode(counts).cpu().numpy()[valid].astype(np.float64)
            if not len(z):
                continue
            if not np.isfinite(z).all():
                raise ContractError("Nonfinite fitting latent state.")
            difference = z.mean(axis=0) - mean
            total = n + len(z)
            m2 += ((z - z.mean(axis=0)) ** 2).sum(axis=0) + difference**2 * n * len(z) / total
            mean += difference * len(z) / total
            n = total
    if n < 2:
        raise ContractError("Latent scaling requires at least two positive-RNA fitting cells.")
    std = np.sqrt(m2 / n)
    floor = max(
        float(np.median(std[std > 0])) * spec.rules.scale_floor_fraction
        if np.any(std > 0)
        else 0.0,
        1e-6,
    )
    model.center.copy_(torch.as_tensor(mean, dtype=torch.float32, device=model.device))
    model.scale.copy_(
        torch.as_tensor(np.maximum(std, floor), dtype=torch.float32, device=model.device)
    )
    return dict(
        exposure=exposure.record(),
        positive_RNA_cells=n,
        population_variance_ddof=0,
        scale_floor=floor,
        scale_floor_rule="max(1e-6,fraction*median_positive_fitting_std)",
        constant_latent_dimensions=int(np.count_nonzero(std == 0)),
    )


def refit_representation(
    root: Path, calibration: Path, destination: Path
) -> RepresentationManifest:
    record = verify(calibration, stage="calibration")
    spec = CountRepresentationSpec.model_validate_json(
        (calibration / "specification.json").read_text()
    )
    runtime_check(spec)
    if record.specification_sha256 != spec.identity() or record.parents != {
        "fitting": spec.fitting.identity()
    }:
        raise ContractError("Calibration fitting/specification identity mismatch.")
    output_check(root, destination)
    result = json.loads((calibration / "calibration.json").read_text())
    epochs = result["selected_epochs"]["autoencoder"]
    if (
        epochs not in spec.rules.candidate_epochs
        or record.facts["selected_epochs"] != result["selected_epochs"]
    ):
        raise ContractError("Calibration exposure does not match the frozen candidate selection.")
    model = initialize(spec.rules, len(spec.rna_positions))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.rules.learning_rate, weight_decay=spec.rules.weight_decay
    )
    history = [
        train_epoch(root, spec, {"autoencoder": model}, {"autoencoder": optimizer}, epoch=epoch)
        for epoch in range(epochs)
    ]
    scaling = fit_scaling(root, spec, model)
    expected = sum(s.selected_rows for s in spec.fitting.shards)
    if (
        any(row["exposure"]["rows"] != expected for row in history)
        or scaling["exposure"]["rows"] != expected
    ):
        raise ContractError("Fresh refit did not consume every authorized row per epoch.")

    def writer(path: Path) -> dict[str, Any]:
        save_tensor_file(path / "model.safetensors", model.state_dict())
        write_json(
            path / "exposure.json",
            dict(
                epochs=history,
                scaling=scaling,
                initialization="fresh_from_prespecified_seed_not_calibration_weights",
                audit_cells_now_used_in_refit=True,
                heldout_score_belongs_to_calibration_model=True,
            ),
        )
        return dict(
            selected_epochs=epochs,
            fitting_rows=expected,
            query_cells_used=0,
            model_numerical_sha256=sha256_file(path / "model.safetensors"),
            latent_dim=spec.rules.latent_dim,
            heldout_count_gate=result["heldout_count_gate"],
            representation_qualified=False,
            frozen_encoder=True,
            frozen_decoder=True,
            frozen_fitting_geometry=True,
            scientific_promotion=False,
        )

    runtime_check(spec)
    return publish(
        destination,
        spec,
        stage="fitted_representation",
        parents={"calibration": record.identity(), "fitting": spec.fitting.identity()},
        writer=writer,
    )


def load_representation(
    root: Path,
) -> tuple[SparseCountAutoencoder, CountRepresentationSpec, RepresentationManifest]:
    record = verify(root, stage="fitted_representation")
    spec = CountRepresentationSpec.model_validate_json((root / "specification.json").read_text())
    runtime_check(spec)
    if (
        record.specification_sha256 != spec.identity()
        or record.parents.get("fitting") != spec.fitting.identity()
    ):
        raise ContractError("Fitted representation specification/fitting mismatch.")
    if record.facts["model_numerical_sha256"] != sha256_file(root / "model.safetensors"):
        raise ContractError("Fitted representation numerical identity mismatch.")
    model = initialize(spec.rules, len(spec.rna_positions))
    state = load_tensor_file(root / "model.safetensors", device=spec.rules.device)
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ContractError("Nonfinite fitted representation tensor.")
    model.load_state_dict(state, strict=True)
    if bool(torch.any(model.scale <= 0)):
        raise ContractError("Fitted geometry scale must be positive.")
    model.eval().requires_grad_(False)
    return model, spec, record
