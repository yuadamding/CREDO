from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import Tensor, nn

from camfnd.data.contract import PerturbationCatalog


class ControlAnchoredEmbeddingStore(nn.Module):
    """Trainable perturbation embeddings with an exact zero control anchor.

    For the Stage-I benchmark, a one-hot initialization for non-controls makes
    the scalar control-anchored coefficient heads easy to interpret and fit.
    """

    def __init__(self, catalog: PerturbationCatalog, embedding_dim: int = 3, init_scale: float = 0.05) -> None:
        super().__init__()
        catalog.validate()
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.catalog = catalog
        self.embedding_dim = int(embedding_dim)
        self.controls = set(catalog.controls)
        self.non_control_ids = [pid for pid in catalog.perturbation_ids if pid not in self.controls]
        self._non_control_index = {pid: idx for idx, pid in enumerate(self.non_control_ids)}

        if self.non_control_ids:
            if self.embedding_dim >= len(self.non_control_ids):
                initial = torch.zeros(len(self.non_control_ids), self.embedding_dim, dtype=torch.float64)
                for idx in range(len(self.non_control_ids)):
                    initial[idx, idx] = 1.0
            else:
                initial = init_scale * torch.randn(len(self.non_control_ids), self.embedding_dim, dtype=torch.float64)
            self.non_control_embeddings = nn.Parameter(initial)
        else:
            self.register_parameter("non_control_embeddings", None)
        self.register_buffer("_zero", torch.zeros(self.embedding_dim, dtype=torch.float64), persistent=False)

    def forward_one(self, perturbation_id: str) -> Tensor:
        perturbation_id = str(perturbation_id)
        if perturbation_id in self.controls:
            return self._zero.clone()
        if perturbation_id not in self._non_control_index:
            raise KeyError(f"Unknown perturbation_id {perturbation_id!r}.")
        return self.non_control_embeddings[self._non_control_index[perturbation_id]]

    def forward_many(self, perturbation_ids: Iterable[str]) -> Tensor:
        rows = [self.forward_one(pid) for pid in perturbation_ids]
        return torch.stack(rows, dim=0)

    def regularization(self) -> Tensor:
        if self.non_control_embeddings is None:
            return torch.zeros((), dtype=torch.float64, device=self._zero.device)
        return (self.non_control_embeddings ** 2).mean()

    def control_anchor_is_exact(self, atol: float = 0.0) -> bool:
        for perturbation_id in self.controls:
            if not torch.allclose(self.forward_one(perturbation_id), self._zero, atol=atol, rtol=0.0):
                return False
        return True

    def snapshot(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for perturbation_id in self.catalog.perturbation_ids:
            out[perturbation_id] = self.forward_one(perturbation_id).detach().cpu().tolist()
        return out
