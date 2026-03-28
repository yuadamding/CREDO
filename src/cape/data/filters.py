"""Sparse support filtering for state endpoint problems.

Only perturbations with sufficient cells at both endpoints enter the
state-transport UOT loss. Low-support perturbations may still enter
the count model.
"""
from __future__ import annotations

from typing import List, Optional

from .core import PerturbSeqDynamicsData


def filter_state_supported_perturbations(
    data: PerturbSeqDynamicsData,
    min_cells_p4: int = 20,
    min_cells_p60: int = 20,
    min_total_mass: Optional[float] = None,
    initial_label: Optional[str] = None,
    terminal_label: Optional[str] = None,
) -> List[str]:
    """Return perturbation_ids with sufficient cell support at both endpoints.

    Parameters
    ----------
    data:
        The canonical study object.
    min_cells_p4:
        Minimum cells at the initial timepoint.
    min_cells_p60:
        Minimum cells at the terminal timepoint.
    min_total_mass:
        Optional lower bound on pooled mass at each endpoint.
    initial_label, terminal_label:
        Override time labels; defaults to first/last in time_axis.

    Returns
    -------
    List of perturbation_ids that pass the filter.
    """
    init_label = initial_label or data.time_axis.labels[0]
    term_label = terminal_label or data.time_axis.labels[-1]
    df = data.cell_state.df

    supported = []
    for pid in data.catalog.perturbation_ids:
        n_init = int(((df["perturbation_id"] == pid) & (df["time_label"] == init_label)).sum())
        n_term = int(((df["perturbation_id"] == pid) & (df["time_label"] == term_label)).sum())

        if n_init < min_cells_p4 or n_term < min_cells_p60:
            continue

        if min_total_mass is not None:
            m_init = data.mass_table.get_pooled(pid, init_label)
            m_term = data.mass_table.get_pooled(pid, term_label)
            if m_init < min_total_mass or m_term < min_total_mass:
                continue

        supported.append(pid)

    return supported
