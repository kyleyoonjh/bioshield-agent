"""
Multi-candidate drug screening — dock every candidate in a curated library
against one prepared receptor, then filter out failed/non-drug-like hits.

Measured real AutoDock Vina timing (see refs/docking/README.md's original
single-ligand benchmark plus a direct blind-box timing check done while
building this module) showed ~2.6s per ligand even against a full-protein
blind-docking box — fast enough to screen the whole curated library in a
single pass at standard exhaustiveness, so this does not need a
fast-then-refine two-tier screen.
"""
from __future__ import annotations

import logging

from services import docking_engine

logger = logging.getLogger(__name__)

_MAX_LIPINSKI_VIOLATIONS = 1


def screen_candidates(
    receptor_pdbqt_path: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    candidates: list[dict],
    exhaustiveness: int = 8,
) -> list[dict]:
    """
    candidates: list of {"name": str, "smiles": str, "category": str}.
    Returns one result dict per candidate (docking failures included, with
    docked=False) — filtering happens separately in filter_candidates().
    A failure on one candidate never aborts the rest of the batch.
    """
    results = []
    for candidate in candidates:
        try:
            docking = docking_engine.dock_ligand(
                receptor_pdbqt_path=receptor_pdbqt_path,
                ligand_smiles=candidate["smiles"],
                center=center,
                box_size=box_size,
                exhaustiveness=exhaustiveness,
            )
        except Exception as exc:
            logger.warning("[screening] docking failed for %s | error=%s", candidate["name"], exc)
            docking = {"docked": False, "error": str(exc), "source": "none"}

        results.append({
            "name":     candidate["name"],
            "smiles":   candidate["smiles"],
            "category": candidate.get("category"),
            "docking":  docking,
        })
    return results


def filter_candidates(results: list[dict]) -> list[dict]:
    """Deterministic quality gate — drop candidates that failed to dock or
    are not drug-like. No LLM involvement."""
    kept = []
    for result in results:
        docking = result["docking"]
        if not docking.get("docked"):
            continue
        ligand_analysis = docking.get("ligand_analysis") or {}
        if ligand_analysis.get("lipinski_violations", 99) > _MAX_LIPINSKI_VIOLATIONS:
            continue
        kept.append(result)
    return kept
