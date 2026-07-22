"""
SAR Optimization Service — orchestrates real bioisosteric analog
generation (sar_optimization_engine.py) with a real re-docking comparison,
for a COMPLETED drug discovery job.

The job's original prepared receptor is already cleaned up by the time a
job is COMPLETED (receptor_prep_engine.cleanup_receptor() runs in
run_drug_discovery_pipeline()'s finally block) — this re-prepares a fresh
receptor from the job's own stored real structure["pdb_text"], which is
kept in the result indefinitely, so this works identically for both
curated and live-search-resolved targets without depending on the Tier-1
cache.

Every analog is re-docked for real at exhaustiveness=4 ("fast" — an
explicitly disclosed, quicker on-demand comparison, not the original
screen's full rigor) — never a predicted/estimated affinity. If Vina is
unavailable, dock_ligand()'s own heuristic fallback still runs and is
labeled "heuristic", same as everywhere else in this app; this service
never hides which one produced a given number.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from services import admet_engine, docking_engine, receptor_prep_engine
from services.sar_optimization_engine import generate_analogs

logger = logging.getLogger(__name__)

_SAR_EXHAUSTIVENESS = 4
_MAX_ANALOGS = 3

# Same reasoning as drug_discovery_pipeline.py's _SCREEN_CONCURRENCY:
# docking_engine._VINA_CPU_LIMIT already halves the core count per Vina
# subprocess so 2 concurrent calls exactly saturate the machine.
_SCREEN_CONCURRENCY = 2


def run_sar_optimization(job_result: dict) -> dict:
    """
    Returns {"available": True, "base_name", "base_smiles",
    "base_affinity_kcal_mol", "analogs": [...], "note": ...} on success, or
    {"available": False, "reason": ...} when there's nothing real to
    compute from (no stored structure, no candidate SMILES, no applicable
    bioisostere for this specific molecule, or receptor re-prep failure).
    """
    structure = job_result.get("structure") or {}
    pdb_text = structure.get("pdb_text")
    if not pdb_text:
        return {"available": False, "reason": "원본 타겟 구조 데이터를 찾을 수 없습니다."}

    mode = job_result.get("mode")
    if mode == "single":
        docking = job_result.get("docking_result") or {}
        base_smiles = (docking.get("ligand_analysis") or {}).get("canonical_smiles")
        base_name = "원본 리간드"
        base_affinity = docking.get("best_affinity_kcal_mol")
    else:
        ranked = job_result.get("ranked_candidates") or []
        if not ranked:
            return {"available": False, "reason": "스크리닝 후보 데이터가 없습니다."}
        top = ranked[0]
        base_smiles = top.get("smiles")
        base_name = top.get("name")
        base_affinity = top.get("best_affinity_kcal_mol")

    if not base_smiles:
        return {"available": False, "reason": "기준 리간드의 SMILES를 찾을 수 없습니다."}

    analogs = generate_analogs(base_smiles)[:_MAX_ANALOGS]
    if not analogs:
        return {
            "available": False,
            "reason": f"{base_name}({base_smiles})에는 현재 지원하는 생물학적 등가체 치환 "
                      "(카르복실산->테트라졸, 메틸->CF3, 방향족 하이드록실->불소)이 적용될 작용기가 없습니다.",
        }

    receptor_prep = receptor_prep_engine.prepare_receptor_from_pdb(pdb_text)
    if not receptor_prep.get("prepared"):
        return {"available": False, "reason": f"리셉터 재준비 실패: {receptor_prep.get('reason')}"}

    def _dock_and_profile(analog: dict) -> dict:
        try:
            docking_result = docking_engine.dock_ligand(
                receptor_pdbqt_path=receptor_prep["pdbqt_path"], ligand_smiles=analog["smiles"],
                center=receptor_prep["center"], box_size=receptor_prep["box_size"],
                exhaustiveness=_SAR_EXHAUSTIVENESS,
            )
        except Exception as exc:
            logger.warning("[sar_optimization] docking failed for analog %s | error=%s", analog["smiles"], exc)
            docking_result = {"docked": False, "error": str(exc), "source": "none"}

        admet = None
        if docking_result.get("docked"):
            try:
                admet = admet_engine.predict_admet_profile(analog["smiles"])
            except Exception:
                logger.warning("[sar_optimization] ADMET failed for analog %s (non-fatal)", analog["smiles"])

        return {
            "transformation": analog["transformation"],
            "rationale": analog["rationale"],
            "smiles": analog["smiles"],
            "docked": docking_result.get("docked", False),
            "source": docking_result.get("source"),
            "best_affinity_kcal_mol": docking_result.get("best_affinity_kcal_mol"),
            "admet": admet,
        }

    try:
        # Same real bottleneck fix as the main screening loop
        # (drug_discovery_pipeline.py's _SCREEN_CONCURRENCY): each analog's
        # real Vina subprocess is independent of the others, so running them
        # one at a time left half the machine's cores idle
        # (docking_engine._VINA_CPU_LIMIT already halves the core count per
        # call specifically to allow this). This function runs inside a
        # single worker thread already (dispatched via asyncio.to_thread by
        # the router), so a nested ThreadPoolExecutor here gives real
        # OS-level parallelism for the (at most 3) analog dockings.
        with ThreadPoolExecutor(max_workers=min(len(analogs), _SCREEN_CONCURRENCY)) as executor:
            results = list(executor.map(_dock_and_profile, analogs))
    finally:
        receptor_prep_engine.cleanup_receptor(receptor_prep)

    return {
        "available": True,
        "base_name": base_name,
        "base_smiles": base_smiles,
        "base_affinity_kcal_mol": base_affinity,
        "analogs": results,
        "note": (
            f"각 유사체는 실제로 재도킹되었습니다 (exhaustiveness={_SAR_EXHAUSTIVENESS}, 빠른 비교용 — "
            "원본 스크리닝보다 낮은 정밀도일 수 있음). '예상 효과'를 추정하지 않고 실제 재계산된 "
            "결합친화도만 보고합니다."
        ),
    }
