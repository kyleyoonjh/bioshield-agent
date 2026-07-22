"""
Explainable multi-factor ranking for screened drug candidates.

Mirrors services/ranking_engine.py's style (positive 0-100 "goodness"
components, weighted sum, rule-based strength/weakness narrative) but is a
fully separate implementation with its own domain-specific formula — the
primer-design ranking weights/criteria (Tm, dG, specificity, ...) do not
apply here.

Final Score = 0.7 * Affinity Score + 0.3 * Drug-Likeness Score

Affinity carries the larger weight since binding affinity is the primary
signal that a candidate could plausibly work at all; drug-likeness is a
secondary filter on developability (a very potent binder that also
violates Lipinski's Rule of Five is still a real, weaker candidate, not
an invalid one — this is why non-drug-like candidates are down-weighted
here rather than excluded, unlike the harder exclusion filter in
drug_screening_engine.filter_candidates for candidates that are severely
non-drug-like or where docking failed outright).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_W_AFFINITY      = 0.7
_W_DRUG_LIKENESS = 0.3
assert abs(_W_AFFINITY + _W_DRUG_LIKENESS - 1.0) < 1e-9

# Typical small-molecule Vina affinities run roughly -4 (weak) to -12 (very
# strong) kcal/mol; this range is what the heuristic fallback in
# docking_engine.py is also deliberately scaled to match.
_AFFINITY_WEAK_KCAL_MOL   = -4.0
_AFFINITY_STRONG_KCAL_MOL = -12.0

_LIPINSKI_VIOLATION_SCORE = {0: 100.0, 1: 70.0, 2: 40.0}
_LIPINSKI_VIOLATION_FLOOR = 10.0


def _affinity_score(best_affinity_kcal_mol: float) -> float:
    span = _AFFINITY_WEAK_KCAL_MOL - _AFFINITY_STRONG_KCAL_MOL
    score = (_AFFINITY_WEAK_KCAL_MOL - best_affinity_kcal_mol) / span * 100.0
    return max(0.0, min(100.0, score))


def _drug_likeness_score(lipinski_violations: int) -> float:
    return _LIPINSKI_VIOLATION_SCORE.get(lipinski_violations, _LIPINSKI_VIOLATION_FLOOR)


def calculate_final_score(docking: dict, ligand_analysis: dict) -> dict:
    affinity_score = _affinity_score(docking["best_affinity_kcal_mol"])
    likeness_score = _drug_likeness_score(ligand_analysis.get("lipinski_violations", 99))
    final_score = _W_AFFINITY * affinity_score + _W_DRUG_LIKENESS * likeness_score
    return {
        "final_score":         round(final_score, 1),
        "affinity_score":      round(affinity_score, 1),
        "drug_likeness_score": round(likeness_score, 1),
    }


def rank_candidates(screened: list[dict]) -> list[dict]:
    """
    screened: filtered output of drug_screening_engine.screen_candidates()
    (every entry has docked=True). Returns candidates sorted by final_score
    desc, each annotated with "rank" and "breakdown".
    """
    scored = []
    for candidate in screened:
        docking = candidate["docking"]
        ligand_analysis = docking.get("ligand_analysis") or {}
        breakdown = calculate_final_score(docking, ligand_analysis)
        scored.append((candidate, breakdown))

    ordered = sorted(scored, key=lambda pair: pair[1]["final_score"], reverse=True)
    results = []
    for idx, (candidate, breakdown) in enumerate(ordered, start=1):
        entry = dict(candidate)
        entry["rank"] = idx
        entry["breakdown"] = breakdown
        results.append(entry)
    return results


def admet_strengths_weaknesses(admet: dict) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    if not admet.get("valid"):
        return strengths, weaknesses

    if admet["oral_absorption"]["prediction"] == "high":
        strengths.append("Veber's rule 충족 (경구 생체이용률 양호 예상)")
    else:
        weaknesses.append("Veber's rule 미충족 (TPSA 또는 회전 가능 결합 수 과다 — 경구 흡수 저하 우려)")

    if admet["pains"]["flagged"]:
        weaknesses.append(f"PAINS 구조 경보 검출: {', '.join(admet['pains']['alerts'])} (어세이 방해 가능성, RDKit 공식 카탈로그)")

    sa_score = admet["synthesis"]["score"]
    if sa_score is not None:
        if sa_score <= 4.0:
            strengths.append(f"합성 접근성 양호 (SA score {sa_score}, 1=쉬움~10=어려움)")
        elif sa_score >= 7.0:
            weaknesses.append(f"합성 난이도 높음 (SA score {sa_score}, 1=쉬움~10=어려움)")

    if admet["hepatotoxicity"]["risk"] == "flagged":
        weaknesses.append(f"간독성 구조 경보 다수 검출: {', '.join(admet['hepatotoxicity']['alerts'])} (실험적 검증 필요, 임상 독성 예측 아님)")

    return strengths, weaknesses


def generate_explainable_report(candidate: dict) -> dict:
    """candidate: one entry from rank_candidates()'s output."""
    docking = candidate["docking"]
    ligand_analysis = docking.get("ligand_analysis") or {}
    breakdown = candidate["breakdown"]

    strengths: list[str] = []
    weaknesses: list[str] = []

    if docking.get("source") == "vina":
        strengths.append(f"Real AutoDock Vina docking score: {docking['best_affinity_kcal_mol']} kcal/mol")
    elif docking.get("source") == "heuristic":
        weaknesses.append("AutoDock Vina unavailable for this run — affinity is a deterministic heuristic proxy, not a physical simulation")

    # Real docking-confidence signal from Vina's own reported alternate
    # poses (see docking_engine._compute_docking_confidence) — None when
    # Vina reported fewer than 2 modes, or for a cached pre-v2 result that
    # predates this field (both handled the same way: just omitted, not
    # treated as an error).
    docking_confidence = docking.get("docking_confidence")
    if docking_confidence:
        score = docking_confidence["pose_consistency_score"]
        if score >= 70:
            strengths.append(f"블라인드 도킹 포즈 일관성 높음 (재현 신뢰도 {score}/100, 상위 포즈 간 평균 RMSD {docking_confidence['mean_rmsd_top_poses_angstrom']}Å)")
        elif score <= 30:
            weaknesses.append(f"블라인드 도킹 포즈 일관성 낮음 (재현 신뢰도 {score}/100) — 단백질 표면의 서로 다른 위치에서 비슷한 점수의 포즈가 발견되어, 결합 부위가 모호할 수 있습니다")

    if ligand_analysis.get("lipinski_violations", 99) == 0:
        strengths.append("Satisfies Lipinski Rule of Five with zero violations")
    elif ligand_analysis.get("lipinski_violations", 99) <= 1:
        strengths.append(f"Drug-like ({ligand_analysis['lipinski_violations']} Lipinski violation)")
    else:
        weaknesses.append(f"Violates Lipinski Rule of Five ({ligand_analysis.get('lipinski_violations')} violations)")

    # Real-computable ADMET subset (Veber/PAINS/SA score/hepatotoxicity
    # screen) — see services/admet_engine.py. Wrapped in try/except since
    # this is a best-effort enrichment on top of the already-complete
    # docking/ranking result, not something that should fail the whole
    # candidate if RDKit's Contrib SA_Score import has an environment issue.
    admet = None
    smiles_for_admet = ligand_analysis.get("canonical_smiles") or candidate.get("smiles")
    if smiles_for_admet:
        try:
            from services.admet_engine import predict_admet_profile
            admet = predict_admet_profile(smiles_for_admet)
            admet_strengths, admet_weaknesses = admet_strengths_weaknesses(admet)
            strengths.extend(admet_strengths)
            weaknesses.extend(admet_weaknesses)
        except Exception:
            logger.warning("[drug_ranking] ADMET profile failed for %s (non-fatal)", candidate.get("name"))

    return {
        "rank":     candidate["rank"],
        "name":     candidate["name"],
        "category": candidate.get("category"),
        "smiles":   ligand_analysis.get("canonical_smiles", candidate.get("smiles")),
        "score":    breakdown["final_score"],
        "score_breakdown": {
            "affinity":      breakdown["affinity_score"],
            "drug_likeness": breakdown["drug_likeness_score"],
        },
        "docking_source":          docking.get("source"),
        "best_affinity_kcal_mol":  docking.get("best_affinity_kcal_mol"),
        "docking_confidence": docking_confidence,
        "admet":    admet,
        "strength": strengths,
        "weakness": weaknesses,
    }
