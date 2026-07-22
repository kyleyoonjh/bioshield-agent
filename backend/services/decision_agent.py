"""
Decision Agent — Drug Discovery Assistant (Phase 11 of the Agentic Drug
Discovery AI Platform master plan).

Aggregates every real signal this system has already computed for a
completed job's top candidate into one transparent recommendation:

  - Priority Score: a disclosed, deterministic formula (never an LLM
    guess) starting from the already-real ranking score (drug_ranking_
    engine.calculate_final_score's 0.7 affinity + 0.3 drug-likeness) and
    applying real ADMET-derived modifiers plus real evidence-availability
    signals. "Evidence" here means the TARGET has real literature/
    clinical-trial hits — it is explicitly NOT a claim that this specific
    candidate has been studied or validated; generate_decision_report()'s
    system prompt enforces that distinction in the narrative too.
  - Overall Recommendation / Development Risk / Recommended Next
    Experiment: an LLM narrative, strictly grounded in the real gathered
    data passed in — same discipline as drug_discovery_literature_agent.py
    /drug_discovery_clinical_agent.py. Demo-mode fallback is a real
    deterministic summary built from the same inputs, never a fabricated
    narrative.

This module makes zero network calls itself — get_top_candidate_scored()
only reshapes data the pipeline already computed, and
generate_decision_report() only consumes data the caller
(drug_discovery_chat.py's answer_decision_report_question()) already
fetched via the real literature/clinical engines. Kept this way so the
scoring formula and narrative logic are trivially unit-testable without
mocking any API.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Every weight below is disclosed in calculate_priority_score()'s returned
# "breakdown", not tuned/fitted to any dataset — same transparent-formula
# discipline as drug_ranking_engine.py's 0.7/0.3 affinity/drug-likeness split.
_PAINS_PENALTY = 15
_VEBER_NONCOMPLIANT_PENALTY = 10
_SA_EASY_BONUS = 5
_SA_HARD_PENALTY = 10
_LITERATURE_EVIDENCE_BONUS = 5
_CLINICAL_EVIDENCE_BONUS = 5
_HEURISTIC_DOCKING_PENALTY = 20


def get_top_candidate_scored(job_result: dict) -> dict | None:
    """
    Returns a candidate-shaped dict {"name", "smiles",
    "best_affinity_kcal_mol", "score", "docking_source", "admet"} built
    identically for both "screen" and "single" modes, using the SAME real
    ranking formula (drug_ranking_engine.calculate_final_score) in both
    cases — so the Decision Agent's priority score is computed
    consistently regardless of which mode the job ran in. Returns None
    when there's genuinely no successfully-docked candidate to evaluate.
    """
    mode = job_result.get("mode")
    if mode == "screen":
        ranked = job_result.get("ranked_candidates") or []
        return ranked[0] if ranked else None

    docking = job_result.get("docking_result") or {}
    if not docking.get("docked"):
        return None
    ligand_analysis = docking.get("ligand_analysis") or {}
    from services.drug_ranking_engine import calculate_final_score
    breakdown = calculate_final_score(docking, ligand_analysis)
    return {
        "name": "원본 리간드",
        "smiles": ligand_analysis.get("canonical_smiles"),
        "best_affinity_kcal_mol": docking.get("best_affinity_kcal_mol"),
        "score": breakdown["final_score"],
        "docking_source": docking.get("source"),
        "admet": docking.get("admet"),
    }


def calculate_priority_score(candidate: dict, has_literature_evidence: bool, has_clinical_evidence: bool) -> dict:
    """candidate: get_top_candidate_scored()'s output. Returns
    {"priority_score": float, "breakdown": {term: delta, ...}} — every
    term that actually applied is listed, so the final number is never
    opaque."""
    base_score = candidate.get("score") or 0.0
    breakdown: dict[str, float] = {"base_ranking_score": base_score}
    score = base_score

    admet = candidate.get("admet") or {}
    if admet.get("valid"):
        if admet["pains"]["flagged"]:
            score -= _PAINS_PENALTY
            breakdown["pains_penalty"] = -_PAINS_PENALTY
        if admet["oral_absorption"]["prediction"] != "high":
            score -= _VEBER_NONCOMPLIANT_PENALTY
            breakdown["veber_noncompliant_penalty"] = -_VEBER_NONCOMPLIANT_PENALTY
        sa_score = admet["synthesis"]["score"]
        if sa_score is not None:
            if sa_score <= 4.0:
                score += _SA_EASY_BONUS
                breakdown["synthesis_ease_bonus"] = _SA_EASY_BONUS
            elif sa_score >= 7.0:
                score -= _SA_HARD_PENALTY
                breakdown["synthesis_difficulty_penalty"] = -_SA_HARD_PENALTY

    if candidate.get("docking_source") == "heuristic":
        score -= _HEURISTIC_DOCKING_PENALTY
        breakdown["heuristic_docking_penalty"] = -_HEURISTIC_DOCKING_PENALTY

    if has_literature_evidence:
        score += _LITERATURE_EVIDENCE_BONUS
        breakdown["target_literature_evidence_bonus"] = _LITERATURE_EVIDENCE_BONUS
    if has_clinical_evidence:
        score += _CLINICAL_EVIDENCE_BONUS
        breakdown["target_clinical_evidence_bonus"] = _CLINICAL_EVIDENCE_BONUS

    score = max(0.0, min(100.0, score))
    breakdown["final_priority_score"] = round(score, 1)
    return {"priority_score": round(score, 1), "breakdown": breakdown}


def _get_client():
    from openai import OpenAI
    return OpenAI(api_key=_OPENAI_API_KEY)


def _use_demo() -> bool:
    return not bool(_OPENAI_API_KEY)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


_SYSTEM_PROMPT = """당신은 신약개발 파이프라인의 여러 실제 계산 결과를 종합해 연구자에게 하나의
의사결정 요약을 제공하는 분석가입니다. 아래 JSON으로 주어지는 실제 데이터(candidate, priority_score,
literature_papers, clinical_trials)만 근거로 답하세요. 새로운 사실이나 수치를 절대 지어내지 마세요.

매우 중요: literature_papers/clinical_trials는 이 "타겟 단백질"에 대한 기존 연구/임상 현황일 뿐,
이 특정 "후보 화합물"이 검증되었다는 근거가 아닙니다 — 후보 화합물 자체의 근거는 오직 docking/ADMET
실측값뿐입니다. 이 둘을 절대 혼동해서 서술하지 마세요 (예: "임상시험이 있으므로 이 화합물이 효과적이다"
같은 문장 금지).

반드시 아래 JSON 형식으로만 답하세요:
{
  "overall_recommendation": "한국어 2-3문장 종합 추천 (신중한 톤, 전임상 초기 단계임을 명시)",
  "development_risk": "low 또는 moderate 또는 high",
  "risk_rationale": "위험도 평가 근거 한국어 1-2문장 (실제 데이터 기반)",
  "recommended_next_experiment": "한국어 1-2문장, 실제 데이터의 한계를 메울 다음 실험 제안 (예: 실제 결합 검증 어세이, ADMET 정밀 프로파일링 등)"
}"""


def _demo_report(candidate: dict, scoring: dict, papers: list[dict], trials: list[dict]) -> dict:
    admet = candidate.get("admet") or {}
    risk_flags = []
    if candidate.get("docking_source") == "heuristic":
        risk_flags.append("Vina 미사용(휴리스틱 점수)")
    if admet.get("valid") and admet["pains"]["flagged"]:
        risk_flags.append("PAINS 구조 경보")
    risk = "high" if len(risk_flags) >= 2 else ("moderate" if risk_flags else "low")
    return {
        "overall_recommendation": (
            f"{candidate.get('name')}의 종합 우선순위 점수는 {scoring['priority_score']}점입니다 "
            f"(실제 도킹/ADMET 계산 기반, 전임상 초기 스크리닝 단계 — OPENAI_API_KEY 미설정으로 서술형 "
            f"종합 없이 수치만 제공하는 데모 모드입니다)."
        ),
        "development_risk": risk,
        "risk_rationale": f"위험 요인: {', '.join(risk_flags) if risk_flags else '식별된 주요 위험 요인 없음'}.",
        "recommended_next_experiment": "실제 결합 검증을 위한 in vitro 어세이(예: SPR/ITC) 및 세포 기반 활성 확인을 권장합니다.",
    }


def generate_decision_report(candidate: dict, scoring: dict, papers: list[dict], trials: list[dict]) -> dict:
    """
    Returns {"overall_recommendation", "development_risk",
    "risk_rationale", "recommended_next_experiment", "priority_score",
    "breakdown"} — the last two are always the real calculate_priority_score()
    output, never touched by the LLM.
    """
    if _use_demo():
        narrative = _demo_report(candidate, scoring, papers, trials)
    else:
        try:
            client = _get_client()
            context = {
                "candidate": {k: v for k, v in candidate.items() if k != "admet"} | {
                    "admet_summary": candidate.get("admet"),
                },
                "priority_score": scoring,
                "literature_papers": [{"pmid": p["pmid"], "title": p["title"]} for p in papers[:3]],
                "clinical_trials": [{"nct_id": t["nct_id"], "title": t.get("brief_title"),
                                      "status": t.get("overall_status")} for t in trials[:3]],
            }
            resp = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            parsed = _parse_json(resp.choices[0].message.content or "")
            if parsed.get("development_risk") not in ("low", "moderate", "high"):
                parsed["development_risk"] = "moderate"
            narrative = {
                "overall_recommendation": parsed.get("overall_recommendation") or _demo_report(candidate, scoring, papers, trials)["overall_recommendation"],
                "development_risk": parsed.get("development_risk"),
                "risk_rationale": parsed.get("risk_rationale") or "",
                "recommended_next_experiment": parsed.get("recommended_next_experiment") or "",
            }
        except Exception as exc:
            logger.warning("[decision_agent] LLM synthesis failed, using demo fallback | error=%s", exc)
            narrative = _demo_report(candidate, scoring, papers, trials)

    return {**narrative, "priority_score": scoring["priority_score"], "breakdown": scoring["breakdown"]}
