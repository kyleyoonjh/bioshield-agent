"""
Drug Discovery Assistant's Planner / Evaluator / Reflector / DecisionLog.

Mirrors the concept structure already proven in agent/__init__.py's
primer-design loop (ScientificDesignPlanner / ScientificEvaluator /
ScientificReflectionAgent / DecisionLog) but is a fully separate
implementation — no imports from agent/__init__.py, no shared state, own
standalone OpenAI client (same convention already used by
drug_discovery_intent.py). Numbers are never invented here: the Evaluator
and Reflector only ever read real EngineResult/docking/ranking output; an
LLM (when available) is used solely to narrate *why*, never to decide
scientific values.

Evaluator judges *execution health* (did receptor prep succeed, did real
Vina run, did enough candidates survive filtering) rather than whether the
science outcome itself is "impressive" — mirroring the primer agent's
principle that the Evaluator only ever checks a pipeline-quality metric
(coverage), never a science-outcome score.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_openai_client = None


def _use_demo() -> bool:
    return not bool(os.getenv("OPENAI_API_KEY"))


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 300) -> str:
    client = _get_openai_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5)
    return ""


# Escalation ladder: each retry moves one rung up, never repeats a lower
# rung, caps at "broad". Chosen deterministically (never LLM-picked) so
# retries can never select an invalid/hallucinated strategy.
STRATEGY_TEMPLATES = {
    "fast":     {"exhaustiveness": 4,  "padding": 6.0,  "min_survivors": 3},
    "standard": {"exhaustiveness": 8,  "padding": 8.0,  "min_survivors": 2},
    "broad":    {"exhaustiveness": 16, "padding": 12.0, "min_survivors": 1},
}
_STRATEGY_ORDER = ["fast", "standard", "broad"]


class DrugDiscoveryDecisionLog:
    """Same shape as agent/__init__.py's DecisionLog — append-only, JSON-serializable."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.started_at = time.time()
        self.entries: list[dict] = []

    def add(self, step: str, decision: str, tool: str, outcome: str, details: dict | None = None) -> None:
        self.entries.append({
            "step": step, "decision": decision, "tool": tool, "outcome": outcome,
            "timestamp": time.time(), "details": details or {},
        })

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "started_at": self.started_at, "entries": self.entries}


_LARGE_TARGET_RESIDUE_THRESHOLD = 500


class DrugDiscoveryPlanner:
    """
    Picks a strategy template (search rigor: exhaustiveness / blind-box
    padding / how many screened candidates must survive filtering to call
    the run healthy). Deterministic rules — this is a low-stakes
    engineering-parameter choice, not a judgment call that benefits from
    LLM reasoning, so no LLM call is made here (kept transparent rather
    than adding latency for no real benefit).

    Real measured timing (see refs/docking/README.md's benchmarks) showed a
    single ligand against a large blind box (e.g. full-length Spike,
    ~1273 residues, box ~167x119x168A) takes ~45s at exhaustiveness=8 —
    screening a ~20-candidate library at that rate risks the job's timeout.
    residue_count (a real measured property of the resolved structure, not
    a guess) lets the Planner default to "fast" for large targets instead
    of always assuming "standard" fits every target size.
    """

    def plan(self, goal_text: str, residue_count: int = 0) -> dict:
        text = (goal_text or "").lower()
        if any(kw in text for kw in ("정밀", "철저", "thorough", "exhaustive")):
            strategy_id = "broad"
        elif any(kw in text for kw in ("빠르게", "quick", "fast", "대충")):
            strategy_id = "fast"
        elif residue_count > _LARGE_TARGET_RESIDUE_THRESHOLD:
            strategy_id = "fast"
        else:
            strategy_id = "standard"
        return {"strategy_id": strategy_id, "template": STRATEGY_TEMPLATES[strategy_id]}


class DrugDiscoveryEvaluator:
    def evaluate(self, mode: str, receptor_prep: dict, outcome, template: dict, iteration: int) -> dict:
        """
        outcome: for mode="single", a docking dict; for mode="screen", the
        filtered+ranked candidate list.
        """
        force_pass = iteration >= 3

        if not receptor_prep.get("prepared"):
            return self._verdict(force_pass, "receptor_prep", receptor_prep.get("reason"))

        if mode == "single":
            docking = outcome
            if not docking.get("docked"):
                return self._verdict(force_pass, "docking_failed", docking.get("error"))
            if docking.get("source") != "vina":
                return self._verdict(force_pass, "vina_unavailable", docking.get("note"))
            return self._verdict(True, None, None)

        # mode == "screen"
        ranked = outcome
        min_survivors = template["min_survivors"]
        if len(ranked) < min_survivors:
            return self._verdict(force_pass, "insufficient_survivors",
                                  f"{len(ranked)}/{min_survivors} required candidates survived filtering")
        if ranked and ranked[0]["docking"].get("source") != "vina":
            return self._verdict(force_pass, "vina_unavailable", "top candidate used heuristic fallback")
        return self._verdict(True, None, None)

    @staticmethod
    def _verdict(passed: bool, failed_metric: str | None, failed_detail: str | None) -> dict:
        return {
            "verdict": "pass" if passed else "retry",
            "failed_metric": failed_metric,
            "failed_detail": failed_detail,
        }


class DrugDiscoveryReflectionAgent:
    """
    Root-cause narrative (LLM when available, deterministic template
    otherwise) + a strictly deterministic next-strategy pick. The LLM is
    used only to phrase the explanation — it never chooses next_strategy
    or touches any numeric value.
    """

    _ROOT_CAUSE_TEMPLATES = {
        "receptor_prep":            "타겟 구조로부터 리셉터(PDBQT) 준비에 실패했습니다: {detail}",
        "docking_failed":           "리간드 도킹 자체가 실패했습니다: {detail}",
        "vina_unavailable":         "이번 실행에서 실제 AutoDock Vina 대신 휴리스틱 대체값이 사용되었습니다: {detail}",
        "insufficient_survivors":   "필터(도킹 성공 + Lipinski 기준)를 통과한 후보가 부족합니다: {detail}",
    }

    def reflect(self, evaluation: dict, current_strategy_id: str, iteration: int) -> dict:
        failed_metric = evaluation.get("failed_metric") or "unknown"
        detail = evaluation.get("failed_detail") or "명시된 세부 사유 없음"
        root_cause = self._ROOT_CAUSE_TEMPLATES.get(
            failed_metric, "품질 기준 미달: {detail}"
        ).format(detail=detail)

        next_strategy = self._next_strategy(current_strategy_id)
        recommendation = f"전략을 '{current_strategy_id}' → '{next_strategy}'로 전환해 재시도합니다."

        reasoning = f"{root_cause} {recommendation}"
        if not _use_demo():
            try:
                reasoning = _chat(
                    system="당신은 신약개발 파이프라인의 원인분석 에이전트입니다. 주어진 실패 사유와 다음 전략을 "
                           "한국어 1-2문장으로 자연스럽게 설명하세요. 절대 새로운 수치나 사실을 지어내지 마세요 — "
                           "주어진 정보만 서술하세요.",
                    user=json.dumps({"root_cause": root_cause, "next_strategy": next_strategy,
                                      "iteration": iteration}, ensure_ascii=False),
                ) or reasoning
            except Exception as exc:
                logger.warning("[drug_discovery_agent] reflection LLM call failed, using template | error=%s", exc)

        return {
            "root_cause": root_cause,
            "next_strategy": next_strategy,
            "reasoning": reasoning,
        }

    @staticmethod
    def _next_strategy(current: str) -> str:
        try:
            idx = _STRATEGY_ORDER.index(current)
        except ValueError:
            return "standard"
        return _STRATEGY_ORDER[min(idx + 1, len(_STRATEGY_ORDER) - 1)]


_AFFINITY_GUIDE = (
    "결합친화도(binding affinity, kcal/mol)는 AutoDock Vina 예측값으로, 더 음수(낮을수록) 강한 결합을 "
    "의미합니다 — 일반적으로 -6 kcal/mol 이하면 유의미한 결합, -8 이하면 강한 결합으로 간주됩니다."
)
_SCORE_GUIDE = (
    "종합 점수(final score, 0-100, 스크리닝 모드)는 결합친화도 점수(가중치 70%)와 drug-likeness 점수"
    "(가중치 30%)의 가중합으로, 100에 가까울수록 더 우수한 후보입니다."
)
_LIKENESS_GUIDE = (
    "drug-likeness 점수(0-100)는 Lipinski's Rule of Five 위반 횟수를 기반으로 합니다 — 위반이 0회에 "
    "가까울수록(점수는 100에 가까울수록) 경구 약물로 개발될 가능성이 높습니다."
)
_SOURCE_GUIDE = (
    "데이터 소스가 'vina'면 실제 물리 기반 도킹 시뮬레이션 결과로 신뢰도가 높고, 'heuristic'이면 Vina를 "
    "사용할 수 없을 때의 분자 특성 기반 근사치로 신뢰도가 상대적으로 낮습니다."
)

# Real capabilities this system can actually run as a follow-up — never
# invented, these are the exact keyword-triggered agents in drug_discovery_
# chat.py (is_literature_query/is_clinical_query/is_target_intelligence_query/
# is_sar_optimization_query/is_decision_report_query) plus the structural-
# analysis Q&A already answered via answer_completed_job_question(). Listed
# here so the "recommended next actions" section of the AI summary only
# ever points at things the system genuinely does, in the user's own words.
_AVAILABLE_FOLLOWUPS = (
    "'결합 포켓 주변 아미노산 잔기와 상호작용을 분석해줘' (실제 도킹 포즈 좌표 기반 결합 포켓 분석, "
    "AlphaFold DB 유래 구조는 PAE 신뢰도까지 포함)",
    "'관련 논문 찾아줘' (PubMed 실시간 검색)",
    "'임상시험 현황 알려줘' (ClinicalTrials.gov 실시간 검색)",
    "'이 타겟의 질병 연관성과 관련 경로 분석해줘' (UniProt/Reactome 실시간 조회)",
    "'이 후보의 구조를 개선할 수 있는 유사체 찾아줘' (실제 생물학적 등가체 치환 + 재도킹 비교)",
    "'종합 평가 리포트 만들어줘' (우선순위 점수·개발 위험도 종합)",
)


def _confidence_band(plddt: float) -> str:
    """AlphaFold's own published pLDDT interpretation bands — not invented."""
    if plddt >= 90:
        return "매우 높음"
    if plddt >= 70:
        return "신뢰할 만함"
    if plddt >= 50:
        return "낮음"
    return "매우 낮음"


def _demo_recommended_actions(pains_flagged: bool, structural_analysis_available: bool) -> str:
    """Deterministic (non-LLM) part D — real conditional logic over the same
    real flags passed to the LLM path, so demo mode still recommends
    something specific to THIS result rather than a generic fixed list."""
    recs: list[str] = []
    if pains_flagged:
        recs.append(
            "PAINS 구조 경보가 있는 후보가 있으므로, \"결합 포켓 주변 아미노산 잔기와 상호작용을 분석해줘\"로 "
            "실제 결합 양상을 직접 확인해 보는 것을 권장합니다."
        )
    elif structural_analysis_available:
        recs.append(
            "\"결합 포켓 주변 아미노산 잔기와 상호작용을 분석해줘\"라고 하시면 이미 계산된 실제 결합 포켓 데이터를 "
            "확인할 수 있습니다."
        )
    recs.append("\"관련 논문 찾아줘\" 또는 \"임상시험 현황 알려줘\"로 이 타겟에 대한 기존 연구 맥락을 파악해 보세요.")
    recs.append("\"종합 평가 리포트 만들어줘\"라고 하시면 우선순위 점수와 개발 위험도를 종합적으로 확인할 수 있습니다.")
    return "\n\n추천 다음 행동:\n- " + "\n- ".join(recs)


def _demo_ai_summary(mode: str, data: dict) -> str:
    """Deterministic template used when no OPENAI_API_KEY is set, or if the
    live call fails — same narrate-only-the-given-numbers content as the
    live path, just without LLM phrasing. Still explains each metric's
    meaning and which direction is better, just via fixed sentences instead
    of free-form LLM prose."""
    if mode == "single":
        confidence = data.get("confidence")
        confidence_note = (
            f" (구조 신뢰도 pLDDT {confidence}, 0-100 스케일 중 '{_confidence_band(confidence)}' 등급 — "
            f"100에 가까울수록 좋습니다)" if confidence else ""
        )
        docking_kind = "실제 AutoDock Vina 도킹" if data.get("docking_source") == "vina" else "분자 특성 기반 휴리스틱(Vina 미사용, 참고용 근사치)"
        lipinski = data.get("lipinski_violations")
        lipinski_note = (
            f" Lipinski's Rule of Five 위반 {lipinski}회입니다(0회에 가까울수록 경구 약물로 유리)."
            if lipinski is not None else ""
        )
        admet = data.get("admet") or {}
        pains_flagged = bool(admet.get("valid") and admet["pains"]["flagged"])
        return (
            f"타겟 구조는 {data.get('target_source')}에서 확보했습니다{confidence_note}. "
            f"결합친화도는 {data.get('affinity')} kcal/mol로 계산되었으며, {docking_kind} 결과입니다. "
            f"{_AFFINITY_GUIDE}{lipinski_note} {_SOURCE_GUIDE} "
            f"본 요약은 자동 생성되었으며, 최종 판단 전 전문가 검토가 필요합니다."
            f"{_demo_recommended_actions(pains_flagged, data.get('structural_analysis_available', False))}"
        )
    top = data.get("top_candidates") or []
    if not top:
        return f"{data.get('n_candidates', 0)}개 후보를 스크리닝했으나 기준을 통과한 후보가 없었습니다."
    lead = top[0]
    breakdown_note = ""
    if lead.get("affinity_score") is not None and lead.get("drug_likeness_score") is not None:
        breakdown_note = (
            f" (친화도 점수 {lead['affinity_score']}/100, drug-likeness 점수 {lead['drug_likeness_score']}/100)"
        )
    lead_admet = lead.get("admet") or {}
    pains_flagged = bool(lead_admet.get("valid") and lead_admet["pains"]["flagged"])
    return (
        f"{data.get('n_candidates')}개 후보 중 {lead['name']}이(가) 친화도 {lead.get('affinity_kcal_mol')} kcal/mol로 "
        f"가장 높은 종합 점수({lead.get('final_score')}/100){breakdown_note}를 기록했습니다. "
        f"{_AFFINITY_GUIDE} {_SCORE_GUIDE} {_LIKENESS_GUIDE} "
        f"이 결과는 결정론적 도킹/랭킹 엔진의 산출값을 그대로 요약한 것이며, 실험적 검증이 필요합니다."
        f"{_demo_recommended_actions(pains_flagged, data.get('structural_analysis_available', False))}"
    )


def generate_ai_summary(
    mode: str, structure: dict, result_payload, report: dict,
    structural_analysis: dict | None = None,
) -> str:
    """
    Narrates the pipeline's own already-computed numbers in plain language
    for the final report — never a source of new scientific values (mirrors
    ScientificReflectionAgent.reflect()'s same discipline). result_payload is
    a single docking dict for mode="single", or the ranked candidate list for
    mode="screen". Explicitly explains each metric's meaning and which
    direction is better (not just narrating the numbers), per the report's
    "AI 해설" needing to double as a reader's interpretation guide.

    Also appends a real "recommended next actions" section (part of the
    "acts like a research collaborator, not just a docking tool" UX goal) —
    grounded only in this job's real ADMET/structural-analysis flags and
    _AVAILABLE_FOLLOWUPS' real, already-built follow-up capabilities. Never
    invents a capability or a flag that isn't actually true of this result.
    """
    structural_analysis_available = bool(
        structural_analysis and (structural_analysis.get("binding_pocket") or {}).get("available")
    )
    if mode == "single":
        docking = result_payload or {}
        ligand = docking.get("ligand_analysis") or {}
        data = {
            "target_source":       structure.get("source"),
            "confidence":          structure.get("confidence"),
            "affinity":            docking.get("best_affinity_kcal_mol"),
            "docking_source":      docking.get("source"),
            "lipinski_violations": ligand.get("lipinski_violations"),
            "strengths":           report.get("strengths"),
            "weaknesses":          report.get("weaknesses"),
            "admet":               docking.get("admet"),
            "structural_analysis_available": structural_analysis_available,
        }
    else:
        ranked = result_payload or []
        data = {
            "target_source": structure.get("source"),
            "confidence":    structure.get("confidence"),
            "n_candidates":  len(ranked),
            "top_candidates": [
                {
                    "name":                c["name"],
                    "affinity_kcal_mol":   c.get("best_affinity_kcal_mol"),
                    "final_score":         c.get("score"),
                    "affinity_score":      (c.get("score_breakdown") or {}).get("affinity"),
                    "drug_likeness_score": (c.get("score_breakdown") or {}).get("drug_likeness"),
                    "docking_source":      c.get("docking_source"),
                    "admet":               c.get("admet"),
                }
                for c in ranked[:5]
            ],
            "strengths":  report.get("strengths"),
            "weaknesses": report.get("weaknesses"),
            "structural_analysis_available": structural_analysis_available,
        }

    if _use_demo():
        return _demo_ai_summary(mode, data)

    try:
        return _chat(
            system=(
                "당신은 신약개발 결과를 연구자에게 설명하는 과학 커뮤니케이터입니다. 주어진 실제 데이터만 근거로 "
                "한국어로 자세히 해설하세요 (7-10문장). 이 해설은 리포트 독자가 각 수치를 어떻게 해석해야 "
                "하는지 알려주는 가이드 역할도 해야 하므로, 등장하는 지표마다 (1) 그 값이 무엇을 의미하는지와 "
                "(2) 어떤 방향/어느 값에 가까울수록 더 좋은 결과인지를 반드시 함께 설명하세요:\n"
                f"- {_AFFINITY_GUIDE}\n"
                f"- {_SCORE_GUIDE}\n"
                f"- {_LIKENESS_GUIDE}\n"
                "- 구조 신뢰도(confidence, pLDDT, 값이 주어진 경우): 0-100 스케일이며, AlphaFold 공식 기준으로 "
                "90 이상이면 매우 높은 신뢰도, 70-90이면 신뢰할 만함, 70 미만이면 예측 신뢰도가 낮습니다. "
                "100에 가까울수록 좋습니다.\n"
                f"- {_SOURCE_GUIDE}\n"
                "위 지표들의 실제 수치(특히 상위 후보들)를 구체적으로 인용하며 설명하고, 한계점도 함께 짚어 "
                "주세요. 새로운 수치나 사실을 절대 지어내지 마세요 — 오직 주어진 값만 서술하세요.\n\n"
                "마지막에 반드시 '추천 다음 행동' 섹션을 추가하세요 — 아래 실제로 시스템이 수행 가능한 "
                "기능 목록에서만 골라 2-3개를 이 결과의 실제 데이터(admet.pains.flagged, "
                "structural_analysis_available 등)에 근거해 우선순위를 매겨 추천하세요. 목록에 없는 기능을 "
                "지어내지 마세요 — 사용자가 그대로 입력할 수 있는 문구를 따옴표로 인용하세요:\n"
                + "\n".join(f"- {a}" for a in _AVAILABLE_FOLLOWUPS)
            ),
            user=json.dumps(data, ensure_ascii=False),
            temperature=0.3,
            max_tokens=700,
        ) or _demo_ai_summary(mode, data)
    except Exception as exc:
        logger.warning("[drug_discovery_agent] AI summary generation failed, using template | error=%s", exc)
        return _demo_ai_summary(mode, data)
