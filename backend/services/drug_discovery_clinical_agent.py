"""
Drug Discovery Assistant — Clinical Intelligence Agent (Phase 2 of the
Agentic Drug Discovery AI Platform master plan).

Standalone: own OpenAI client, no shared code with agent/__init__.py or
api/agent_router.py (off-limits per this project's isolation requirement) —
mirrors drug_discovery_literature_agent.py's structure (real fetch +
grounded LLM narrative + real extractive demo fallback), applied to
ClinicalTrials.gov data instead of PubMed.

summarize_clinical_landscape() is grounded ONLY in the real trials
services/clinical_trials_engine.search_clinical_trials() actually fetched —
every cited trial must be a real NCT ID from that list. The LLM never
invents a trial, NCT ID, phase, or outcome.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter

logger = logging.getLogger(__name__)

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_SYSTEM_PROMPT = """당신은 신약개발 연구자를 위한 임상 인텔리전스 분석가입니다.
아래 JSON으로 주어지는 실제 ClinicalTrials.gov 임상시험 목록(NCT ID/제목/상태/단계/중재/스폰서)만
근거로 답하세요. 이 목록에 없는 임상시험, NCT ID, 결과, 승인 여부는 절대 지어내지 마세요 —
목록이 비어있거나 관련성이 낮으면 그 사실을 정직하게 말하세요. ClinicalTrials.gov에 등록된
시험이 있다고 해서 그 약물이 승인되었다는 뜻은 아니며, 이 데이터만으로 규제 승인 여부를
단정하지 마세요.

각 핵심 시험(key_trials)에는 반드시 그 근거가 된 NCT ID를 명시하세요.

반드시 아래 JSON 형식으로만 답하세요:
{
  "landscape_summary": "임상 개발 현황을 종합한 한국어 요약 (3-5문장)",
  "key_trials": [{"note": "한국어 서술", "nct_id": "해당 NCT ID"}],
  "development_stage_assessment": "가장 진전된 단계와 전반적 개발 단계에 대한 한국어 평가 (1-2문장)"
}"""


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


def _demo_summary(trials: list[dict]) -> dict:
    if not trials:
        return {
            "landscape_summary": "ClinicalTrials.gov에서 관련 임상시험을 찾지 못했습니다.",
            "key_trials": [],
            "development_stage_assessment": "등록된 임상시험이 없어 개발 단계를 평가할 수 없습니다.",
        }
    status_counts = Counter(t.get("overall_status") for t in trials if t.get("overall_status"))
    phase_counts = Counter(p for t in trials for p in (t.get("phases") or []))
    key_trials = [
        {"note": f"{t.get('brief_title', '')} — {t.get('overall_status', '?')}", "nct_id": t["nct_id"]}
        for t in trials
    ]
    return {
        "landscape_summary": (
            f"ClinicalTrials.gov에서 관련 임상시험 {len(trials)}건을 찾았습니다 "
            f"(상태 분포: {dict(status_counts)}, 단계 분포: {dict(phase_counts)}) "
            f"(OPENAI_API_KEY 미설정으로 실제 데이터 집계만 제공하는 데모 모드입니다)."
        ),
        "key_trials": key_trials,
        "development_stage_assessment": (
            f"가장 흔한 단계: {phase_counts.most_common(1)[0][0] if phase_counts else '정보 없음'}."
        ),
    }


def summarize_clinical_landscape(target_query: str, trials: list[dict]) -> dict:
    """
    Returns {"landscape_summary": str, "key_trials": [{"note","nct_id"}],
    "development_stage_assessment": str, "references": [...]} —
    "references" is always the real trials list unmodified.
    """
    if _use_demo():
        result = _demo_summary(trials)
    elif not trials:
        result = {
            "landscape_summary": f"'{target_query}'에 대해 ClinicalTrials.gov에서 관련 임상시험을 찾지 못했습니다.",
            "key_trials": [],
            "development_stage_assessment": "등록된 임상시험이 없어 개발 단계를 평가할 수 없습니다.",
        }
    else:
        try:
            client = _get_client()
            user_content = json.dumps({"target_query": target_query, "trials": trials}, ensure_ascii=False)
            resp = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                # Same reasoning as the literature agent: generated tokens are
                # what this call spends its time on, and a stalled OpenAI request
                # must fail over to the extractive fallback rather than burn the
                # tool's whole budget under Kakao's ~10s timeout.
                max_tokens=450,
                response_format={"type": "json_object"},
                timeout=6.0,
            )
            parsed = _parse_json(resp.choices[0].message.content or "")
            real_nct_ids = {t["nct_id"] for t in trials}
            key_trials = [
                kt for kt in (parsed.get("key_trials") or [])
                if isinstance(kt, dict) and kt.get("nct_id") in real_nct_ids
            ]
            result = {
                "landscape_summary": parsed.get("landscape_summary") or _demo_summary(trials)["landscape_summary"],
                "key_trials": key_trials,
                "development_stage_assessment": parsed.get("development_stage_assessment") or "",
            }
        except Exception as exc:
            logger.warning("[drug_discovery_clinical] LLM summarization failed, using extractive fallback | error=%s", exc)
            result = _demo_summary(trials)

    result["references"] = trials
    return result
