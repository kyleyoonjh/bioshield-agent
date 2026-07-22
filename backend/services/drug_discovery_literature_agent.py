"""
Drug Discovery Assistant — Literature Agent (Phase 2, "Literature Agent" of
the Agentic Drug Discovery AI Platform master plan).

Standalone: own OpenAI client, no shared code with agent/__init__.py or
api/agent_router.py (off-limits per this project's isolation requirement),
matching the same convention as drug_discovery_intent.py/drug_discovery_
chat.py.

summarize_literature() is grounded ONLY in the real papers
services/literature_engine.search_pubmed() actually fetched — every
finding must cite a PMID from that real list. The LLM never invents a
paper, PMID, or scientific claim; when OPENAI_API_KEY is unset, the demo
fallback is a real extractive summary (first real sentence of each real
abstract), not a fabricated narrative.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_SYSTEM_PROMPT = """당신은 신약개발 연구자를 위한 문헌 리뷰어입니다.
아래 JSON으로 주어지는 실제 PubMed 논문 목록(제목/초록/저널/연도/PMID)만 근거로 답하세요.
이 목록에 없는 논문, PMID, 저자, 수치, 결론은 절대 지어내지 마세요 — 목록이 비어있거나
관련성이 낮으면 그 사실을 정직하게 말하세요.

각 핵심 발견(key_findings)에는 반드시 그 근거가 된 논문의 PMID를 명시하세요.
limitations에는 이 논문 목록만으로 알 수 없는 것(예: 검색된 논문 수가 적음, 최신 연구가
없을 수 있음, 전임상 단계뿐일 수 있음 등)을 실제 목록 상태에 근거해 서술하세요.

간결하게 쓰세요 — 핵심 발견은 논문 1건당 1개, 한 문장으로 씁니다.

반드시 아래 JSON 형식으로만 답하세요:
{
  "evidence_summary": "전체 논문 목록을 종합한 한국어 요약 (2-4문장)",
  "key_findings": [{"finding": "한국어 한 문장", "pmid": "해당 근거 PMID"}],
  "limitations": "이 검색 결과의 한계 (한국어 1-2문장)"
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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_real_sentence(abstract: str | None) -> str:
    if not abstract:
        return ""
    # Structured abstracts carry real section labels ("BACKGROUND: ...")
    # from literature_engine._parse_article — strip only that prefix, not
    # any part of the actual sentence content.
    text = re.sub(r"^[A-Z][A-Z /-]+:\s*", "", abstract.strip(), count=1)
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return sentences[0].strip() if sentences else text.strip()


# A PubMed abstract runs 1,500-2,500 characters, and the whole set used to be
# pasted into the prompt verbatim — 1,634 input tokens for three papers. The
# summary only needs each paper's claim, which lives in its opening lines; the
# methods and statistics that fill the tail cost tokens (and time, and 429s)
# without changing what gets written. The papers themselves are returned to the
# caller intact under "references" — only the LLM's copy is shortened.
_PROMPT_ABSTRACT_CHARS = 700


def _for_prompt(papers: list[dict]) -> list[dict]:
    return [
        {**p, "abstract": (p.get("abstract") or "")[:_PROMPT_ABSTRACT_CHARS]}
        for p in papers
    ]


def _demo_summary(papers: list[dict]) -> dict:
    if not papers:
        return {
            "evidence_summary": "PubMed에서 관련 논문을 찾지 못했습니다.",
            "key_findings": [],
            "limitations": "검색된 논문이 없어 근거 기반 요약을 제공할 수 없습니다.",
        }
    key_findings = [
        {"finding": _first_real_sentence(p.get("abstract")) or p.get("title", ""), "pmid": p["pmid"]}
        for p in papers if p.get("abstract") or p.get("title")
    ]
    return {
        "evidence_summary": (
            f"PubMed에서 관련 논문 {len(papers)}건을 찾았습니다 (OPENAI_API_KEY 미설정으로 "
            f"실제 초록의 첫 문장만 발췌하는 추출 요약입니다 — 종합 서술은 아닙니다)."
        ),
        "key_findings": key_findings,
        "limitations": "LLM 종합 요약이 비활성화된 데모 모드이므로, 각 논문 초록을 직접 확인하는 것을 권장합니다.",
    }


def summarize_literature(target_query: str, papers: list[dict]) -> dict:
    """
    Returns {"evidence_summary": str, "key_findings": [{"finding","pmid"}],
    "limitations": str, "references": [...]} — "references" is always the
    real papers list unmodified (never touched by the LLM), so callers can
    always show real PMIDs/links regardless of whether the narrative
    generation succeeded.
    """
    if _use_demo():
        result = _demo_summary(papers)
    elif not papers:
        result = {
            "evidence_summary": f"'{target_query}'에 대해 PubMed에서 관련 논문을 찾지 못했습니다.",
            "key_findings": [],
            "limitations": "검색된 논문이 없어 근거 기반 요약을 제공할 수 없습니다.",
        }
    else:
        try:
            client = _get_client()
            user_content = json.dumps(
                {"target_query": target_query, "papers": _for_prompt(papers)}, ensure_ascii=False)
            resp = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                # Latency here is dominated by tokens GENERATED, not by thinking:
                # the old 800-token ceiling let the model write a 372-token essay
                # when the schema calls for a 3-5 sentence summary and one finding
                # per paper. Capping it is the single biggest lever on this tool's
                # response time. JSON mode also removes the ``` fences _parse_json
                # otherwise has to strip.
                max_tokens=450,
                response_format={"type": "json_object"},
                # An OpenAI stall must not eat the caller's whole budget — the
                # extractive fallback below is a real summary, so failing over to
                # it fast beats waiting.
                timeout=6.0,
            )
            parsed = _parse_json(resp.choices[0].message.content or "")
            real_pmids = {p["pmid"] for p in papers}
            key_findings = [
                kf for kf in (parsed.get("key_findings") or [])
                if isinstance(kf, dict) and kf.get("pmid") in real_pmids
            ]
            result = {
                "evidence_summary": parsed.get("evidence_summary") or _demo_summary(papers)["evidence_summary"],
                "key_findings": key_findings,
                "limitations": parsed.get("limitations") or "",
            }
        except Exception as exc:
            logger.warning("[drug_discovery_literature] LLM summarization failed, using extractive fallback | error=%s", exc)
            result = _demo_summary(papers)

    result["references"] = papers
    return result
