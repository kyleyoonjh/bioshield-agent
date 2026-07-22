"""
Drug Discovery Assistant — conversational Q&A layer.

Mirrors agent/__init__.py's ScientificChat / running_job_reply /
classify_action pattern in CONCEPT only — zero shared code, own standalone
OpenAI client (same convention as drug_discovery_intent.py/drug_discovery_
agent.py), per this feature's isolation requirement.

This closes a real, concrete gap: drug_discovery_intent.py only ever
classifies a message into "start_design" or "chat" (slot-filling), so once
a job started, every subsequent message got the same canned "이미 진행
중입니다..." line no matter what was actually asked, and once a job
completed there was no way to ask about its results at all — the assistant
behaved like a fixed pipeline rather than a research collaborator, unlike
the primer-design agent's ScientificChat.ask()/running_job_reply().

Two real capabilities added:
  - answer_completed_job_question(): real, cited Q&A about a job's ACTUAL
    computed results (ranked_candidates/ai_summary/structure/evaluation) —
    never fabricates a value that isn't already in that data.
  - answer_running_job_question(): conversational replies while a job is
    still RUNNING, using live job state, instead of one static line.

classify_drug_discovery_action() decides start_design / ask_question / chat
for a given message + current job state (mirrors classify_action()).
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_KOREAN_CHARS_RE = re.compile(r"[가-힣]+")

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

_openai_client = None


def _use_demo() -> bool:
    return not bool(_OPENAI_API_KEY)


# Every LLM call in the project funnels through this client — the neoantigen
# interpretation, the decision narrative, the Korean->English query translation,
# intent parsing. It had no timeout and the SDK's default retry policy, so a burst
# of OpenAI 429s (easy to hit: the vaccine pipeline runs right after the literature
# and clinical tools, all on the same token budget) made the SDK sit in backoff for
# tens of seconds with nothing to stop it. Measured: a vaccine job that normally
# finishes in 5.6s took over 41 seconds and was still RUNNING — on Kakao that is a
# model polling forever and a user watching nothing happen.
#
# The narration is a garnish on numbers that are already computed. It is not worth
# one second of a stalled job, let alone forty: bound it, and let the callers fall
# back to their deterministic summaries (every one of them has one).
_LLM_TIMEOUT_S = 8.0
_LLM_MAX_RETRIES = 1


def _get_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=_OPENAI_API_KEY,
            timeout=_LLM_TIMEOUT_S,
            max_retries=_LLM_MAX_RETRIES,
        )
    return _openai_client


def _chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# A small curated map of very common disease/pathogen terms — checked
# first (free, instant, no LLM round-trip) before falling back to a real
# LLM translation. Real terms only, no invented synonyms.
_KOREAN_TERM_MAP = {
    "결핵": "tuberculosis", "결핵균": "tuberculosis Mycobacterium",
    "코로나": "COVID-19 SARS-CoV-2", "코로나바이러스": "coronavirus SARS-CoV-2", "코비드": "COVID-19",
    "독감": "influenza", "인플루엔자": "influenza",
    "암": "cancer", "폐암": "lung cancer", "유방암": "breast cancer", "위암": "gastric cancer",
    "대장암": "colorectal cancer", "간암": "liver cancer",
    # Real reported gap: only 4 specific cancer types were curated, so any
    # other real "-암" cancer term (e.g. "췌장암 논문 찾아줘") fell back to
    # the generic "암"->"cancer" entry below instead of its actual type —
    # see _translate_korean_query()'s _CANCER_TERM_RE handling for the
    # general-purpose fix (routes any uncurated "-암" term through the LLM
    # instead of silently defaulting to "cancer"); these are just the fast,
    # no-LLM-round-trip common cases.
    "췌장암": "pancreatic cancer", "전립선암": "prostate cancer", "난소암": "ovarian cancer",
    "신장암": "kidney cancer renal cell carcinoma", "방광암": "bladder cancer",
    "갑상선암": "thyroid cancer", "식도암": "esophageal cancer", "자궁경부암": "cervical cancer",
    "자궁내막암": "endometrial cancer", "구강암": "oral cancer", "피부암": "skin cancer",
    "흑색종": "melanoma", "백혈병": "leukemia", "림프종": "lymphoma", "다발골수종": "multiple myeloma",
    "뇌종양": "brain tumor glioma", "골육종": "osteosarcoma",
    "당뇨": "diabetes", "당뇨병": "diabetes mellitus",
    "에이즈": "HIV AIDS", "말라리아": "malaria",
}

# Matches any real Korean "-암" (cancer) compound word of 2+ characters
# (e.g. "췌장암", "담낭암") but NOT the standalone generic term "암" itself
# (which needs at least 1 preceding Hangul character to match) — used by
# _translate_korean_query() to detect "this names a SPECIFIC cancer type,
# just not one in the curated map above" so it isn't silently flattened to
# generic "cancer".
_CANCER_TERM_RE = re.compile(r"[가-힣]{1,6}암")

_QUERY_TRANSLATE_SYSTEM = """당신은 생의학 검색어 번역기입니다. 사용자의 한국어 문장에서 연구 주제(질병/
병원체/약물/유전자 등)만 추출해 PubMed·ClinicalTrials.gov 검색에 적합한 간결한 영어 검색어로 번역하세요.
"찾아줘"/"논문"/"알려줘"/"현황"/"임상시험" 같은 지시어·메타 표현은 제외하고, 실제 존재하는 의학/생물학
용어만 사용하세요 — 지어내지 마세요. 문장에 실제 질병/병원체/약물/유전자 이름이 전혀 없다면 (예: "관련
논문 찾아줘", "임상시험 현황 알려줘"처럼 지시어만 있는 경우) 절대 추측하지 말고 정확히 NONE 이라고만
답하세요. 설명 없이 검색어 문자열 또는 NONE만 출력하세요 (예: "결핵 관련 논문 찾아줘" -> "tuberculosis")."""


def _translate_korean_query(message: str) -> str:
    """
    Real translation of a Korean research topic into an English search
    query — PubMed/ClinicalTrials.gov's search indexes are effectively
    English-only, so a bare Korean term (e.g. "결핵 논문 찾아줘") returns
    zero real hits even though real papers/trials exist under the English
    name (confirmed live: same bug pattern already fixed for live UniProt
    search in drug_discovery_intent.py's organism_query translation).
    Checks the small curated term map first; falls back to a real LLM call
    when available. In demo mode with no map hit, returns the original
    text untranslated (honest limitation, never a guessed translation).

    Returns "" (not a guess) when the message has no real topic to
    translate — real reported bug: a topic-less message like "관련 논문
    찾아줘" or "임상시험 현황 알려줘" previously made the LLM invent a
    plausible-looking but wrong "translation" (either a Korean refusal
    sentence like "관련 주제를 제공해 주세요", or, worse, non-Korean text
    like "clinical trials" — a description of the REQUEST, not an actual
    topic — that slipped past a Korean-character-only check and got
    treated as if the message named a genuine new topic). The prompt now
    requires an explicit "NONE" sentinel instead of guessing.
    """
    # Real reported bug: "대장암 문헌 검색" returned generic "cancer" papers
    # instead of "colorectal cancer" ones — "암" ("cancer") is itself a
    # substring of "대장암" ("colorectal cancer") and sat earlier in the
    # dict literal, so simple first-match-wins iteration matched the
    # generic term before ever reaching the more specific one. Same bug
    # would hit every other specific cancer type in this map (폐암/유방암/
    # 위암/간암 all contain "암"). Prefers the LONGEST matching key instead —
    # correct regardless of dict insertion order. Excludes the generic "암"
    # entry from this pass on purpose (handled separately below) — real
    # follow-up bug: "췌장암 논문 찾아줘" (pancreatic cancer, not in the
    # curated map) still matched "암" as a substring and short-circuited to
    # generic "cancer" here, never reaching the LLM translation that would
    # have correctly identified it.
    matches = [(kor, eng) for kor, eng in _KOREAN_TERM_MAP.items() if kor in message and kor != "암"]
    if matches:
        _, eng = max(matches, key=lambda pair: len(pair[0]))
        return eng
    # The message names a specific "-암" cancer type this curated map
    # doesn't cover (e.g. "췌장암") — let the LLM translate it properly
    # instead of falling back to generic "cancer" immediately below. Only
    # standalone "암" (no specific type attached) uses the generic fallback.
    names_uncurated_cancer_type = bool(_CANCER_TERM_RE.search(message))
    if "암" in message and not names_uncurated_cancer_type:
        return _KOREAN_TERM_MAP["암"]
    if _use_demo():
        # No LLM available to translate an uncurated "-암" type correctly —
        # honest limitation (see docstring): return the original text
        # untranslated rather than guess generic "cancer", which is exactly
        # the wrong-answer pattern this function exists to avoid.
        return message
    try:
        translated = _chat(_QUERY_TRANSLATE_SYSTEM, message, temperature=0.0, max_tokens=30).strip()
        if translated.upper() == "NONE":
            return ""
        return translated or message
    except Exception as exc:
        logger.warning("[drug_discovery_chat] query translation failed, using original text | error=%s", exc)
        return message


def _clean_external_search_query(message: str, target_name: str | None) -> str:
    """
    Confirmed via direct live testing: ClinicalTrials.gov's search API
    returns ZERO results when Korean text is mixed into query.term, even
    though the same query in clean English finds real trials (e.g.
    "SARS-CoV-2 spike protein inhibitor" -> 17 real hits; the same string
    with trailing Korean instruction words like "관련 임상시험 현황 알려줘"
    appended -> 0 hits). PubMed happened to tolerate the same noisy string
    in one test, but that tolerance isn't guaranteed, so both external
    search agents route through this same cleaning step. Prefers the
    already-English target_name (ground-truth canonical name, set only by
    drug_discovery_intent.py's deterministic resolution paths, never
    LLM-guessed) when available; otherwise strips Korean characters from
    the raw message and keeps only the English/ASCII remainder — and when
    NOTHING survives that strip (the message was entirely Korean, e.g.
    "결핵 논문 찾아줘"), real-translates the original message instead of
    silently searching with untranslatable Korean text (confirmed real
    bug: this previously always returned "관련 논문을 찾지 못했습니다").
    """
    if target_name:
        return target_name
    cleaned = " ".join(_KOREAN_CHARS_RE.sub(" ", message).split())
    if cleaned:
        return cleaned
    translated = _translate_korean_query(message)
    # Real reported bug: a topic-less message ("관련 논문 찾아줘" — no actual
    # disease/pathogen/gene named) left the LLM nothing real to translate,
    # and instead of a clean search term it sometimes returns a Korean
    # clarifying question of its own (e.g. "관련 주제를 제공해 주세요") —
    # that got used as the literal PubMed/ClinicalTrials.gov query and
    # silently searched, returning a confusing "'관련 주제를 제공해
    # 주세요.'에 대해 찾지 못했습니다" reply. If the "translation" still has
    # Korean characters, it demonstrably isn't a real English search term —
    # treat this as "no topic" (empty string) so the caller can ask instead
    # of searching with garbage.
    if _KOREAN_CHARS_RE.search(translated):
        return ""
    return translated


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


# ── Action classification ────────────────────────────────────────────────────

_RESULT_QUESTION_KEYWORDS = (
    "점수", "친화도", "affinity", "score", "순위", "rank", "1위", "몇위", "몇 위", "왜",
    "drug-likeness", "admet", "결과", "이 후보", "top", "구조 신뢰도", "plddt", "신뢰도",
    "결합 포켓", "바인딩 포켓", "binding pocket", "포켓", "잔기", "residue", "수소결합", "hydrogen bond",
    "상호작용", "interaction", "pae", "정렬 오차",
)

_ROUTER_SYSTEM_TEMPLATE = """당신은 신약개발 어시스턴트의 의도 라우터입니다. 연구자의 최신 메시지와 현재 작업 상태를 보고
정확히 하나의 액션을 결정하세요:
- "start_design": current_target과 다른 새로운 타겟/병원체/변이에 대한 도킹·스크리닝을 시작하려는 요청
  (예: current_target이 "SARS-CoV-2 Spike glycoprotein"인데 메시지가 "결핵균을 억제할 수 있는 승인된
  약물을 찾아줘"라면, 결핵균은 전혀 다른 타겟이므로 반드시 start_design입니다 — current_target의 기존
  후보 물질에 대한 질문으로 오독하지 마세요.)
- "ask_question": 이미 완료된(COMPLETED) 작업의 실제 결과(current_target 및 그 후보 물질)에 대한 질문
- "chat": 인사, 잡담, 또는 새 설계 요청이 아닌 일반 과학 질문

현재 상태: {state}

반드시 JSON만 반환: {{"action": "start_design" 또는 "ask_question" 또는 "chat"}}"""


def classify_drug_discovery_action(
    message: str, has_job: bool, job_status: str | None, target_name: str | None = None,
) -> str:
    """Returns "start_design" | "ask_question" | "chat"."""
    if _use_demo():
        return _classify_demo(message, has_job, job_status)

    # Real reported bug: the state string never told the LLM WHAT the
    # current target actually is, so it had no way to judge whether a
    # message naming a target ("결핵균을 억제할 수 있는 승인된 약물을
    # 찾아줘") was a NEW design request or a question about the EXISTING
    # one — confirmed live: this exact message got misclassified as
    # ask_question right after a SARS-CoV-2 Spike job completed, and got
    # answered as if asking whether the Spike job's own candidates also
    # inhibit tuberculosis (a real but wrong-question answer) instead of
    # being redirected to the "새 연구" button. Including current_target
    # lets the LLM actually compare the mentioned target against it.
    state = f"active_job={has_job}, status={job_status}, current_target={target_name or 'unknown'}"
    try:
        raw = _chat(_ROUTER_SYSTEM_TEMPLATE.format(state=state), message, temperature=0.0, max_tokens=60)
        action = _parse_json(raw).get("action", "chat")
    except Exception as exc:
        logger.warning("[drug_discovery_chat] router classification failed, defaulting to chat: %s", exc)
        action = "chat"
    if action not in ("start_design", "ask_question", "chat"):
        action = "chat"

    # Safety net, same discipline as agent/__init__.py's classify_action():
    # the LLM router can miss Korean result-lookup phrasing even at temp=0.
    if action == "chat" and has_job and job_status == "COMPLETED":
        low = message.lower()
        if any(k in message or k in low for k in _RESULT_QUESTION_KEYWORDS):
            action = "ask_question"
    return action


def _classify_demo(message: str, has_job: bool, job_status: str | None) -> str:
    if has_job and job_status == "COMPLETED":
        low = message.lower()
        if any(k in message or k in low for k in _RESULT_QUESTION_KEYWORDS):
            return "ask_question"
    return "chat"


# ── Q&A about a COMPLETED job's real results ────────────────────────────────

_QA_SYSTEM = """당신은 신약개발 결과에 대해 연구자와 대화하는 과학 커뮤니케이터입니다.
아래 주어진 실제 계산 결과(ranked_candidates, ai_summary, structure, evaluation, structural_analysis,
admet, 또는 mode="neoantigen"인 경우 candidates/hla_note/literature_by_gene)만 근거로 답하세요.
새로운 수치나 사실을 절대 지어내지 마세요 — 데이터에 없는 내용은 "그 정보는
이 결과에 없습니다"라고 솔직히 답하세요. admet은 Veber's rule/PAINS/합성 접근성(SA score)/간독성
구조 경보만 실제로 계산된 것이며, hERG/CYP 억제/BBB 투과성/독성은 검증된 모델이 없어 절대 지어내지
마세요 — 질문받으면 "제공되지 않습니다"라고 답하세요. mode="neoantigen"인 경우 hla_note에 명시된 대로
BAM 기반 실제 HLA 타이핑은 수행되지 않았고 population-common allele을 사용했다는 점을 반드시
전제로 깔고 답하세요 — 이 환자의 실제 HLA 유전형인 것처럼 말하지 마세요. 답할 때는 관련 후보의
순위(#)나 실제 수치를 구체적으로 인용하세요. 한국어로 2-4문장, 간결하게 답하세요."""


def answer_completed_job_question(question: str, job_result: dict) -> str:
    if _use_demo():
        return _qa_demo(question, job_result)

    if job_result.get("mode") == "neoantigen":
        context = {
            "mode": "neoantigen",
            "mutations_analyzed": job_result.get("mutations_analyzed"),
            "hla_alleles": job_result.get("hla_alleles"),
            "hla_note": job_result.get("hla_note"),
            "bam_summary": job_result.get("bam_summary"),
            "candidates": job_result.get("candidates"),
            "ai_interpretation": job_result.get("ai_interpretation"),
            "algorithm_explanation": job_result.get("algorithm_explanation"),
            "prediction_errors": job_result.get("prediction_errors"),
            "literature_by_gene": job_result.get("literature_by_gene"),
        }
        try:
            return _chat(
                _QA_SYSTEM,
                f"실제 결과 데이터:\n{json.dumps(context, ensure_ascii=False, default=str)}\n\n연구자 질문: {question}",
                temperature=0.3, max_tokens=500,
            ) or "죄송합니다, 답변을 생성하지 못했습니다."
        except Exception as exc:
            logger.warning("[drug_discovery_chat] answer_completed_job_question (neoantigen) failed: %s", exc)
            return "죄송합니다, 답변을 생성하지 못했습니다."

    ranked = job_result.get("ranked_candidates") or []
    top5 = [
        {
            "rank": c.get("rank"), "name": c.get("name"),
            "affinity_kcal_mol": c.get("best_affinity_kcal_mol"), "final_score": c.get("score"),
            "score_breakdown": c.get("score_breakdown"),
            "strength": c.get("strength"), "weakness": c.get("weakness"),
            "admet": c.get("admet"),
        }
        for c in ranked[:5]
    ]
    context = {
        "mode": job_result.get("mode"),
        "structure_source": job_result.get("structure_source"),
        "structure_confidence": (job_result.get("structure") or {}).get("confidence"),
        "ai_summary": job_result.get("ai_summary"),
        "evaluation": job_result.get("evaluation"),
        "top_candidates": top5,
        "docking_result": job_result.get("docking_result"),
        "variant_context": job_result.get("variant_context"),
        # Real geometry-derived binding-pocket contacts (from the actual
        # docked pose) and real AlphaFold PAE summary when available — both
        # None/"available": False when not computed for this job (e.g. Vina
        # unavailable, or an ESMFold-predicted target with no PAE data), so
        # the LLM must say "이 결과에 없습니다" rather than invent numbers.
        "structural_analysis": job_result.get("structural_analysis"),
    }
    try:
        return _chat(
            _QA_SYSTEM,
            f"실제 결과 데이터:\n{json.dumps(context, ensure_ascii=False, default=str)}\n\n연구자 질문: {question}",
            temperature=0.3, max_tokens=500,
        ) or _qa_demo(question, job_result)
    except Exception as exc:
        logger.warning("[drug_discovery_chat] answer_completed_job_question failed, using demo fallback: %s", exc)
        return _qa_demo(question, job_result)


def _qa_demo(question: str, job_result: dict) -> str:
    q = question.lower()
    ranked = job_result.get("ranked_candidates") or []
    top = ranked[0] if ranked else {}

    if any(w in q for w in ("왜", "1위", "순위", "rank", "why")):
        if top:
            return (
                f"{top.get('name')}이(가) 결합친화도 {top.get('best_affinity_kcal_mol')} kcal/mol, "
                f"종합 점수 {top.get('score')}(100점 만점)로 가장 높아 1위입니다."
            )
        return "완료된 스크리닝 후보 데이터가 없습니다."
    if "drug-likeness" in q or "약물유사성" in q:
        return ("drug-likeness 점수는 Lipinski's Rule of Five 위반 횟수를 기반으로 하며, "
                "100에 가까울수록(위반 0회) 경구 약물로 개발될 가능성이 높다는 뜻입니다.")
    if "admet" in q:
        top_admet = (top or {}).get("admet")
        docking_admet = (job_result.get("docking_result") or {}).get("admet")
        admet = top_admet or docking_admet
        if admet and admet.get("valid"):
            pains = admet["pains"]
            return (
                f"실제 계산된 ADMET 하위 지표: Veber's rule {'충족' if admet['oral_absorption']['prediction'] == 'high' else '미충족'}, "
                f"PAINS 구조 경보 {'있음 (' + ', '.join(pains['alerts']) + ')' if pains['flagged'] else '없음'}, "
                f"합성 접근성(SA score) {admet['synthesis']['score']} (1=쉬움~10=어려움). "
                "hERG/CYP/BBB 투과성/독성 예측은 검증된 모델이 없어 제공하지 않습니다."
            )
        return "이 결과에는 ADMET 데이터가 없습니다 (RDKit 계산이 수행되지 않았을 수 있습니다)."
    if any(w in q for w in ("포켓", "pocket", "잔기", "residue", "수소결합", "hydrogen bond", "상호작용", "interaction")):
        pocket = (job_result.get("structural_analysis") or {}).get("binding_pocket")
        if pocket and pocket.get("available"):
            residues = ", ".join(f"{r['chain']}:{r['resnum']}" for r in pocket["residues_in_contact"][:5])
            return (
                f"실제 도킹 포즈 좌표 기준, 접촉 잔기(가까운 순): {residues}. "
                f"수소결합 후보 {len(pocket['hydrogen_bond_candidates'])}건, "
                f"소수성 접촉 {len(pocket['hydrophobic_contacts'])}건 (거리 기반 추정, {pocket['method']})"
            )
        return "이 작업에는 결합 포켓 분석 데이터가 없습니다 (Vina 실제 도킹이 수행되지 않았을 수 있습니다)."
    if "pae" in q or "정렬 오차" in q:
        pae = (job_result.get("structural_analysis") or {}).get("pae_summary")
        if pae and pae.get("available"):
            return (
                f"AlphaFold 실제 PAE 기준 전체 평균 {pae['overall_mean_pae']}Å, "
                f"결합 포켓 잔기 간 평균 {pae['pocket_residue_mean_pae']}Å 입니다."
            )
        reason = (pae or {}).get("reason", "이 타겟 구조에는 PAE 데이터가 없습니다 (AlphaFold DB 유래 구조만 지원).")
        return reason
    if job_result.get("ai_summary"):
        return job_result["ai_summary"]
    return "완료된 작업의 결과 데이터를 찾지 못했습니다."


# ── Conversational replies WHILE a job is RUNNING ───────────────────────────

_RUNNING_STATUS_SYSTEM = """당신은 신약개발 어시스턴트의 진행상황 안내자입니다.
현재 실제로 도킹/스크리닝 작업이 RUNNING 상태입니다. 주어진 실제 진행 상태를 바탕으로
연구자의 메시지에 대화체로 구체적으로 답하세요 — 매번 똑같은 정형화된 문장을 반복하지 마세요.
파이프라인 단계: 1) 타겟 구조 확보(AlphaFold DB/ESMFold) 2) 리셉터 준비(블라인드 도킹 박스)
3) 후보 도킹/스크리닝(AutoDock Vina) 4) 필터링 5) 랭킹 6) AI 해설 및 리포트 생성.
진행 중에는 전략을 바꿀 수 없습니다 — 바꾸고 싶으면 중지("중지"라고 말하면 됩니다) 후 새로
시작해야 한다고 안내하세요. 아직 계산되지 않은 최종 점수·순위를 절대 지어내지 마세요.
단백질 정체를 절대 추측하지 마세요: 주어진 상태에 target_name이 있으면 그 이름을 그대로 쓰고,
target_name이 비어 있으면 그 UniProt ID가 어떤 단백질인지 이름 붙이거나 설명하지 말고 "UniProt {id}"
로만 지칭하세요 (배경지식으로 지어내면 실제로 틀린 단백질을 말하는 사고로 이어집니다).
한국어로 1-3문장, 간결하고 친근하게 답하세요."""


def answer_running_job_question(message: str, job: dict) -> str:
    if _use_demo():
        return (
            f"지금 '{job.get('uniprot_id', '')}' 타겟에 대해 작업이 진행 중입니다 "
            f"({job.get('current_message', '')}, {job.get('current_step', 0)}/{job.get('total_steps', 3)}단계). "
            "완료되면 바로 알려드릴게요!"
        )
    state = (
        f"uniprot_id={job.get('uniprot_id')}, target_name={job.get('target_name') or '(미확인)'}, "
        f"mode={job.get('mode')}, current_step={job.get('current_step')}/{job.get('total_steps')}, "
        f"current_message={job.get('current_message')}"
    )
    try:
        return _chat(_RUNNING_STATUS_SYSTEM, f"실시간 작업 상태: {state}\n\n연구자: {message}",
                     temperature=0.4, max_tokens=300)
    except Exception as exc:
        logger.warning("[drug_discovery_chat] answer_running_job_question failed, using canned fallback: %s", exc)
        return (
            f"지금 진행 중입니다 ({job.get('current_message', '')}, "
            f"{job.get('current_step', 0)}/{job.get('total_steps', 3)}단계). 완료되면 바로 알려드릴게요!"
        )


# ── Next-step suggestion (shared by Literature/Clinical/Compound Discovery) ──
#
# generate_ai_summary() in drug_discovery_agent.py already recommends a next
# action after a COMPLETED screening job; these job-independent Q&A agents
# previously just dumped data and stopped. This mirrors that same discipline
# one step earlier in the pipeline — every suggestion names only real,
# already-built capabilities (never generic filler), and is skipped for
# whichever agent just answered so the suggestion never repeats the action
# the user just took.
_NEXT_STEP_ACTIONS = {
    "literature": "관련 논문",
    "clinical": "임상시험 현황",
    "target_intelligence": "타겟 우선순위(질병 연관성·경로)",
    "screening": "억제 가능한 승인 약물 스크리닝",
}


def _next_step_suggestion(target_name: str | None, exclude: str) -> str:
    if not target_name:
        return (
            "\n\n다음 단계: 특정 타겟 단백질을 알려주시면 관련 논문·임상시험 현황·타겟 우선순위를 "
            "확인하거나, 바로 억제 가능한 승인 약물 스크리닝을 시작할 수 있습니다."
        )
    others = [label for key, label in _NEXT_STEP_ACTIONS.items() if key != exclude]
    return f"\n\n다음 단계: '{target_name}'에 대해 {' / '.join(others)}도 확인해보시겠어요?"


# ── Neoantigen / mRNA vaccine intent detection ──────────────────────────────
#
# Real reported bug: typing "신항원 후보 찾기" (or similar) as a plain chat
# message had no route at all — it fell through to generic intent-parsing,
# which (with a cancer topic like "대장암" still remembered from goal_text)
# produced a confusing "무엇을 하고 싶은지 구체적으로 말씀해 주세요"-style
# reply instead of actually doing anything, leaving the conversation stuck.
# The real neoantigen pipeline (services/neoantigen_pipeline.py) always
# needs an actual VCF file, which can't be conjured from chat text alone —
# so the correct response to this intent, exactly like the existing
# cancer-topic-without-VCF branch (drug_discovery_intent.py), is to ask for
# the VCF/BAM upload, not attempt anything itself. drug_discovery_router.py
# checks this before intent-parsing so it works regardless of whatever
# stale goal_text/topic happens to be remembered in the session.
_NEOANTIGEN_KEYWORDS = (
    "신항원", "neoantigen", "자가 백신", "백신 후보", "맞춤형 백신", "personalized vaccine",
    "암 백신", "종양 백신", "cancer vaccine", "tumor vaccine",
)


def is_neoantigen_query(message: str) -> bool:
    low = message.lower()
    if any(k in message or k in low for k in _NEOANTIGEN_KEYWORDS):
        return True
    return "mrna" in low and ("백신" in message or "vaccine" in low)


# Real reported gap: typing the demo button's own label text ("암 mRNA 자가
# 백신 데모" / "🧬 mRNA 암 백신") as a chat message matched is_neoantigen_
# query() above, but that only re-showed the exact same "please click the
# button or upload a file" prompt — since is_neoantigen_query() alone can't
# distinguish general neoantigen interest from an explicit request to run
# the sample demo. This lets drug_discovery_router.py start the actual
# sample/NSCLC_variants.vcf + sample/NSCLC.bam job directly from text,
# matching what clicking the demo button does, instead of looping the same
# prompt back — and per a follow-up request, without requiring the message
# to closely match the button's exact label.
#
# "데모"/"demo"/"시연" alone is treated as sufficient on its own — it's the
# ONLY feature in this chat panel actually labeled "데모" (the other sample
# button, small-molecule VCF screening, is labeled "샘플 VCF 사용", never
# "데모" — see handleSampleVcf in DrugDiscoveryChatPanel.tsx), so the word is
# an unambiguous signal within this app regardless of surrounding wording.
# "샘플"/"sample"/"체험"/"테스트"/"예시" are more ambiguous (that other
# button also says "샘플"), so those only count when paired with an actual
# neoantigen/vaccine keyword via is_neoantigen_query().
_NEOANTIGEN_DEMO_UNAMBIGUOUS_KEYWORDS = ("데모", "demo", "시연")
_NEOANTIGEN_DEMO_CONTEXTUAL_KEYWORDS = ("샘플", "sample", "체험", "테스트", "예시")


def is_neoantigen_demo_query(message: str) -> bool:
    low = message.lower()
    if any(k in message or k in low for k in _NEOANTIGEN_DEMO_UNAMBIGUOUS_KEYWORDS):
        return True
    if not is_neoantigen_query(message):
        return False
    return any(k in message or k in low for k in _NEOANTIGEN_DEMO_CONTEXTUAL_KEYWORDS)


# Real reported bug: a genuine conceptual question ("mRNA 암백신이 뭐야?")
# also matches is_neoantigen_query() (contains "mRNA"+"백신"), so it got the
# SAME "실제 VCF 파일이 필요합니다... 데모 버튼을 눌러보세요" action prompt as
# an actual request to run something — the user was asking what the thing
# IS, not asking to identify candidates. Checked in drug_discovery_router.py
# before is_neoantigen_query()'s action-prompt branch (but after is_
# neoantigen_demo_query(), which already requires an explicit "데모"-class
# keyword and so never collides with a plain question) so an explanatory
# question gets an actual explanation instead of being misread as an action.
_NEOANTIGEN_QUESTION_MARKERS = (
    "뭐야", "뭔가요", "뭔지", "무엇인가", "무엇이야", "란 무엇", "이란", "설명해", "원리가",
    "어떻게 작동", "차이가 뭐", "궁금", "what is", "explain",
)


def is_neoantigen_question(message: str) -> bool:
    if not is_neoantigen_query(message):
        return False
    low = message.lower()
    return any(k in message or k in low for k in _NEOANTIGEN_QUESTION_MARKERS)


_NEOANTIGEN_EXPLAINER_SYSTEM = (
    "당신은 신약개발 어시스턴트의 과학 커뮤니케이터입니다. 사용자가 '신항원(neoantigen)' 또는 "
    "'mRNA 암 백신'이 무엇인지 개념을 물었습니다. 확립된 일반 생의학 지식만으로 2-4문장으로 쉽게 "
    "설명하세요 (종양 세포의 체세포 변이로 생긴, 정상 세포에는 없는 새로운 단백질 조각을 면역계가 "
    "인식하도록 mRNA로 전달해 맞춤형 면역반응을 유도하는 원리). 특정 임상 성공률이나 승인 여부 등 "
    "사실이 아닌 수치는 절대 지어내지 마세요. 마지막에 이 앱이 실제로 할 수 있는 것 — 사용자가 "
    "업로드한 VCF(체세포 변이) 파일에서 실제 MHCflurry 모델로 신항원 후보를 예측하는 기능과, 실제 "
    "NSCLC 샘플 데모로 체험해볼 수 있다는 점 — 을 한 문장으로 자연스럽게 안내하세요. 한국어로 "
    "작성하세요."
)

_NEOANTIGEN_EXPLAINER_FALLBACK = (
    "신항원(neoantigen)은 암세포에만 생긴 체세포 변이로 만들어지는, 정상 세포에는 없는 새로운 단백질 "
    "조각입니다. mRNA 암 백신은 이 신항원의 정보를 mRNA로 전달해 면역계가 암세포를 정상 세포와 "
    "구별해 공격하도록 유도하는 환자 맞춤형 치료 접근법입니다. 이 앱은 실제 VCF(체세포 변이) 파일을 "
    "업로드하면 실제 MHCflurry 모델로 신항원 후보를 예측할 수 있습니다 — 아래 '🧬 mRNA 암 백신' 데모 "
    "버튼으로 실제 NSCLC 샘플을 먼저 체험해 보실 수 있습니다."
)


def answer_neoantigen_question(message: str) -> dict:
    if _use_demo():
        return {"text": _NEOANTIGEN_EXPLAINER_FALLBACK}
    try:
        text = _chat(_NEOANTIGEN_EXPLAINER_SYSTEM, message, temperature=0.3, max_tokens=300).strip()
        return {"text": text or _NEOANTIGEN_EXPLAINER_FALLBACK}
    except Exception as exc:
        logger.warning("[drug_discovery_chat] neoantigen explainer failed, using fallback | error=%s", exc)
        return {"text": _NEOANTIGEN_EXPLAINER_FALLBACK}


# ── Literature Agent (Phase 2 of the Agentic Drug Discovery AI Platform) ───
#
# Real live PubMed search (services/literature_engine.py) + grounded
# summary (services/drug_discovery_literature_agent.py) — checked FIRST in
# /converse, independent of any active job or intent-parsing slot state,
# since a literature question ("관련 논문 찾아줘") is meaningful whether or
# not a design job is running/completed/nonexistent. Keyword-based
# detection (not an LLM classification call) mirrors the cheap, deterministic
# stop-phrase detection already used the same way in drug_discovery_router.py.

_LITERATURE_KEYWORDS = (
    "논문", "문헌", "선행 연구", "선행연구", "연구 결과가", "학술 자료",
    "paper", "literature", "publication", "pubmed",
)


def is_literature_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _LITERATURE_KEYWORDS)


# Real reported bug: "결핵에 대해 설명해줘" / "결핵균에 대해 알려줘" — a
# general request to be told about a NAMED topic — don't match any of the
# more specific classifiers (literature/clinical/target-intelligence/
# compound-discovery all require narrower action keywords like "논문"/"임상
# 시험"/"경로"), so they fell through to drug_discovery_intent.py's
# mentions_research_topic branch and got the same 1/2/3 action-clarification
# menu as a bare topic mention ("결핵") — the user was asking to be TOLD
# about the topic, not asking which pipeline to run. Per explicit follow-up
# feedback, answered as a direct general-AI explanation (answer_general_
# explain_question below), NOT the literature-search/PubMed-summary format —
# those are different requests, handled by two different functions.
#
# Deliberately requires the "~에 대해" ("about ~") pattern rather than bare
# "뭐야"/"알려줘" alone, which would otherwise collide with completed-job
# Q&A phrasing like "이게 뭐야?"/"결과 좀 알려줘" that should stay routed to
# answer_completed_job_question() instead (that phrasing refers to "this"/
# "the result", not a named external topic).
_GENERAL_EXPLAIN_RE = re.compile(r"에\s*대해\s*(설명|알려|소개)")


def is_general_explain_query(message: str) -> bool:
    # Only a catch-all for messages the more specific classifiers don't
    # already claim — never steals clinical/target-intelligence/compound-
    # discovery traffic (e.g. "결핵 임상시험 현황에 대해 알려줘" still
    # correctly matches is_clinical_query and must stay clinical).
    if is_clinical_query(message) or is_target_intelligence_query(message) or is_compound_discovery_query(message):
        return False
    return bool(_GENERAL_EXPLAIN_RE.search(message))


_GENERAL_EXPLAIN_SYSTEM = (
    "당신은 신약개발 어시스턴트의 과학 커뮤니케이터입니다. 사용자가 특정 질병/병원체/유전자에 대해 "
    "일반적인 설명을 요청했습니다. 확립된 일반 생의학 지식만으로 2-4문장으로 쉽고 정확하게 설명하세요 "
    "(원인, 특징, 왜 중요한지 등). 최신 통계나 구체적 임상 수치는 절대 지어내지 마세요 — 최신 연구/임상 "
    "현황이 필요하면 '관련 논문 찾아줘'/'임상시험 현황 알려줘'로 실제 데이터를 확인할 수 있다고 마지막에 "
    "한 문장으로 안내하세요. 한국어로 작성하세요."
)


def answer_general_explain_question(message: str) -> dict:
    """
    Direct general-AI conceptual explanation (established biomedical
    knowledge, no fabricated statistics) — deliberately NOT the PubMed-
    citation format answer_literature_question() returns, per explicit user
    feedback that a "what is X" question should get a general AI answer,
    not a literature summary. Falls back to a real literature-search answer
    only when no LLM is available at all (demo mode) — an honest citation-
    backed answer is preferable to silently returning nothing.
    """
    if _use_demo():
        return answer_literature_question(message, None)
    try:
        text = _chat(_GENERAL_EXPLAIN_SYSTEM, message, temperature=0.3, max_tokens=300).strip()
        if not text:
            return answer_literature_question(message, None)
        return {"text": text}
    except Exception as exc:
        logger.warning("[drug_discovery_chat] general explain failed, falling back to literature | error=%s", exc)
        return answer_literature_question(message, None)


# ── Assistant self-identity ("what are you?") ───────────────────────────────
#
# Real reported request: "너는 무슨 프로그램이니?"-style questions about the
# assistant ITSELF (not a research topic) don't match is_general_question
# (that classifier only extracts named disease/pathogen/gene topics) or any
# other classifier here, so they fell through to generic intent-parsing and
# got an unhelpful "구체적으로 말씀해주세요"-style reply. Answered with a
# static, real capability description — never fabricated, just restates what
# this app actually does (see the classifiers/answer functions throughout
# this file). Keep in sync by hand with GREETING_MESSAGE in
# DrugDiscoveryChatPanel.tsx — same content, duplicated because one is
# Python (backend-answered mid-conversation) and one is TypeScript
# (frontend-rendered on first load), with no shared build step between them.
_IDENTITY_KEYWORDS = (
    "너는 뭐", "넌 뭐", "너 뭐", "무슨 프로그램", "무슨 앱", "무슨 서비스", "어떤 프로그램", "어떤 서비스",
    "정체가 뭐", "자기소개", "네 소개", "너 소개", "뭐 하는 애", "뭐하는 애", "뭐 하는 곳", "뭐하는 곳",
    "누구야", "누구니", "what are you", "who are you", "introduce yourself",
)


def is_identity_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _IDENTITY_KEYWORDS)


_IDENTITY_ANSWER = (
    "저는 AiRemedy(AI신약)의 신약개발(Drug Discovery) 어시스턴트입니다 — 실제 데이터와 "
    "계산만 사용하는 신약 개발 리서치 도구예요. 이런 걸 할 수 있습니다:\n\n"
    "📚 문헌 검색 — PubMed 실시간 논문 조회\n"
    "🧬 신약 후보 탐색 — AlphaFold/ESMFold 구조 예측 기반 실제 도킹·스크리닝\n"
    "🏥 임상시험 조회 — ClinicalTrials.gov 실시간 현황\n"
    "🎯 타겟 분석 — 질병 연관성과 생물학적 경로 (UniProt/Reactome)\n"
    "🧫 화합물 검색 — 유사 화합물/알려진 억제제 (PubChem/ChEMBL)\n"
    "📊 최종 레포트 — 우선순위 점수·개발 위험도 종합 평가\n"
    "🧬 mRNA 암 백신(신항원) 후보 식별 — 실제 VCF 파일 기반 MHCflurry 예측\n\n"
    "궁금하신 병원체/타겟/약물 이름을 말씀해 주세요!"
)


def answer_identity_question() -> dict:
    return {"text": _IDENTITY_ANSWER}


def answer_literature_question(message: str, target_name: str | None = None) -> dict:
    """
    target_name, when available (the ground-truth resolved protein/pathogen
    name — see drug_discovery_intent.py's target_name field, never an
    LLM-guessed identity), narrows the real PubMed query so a bare "관련
    논문 찾아줘" after a target was already established searches for the
    right protein instead of just the literal user message text.

    Returns {"text": str, "available": bool, "papers": [...],
    "key_findings": [...], "evidence_summary": str, "limitations": str,
    "query": str} — "text" is a plain-text rendering of the same real data
    for logging/non-card clients; the frontend's dedicated literature card
    renders the structured fields directly instead of parsing "text".
    """
    from services.literature_engine import search_pubmed
    from services.drug_discovery_literature_agent import summarize_literature

    query = _clean_external_search_query(message, target_name)
    if not query:
        text = "어떤 병원균, 질병, 유전자, 또는 약물에 대한 논문을 찾아드릴까요? 이름을 말씀해 주시면 바로 검색하겠습니다."
        return {"text": text, "available": False, "query": "", "papers": []}
    fetched = search_pubmed(query, max_results=5)
    if fetched.get("error"):
        text = f"PubMed 검색 중 오류가 발생했습니다: {fetched['error']}"
        return {"text": text, "available": False, "query": query, "papers": []}
    if not fetched["papers"]:
        text = f"'{query}'에 대해 PubMed에서 관련 논문을 찾지 못했습니다."
        return {"text": text, "available": False, "query": query, "papers": []}

    summary = summarize_literature(query, fetched["papers"])
    lines = [summary["evidence_summary"]]
    if summary["key_findings"]:
        lines.append("\n주요 발견:")
        lines.extend(f"- {kf['finding']} (PMID: {kf['pmid']})" for kf in summary["key_findings"][:5])
    if summary.get("limitations"):
        lines.append(f"\n한계: {summary['limitations']}")
    lines.append("\n참고 문헌:")
    lines.extend(
        f"- [{p['pmid']}] {p['title']} ({p.get('journal') or '?'}, {p.get('year') or '?'}) {p['url']}"
        for p in fetched["papers"][:5]
    )
    lines.append(_next_step_suggestion(target_name, exclude="literature"))
    return {
        "text": "\n".join(lines),
        "available": True,
        "query": query,
        "evidence_summary": summary["evidence_summary"],
        "key_findings": summary["key_findings"][:5],
        "limitations": summary.get("limitations") or "",
        "papers": fetched["papers"][:5],
    }


# ── Clinical Intelligence Agent (Phase 2 of the Agentic Drug Discovery AI
# Platform master plan) ─────────────────────────────────────────────────────
#
# Same real-data-only pattern as the Literature Agent above: real live
# ClinicalTrials.gov v2 search (services/clinical_trials_engine.py) + a
# summary grounded strictly in the real fetched trials
# (services/drug_discovery_clinical_agent.py) — checked in /converse
# alongside the literature check, independent of job state.

_CLINICAL_KEYWORDS = (
    "임상시험", "임상 시험", "임상시험 현황", "임상 현황", "임상 개발", "승인 현황",
    "clinical trial", "clinical trials", "nct", "clinicaltrials",
)


def is_clinical_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _CLINICAL_KEYWORDS)


def answer_clinical_question(message: str, target_name: str | None = None) -> dict:
    """Mirrors answer_literature_question()'s target_name-context, return
    shape, and real-data-only citation discipline, applied to
    ClinicalTrials.gov data."""
    from services.clinical_trials_engine import search_clinical_trials
    from services.drug_discovery_clinical_agent import summarize_clinical_landscape

    query = _clean_external_search_query(message, target_name)
    if not query:
        text = "어떤 병원균, 질병, 유전자, 또는 약물에 대한 임상시험을 찾아드릴까요? 이름을 말씀해 주시면 바로 검색하겠습니다."
        return {"text": text, "available": False, "query": "", "trials": []}
    fetched = search_clinical_trials(query, max_results=5)
    if fetched.get("error"):
        text = f"ClinicalTrials.gov 검색 중 오류가 발생했습니다: {fetched['error']}"
        return {"text": text, "available": False, "query": query, "trials": []}
    if not fetched["trials"]:
        text = f"'{query}'에 대해 ClinicalTrials.gov에서 관련 임상시험을 찾지 못했습니다."
        return {"text": text, "available": False, "query": query, "trials": []}

    summary = summarize_clinical_landscape(query, fetched["trials"])
    lines = [summary["landscape_summary"]]
    if summary["key_trials"]:
        lines.append("\n주요 임상시험:")
        lines.extend(f"- {kt['note']} (NCT ID: {kt['nct_id']})" for kt in summary["key_trials"][:5])
    if summary.get("development_stage_assessment"):
        lines.append(f"\n개발 단계 평가: {summary['development_stage_assessment']}")
    lines.append("\n참고 임상시험:")
    lines.extend(
        f"- [{t['nct_id']}] {t.get('brief_title') or '?'} — {t.get('overall_status') or '?'} "
        f"({', '.join(t.get('phases') or []) or '단계 정보 없음'}) {t['url']}"
        for t in fetched["trials"][:5]
    )
    lines.append(_next_step_suggestion(target_name, exclude="clinical"))
    return {
        "text": "\n".join(lines),
        "available": True,
        "query": query,
        "landscape_summary": summary["landscape_summary"],
        "key_trials": summary["key_trials"][:5],
        "development_stage_assessment": summary.get("development_stage_assessment") or "",
        "trials": fetched["trials"][:5],
    }


# ── Compound Discovery Agent (Phase 2 of the Agentic Drug Discovery AI
# Platform master plan) ─────────────────────────────────────────────────────
#
# No LLM anywhere in this path, unlike the Literature/Clinical agents above
# — this data is inherently tabular (real PubChem similarity hits, real
# ChEMBL measured IC50 values), so a narrative layer would only add
# hallucination risk for no benefit. Two distinct real capabilities,
# distinguished by keyword since they need different inputs: known
# inhibitors needs the resolved uniprot_id (ChEMBL target lookup requires a
# real accession, not just a display name); similar compounds needs a
# reference compound resolvable to a real SMILES (a known drug name from
# drug_discovery_intent.py's curated lookup, or a raw SMILES the user
# pasted directly, validated via RDKit before ever hitting PubChem).

_INHIBITOR_KEYWORDS = ("알려진 억제제", "알려진 저해제", "기존 억제제", "known inhibitor", "chembl", "ic50")
_SIMILAR_COMPOUND_KEYWORDS = ("유사 화합물", "비슷한 화합물", "유사한 화합물", "similar compound", "화합물 검색", "pubchem")


def is_compound_discovery_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _INHIBITOR_KEYWORDS + _SIMILAR_COMPOUND_KEYWORDS)


def _resolve_reference_smiles(message: str) -> tuple[str | None, str | None]:
    """Returns (smiles, label) for a compound named/pasted in the message,
    or (None, None) if nothing resolvable was found. Reuses drug_discovery_
    intent.py's curated name->SMILES table (ground-truth, not LLM-guessed)
    and falls back to treating the raw message as a possible pasted SMILES,
    validated via RDKit before being trusted (same discipline as the intent
    parser's own SMILES validation gate)."""
    from services.drug_discovery_intent import _KNOWN_LIGANDS
    from services.docking_engine import analyze_ligand

    low = message.lower()
    for name, smiles in _KNOWN_LIGANDS.items():
        if name.lower() in low:
            return smiles, name

    # Try the message whole, then each whitespace-separated token, so a
    # pasted SMILES embedded in a Korean sentence (e.g. "이 화합물과 유사한
    # 거 찾아줘 CC(=O)OC1=CC=CC=C1C(=O)O") is still found — each candidate is
    # validated via RDKit before being trusted, same as the intent parser's
    # own SMILES gate, so a token that merely looks SMILES-shaped but isn't
    # a real valid structure is rejected, not guessed at.
    for token in [message.strip(), *message.split()]:
        check = analyze_ligand(token)
        if check.get("valid"):
            return check["canonical_smiles"], token
    return None, None


def answer_compound_discovery_question(message: str, target_name: str | None, uniprot_id: str | None) -> dict:
    """
    Returns {"text": str, "available": bool, "kind": "known_inhibitors" |
    "similar_compounds", "label": str, "items": [...]} — "items" is
    ChEMBL inhibitor records for "known_inhibitors" or PubChem compound
    records for "similar_compounds"; the frontend picks which table
    layout to render based on "kind".
    """
    low = message.lower()
    if any(k in message or k in low for k in _INHIBITOR_KEYWORDS):
        if not uniprot_id:
            text = "먼저 타겟 단백질을 지정해주세요 (예: 'SARS-CoV-2 스파이크 단백질' 또는 UniProt ID)."
            return {"text": text, "available": False, "kind": "known_inhibitors", "items": []}
        from services.compound_discovery_engine import search_known_inhibitors_chembl
        result = search_known_inhibitors_chembl(uniprot_id, max_results=5)
        label = target_name or uniprot_id
        if result["error"]:
            text = f"ChEMBL 검색 중 오류가 발생했습니다: {result['error']}"
            return {"text": text, "available": False, "kind": "known_inhibitors", "label": label, "items": []}
        if not result["inhibitors"]:
            text = f"ChEMBL에서 '{label}'에 대한 실측 IC50 억제제 데이터를 찾지 못했습니다."
            return {"text": text, "available": False, "kind": "known_inhibitors", "label": label, "items": []}
        lines = [f"ChEMBL 실측 데이터 기준, {label}에 대한 IC50 억제제 {len(result['inhibitors'])}건 (역가 순, IC50 낮을수록 강함):"]
        lines.extend(
            f"- {inh['name'] or inh['chembl_id']}: IC50 {inh['ic50_nm']:.1f} nM "
            f"({inh['document_year'] or '연도 정보 없음'}) {inh['url']}"
            for inh in result["inhibitors"]
        )
        top_name = result["inhibitors"][0]["name"] or result["inhibitors"][0]["chembl_id"]
        lines.append(
            f"\n\n다음 단계: '{top_name}을(를) {label}에 도킹해줘'처럼 가장 강한 억제제를 실제로 도킹해보거나, "
            f"'{label}'의 타겟 우선순위도 확인해볼 수 있습니다."
        )
        return {
            "text": "\n".join(lines), "available": True, "kind": "known_inhibitors",
            "label": label, "items": result["inhibitors"],
        }

    ref_smiles, ref_label = _resolve_reference_smiles(message)
    if not ref_smiles:
        text = "유사 화합물을 검색하려면 기준이 될 약물명(예: 아스피린) 또는 SMILES 문자열을 알려주세요."
        return {"text": text, "available": False, "kind": "similar_compounds", "items": []}
    from services.compound_discovery_engine import search_similar_compounds_pubchem
    result = search_similar_compounds_pubchem(ref_smiles, max_results=5)
    if result["error"]:
        text = f"PubChem 검색 중 오류가 발생했습니다: {result['error']}"
        return {"text": text, "available": False, "kind": "similar_compounds", "label": ref_label, "items": []}
    if not result["compounds"]:
        text = f"'{ref_label}'과(와) 구조적으로 유사한 화합물을 PubChem에서 찾지 못했습니다."
        return {"text": text, "available": False, "kind": "similar_compounds", "label": ref_label, "items": []}
    lines = [f"PubChem 실제 데이터 기준, '{ref_label}'과(와) 구조적으로 유사한 화합물 {len(result['compounds'])}건 "
             "(유사도 순 정렬은 아님):"]
    lines.extend(
        f"- {c['iupac_name'] or c['smiles']} (CID {c['cid']}, MW {c['molecular_weight']}) {c['url']}"
        for c in result["compounds"]
    )
    top_name = result["compounds"][0]["iupac_name"] or result["compounds"][0]["smiles"]
    if target_name:
        lines.append(f"\n\n다음 단계: '{top_name}을(를) {target_name}에 도킹해줘'처럼 이 중 하나를 실제로 도킹해볼 수 있습니다.")
    else:
        lines.append(f"\n\n다음 단계: 타겟 단백질을 알려주시면 '{top_name}'을(를) 그 타겟에 실제로 도킹해볼 수 있습니다.")
    return {
        "text": "\n".join(lines), "available": True, "kind": "similar_compounds",
        "label": ref_label, "items": result["compounds"],
    }


# ── Target Intelligence Agent (Phase 3 of the Agentic Drug Discovery AI
# Platform master plan) ─────────────────────────────────────────────────────
#
# No LLM here either — real UniProt DISEASE/FUNCTION comments and real
# Reactome pathway names are already citable factual text straight from
# the source databases (services/target_intelligence_engine.py); a
# narrative layer would only risk paraphrasing away precision.

_TARGET_INTELLIGENCE_KEYWORDS = (
    "질병 연관성", "질환 연관", "타겟 분석", "타겟 인텔리전스", "경로 분석", "신호전달 경로",
    "druggability", "disease association", "pathway", "target intelligence", "reactome",
)


def is_target_intelligence_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _TARGET_INTELLIGENCE_KEYWORDS)


def answer_target_intelligence_question(target_name: str | None, uniprot_id: str | None) -> dict:
    """Returns {"text": str, "available": bool, "target_name": str,
    "uniprot_id": str, "function_summary": str, "diseases": [...],
    "pathways": [...], "opentargets_diseases": [...],
    "opentargets_tractability": [...]}."""
    if not uniprot_id:
        text = "먼저 타겟 단백질을 지정해주세요 (예: 'SARS-CoV-2 스파이크 단백질' 또는 UniProt ID)."
        return {"text": text, "available": False}

    from concurrent.futures import ThreadPoolExecutor
    from services.target_intelligence_engine import (
        get_target_disease_associations, get_target_pathways, calculate_target_priority_score,
    )
    from services.opentargets_engine import get_opentargets_profile
    from services.compound_discovery_engine import search_known_inhibitors_chembl

    label = target_name or uniprot_id
    # Four independent, unrelated real network calls (UniProt, Reactome,
    # OpenTargets, ChEMBL) — run concurrently rather than sequentially, same
    # reasoning as the Decision Agent's literature+clinical fetch.
    with ThreadPoolExecutor(max_workers=4) as executor:
        disease_future = executor.submit(get_target_disease_associations, uniprot_id)
        pathway_future = executor.submit(get_target_pathways, uniprot_id)
        ot_future = executor.submit(get_opentargets_profile, uniprot_id)
        inhibitor_future = executor.submit(search_known_inhibitors_chembl, uniprot_id, 5)
        disease_result = disease_future.result()
        pathway_result = pathway_future.result()
        ot_result = ot_future.result()
        inhibitor_result = inhibitor_future.result()

    known_inhibitor_count = len(inhibitor_result.get("inhibitors") or [])
    priority = calculate_target_priority_score(ot_result, known_inhibitor_count)

    lines = [f"UniProt/Reactome/OpenTargets 실제 데이터 기준, {label} (UniProt {uniprot_id}) 타겟 인텔리전스:"]

    if disease_result.get("error"):
        lines.append(f"\n(UniProt 조회 오류: {disease_result['error']})")
    else:
        if disease_result["function_summary"]:
            lines.append(f"\n기능: {disease_result['function_summary'][:400]}")
        if disease_result["diseases"]:
            lines.append("\n질병 연관성 (UniProt 큐레이션):")
            lines.extend(
                f"- {d['name']}" + (f" (MIM:{d['mim_id']})" if d.get("mim_id") else "")
                for d in disease_result["diseases"][:5]
            )
        else:
            lines.append("\n질병 연관성: UniProt에 등재된 DISEASE 코멘트가 없습니다 (비-인간/바이러스 단백질이거나 멘델리안 질환 연관이 없는 경우 흔함).")

    if pathway_result.get("error"):
        lines.append(f"\n(Reactome 조회 오류: {pathway_result['error']})")
    elif pathway_result["pathways"]:
        lines.append(f"\n관련 경로 (Reactome, {len(pathway_result['pathways'])}건 중 상위 5):")
        lines.extend(
            f"- {p['name']}" + (" [질병 관련]" if p["in_disease"] else "") + f" {p['url']}"
            for p in pathway_result["pathways"][:5]
        )
    else:
        lines.append("\n관련 경로: Reactome에 색인된 경로가 없습니다.")

    if ot_result.get("available"):
        if ot_result["diseases"]:
            lines.append("\n질병 연관성 점수 (OpenTargets, 유전학·문헌·발현 등 종합, 0~1):")
            lines.extend(f"- {d['name']}: {d['score']}" for d in ot_result["diseases"])
        if ot_result["tractability_small_molecule"]:
            lines.append("\n소분자 약물 가능성(Tractability, OpenTargets):")
            lines.append(", ".join(ot_result["tractability_small_molecule"]))
    else:
        lines.append(f"\nOpenTargets: {ot_result.get('error') or '데이터 없음'} (인간 유전자가 아닌 경우 흔함 — 바이러스 단백질 등)")

    lines.append(
        f"\n타겟 우선순위 점수: {priority['priority_score']}/100 "
        f"(질병연관성 {priority['breakdown']['disease_association_component']} + "
        f"약물가능성 {priority['breakdown']['tractability_component']} + "
        f"기존억제제 {priority['breakdown']['known_inhibitor_component']}, ChEMBL 실측 억제제 {known_inhibitor_count}건 반영)"
    )

    # Real, tiered next-step recommendation grounded in the priority_score
    # actually computed above (not generic filler) — same disclosed-formula
    # discipline as decision_agent.py's recommendation.
    score = priority["priority_score"]
    if score >= 70:
        lines.append(f"\n다음 단계: 우선순위 점수가 높습니다 — '{label}을(를) 억제할 수 있는 승인된 약물을 찾아줘'로 바로 스크리닝을 시작하는 것을 권장합니다.")
    elif score >= 40:
        lines.append(f"\n다음 단계: 중간 수준의 우선순위입니다 — '{label} 관련 논문 찾아줘'로 근거를 더 확인하거나, 준비되셨다면 바로 스크리닝을 시작할 수 있습니다.")
    else:
        lines.append(f"\n다음 단계: 우선순위 점수가 낮은 편입니다 (질병 연관성/약물 가능성/기존 억제제 데이터 부족) — 다른 타겟을 고려하거나, '{label} 관련 논문 찾아줘'로 먼저 근거를 확인해보시길 권장합니다.")

    return {
        "text": "\n".join(lines),
        "available": True,
        "target_name": label,
        "uniprot_id": uniprot_id,
        "function_summary": disease_result.get("function_summary") or "",
        "diseases": disease_result.get("diseases") or [],
        "pathways": (pathway_result.get("pathways") or [])[:5],
        "opentargets_diseases": ot_result.get("diseases") or [],
        "opentargets_tractability": ot_result.get("tractability_small_molecule") or [],
        "priority_score": priority["priority_score"],
        "priority_breakdown": priority["breakdown"],
        "known_inhibitor_count": known_inhibitor_count,
    }


# ── SAR Optimization Agent (Phase 9 of the Agentic Drug Discovery AI
# Platform master plan) ─────────────────────────────────────────────────────
#
# Unlike every other agent above, this one requires a COMPLETED job's full
# result (a real candidate to generate analogs from and a real target
# structure to re-dock against) — checked in drug_discovery_router.py's
# COMPLETED-job branch, not the job-independent block those agents share.
# No LLM anywhere in this path either: services/sar_optimization_service.py
# reports real recomputed Vina/ADMET numbers, never a predicted effect.

_SAR_OPTIMIZATION_KEYWORDS = (
    "구조 개선", "구조 최적화", "sar 최적화", "유사체", "치환", "생물학적 등가체", "bioisostere",
    "sar optimization", "구조활성", "structural improvement", "analog",
)


def is_sar_optimization_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _SAR_OPTIMIZATION_KEYWORDS)


def answer_sar_optimization_question(job_result: dict) -> dict:
    """Returns {"text": str, "available": bool, "base_name", "base_smiles",
    "base_affinity_kcal_mol", "analogs" (each annotated with real
    "delta_kcal_mol"/"improved"), "note"} or {"text", "available": False,
    "reason"} when there's nothing to compute."""
    from services.sar_optimization_service import run_sar_optimization

    result = run_sar_optimization(job_result)
    if not result["available"]:
        text = f"SAR 최적화를 수행할 수 없습니다: {result['reason']}"
        return {"text": text, "available": False, "reason": result["reason"]}

    base_affinity = result.get("base_affinity_kcal_mol")
    lines = [
        f"'{result['base_name']}' 기준 실제 재도킹 기반 SAR 최적화 결과 "
        f"(원본 결합친화도: {base_affinity} kcal/mol):",
    ]
    annotated_analogs = []
    for a in result["analogs"]:
        entry = dict(a)
        if a["docked"]:
            delta_str = ""
            if isinstance(base_affinity, (int, float)) and isinstance(a["best_affinity_kcal_mol"], (int, float)):
                diff = a["best_affinity_kcal_mol"] - base_affinity
                entry["delta_kcal_mol"] = round(diff, 2)
                entry["improved"] = diff < 0
                delta_str = f" ({'개선' if diff < 0 else '악화'}, Δ{diff:+.2f})"
            else:
                entry["delta_kcal_mol"] = None
                entry["improved"] = None
            admet_note = ""
            if a["admet"] and a["admet"].get("valid"):
                admet_note = f", PAINS {'있음' if a['admet']['pains']['flagged'] else '없음'}, SA score {a['admet']['synthesis']['score']}"
            lines.append(
                f"\n[{a['transformation']}] {a['smiles']}\n  {a['rationale']}\n"
                f"  실제 재도킹 결합친화도: {a['best_affinity_kcal_mol']} kcal/mol{delta_str} (source: {a['source']}){admet_note}"
            )
        else:
            entry["delta_kcal_mol"] = None
            entry["improved"] = None
            lines.append(f"\n[{a['transformation']}] {a['smiles']} — 재도킹 실패")
        annotated_analogs.append(entry)
    lines.append(f"\n{result['note']}")

    return {
        "text": "\n".join(lines),
        "available": True,
        "base_name": result["base_name"],
        "base_smiles": result["base_smiles"],
        "base_affinity_kcal_mol": base_affinity,
        "analogs": annotated_analogs,
        "note": result["note"],
    }


# ── Decision Agent (Phase 11 of the Agentic Drug Discovery AI Platform
# master plan) ───────────────────────────────────────────────────────────
#
# Like SAR optimization, this needs a COMPLETED job's full result — checked
# in drug_discovery_router.py's COMPLETED-job branch. Unlike SAR, it also
# needs target context (target_name for real literature/clinical evidence
# lookups), so it takes the same (target_name, uniprot_id) pair the
# job-independent agents use.

_DECISION_REPORT_KEYWORDS = (
    "종합 평가", "종합 리포트", "최종 추천", "의사결정", "우선순위 점수", "decision report",
    "overall recommendation", "priority score", "개발 위험도",
)


def is_decision_report_query(message: str) -> bool:
    low = message.lower()
    return any(k in message or k in low for k in _DECISION_REPORT_KEYWORDS)


def answer_decision_report_question(job_result: dict, target_name: str | None) -> dict:
    """Returns {"text": str, "available": bool, "candidate_name",
    "priority_score", "breakdown", "development_risk", "risk_rationale",
    "overall_recommendation", "recommended_next_experiment",
    "target_name", "has_target_context"}."""
    from services.decision_agent import get_top_candidate_scored, calculate_priority_score, generate_decision_report

    candidate = get_top_candidate_scored(job_result)
    if not candidate:
        text = "종합 평가를 수행할 후보가 없습니다 (도킹이 실패했거나 완료된 결과가 없습니다)."
        return {"text": text, "available": False}

    papers, trials = [], []
    if target_name:
        from concurrent.futures import ThreadPoolExecutor
        from services.literature_engine import search_pubmed
        from services.clinical_trials_engine import search_clinical_trials
        # Two independent, unrelated real network calls (PubMed + Clinical
        # Trials.gov) — previously awaited one after another even though
        # neither depends on the other's result, needlessly adding their
        # latencies together. Run concurrently instead (this whole function
        # is already dispatched off the event loop via asyncio.to_thread by
        # the router, so a plain ThreadPoolExecutor here is real OS-level
        # concurrency, not just cooperative scheduling).
        with ThreadPoolExecutor(max_workers=2) as executor:
            papers_future = executor.submit(search_pubmed, target_name, 3)
            trials_future = executor.submit(search_clinical_trials, target_name, 3)
            papers = papers_future.result().get("papers") or []
            trials = trials_future.result().get("trials") or []

    scoring = calculate_priority_score(candidate, has_literature_evidence=bool(papers), has_clinical_evidence=bool(trials))
    report = generate_decision_report(candidate, scoring, papers, trials)

    breakdown_str = ", ".join(f"{k}: {v:+}" if k != "base_ranking_score" else f"{k}: {v}"
                               for k, v in report["breakdown"].items() if k != "final_priority_score")
    lines = [
        f"=== 종합 의사결정 리포트: {candidate['name']} ===",
        f"\n우선순위 점수: {report['priority_score']}/100 (실제 계산 근거: {breakdown_str})",
        f"\n개발 위험도: {report['development_risk']} — {report['risk_rationale']}",
        f"\n종합 추천: {report['overall_recommendation']}",
        f"\n권장 다음 실험: {report['recommended_next_experiment']}",
    ]
    if not target_name:
        lines.append("\n(타겟이 식별되지 않아 문헌/임상 근거는 조회하지 못했습니다 — 후보 자체의 실측 "
                      "도킹/ADMET 데이터만 반영되었습니다.)")
    else:
        lines.append(f"\n※ 위 문헌/임상 근거는 타겟 '{target_name}'에 대한 기존 연구 현황이며, "
                      "이 후보 화합물 자체가 검증되었다는 의미가 아닙니다.")

    return {
        "text": "\n".join(lines),
        "available": True,
        "candidate_name": candidate["name"],
        "priority_score": report["priority_score"],
        "breakdown": report["breakdown"],
        "development_risk": report["development_risk"],
        "risk_rationale": report["risk_rationale"],
        "overall_recommendation": report["overall_recommendation"],
        "recommended_next_experiment": report["recommended_next_experiment"],
        "target_name": target_name,
        "has_target_context": bool(target_name),
    }
