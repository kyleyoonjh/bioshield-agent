"""
Drug Discovery Assistant — natural-language intent parser.

Standalone: does NOT import anything from agent/__init__.py or
api/agent_router.py (off-limits per this feature's isolation requirement).
Reuses the same OPENAI_API_KEY/OPENAI_MODEL env vars via its own minimal
OpenAI client, so it shares configuration but zero code with the existing
primer-design conversational agent.

Extracts a target (UniProt ID or raw sequence) from a free-text message,
plus either:
  - a specific ligand (SMILES or well-known drug name) -> mode="single",
    dock exactly that ligand; or
  - a goal-oriented request with no specific ligand (e.g. "SARS-CoV-2
    주단백질분해효소를 억제할 수 있는 승인된 약물을 찾아줘") -> mode="screen",
    screen the curated drug library instead.
This mirrors MASTER_PLAN.md's Core Philosophy directly: users describe
research goals ("find approved drugs that may inhibit X"), not literal
docking commands.

The LLM may resolve a well-known drug name to its SMILES or a well-known
protein to its UniProt ID (stable public identifiers — the same kind of
lookup IntentAnalyzer already does for organism names), but any returned
SMILES is always re-validated with RDKit before being trusted: if it
doesn't parse, the action is downgraded to a clarification question
instead of feeding a possibly-fabricated structure into the pipeline.

Multi-turn slot accumulation: callers (drug_discovery_router.py) pass in
`known_slots` captured from prior turns in the same chat session, so e.g.
"타겟은 P0DTC1이야" followed by "아스피린" in a later turn correctly combines
into one start_design call instead of losing the first turn's target.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

_UNIPROT_RE = re.compile(
    r"(?<![A-Za-z0-9])([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(?![A-Za-z0-9])"
)  # (?<!...)/(?!...) rather than \b — Python's \b treats Hangul as a "word"
   # char, so \b silently fails to match right before/after a Korean particle
   # with no space (e.g. "P0DTC9를", "P0DTC1이야"), which is the normal way
   # Korean attaches particles to a preceding token.

_SCREEN_KEYWORDS = (
    "찾아줘", "찾아 줘", "후보", "억제할 수 있는", "억제하는", "치료제",
    "find drug", "find approved", "candidate", "screen", "which drugs", "repurpos",
)

# Replies like "몰라"/"아무거나" carry zero extractable entities but do
# signal "just proceed" — treated as an explicit request to fall back to
# session context or a sensible default rather than asking again (see
# _resolve_target_fallback).
_VAGUE_PHRASES = (
    "몰라", "모름", "모르겠어", "모르겠음", "아무거나", "그냥 해줘", "그냥",
    "아무", "알아서 해줘", "you decide", "i don't know", "idk", "whatever", "anything",
)

# Used only to resolve a pending confirmation (see PENDING_CONFIRMATION
# handling below) — never to guess a target/mode, so no risk of the same
# kind of false-positive substring matching that broke the cancer check.
_AFFIRMATIVE_PHRASES = ("네", "응", "그래", "예", "맞아", "진행해", "진행", "좋아", "ok", "okay", "yes", "y", "proceed", "confirm")
_NEGATIVE_PHRASES = ("아니", "아냐", "no", "n", "취소", "그거 아니", "다른", "틀렸")

# target_synonyms.json is SARS-CoV-2-only today. Without this check, a
# message naming a real but unsupported disease (e.g. "대장암") that also
# happens to contain a screening keyword (e.g. "후보") would fall through
# to _default_target() and silently substitute an unrelated SARS-CoV-2
# protein — a real reported bug, not a hypothetical: the fuzzy-match step
# fails (no match in target_synonyms.json), but the vague/screen-signal
# branch doesn't know the difference between "no topic mentioned" and
# "a specific topic was mentioned that we just don't support yet". This
# list lets us tell those two cases apart and refuse to guess in the
# second case instead of defaulting.
_CANCER_SUFFIX_RE = re.compile(r"[가-힣]{2,}암")  # e.g. 대장암, 유방암, 폐암, 위암, 간암
   # No trailing boundary check (unlike _UNIPROT_RE) — Korean particles
   # attach directly to the preceding noun with no space (e.g. "대장암을"),
   # and a false positive here just triggers an honest "not supported yet"
   # clarification instead of a wrong scientific answer, so under- rather
   # than over-constraining this pattern is the safe direction.
_OUT_OF_SCOPE_KEYWORDS = ("cancer", "tumor", "종양", "carcinoma", "sarcoma")


def _is_cancer_topic(text: str) -> bool:
    """Shared by the unsupported-disease branch and the intent-
    clarification menu below (and reused by drug_discovery_router.py's
    choice-2 resolution) — a cancer topic has no single standardized
    target, so it must route toward the real VCF/BAM-driven neoantigen
    (mRNA vaccine candidate) pipeline instead of small-molecule screening
    against a guessed default target."""
    lowered = text.lower()
    return _CANCER_SUFFIX_RE.search(lowered) is not None or any(kw in lowered for kw in _OUT_OF_SCOPE_KEYWORDS)

_TARGET_SYNONYMS_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "target_synonyms.json")
_target_synonyms_cache: dict | None = None


def _load_target_synonyms() -> dict:
    global _target_synonyms_cache
    if _target_synonyms_cache is None:
        with open(_TARGET_SYNONYMS_PATH, encoding="utf-8") as f:
            _target_synonyms_cache = json.load(f)
    return _target_synonyms_cache


def _fuzzy_match_target(lowered_text: str) -> tuple[str, str] | None:
    """Substring match against knowledge/target_synonyms.json's curated,
    UniProt-verified disease/pathogen/protein name list (mirrors _KNOWN_LIGANDS'
    dict-lookup pattern, applied to targets instead of ligands)."""
    for entry in _load_target_synonyms()["targets"]:
        if any(syn.lower() in lowered_text for syn in entry["synonyms"]):
            return entry["uniprot_id"], entry["canonical_name"]
    return None


def _default_target() -> tuple[str, str]:
    synonyms = _load_target_synonyms()
    default_id = synonyms["default_target"]
    entry = next(t for t in synonyms["targets"] if t["uniprot_id"] == default_id)
    return entry["uniprot_id"], entry["canonical_name"]


def _curated_uniprot_ids() -> set[str]:
    """target_synonyms.json's IDs were each already verified live against
    UniProt once during curation — skip the redundant re-verification
    network call for those specifically."""
    return {t["uniprot_id"] for t in _load_target_synonyms()["targets"]}

# Small, well-known drug name -> canonical SMILES lookup for the demo/no-key
# fallback path. Live mode lets the LLM resolve arbitrary names instead.
_KNOWN_LIGANDS = {
    "aspirin":       "CC(=O)OC1=CC=CC=C1C(=O)O",
    "아스피린":        "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine":      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "카페인":          "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen":     "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "이부프로펜":       "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
    "paracetamol":   "CC(=O)NC1=CC=C(C=C1)O",
    "아세트아미노펜":     "CC(=O)NC1=CC=C(C=C1)O",
    "타이레놀":         "CC(=O)NC1=CC=C(C=C1)O",
}

_SYSTEM_PROMPT = """당신은 OpenBio 신약개발(Drug Discovery) 어시스턴트의 의도 분석기입니다.
사용자는 "이 리간드를 이 단백질에 도킹해줘"처럼 구체적 명령을 할 수도 있고, "SARS-CoV-2 주단백질분해효소를
억제할 수 있는 승인된 약물을 찾아줘"처럼 특정 리간드 없이 연구 목표만 말할 수도 있습니다. 두 경우를 구분하세요.

이전 대화에서 이미 확보된 정보(known_slots)가 함께 주어집니다. 새 메시지의 정보와 합쳐 현재까지 확정된
전체 상태를 반환하세요 (이전에 확보된 정보는 새 메시지가 뒤집지 않는 한 유지하세요).

추출 항목:
1. 타겟 단백질: 잘 알려진 UniProt ID가 언급되면 그대로, 매우 유명해서 확신할 수 있는 단백질/병원체 이름(예: "SARS-CoV-2 메인 프로테아제")이면 널리 알려진 UniProt ID를 채워주세요. 확신이 없으면 null로 두세요 (지어내지 마세요 — 별도 시스템이 실시간으로 UniProt에서 검증합니다).
2. mode: 사용자가 구체적 리간드(SMILES 또는 약물명)를 지정했으면 "single", 특정 리간드 없이 "~를 억제할 약물을 찾아줘" 같은 목표만 말했으면 "screen".
3. 리간드(mode="single"일 때만): SMILES 문자열이 직접 주어지면 그대로, 유명 약물명이면 알려진 표준 SMILES로 채워주세요.
4. goal_text: 사용자의 연구 목표를 요약한 짧은 한국어 문장 (스크리닝 전략 선택에 참고용).
5. organism_query: 사용자가 특정 병원체/생물종을 언급했다면, 그 이름을 UniProt 실시간 검색에 쓸 수 있는 영어 학명 또는 일반명으로 번역해 채워주세요 (예: "결핵균" -> "Mycobacterium tuberculosis", "독감 바이러스" -> "influenza A virus"). 이건 검색을 돕는 참고용 번역일 뿐, 최종 타겟 확정은 실시간 검증이 담당하니 완벽하지 않아도 됩니다. 암/종양 관련 언급이면 null로 두세요 (아직 지원하지 않는 별도 영역입니다).
6. mentions_research_topic: 사용자가 실제 질병/병원체/유전자/단백질 이름(예: "결핵", "코로나", "KRAS")을 언급했지만,
   무엇을 원하는지(신약 스크리닝/문헌 검색/타겟 정보 확인 등 구체적 행동, 또는 아래 is_general_question에 해당하는
   일반 설명 요청)는 전혀 밝히지 않은 "주제만 던진" 메시지인 경우에만 true로 표시하세요. is_general_question이
   true이면 이 항목은 반드시 false로 두세요 (둘은 상호 배타적입니다). 인사말이나 리간드/타겟과 무관한 잡담이면
   false입니다.
7. topic_text: mentions_research_topic이 true일 때만, 그 주제를 그대로(또는 자연스럽게 다듬어) 채우세요 (예:
   "결핵", "코로나 바이러스"). 그 외에는 null.
8. is_general_question: 사용자가 특정 질병/병원체/유전자 주제에 대해 "~가 뭐야", "~란 무엇인가요", "~에 대해
   설명해줘", "~가 궁금해"처럼 일반적인 개념 설명을 묻는 질문이면 true. 신약 스크리닝/문헌 검색/임상시험 현황/
   타겟 정보 확인처럼 구체적 행동을 요청하는 것이면 false입니다.
9. question_topic: is_general_question이 true일 때만, 질문의 대상 주제를 채우세요 (예: "코로나가 뭐야?" ->
   "코로나"). 그 외에는 null.

새로운 과학적 수치(결합 친화도, 도킹 점수 등)는 여기서 절대 만들어내지 마세요 — 이 도구는 오직 타겟/리간드/모드 식별만 담당합니다.

반드시 아래 JSON 형식으로만 답하세요:
{
  "action": "start_design" 또는 "chat",
  "reply": "사용자에게 보여줄 한국어 답변 (1-2문장)",
  "mode": "single" 또는 "screen" 또는 null,
  "uniprot_id": "UniProt ID 문자열 또는 null",
  "target_sequence": "아미노산 서열 문자열 또는 null (사용자가 직접 서열을 붙여넣은 경우만)",
  "ligand_smiles": "SMILES 문자열 또는 null",
  "goal_text": "연구 목표 요약 또는 null",
  "organism_query": "UniProt 검색용 영어 병원체/생물종 이름 또는 null",
  "mentions_research_topic": true 또는 false,
  "topic_text": "주제 이름 문자열 또는 null",
  "is_general_question": true 또는 false,
  "question_topic": "질문 대상 주제 문자열 또는 null"
}

action은 (uniprot_id 또는 target_sequence 중 하나)가 채워지고, mode="single"이면 ligand_smiles까지, mode="screen"이면
목표가 충분히 명확할 때 "start_design"으로 하세요. 타겟이 아직 없으면 "chat"으로 하고 reply에서 부족한 정보를 물어보세요.

참고로 uniprot_id를 채우지 못한 경우, 별도의 결정론적 로직이 known_slots 상속 -> 알려진 동의어 매칭 -> organism_query
기반 실시간 UniProt 검색 -> 기본 타겟 제안 순으로 자동 보완하니, 당신은 억지로 지어내지 말고 정말 모르면 null로 두세요.
uniprot_id를 채운 경우에도 별도 시스템이 실제로 존재하는 ID인지 실시간으로 재검증하니, 확신이 없으면 차라리 null로
남기고 organism_query만 채우는 편이 낫습니다.

중요 — reply 작성 시 단백질 정체를 절대 추측하지 마세요: known_slots에 "target_name"이 이미 채워져 있다면 그 단백질을
가리킬 때 반드시 그 이름을 그대로 사용하세요. target_name이 null/없는데 uniprot_id만 있는 경우, 그 ID가 어떤 단백질인지
당신의 배경지식으로 이름 붙이거나 설명하지 마세요 (예: "P0DTC2는 메인 프로테아제입니다" 같은 문장 절대 금지 — 실제로는
틀린 답일 수 있고, 검증은 별도 결정론적 시스템만 담당합니다). 이 경우 그냥 "UniProt {id}로 설정된 타겟"처럼 ID로만
지칭하세요."""


def _get_client():
    from openai import OpenAI
    return OpenAI(api_key=_OPENAI_API_KEY)


def _use_demo() -> bool:
    return not bool(_OPENAI_API_KEY)


def _merge_slot(new_value, known_slots: dict, key: str):
    return new_value if new_value else known_slots.get(key)


def _extract_demo(message: str, known_slots: dict) -> dict:
    """Rule-based fallback: regex for UniProt IDs, dict lookup for common drug names,
    keyword detection for goal-oriented screening requests."""
    uniprot_match = _UNIPROT_RE.search(message)
    uniprot_id = _merge_slot(uniprot_match.group(0) if uniprot_match else None, known_slots, "uniprot_id")

    ligand_smiles = None
    lowered = message.lower()
    for name, smiles in _KNOWN_LIGANDS.items():
        if name.lower() in lowered:
            ligand_smiles = smiles
            break
    ligand_smiles = _merge_slot(ligand_smiles, known_slots, "ligand_smiles")

    is_screen_request = any(kw.lower() in lowered for kw in _SCREEN_KEYWORDS)
    if ligand_smiles:
        mode = "single"
    elif is_screen_request or known_slots.get("mode") == "screen":
        mode = "screen"
    else:
        mode = known_slots.get("mode")

    goal_text = _merge_slot(message if is_screen_request else None, known_slots, "goal_text")
    # No LLM available in demo mode to translate a pathogen name, so this is
    # a best-effort pass-through: UniProt's search index is English/Latin,
    # so this only helps when the user already typed an English/Latin
    # organism name — a pure-Korean pathogen name here will just yield an
    # empty live-search result and safely fall through to the default
    # target, not a wrong answer.
    organism_query = _merge_slot(message if is_screen_request else None, known_slots, "organism_query")

    if uniprot_id and mode == "single" and ligand_smiles:
        return {
            "action": "start_design", "mode": mode,
            "reply": f"UniProt {uniprot_id}와 해당 리간드로 구조 조회 및 도킹을 시작합니다.",
            "uniprot_id": uniprot_id, "target_sequence": None,
            "ligand_smiles": ligand_smiles, "goal_text": goal_text, "organism_query": organism_query,
        }
    if uniprot_id and mode == "screen":
        return {
            "action": "start_design", "mode": mode,
            "reply": f"UniProt {uniprot_id}에 대해 승인 약물 라이브러리 스크리닝을 시작합니다.",
            "uniprot_id": uniprot_id, "target_sequence": None,
            "ligand_smiles": None, "goal_text": goal_text, "organism_query": organism_query,
        }
    missing = []
    if not uniprot_id:
        missing.append("타겟 단백질의 UniProt ID (또는 서열)")
    if not mode:
        missing.append("구체적인 약물명/SMILES, 또는 '~를 억제할 약물을 찾아줘' 같은 목표")
    return {
        "action": "chat", "mode": mode,
        "reply": f"다음 정보가 더 필요합니다: {', '.join(missing)}.",
        "uniprot_id": uniprot_id, "target_sequence": None,
        "ligand_smiles": ligand_smiles, "goal_text": goal_text, "organism_query": organism_query,
    }


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


def parse_drug_discovery_intent(message: str, known_slots: dict | None = None) -> dict:
    """
    Returns {"action": "start_design"|"chat", "reply": str, "mode": "single"|"screen"|None,
             "uniprot_id": str|None, "target_sequence": str|None,
             "ligand_smiles": str|None, "goal_text": str|None, "organism_query": str|None}.
    """
    known_slots = known_slots or {}
    logger.info("[drug_discovery_intent] parse start | demo=%s known_slots=%s message=%r",
                _use_demo(), known_slots, message[:200])

    # A live-search-resolved target (see the is_screen_signal branch below)
    # is a real but UNVERIFIED guess — unlike a curated fuzzy match, it can
    # be wrong, and previously the pipeline started immediately anyway
    # ("[fast] Preparing receptor..." right after a guess), giving the user
    # no chance to correct it before real (sometimes slow, uncached)
    # computation began. When a prior turn set this, this turn's message
    # must explicitly confirm or reject it before anything runs.
    if known_slots.get("pending_confirmation") and known_slots.get("uniprot_id"):
        lowered_msg = message.strip().lower()
        if any(p in lowered_msg for p in _AFFIRMATIVE_PHRASES):
            logger.info("[drug_discovery_intent] pending confirmation ACCEPTED | uniprot_id=%s", known_slots["uniprot_id"])
            return {
                "action": "start_design",
                "reply": "확인 감사합니다. 설계를 시작합니다.",
                "mode": known_slots.get("mode") or "screen",
                "uniprot_id": known_slots.get("uniprot_id"),
                "target_sequence": known_slots.get("target_sequence"),
                "ligand_smiles": known_slots.get("ligand_smiles"),
                "goal_text": known_slots.get("goal_text"),
                "organism_query": known_slots.get("organism_query"),
                "target_name": known_slots.get("target_name"),
                "pending_confirmation": False,
            }
        if any(p in lowered_msg for p in _NEGATIVE_PHRASES):
            logger.info("[drug_discovery_intent] pending confirmation REJECTED | uniprot_id=%s", known_slots["uniprot_id"])
            return {
                "action": "chat",
                "reply": "알겠습니다. 정확한 UniProt ID나 다른 타겟/병원체 이름을 알려주시겠어요?",
                "mode": None, "uniprot_id": None, "target_sequence": None,
                "ligand_smiles": None, "goal_text": None, "organism_query": None,
                "target_name": None,
                "pending_confirmation": False,
            }
        # Message doesn't clearly confirm or reject — treat it as a brand
        # new request instead of getting stuck (e.g. the user may have just
        # typed a different, more specific UniProt ID directly).

    if _use_demo():
        result = _extract_demo(message, known_slots)
    else:
        try:
            client = _get_client()
            user_content = json.dumps({"known_slots": known_slots, "message": message}, ensure_ascii=False)
            resp = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            parsed = _parse_json(resp.choices[0].message.content or "")
            result = {
                "action":          parsed.get("action") if parsed.get("action") in ("start_design", "chat") else "chat",
                "reply":           parsed.get("reply") or "요청을 이해하지 못했습니다. 다시 말씀해 주세요.",
                "mode":            _merge_slot(parsed.get("mode"), known_slots, "mode"),
                "uniprot_id":      _merge_slot(parsed.get("uniprot_id"), known_slots, "uniprot_id"),
                "target_sequence": _merge_slot(parsed.get("target_sequence"), known_slots, "target_sequence"),
                "ligand_smiles":   _merge_slot(parsed.get("ligand_smiles"), known_slots, "ligand_smiles"),
                "goal_text":       _merge_slot(parsed.get("goal_text"), known_slots, "goal_text"),
                "organism_query":  _merge_slot(parsed.get("organism_query"), known_slots, "organism_query"),
                "mentions_research_topic": bool(parsed.get("mentions_research_topic")),
                "topic_text":      parsed.get("topic_text") if parsed.get("mentions_research_topic") else None,
                "is_general_question": bool(parsed.get("is_general_question")),
                "question_topic":  parsed.get("question_topic") if parsed.get("is_general_question") else None,
            }
        except Exception as exc:
            logger.warning("[drug_discovery_intent] LLM call failed, falling back to demo rules | error=%s", exc)
            result = _extract_demo(message, known_slots)

    # Default: no confirmation pending unless the live-search branch below
    # sets one fresh this turn. A curated fuzzy match is definitive and
    # never needs confirmation, so this stays False in that case even if a
    # stale pending_confirmation was carried in from known_slots above.
    result["pending_confirmation"] = False
    # demo mode (_extract_demo) has no LLM to judge "does this message name
    # a real topic without a clear action" — never guessed heuristically,
    # just left off (existing generic "more info needed" reply still
    # applies in demo mode, no regression).
    result.setdefault("mentions_research_topic", False)
    result.setdefault("topic_text", None)
    result.setdefault("is_general_question", False)
    result.setdefault("question_topic", None)

    # Curated fuzzy-match ALWAYS takes priority over whatever uniprot_id the
    # LLM/regex path already filled in — never just a fallback for when
    # nothing was found. Confirmed as a real, serious gap (not hypothetical):
    # asked for "인플루엔자 M2 단백질" with M2 already curated as P06821, the
    # LLM confidently returned uniprot_id=P05815 instead — a real, valid
    # UniProt entry, so the "does this ID exist" check below would have
    # happily approved it, except P05815 is Zein-alpha B49, a corn (Zea
    # mays) seed storage protein with zero relation to influenza. Existence
    # is not the same as relevance, and only a curated, hand-verified
    # synonym match can catch that class of confident-but-wrong LLM answer.
    # Ground-truth identity label, carried forward via known_slots but NEVER
    # authored by the LLM (only the deterministic branches below — fuzzy
    # match, default target, organism search — ever set it). Confirmed real
    # bug: without this, once a turn didn't re-trigger one of those
    # deterministic branches, the LLM had to compose its reply mentioning a
    # bare UniProt ID from its own parametric memory alone and confidently
    # hallucinated the wrong protein identity (claimed P0DTC2 — Spike — was
    # "SARS-CoV-2 메인 프로테아제"/Mpro, a completely different protein with
    # no curated entry, presumably because Mpro is the far more famous COVID
    # drug target in its training data).
    result["target_name"] = known_slots.get("target_name")
    _prior_uniprot_id = known_slots.get("uniprot_id")

    lowered = message.strip().lower()
    combined_text = f"{lowered} {(known_slots.get('goal_text') or '').lower()}"
    fuzzy = _fuzzy_match_target(combined_text)
    if fuzzy:
        fuzzy_id, target_name = fuzzy
        if result["uniprot_id"] and result["uniprot_id"] != fuzzy_id:
            logger.warning("[drug_discovery_intent] overriding non-curated uniprot_id=%s with curated "
                            "fuzzy match=%s (%s) for message=%r",
                            result["uniprot_id"], fuzzy_id, target_name, message[:200])
        result["uniprot_id"] = fuzzy_id
        result["target_sequence"] = None
        result["target_name"] = target_name
        result["reply"] = f"'{target_name}'(UniProt {fuzzy_id})로 인식하여 진행합니다."

    # Never trust an unverified UniProt ID — whether it came from the regex
    # match (demo mode: any string shaped like an ID is accepted, real or
    # not) or an LLM guess. Confirmed once via a real UniProt lookup before
    # the pipeline ever tries to fetch a structure for it — the same
    # "validate before trusting" discipline already applied to LLM-supplied
    # SMILES via RDKit, below. Curated target_synonyms.json IDs (including
    # one just assigned by the fuzzy match above) are already pre-verified,
    # so skip those. This still can't catch a wrong-but-real ID for a
    # target with NO curated entry — that residual risk is why the fuzzy
    # match above must run first and win whenever it has an opinion.
    if result["uniprot_id"] and result["uniprot_id"] not in _curated_uniprot_ids():
        from services.uniprot_search_engine import verify_uniprot_id_exists
        verified = verify_uniprot_id_exists(result["uniprot_id"])
        if not verified:
            logger.warning("[drug_discovery_intent] discarding unverifiable uniprot_id=%s", result["uniprot_id"])
            result["uniprot_id"] = None

    # Target-resolution fallback (applies identically to demo and LLM paths,
    # so behavior never depends on which one ran): if the target is still
    # unresolved, try in order (1) refuse rather than guess for a
    # named-but-unsupported disease (e.g. cancer — needs a different,
    # mutation-aware pipeline, not built yet), (2) a live UniProt search for
    # any other named pathogen/organism (real, verifiable — not limited to
    # the small curated list), and only then (3) if the request is truly
    # vague or the search above found nothing, proactively assume the
    # most-studied default target instead of blocking — matching
    # MASTER_PLAN's "선제적 제안 체계": never dead-end the conversation on a
    # missing ID the system can reasonably infer, search for, or default.
    if not result["uniprot_id"] and not result["target_sequence"]:
        is_vague = any(vp in lowered for vp in _VAGUE_PHRASES)
        is_screen_signal = any(kw.lower() in combined_text for kw in _SCREEN_KEYWORDS) or result.get("mode") == "screen"

        # Checked against the RAW CURRENT MESSAGE ONLY, not combined_text —
        # confirmed real bug: combined_text includes goal_text persisted
        # from a PRIOR turn's unsupported-disease mention (e.g. "대장암"),
        # and since that persistence is never cleared except by a
        # successful start_design, a totally unrelated later message in the
        # same session (e.g. "결핵", a real resolvable pathogen) still had
        # "대장암" mixed into the text checked here and got wrongly refused
        # as if it were the same unsupported cancer request. A stale
        # mention should never be able to override what the CURRENT message
        # actually says on its own.
        names_unsupported_disease = _is_cancer_topic(lowered)
        if names_unsupported_disease:
            # A specific (but unsupported) disease was named — never silently
            # substitute the unrelated default target here. Falls through to
            # action="chat" below since uniprot_id stays unresolved. Persist
            # the mention into goal_text so the next turn (e.g. a bare
            # follow-up like "후보 타겟 알려줘") still remembers it instead of
            # looking like an unrelated, unscoped screening request.
            result["goal_text"] = result.get("goal_text") or message
            # Explicitly invites the real next step (VCF upload) instead of
            # just stating what's unsupported —암은 코로나처럼 단일 표준
            # 타겟이 없어 이름만으로는 절대 추정하지 않지만, 실제 VCF가 있으면
            # (services/vcf_annotation_engine.py의 실시간 VEP 검증 경유) 어떤
            # 암/변이든 이미 정상적으로 처리 가능 — 그 경로로 안내.
            result["reply"] = (
                "말씀하신 질환은 코로나와 달리 표준화된 단일 타겟이 없어 (암은 환자/변이마다 원인 "
                "단백질이 다릅니다), 이름만으로 임의의 타겟을 추정해 진행하지 않습니다. "
                "실제 VCF(유전체 변이) 파일을 업로드해주시겠어요? 📎 버튼으로 직접 업로드하시거나, "
                "아래 '🧬 암 mRNA 자가 백신 데모' 버튼으로 실제 NSCLC(폐암) 샘플(KRAS G12D 변이 포함, "
                "Ensembl VEP로 실시간 검증됨)을 먼저 체험해 보실 수 있습니다."
            )
            # Lets the frontend show ONLY the VCF-upload quick-action here
            # instead of the (irrelevant in this context) SARS-CoV-2/
            # Influenza pathogen target buttons — those buttons were
            # showing up after every "chat" response regardless of context,
            # including right after telling the user "upload a VCF."
            result["needs_vcf"] = True
            # Real reported bug: a cancer topic's VCF upload was routing
            # into the small-molecule docking/screening pipeline (produced
            # an irrelevant "top candidate: Warfarin" result against some
            # guessed default target) instead of the real neoantigen/mRNA
            # vaccine pipeline this project actually built for cancer's
            # per-patient variant data. This flag tells the frontend (see
            # DrugDiscoveryChatPanel.tsx's neoantigenMode) to route the
            # VCF/BAM upload to /design-from-bam (run_neoantigen_pipeline)
            # instead of /design-from-vcf (run_drug_discovery_from_vcf).
            result["neoantigen_mode"] = True
        elif is_vague:
            # No specific topic named at all — nothing to search for, go
            # straight to the safe default rather than wasting a network
            # call on empty/filler text.
            default_id, target_name = _default_target()
            result["uniprot_id"] = default_id
            result["target_name"] = target_name
            result["reply"] = (
                f"타겟이 명확하지 않아, 가장 널리 연구된 {target_name}(UniProt {default_id})을 "
                f"기준으로 가정하고 진행합니다. 다른 타겟을 원하시면 말씀해 주세요."
            )
        elif is_screen_signal:
            # A screening goal was stated but names a pathogen/organism
            # outside the curated 6 — try a real, live UniProt search
            # before falling back to the default (prefers the LLM's
            # English/Latin organism_query translation when available,
            # otherwise searches the raw text directly, which still works
            # for organism names already typed in English/Latin).
            from services.uniprot_search_engine import search_reviewed_proteins
            search_query = result.get("organism_query") or combined_text
            candidates = search_reviewed_proteins(search_query, limit=3)
            if candidates:
                top = candidates[0]
                result["uniprot_id"] = top["uniprot_id"]
                protein_label = top.get("protein_name") or "단백질"
                organism_label = top.get("organism") or ""
                result["target_name"] = f"{protein_label} ({organism_label})" if organism_label else protein_label
                # This is a real but UNVERIFIED guess (unlike a curated
                # fuzzy match) — must not auto-start the (sometimes slow,
                # uncached) real pipeline before the user gets a chance to
                # correct it. Confirmed real complaint: the pipeline began
                # ("[fast] Preparing receptor...") immediately after a
                # guess like this, with no pause.
                result["pending_confirmation"] = True
                result["reply"] = (
                    f"UniProt 실시간 검색으로 '{protein_label}' ({organism_label}, UniProt {top['uniprot_id']})를 "
                    f"찾았습니다. 이 타겟으로 진행할까요? ('네'/'아니오'로 답해주세요)"
                )
            else:
                default_id, target_name = _default_target()
                result["uniprot_id"] = default_id
                result["target_name"] = target_name
                result["reply"] = (
                    f"말씀하신 타겟을 UniProt에서 실시간으로 찾지 못해, 가장 널리 연구된 "
                    f"{target_name}(UniProt {default_id})을 기준으로 가정하고 진행합니다. "
                    f"정확한 UniProt ID를 알고 계시면 알려주세요."
                )
        elif result.get("mentions_research_topic") and result.get("topic_text"):
            # Real gap: a bare topic mention with no stated goal ("결핵")
            # previously fell through to a generic LLM "more info needed"
            # reply instead of a clear menu of what this assistant can
            # actually do for that topic. Never silently assumes screening
            # here (mirrors "the Intent Agent should never guess") — always
            # asks explicitly. The raw topic is persisted into goal_text so
            # a follow-up choice (typed or via a frontend quick-pick button)
            # doesn't need to repeat it.
            topic = result["topic_text"]
            result["goal_text"] = result.get("goal_text") or topic
            result["needs_intent_clarification"] = True
            result["intent_topic"] = topic
            # Real reported bug: this menu offered "신약 스크리닝" (small-
            # molecule screening against a guessed default target) even for
            # cancer topics, which have no single standardized target (see
            # _is_cancer_topic/the needs_vcf branch above) — picking it
            # silently ran a docking screen against an unrelated default
            # target and returned irrelevant results (e.g. "1위: Warfarin"
            # for "대장암"). Cancer topics get option 2 replaced with the
            # real neoantigen/mRNA vaccine pipeline instead, which needs a
            # VCF and is why this also sets needs_vcf/neoantigen_mode so
            # the frontend can route straight to it once picked (see
            # drug_discovery_router.py's _resolve_intent_clarification_choice).
            is_cancer = _is_cancer_topic(topic)
            option_2 = (
                "2) mRNA 자가 백신 후보 찾기 — 신항원(neoantigen) 식별 (VCF 필요)\n"
                if is_cancer else
                "2) 신약 스크리닝 — 억제할 수 있는 승인 약물 후보 찾기\n"
            )
            result["neoantigen_mode"] = is_cancer
            result["reply"] = (
                f"'{topic}'에 대해 어떤 걸 도와드릴까요?\n"
                # 문헌 검색을 1순위로: 아직 아무 정보도 없는 주제라면 비용이 드는
                # 스크리닝(실제 도킹 계산)보다 먼저 관련 연구를 파악하는 게
                # 자연스러운 연구 순서라는 실제 사용자 피드백 반영.
                "1) 관련 논문 검색\n"
                + option_2 +
                "3) 타겟 정보 확인 — 질병 연관성/관련 경로\n\n"
                "원하시는 번호나 방식을 말씀해 주세요."
            )

    # If the resolved uniprot_id ended up different from the one already
    # known (e.g. the LLM directly returned a new, already-curated/verified
    # ID without going through the fuzzy-match/organism-search branches that
    # freshly set target_name above) but target_name is still the OLD value
    # carried over from known_slots, it would now mislabel the NEW target —
    # the exact class of stale-mismatch that caused the P0DTC2/메인 프로테아제
    # hallucination. Safer to drop it than let a stale name survive a target
    # change; the LLM instructions above already handle target_name=None by
    # referring to the ID only, never inventing an identity.
    if result["uniprot_id"] != _prior_uniprot_id and result["target_name"] == known_slots.get("target_name"):
        result["target_name"] = None

    # A target is resolved (by ANY path — curated fuzzy match, live organism
    # search, or vague-default) but no ligand/goal was ever stated — default
    # to "screen" (find approved drugs for this target) rather than dead-
    # ending the conversation. Previously this only ran inside the "target
    # was still unresolved before this function" branch above, so a curated
    # fuzzy match (which resolves uniprot_id earlier, unconditionally) never
    # got a mode default and could never reach start_design no matter how
    # many times the user said "진행해줘" — a real, confirmed bug: naming a
    # known target with no explicit ligand is exactly the screening case,
    # same as the other two resolution paths already handle correctly.
    if (result["uniprot_id"] or result["target_sequence"]) and not result.get("mode"):
        result["mode"] = "screen"

    # Validation gate: never trust an LLM-produced SMILES without RDKit parsing it first.
    if result["mode"] == "single" and result["ligand_smiles"]:
        from services.docking_engine import analyze_ligand
        check = analyze_ligand(result["ligand_smiles"])
        if not check["valid"]:
            logger.warning("[drug_discovery_intent] rejected invalid SMILES from intent parse: %s", result["ligand_smiles"])
            result["action"] = "chat"
            result["reply"] = "리간드 구조를 정확히 인식하지 못했습니다. SMILES 문자열을 직접 붙여넣어 주시겠어요?"
            result["ligand_smiles"] = None

    # Recomputed fully (not just downgraded) so the target-resolution fallback
    # above can upgrade an initial "chat" into "start_design" once a target
    # was inferred, in addition to the existing downgrade cases (invalid
    # SMILES, missing mode).
    target_ok = bool(result["uniprot_id"] or result["target_sequence"])
    mode_ok = (result["mode"] == "single" and result["ligand_smiles"]) or (result["mode"] == "screen")
    result["action"] = "start_design" if (target_ok and mode_ok and not result["pending_confirmation"]) else "chat"

    logger.info("[drug_discovery_intent] parse result | action=%s mode=%s uniprot_id=%s pending_confirmation=%s",
                result["action"], result["mode"], result["uniprot_id"], result["pending_confirmation"])
    return result
