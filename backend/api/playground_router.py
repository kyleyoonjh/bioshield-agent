"""
Local MCP playground — a KakaoTalk-style chat UI to test the MCP server the
same way Kakao PlayMCP does, on localhost.

GET  /playground        -> the chat page (self-contained HTML)
POST /playground/chat   -> {message, session_id, last_job_id}

When OPENAI_API_KEY is set the endpoint runs a real LLM assistant ('신약개발')
that reasons over the natural-language message and calls the MCP tools via
function-calling (mcp.call_tool — the same in-process path Kakao's gateway
drives). Without a key it falls back to a keyword router so the demo prompts
still work offline.

Path deliberately NOT under /mcp so the OAuth guard (main.py) never applies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mcp_server import mcp

logger = logging.getLogger("airemedy.playground")

router = APIRouter()

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
_JOB_TOOLS = {"predict_drug_binding", "predict_neoantigen_candidates", "run_sar_optimization"}
_NAMED_SMILES = {
    "아스피린": "CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "이부프로펜": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "카페인": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}


class ChatReq(BaseModel):
    message: str
    session_id: str = "default"
    last_job_id: str = ""
    is_poll: bool = False


def _has(text: str, *keys: str) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keys)


def _extract_text(res) -> str:
    content = res[0] if isinstance(res, tuple) else res
    parts = []
    for block in content or []:
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts) if parts else str(content)


# ── LLM assistant ('신약개발') ────────────────────────────────────────────────

_SYSTEM_PROMPT = """당신은 의학·신약개발 전문 AI 어시스턴트 '신약개발'입니다. AiRemedy MCP 도구들이 연결되어 있습니다: 약물 도킹 가상 스크리닝(predict_drug_binding), mRNA 암백신 신항원 설계(predict_neoantigen_candidates), 작업 상태 조회(get_drug_discovery_job_status), 타겟 질환연관성/경로/OpenTargets 분석, 논문·임상시험 검색, 유사화합물·저해제 검색, ADMET 분석 등.

일반 사용자는 전문 용어(UniProt ID, SMILES, VCF 등)를 모릅니다. 사용자가 일상 언어로 질문하면 당신이 알아서 적절한 도구를 골라 조합하고, 부족한 정보는 친절히 유도하세요.

[작동 지침]
1. 사용자가 암·질병·유전자 이름을 말하면 관련 타겟 단백질 UniProt ID로 매핑해 도구를 호출하세요. 매핑 예: KRAS/췌장암=P01116, TP53/p53/대장암=P04637, EGFR/폐암=P00533, SARS-CoV-2 스파이크/코로나=P0DTC2, HER2/ERBB2/유방암=P04626, BRAF=P15056, BRCA1=P38398, VEGFR2=P35968.
2. predict_drug_binding처럼 오래 걸리는 작업은 한 번만 시작하고, 사용자에게 "가상 스크리닝을 시작했습니다. Job ID: XXX. 잠시 후 상태를 확인해 드릴게요"라고 먼저 안내하세요. 같은 답변 안에서 상태를 반복 폴링하지 마세요 — 앱이 자동으로 get_drug_discovery_job_status를 호출합니다. 단, 사용자가 '상태 확인'을 요청하면 반드시 get_drug_discovery_job_status를 job_id로 호출하고, 완료면 결과를 쉬운 말로 정리하세요.
3. 백신 설계 요청: **절대 VCF 파일을 요구하거나 되묻지 마세요.** 사용자가 문장 안에 변이를 직접 적어준 경우(예: 'chr12 25398283 C A')에만 그것을 최소 VCF 형식(첫 줄 '##fileformat=VCFv4.2', 그다음 '#CHROM\\tPOS\\tID\\tREF\\tALT\\tQUAL\\tFILTER\\tINFO\\tFORMAT\\tSAMPLE' 헤더, 그다음 데이터 줄)으로 만들어 vcf_content에 넣으세요. 그 외의 모든 경우(환자 정보를 안 줬거나 '환자의 암 정보/저장된 환자 정보'를 언급한 경우)에는 **즉시 인자 없이 predict_neoantigen_candidates를 호출**해 저장된 환자 WES 유전체 데이터로 바로 진행하세요.
4. 여러 관점이 필요한 '종합 분석' 요청이면 get_target_disease_associations, get_target_pathways, get_opentargets_profile를 함께 호출해 결과를 합쳐 설명하세요.
5. [치료 모달리티를 절대 바꾸지 마세요 — 가장 중요] mRNA 암 백신(신항원 기반 면역치료)과 저분자 신약(단백질 억제)은 **완전히 다른 치료 전략**입니다. 사용자가 백신을 설계했으면 이후 질문은 **그 백신에 대한 것**입니다. 사용자가 명시적으로 요청하지 않는 한 **predict_drug_binding(저분자 도킹)으로 넘어가지 마세요.** 저분자 탐색이 도움이 될 것 같으면 실행하지 말고 **"추가로 이 유전자를 표적하는 저분자 신약 후보도 탐색해 드릴까요?"라고 먼저 물어보고**, 사용자가 동의할 때만 실행하세요.
6. [백신 개발 가능성 평가] 백신 설계 후 "신약으로 개발 가능한지", "개발 가능성" 등을 물으면 — **저분자 도킹을 돌리지 말고** — 이미 확보한 신항원 결과 자체를 근거로 평가하세요: 결합 친화도(nM)·제시 percentile·foreignness(비자기성)·자기유사성 여부·AI Neo-Score·HLA 커버리지. 여기에 **search_literature / search_clinical_trials**로 해당 변이 표적 백신(예: KRAS G12D)의 **실제 임상·문헌 근거**를 찾아 붙이세요. 그리고 한계(인구집단 표준 HLA 기준, 환자별 HLA 타이핑 필요)를 담백하게 밝히세요.
7. [리포트는 실제 수행한 연구에 맞춰서] "연구 리포트"/"종합 리포트" 요청 시 **직전에 무엇을 연구했는지**로 도구를 고르세요. mRNA 백신 job이면 **generate_vaccine_report(job_id)**, 저분자 도킹 job이면 **generate_decision_report(job_id)**. 백신 연구를 저분자 리포트로 만들면 사용자가 하지도 않은 연구를 보고하는 셈이니 절대 섞지 마세요. 리포트를 전할 때는 도구가 준 **최종_결론(선별 결과·근거·판단·한계·필요한 후속 검증)을 반드시 그대로 마지막에 실어** 주세요 — 결론 없는 연구 리포트는 미완성입니다.
8. [백신 표현의 정확성] 종양 변이는 환자 실제 데이터지만 **HLA는 인구집단 표준 6종**을 씁니다. 따라서 **"환자 개인 맞춤형 백신"이라고 말하지 마세요.** "환자 **종양 변이 기반 예비(preliminary) mRNA 암 백신 후보**"라고 표현하고, 진짜 개인 맞춤 설계에는 **환자 고유 HLA 타이핑이 추가로 필요**하다는 점을 밝히세요.

[중요]
- **[근거 없는 타겟 가정 금지]** 이 대화에서 실제로 수행한 연구(백신/도킹 job)도 없고 사용자가 타겟·질환을 말한 적도 없다면, 특정 유전자·변이(예: KRAS G12D)를 **임의로 가정해서 분석하지 마세요.** "신약으로 개발 가능한지 분석해줘", "종합 리포트 만들어줘" 처럼 대상이 빠진 요청이 오면 **무엇을 연구할지 먼저 되물으세요.** 없는 연구를 있는 것처럼 보고하는 것이 이 시스템에서 가장 큰 실패입니다.
- 도구가 반환한 실제 데이터만 사용하고 수치·결론을 절대 지어내지 마세요. 도구를 부르지 않고 임의로 답하지 마세요.
- **[기억으로 답하지 말 것]** 화합물의 약물성·흡수·독성·안전성을 물으면 반드시 **predict_admet_profile**을 호출해 실제 RDKit 계산으로 답하세요. 타겟의 질환 연관성·경로·유망도를 물으면 반드시 해당 도구를 호출하세요. 당신의 사전 지식으로 의학 정보를 서술하는 것은 **금지**입니다(검증 불가능한 답변이 됩니다). 단, 사용자가 화합물 이름만 말하면 그 SMILES는 당신이 알아내어 도구에 넘기세요(예: 아스피린 = CC(=O)OC1=CC=CC=C1C(=O)O).
- 이 시스템은 연구 도구입니다. **환자 복용 지침 같은 임상적 의학 조언은 하지 마세요.**
- 결과를 설명할 때는 **실제 후보의 구체적 수치를 반드시 포함**하세요. mRNA 백신이면 후보 펩타이드 서열·결합력(nM)·결합하는 HLA형·Neo-Score를, 도킹이면 1위 화합물명·결합에너지(kcal/mol)·약물성 점수를 빠뜨리지 마세요.
- 분석의 기준·한계는 담백하게 사실만 밝히세요(예: "인구집단 표준 HLA 6종 기준으로 분석", 휴리스틱 도킹 여부, 데모용 화합물 라이브러리). "환자의 실제 유전형이 아니다" 같은 경고조 표현은 쓰지 말고, 분석 기준을 설명하듯 자연스럽게 쓰세요.
- 답변은 전문 용어를 풀어 일반인이 이해할 쉬운 한국어로, 핵심을 정리하고 항상 '다음 단계'를 한 줄 제시하세요."""

_openai_tools: list | None = None
_SESSIONS: dict[str, list] = {}
_MAX_HISTORY = 12  # keep system + recent turns (every kept message is re-sent each request)

# Real reported bug (still reproduced after the _trim() fix above, on what
# looked like a short/fresh conversation — meaning _trim wasn't even the
# only cause): the frontend's pollLoop() fires ANOTHER POST /playground/chat
# for the same session_id every 5s while a job is running, with no
# coordination against the user's own in-flight message. Since _SESSIONS
# [session_id] is one shared mutable list, two concurrent requests for the
# same session can interleave their history.append() calls — e.g. request A
# appends its user message and awaits the OpenAI call, request B (the poll)
# appends ITS user message and completes a full assistant(tool_calls)+
# tool(...)+assistant round-trip before A resumes, leaving A's
# subsequently-appended messages out of order relative to what A's own
# OpenAI call actually saw. A per-session lock serializes all processing
# for a given session_id so only one turn ever mutates its history at a
# time, closing off this whole class of corruption instead of patching
# individual symptoms of it.
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}


def _session_lock(session_id: str) -> asyncio.Lock:
    lock = _SESSION_LOCKS.get(session_id)
    if lock is None:
        lock = _SESSION_LOCKS[session_id] = asyncio.Lock()
    return lock


# Token budget, not cosmetics: the full tool schema is re-sent on EVERY
# request, so 13 tools x ~1000-char descriptions alone cost ~3.5k tokens a
# turn. Combined with history that was enough to push a single request past
# 10k tokens and trip the account's 30000 TPM limit (real reported 429:
# "Limit 30000, Used 20498, Requested 10307"). The model only needs enough
# description to PICK the right tool — the full disclosure text still lives
# in the MCP tool itself for real clients.
_TOOL_DESC_CHARS = 400   # per tool, sent every request
_TOOL_OUT_CHARS = 3500   # tool result kept in history (was 12000)
_STATUS_OUT_CHARS = 4000  # completed-job result injected into the prompt (after _slim_status)


async def _get_openai_tools() -> list:
    global _openai_tools
    if _openai_tools is None:
        tools = await mcp.list_tools()
        _openai_tools = [
            {"type": "function", "function": {
                "name": t.name,
                "description": (t.description or "")[:_TOOL_DESC_CHARS],
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            }}
            for t in tools
        ]
    return _openai_tools


def _slim_status(job_id: str) -> str:
    """Compact markdown for a job: real fields only, minus the bulky sections
    (see _slim_job). Returns "" when the job isn't in the store (caller falls
    back to the tool)."""
    from api.drug_discovery_router import _DRUG_DISCOVERY_STORE
    from mcp_server import _md, _slim_job

    job = _DRUG_DISCOVERY_STORE.get(job_id)
    if not job:
        return ""
    return _md(_slim_job(job))


async def _create_with_retry(client, **kwargs):
    """OpenAI's TPM limit is a rolling 60s window, so a 429 here is usually
    transient — the error itself says "try again in 1.61s". Retrying with
    backoff turns what used to surface as a hard "LLM 처리 중 오류" into a
    short pause the user never notices."""
    from openai import RateLimitError

    delay = 2.0
    for attempt in range(4):
        try:
            return await asyncio.to_thread(client.chat.completions.create, **kwargs)
        except RateLimitError:
            if attempt == 3:
                raise
            await asyncio.sleep(delay)
            delay *= 2


def _trim(history: list) -> list:
    """Real reported bug: naively slicing to the last N messages could start
    the kept window in the MIDDLE of an assistant(tool_calls) -> tool(...)
    sequence, leaving a 'tool' message with no preceding tool_calls message
    for it to respond to — OpenAI rejects that with a 400
    invalid_request_error ("messages with role 'tool' must be a response to
    a preceding message with 'tool_calls'"), which killed the whole turn.
    A 'tool' message is only ever valid immediately after its own paired
    assistant tool_calls message (that's the only way this code ever
    appends one — see _llm_chat), so trimming any LEADING 'tool' messages
    off the window always lands on a safe boundary."""
    if len(history) <= _MAX_HISTORY:
        return history
    window = history[-(_MAX_HISTORY - 1):]
    start = 0
    while start < len(window) and window[start]["role"] == "tool":
        start += 1
    return [history[0]] + window[start:]


async def _llm_chat(req: ChatReq) -> dict:
    from openai import OpenAI

    # Real reported bug: the 5s pollLoop() was routing EVERY status check
    # through a full OpenAI function-calling round-trip (full tool schema +
    # entire session history sent every time), just to hear "still running"
    # — burning thousands of TPM tokens per poll with zero new information.
    # Over a long job (many polls within the same rolling 60s window) this
    # alone was enough to trip the account's 30000 TPM limit, independent of
    # how many drug candidates were being docked. Fix: while polling, check
    # the job status directly via the tool (zero LLM tokens) and only fall
    # through to a real LLM call once the job has actually finished, so the
    # LLM is invoked once per job instead of once per 5s poll.
    # Set once the poll path has itself established the job reached a terminal
    # state. Without it `done` stays False on the completion turn (the LLM is
    # handed the result inline, so it never calls get_drug_discovery_job_status
    # and never trips the `done` flag below) — and the frontend's pollLoop only
    # renders the final summary when done=true, so the finished result was
    # never shown and the "⏳ 분석 진행 중…" spinner polled on forever.
    poll_terminal = False
    if req.is_poll and req.last_job_id:
        try:
            status_out = _slim_status(req.last_job_id) or _extract_text(
                await mcp.call_tool("get_drug_discovery_job_status", {"job_id": req.last_job_id})
            )
        except Exception as exc:
            status_out = f"[도구 오류] {exc}"
        # "Job not found" counts as terminal too: the job store is in-memory,
        # so a backend restart mid-job leaves the frontend polling a job_id
        # that no longer exists — without this it would spin on
        # "⏳ 분석 진행 중…" forever instead of telling the user what happened.
        if not _has(status_out, "COMPLETED", "FAILED", "CANCELLED", "Job not found", "도구 오류"):
            return {"reply": "", "tools": ["get_drug_discovery_job_status"],
                    "job_id": req.last_job_id, "started": True, "done": False}
        poll_terminal = True
        req = req.model_copy(update={
            "message": "작업이 완료됐습니다. 아래 결과를 일반인이 이해하도록 정리해줘.\n\n"
                       + status_out[:_STATUS_OUT_CHARS]
        })

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    tools = await _get_openai_tools()

    history = _SESSIONS.setdefault(req.session_id, [{"role": "system", "content": _SYSTEM_PROMPT}])
    user_msg = req.message
    if req.last_job_id:
        user_msg += f"\n\n(참고: 현재 진행 중인 job_id는 {req.last_job_id} 입니다.)"
    history.append({"role": "user", "content": user_msg})

    used_tools: list[str] = []
    job_id = req.last_job_id
    started = done = False
    final = ""

    # Real reported failure: asked "폐암 유전자는 우리 몸에서 무슨 일을 해?" the model
    # skipped the tools entirely and wrote a confident paragraph about EGFR from
    # its own memory — no Reactome call, nothing verifiable, exactly what this
    # system exists NOT to do. A system-prompt rule alone doesn't hold; the
    # model quietly ignores it whenever it "already knows" the answer. So catch
    # the shape of that failure — a long substantive reply with zero tool calls —
    # and make the model try again with tool_choice="required", which cannot be
    # satisfied without actually calling something. A short reply is left alone:
    # that's a clarifying question ("어떤 질환을 볼까요?"), which is legitimate.
    _MEMORY_ANSWER_CHARS = 200
    forced_tool_retry = False
    tool_choice = "auto"

    for _ in range(6):
        resp = await _create_with_retry(
            client,
            model=model, messages=_trim(history), tools=tools, tool_choice=tool_choice, temperature=0.2,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            if (not used_tools and not forced_tool_retry
                    and len(final) > _MEMORY_ANSWER_CHARS):
                logger.info("[playground] answered from memory with no tool call — forcing a tool")
                forced_tool_retry = True
                tool_choice = "required"
                continue  # discard the unverified answer, don't put it in history
            history.append({"role": "assistant", "content": final})
            break
        tool_choice = "auto"  # tools were used; let the model finish normally

        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            used_tools.append(name)
            try:
                out = _extract_text(await mcp.call_tool(name, args))
            except Exception as exc:  # surface, don't crash the turn
                out = f"[도구 오류] {exc}"
            found = _UUID_RE.search(out)
            if found:
                job_id = found.group(0)
            if name in _JOB_TOOLS:
                started = True
            if name == "get_drug_discovery_job_status" and _has(out, "COMPLETED", "FAILED", "CANCELLED"):
                done = True
            history.append({"role": "tool", "tool_call_id": tc.id, "content": out[:_TOOL_OUT_CHARS]})
    else:
        final = "분석 단계가 많아 조금 더 걸리고 있어요. 잠시 후 다시 시도하거나 '결과 확인'을 눌러 주세요."

    _SESSIONS[req.session_id] = _trim(history)
    return {"reply": final, "tools": list(dict.fromkeys(used_tools)),
            "job_id": job_id, "started": started, "done": done or poll_terminal}


# ── Keyword fallback (no LLM key) ─────────────────────────────────────────────

_TARGET_MAP = [
    ("kras", ("P01116", "KRAS")), ("tp53", ("P04637", "TP53")), ("p53", ("P04637", "TP53")),
    ("egfr", ("P00533", "EGFR")), ("braf", ("P15056", "BRAF")), ("erbb2", ("P04626", "HER2")),
    ("her2", ("P04626", "HER2")), ("brca1", ("P38398", "BRCA1")),
    ("스파이크", ("P0DTC2", "SARS-CoV-2 스파이크")), ("spike", ("P0DTC2", "SARS-CoV-2 스파이크")),
    ("코로나", ("P0DTC2", "SARS-CoV-2 스파이크")), ("covid", ("P0DTC2", "SARS-CoV-2 스파이크")),
    ("췌장암", ("P01116", "KRAS")), ("폐암", ("P00533", "EGFR")), ("대장암", ("P04637", "TP53")),
    ("유방암", ("P04626", "HER2")),
]


def _resolve_target(message: str) -> str:
    up = _UNIPROT_RE.search(message)
    if up:
        return up.group(1)
    low = message.lower()
    for key, (uid, _name) in _TARGET_MAP:
        if key in low:
            return uid
    return ""


def _find_smiles(message: str) -> str:
    m = re.search(r"smiles[:\s]+(\S+)", message, re.IGNORECASE)
    if m:
        return m.group(1)
    for name, smi in _NAMED_SMILES.items():
        if name.lower() in message.lower():
            return smi
    return ""


_VARIANT_RE = re.compile(r"\b(?:chr)?([0-9]{1,2}|[XYM])\s+([0-9]{3,})\s+([ACGT]+)\s+([ACGTacgt]+)\b", re.I)


def _variant_to_vcf(message: str) -> str:
    m = _VARIANT_RE.search(message)
    if not m:
        return ""
    chrom = "chr" + m.group(1).upper()
    return ("##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
            f"{chrom}\t{m.group(2)}\t.\t{m.group(3).upper()}\t{m.group(4).upper()}\t100\tPASS\t.\tGT\t0/1\n")


def _route(message: str, last_job_id: str) -> list[tuple[str, dict]]:
    m = message.strip()
    if last_job_id and _has(m, "결과", "상태", "진행", "완료", "확인", "status"):
        return [("get_drug_discovery_job_status", {"job_id": last_job_id})]
    uid = _resolve_target(m)
    if _has(m, "종합", "전체적", "comprehensive") and uid:
        return [("get_target_disease_associations", {"uniprot_id": uid}),
                ("get_target_pathways", {"uniprot_id": uid}),
                ("get_opentargets_profile", {"uniprot_id": uid})]
    if _has(m, "백신", "신항원", "네오안티젠", "neoantigen", "mrna"):
        vcf = _variant_to_vcf(m)
        return [("predict_neoantigen_candidates", {"vcf_content": vcf} if vcf else {})]
    if _has(m, "치료제", "신약", "후보 물질", "후보물질", "도킹", "docking", "결합", "약물") or (_has(m, "찾아") and uid):
        args = {"screen_library": True, "goal_text": m}
        if uid:
            args["uniprot_id"] = uid
        return [("predict_drug_binding", args)]
    if _has(m, "저해제", "억제제", "ic50", "inhibitor"):
        return [("search_known_inhibitors", {"uniprot_id": uid or "P00533"})]
    if _has(m, "admet", "독성", "약물성"):
        smi = _find_smiles(m)
        return [("predict_admet_profile", {"smiles": smi})] if smi else []
    if _has(m, "논문", "문헌", "pubmed"):
        return [("search_literature", {"query": m})]
    if _has(m, "임상", "clinical", "trial"):
        return [("search_clinical_trials", {"query": m})]
    return []


_HELP = ("무엇을 도와드릴까요? 예: **췌장암 치료제 후보 찾아줘**, "
         "**mRNA 암 백신 분석해줘**, **대장암 유전자 p53 종합 분석해줘**, "
         "**아스피린 ADMET 분석해줘**")


async def _keyword_chat(req: ChatReq) -> dict:
    calls = _route(req.message, req.last_job_id)
    if not calls:
        return {"reply": _HELP, "tools": [], "job_id": req.last_job_id, "started": False, "done": True}
    replies, used, job_id, started, done = [], [], req.last_job_id, False, False
    for name, args in calls:
        if name == "predict_drug_binding" and not args.get("uniprot_id"):
            replies.append("어떤 질환이나 표적 단백질을 대상으로 할까요? (예: 췌장암, 폐암, KRAS)")
            continue
        try:
            out = _extract_text(await mcp.call_tool(name, args))
        except Exception as exc:
            out = f"[도구 오류] {exc}"
        used.append(name)
        replies.append((f"■ {name}\n" if len(calls) > 1 else "") + out)
        found = _UUID_RE.search(out)
        if found:
            job_id = found.group(0)
        if name in _JOB_TOOLS:
            started = True
        if name == "get_drug_discovery_job_status" and _has(out, "COMPLETED", "FAILED", "CANCELLED"):
            done = True
    return {"reply": "\n\n".join(replies), "tools": used, "job_id": job_id, "started": started, "done": done}


def _still_running(req: ChatReq) -> dict | None:
    """The 'job hasn't finished yet' answer to a poll. It reads the job store
    and touches no conversation state, so it deliberately runs OUTSIDE the
    session lock: while a job was running, the frontend's 5s pollLoop held that
    lock on every tick, and a message the user typed in the meantime queued
    behind it and sat on '생각 중…'. Returns None once the job is terminal, so
    the caller falls through to the real (locked, history-mutating) turn."""
    if not (req.is_poll and req.last_job_id):
        return None
    try:
        status_out = _slim_status(req.last_job_id)
    except Exception as exc:
        status_out = f"[도구 오류] {exc}"
    if not status_out:
        return None  # not in the store — let the locked path report it properly
    if _has(status_out, "COMPLETED", "FAILED", "CANCELLED", "Job not found", "도구 오류"):
        return None
    return {"reply": "", "tools": ["get_drug_discovery_job_status"],
            "job_id": req.last_job_id, "started": True, "done": False}


@router.post("/playground/chat")
async def playground_chat(req: ChatReq) -> dict:
    pending = _still_running(req)
    if pending is not None:
        return pending

    async with _session_lock(req.session_id):
        if os.getenv("OPENAI_API_KEY"):
            try:
                return await _llm_chat(req)
            except Exception as exc:
                return {"reply": f"LLM 처리 중 오류가 발생했습니다: {exc}", "tools": [],
                        "job_id": req.last_job_id, "started": False, "done": True}
        return await _keyword_chat(req)


@router.get("/playground", response_class=HTMLResponse)
async def playground_page() -> HTMLResponse:
    # no-store: the page's JS is inlined here, so a browser holding a cached
    # copy keeps running an OLD polling loop against a NEW backend. That is
    # exactly what made a finished job look stuck — the server had already
    # returned done=true with the full result, but the stale script never
    # rendered it. Never let this page be cached.
    return HTMLResponse(
        _PAGE,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


_PAGE = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>신약개발 · AiRemedy</title>
<style>
  :root { --kakao:#FEE500; --bg:#b2c7d9; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; font-family:'Apple SD Gothic Neo','Malgun Gothic',system-ui,sans-serif;
         background:var(--bg); height:100vh; display:flex; flex-direction:column; }
  header { background:#a2b8ca; color:#25333f; padding:12px 16px; font-weight:700; font-size:15px;
           display:flex; align-items:center; gap:8px; box-shadow:0 1px 3px rgba(0,0,0,.15); z-index:2; }
  header small { font-weight:500; opacity:.7; font-size:12px; }
  header button { margin-left:auto; background:#fff; border:1px solid #cfd9e2; border-radius:14px;
                  padding:4px 10px; font-size:11.5px; color:#25333f; cursor:pointer; }
  header button:active { transform:scale(.97); }
  #log { flex:1; overflow-y:auto; padding:14px 12px 4px; }
  .row { display:flex; margin:6px 0; align-items:flex-end; gap:6px; }
  .row.me { justify-content:flex-end; }
  .bubble { max-width:80%; padding:9px 12px; border-radius:14px; font-size:14px; line-height:1.55;
            word-break:break-word; box-shadow:0 1px 1px rgba(0,0,0,.08); }
  .bot .bubble { background:#fff; border-top-left-radius:4px; color:#1a1a1a; }
  .me .bubble  { background:var(--kakao); border-top-right-radius:4px; color:#1a1a1a; }
  .name { font-size:11px; color:#33475b; margin:0 0 2px 4px; }
  .bubble b { font-weight:700; }
  .bubble ul { margin:4px 0; padding-left:18px; }
  .tool-tag { display:inline-block; font-size:11px; background:#eef3f8; color:#3a5a78;
              border:1px solid #d6e0ea; border-radius:10px; padding:1px 7px; margin:0 3px 4px 0; }
  .typing { color:#8a8a8a; font-style:italic; }
  .chips { padding:8px 10px; background:#a9bccd; max-height:38vh; overflow-y:auto; }
  .grp { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin-bottom:6px; }
  .grp:last-child { margin-bottom:0; }
  .grp-label { font-size:10.5px; font-weight:700; color:#33475b; background:rgba(255,255,255,.45);
               border-radius:8px; padding:2px 7px; white-space:nowrap; }
  .chip { background:#fff; border:none; border-radius:16px; padding:7px 11px; font-size:12.5px;
          color:#25333f; cursor:pointer; box-shadow:0 1px 2px rgba(0,0,0,.1); }
  .chip:active { transform:scale(.97); }
  .chip.seq { background:#FEE500; font-weight:700; }
  form { display:flex; gap:8px; padding:8px 10px; background:#f4f4f4; border-top:1px solid #ddd; }
  input { flex:1; border:1px solid #ccc; border-radius:20px; padding:10px 14px; font-size:14px; outline:none; }
  button.send { background:var(--kakao); border:none; border-radius:20px; padding:0 18px; font-weight:700;
                cursor:pointer; font-size:14px; }
</style></head>
<body>
  <header>🧬 신약개발 <small>AiRemedy MCP 어시스턴트</small>
    <button id="reset" title="대화와 연구 기록을 초기화합니다">🆕 새 연구</button>
  </header>
  <div id="log"></div>
  <!-- Every button is a layperson sentence — no UniProt IDs, no SMILES, no VCF.
       Between them they exercise all 14 MCP tools; the assistant does the
       translation into accessions/structures, which is the whole point. -->
  <div class="chips" id="chips">
    <div class="grp">
      <span class="grp-label">암 백신 시나리오</span>
      <button class="chip seq">① 환자의 암 정보를 분석해서 맞춤형 mRNA 암 백신을 설계해줘.</button>
      <button class="chip seq">② 백신 후보 논문과 임상 근거를 분석해줘.</button>
      <button class="chip seq">③ 종합 연구 리포트로 만들어줘.</button>
    </div>
    <div class="grp">
      <span class="grp-label">저분자 신약</span>
      <button class="chip">코로나 스파이크 단백질을 막는 약 후보를 찾아줘.</button>
      <button class="chip">1위 후보를 더 좋게 개선할 수 있어?</button>
      <button class="chip">아스피린이 신약 후보로 적합한지 분석해줘.</button>
      <button class="chip">아스피린이랑 비슷한 물질을 찾아줘.</button>
      <button class="chip">폐암 유전자를 잘 막는 약 중에 효과가 센 걸 알려줘.</button>
    </div>
    <div class="grp">
      <span class="grp-label">표적 알아보기</span>
      <button class="chip">폐암 유전자는 어떤 병들과 관련이 있어?</button>
      <button class="chip">폐암 유전자는 우리 몸에서 무슨 일을 해?</button>
      <button class="chip">폐암 유전자가 신약 표적으로 유망한지 알려줘.</button>
    </div>
    <div class="grp">
      <span class="grp-label">근거 찾기</span>
      <button class="chip">췌장암 KRAS 백신 관련 논문 찾아줘.</button>
      <button class="chip">췌장암 KRAS 백신 임상시험이 진행 중인 게 있어?</button>
      <button class="chip">지금 분석 진행 상황 알려줘.</button>
    </div>
  </div>
  <form id="f">
    <input id="msg" placeholder="편하게 물어보세요… (예: 폐암 신약 후보 찾아줘)" autocomplete="off"/>
    <button class="send" type="submit">전송</button>
  </form>
<script>
const log=document.getElementById('log'), form=document.getElementById('f'), input=document.getElementById('msg');
const GREETING = '안녕하세요! 신약개발 AI 어시스턴트입니다. 😊 아래 ①→②→③ 순서로 눌러 보세요.\\n\\n- **① 환자의 암 정보를 분석해서 맞춤형 mRNA 암 백신을 설계해줘.**\\n- **② 백신 후보 논문과 임상 근거를 분석해줘.**\\n- **③ 종합 연구 리포트로 만들어줘.**\\n\\n연구를 새로 시작하려면 우측 상단 **🆕 새 연구**를 누르세요.';
let sessionId = 's-' + Math.random().toString(36).slice(2);
let lastJobId="", progressEl=null, polling=false;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function mdToHtml(s){
  const lines=esc(s).split('\\n'); let html='',inList=false;
  for(let ln of lines){
    ln=ln.replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>');
    if(/^\\s*[-*]\\s+/.test(ln)){ if(!inList){html+='<ul>';inList=true;} html+='<li>'+ln.replace(/^\\s*[-*]\\s+/,'')+'</li>'; }
    else { if(inList){html+='</ul>';inList=false;} html+= ln.trim()===''?'<br/>':'<div>'+ln+'</div>'; }
  }
  if(inList) html+='</ul>'; return html;
}
function tagsHtml(tools){ return (tools&&tools.length)? tools.map(t=>'<span class="tool-tag">🛠 '+t+'</span>').join('')+'<br/>':''; }
function addRow(who, html, tools){
  const row=document.createElement('div'); row.className='row '+(who==='me'?'me':'bot');
  const wrap=document.createElement('div');
  if(who==='bot'){ const n=document.createElement('div'); n.className='name'; n.textContent='신약개발'; wrap.appendChild(n); }
  const b=document.createElement('div'); b.className='bubble'; b.innerHTML=tagsHtml(tools)+html;
  wrap.appendChild(b); row.appendChild(wrap); log.appendChild(row); log.scrollTop=log.scrollHeight; return b;
}
async function post(message, isPoll){
  const r=await fetch('/playground/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message, session_id:sessionId, last_job_id:lastJobId, is_poll:!!isPoll})});
  return r.json();
}
function setProgress(text){
  if(!text){ if(progressEl){progressEl.parentElement.parentElement.remove(); progressEl=null;} return; }
  if(!progressEl) progressEl=addRow('bot','<span class="typing">'+text+'</span>');
  else progressEl.innerHTML='<span class="typing">'+text+'</span>';
  log.scrollTop=log.scrollHeight;
}
async function pollLoop(n){
  if(n>30||!polling) return;
  await sleep(5000);
  if(!polling) return;
  const d=await post('작업 상태를 확인하고, 완료됐으면 결과를 일반인이 이해하도록 정리해줘.', true);
  if(d.job_id) lastJobId=d.job_id;
  if(d.done){ setProgress(null); polling=false; addRow('bot',mdToHtml(d.reply),d.tools); }
  else { setProgress('⏳ 분석 진행 중… 결과를 기다리는 중입니다 ('+(n+1)+')'); pollLoop(n+1); }
}
async function send(text){
  addRow('me', esc(text));
  const typing=addRow('bot','<span class="typing">생각 중…</span>');
  try{
    const d=await post(text);
    typing.parentElement.parentElement.remove();
    if(d.job_id) lastJobId=d.job_id;
    addRow('bot', mdToHtml(d.reply), d.tools);
    if(d.started && d.job_id && !polling){ polling=true; setProgress('⏳ 분석 진행 중… 결과를 기다리는 중입니다 (1)'); pollLoop(1); }
  }catch(e){ typing.innerHTML='요청 실패: '+esc(String(e)); }
}
// A new session id is the only way to drop the server-side conversation: the
// backend keys history by it, so without this the next question is answered
// against the previous study's context — a fresh-looking chat that still
// "remembers" a KRAS vaccine you never designed in it.
document.getElementById('reset').addEventListener('click', ()=>{
  sessionId = 's-' + Math.random().toString(36).slice(2);
  lastJobId=''; polling=false; progressEl=null;
  log.innerHTML='';
  addRow('bot', mdToHtml(GREETING));
});
form.addEventListener('submit',e=>{e.preventDefault(); const t=input.value.trim(); if(!t)return; input.value=''; send(t);});
document.getElementById('chips').addEventListener('click',e=>{ if(e.target.classList.contains('chip')) send(e.target.textContent); });
addRow('bot', mdToHtml(GREETING));
</script>
</body></html>"""
