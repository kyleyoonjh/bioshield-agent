"""
AiRemedy MCP Server — FastMCP Streamable HTTP
Protocol version: 2025-03-26 (PlayMCP compatible)
Tools carry full ToolAnnotations (title/readOnlyHint/destructiveHint/
openWorldHint/idempotentHint) as required by PlayMCP.

Mount at /mcp in main.py:
    app.mount("/mcp", mcp_app)

PlayMCP tool-count note: PlayMCP's submission guide caps tools at 20
(recommends 3-10) per server. This server exposes the drug-discovery and
mRNA-vaccine tools only; the earlier primer/oligo molecular-diagnostics
tooling was removed from the project entirely.

PlayMCP response-time note (2026-07-08): PlayMCP requires avg tool response
<=100ms and p99 <=3000ms. predict_drug_binding, run_sar_optimization, and
predict_neoantigen_candidates wrap real multi-second-to-30-minute pipelines
(docking/re-docking/MHCflurry inference), so each now starts its pipeline as
a background asyncio task via api/drug_discovery_router.py's existing
_DRUG_DISCOVERY_STORE job store and returns a job_id immediately — poll the
new get_drug_discovery_job_status tool until COMPLETED/FAILED. The other
tools (search_literature, search_clinical_trials, etc.) are single real
external-API calls with no long-running computation to defer; they still
exceed the 100ms average in practice (real network round-trip time), which
is not something further code changes here can fix.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

logger = logging.getLogger("openbioshield.mcp")

mcp = FastMCP(
    "open-bioshield",
    streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# PlayMCP guidance: keep tool-call results minimal, and for non-widget text
# content return refined text (markdown) rather than a raw API-response dump —
# a full JSON dump carries unnecessary fields that make for worse answers.
# _md() renders an engine result as compact markdown: it reflects every real
# field (so nothing is silently dropped or fabricated, unlike hand-picking
# keys) but omits null/empty values, collapses whitespace, truncates long
# strings (e.g. abstracts), caps long lists, and bounds nesting depth — so the
# payload stays small without losing the honest "0건"/empty-is-a-real-result
# signal the engines deliberately return.
_MD_MAX_DEPTH, _MD_MAX_LIST, _MD_MAX_STR = 5, 12, 300


def _md(obj, depth: int = 0) -> str:
    if obj is None:
        return "_(없음)_"
    if isinstance(obj, bool):
        return "예" if obj else "아니오"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        s = " ".join(obj.split())
        return s if len(s) <= _MD_MAX_STR else s[:_MD_MAX_STR].rstrip() + "…"
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= _MD_MAX_DEPTH:
            return f"{pad}…"
        lines = []
        for k, v in obj.items():
            if v is None or v == "" or v == {}:
                continue
            if isinstance(v, list) and len(v) == 0:
                lines.append(f"{pad}- **{k}**: 0건")
            elif isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{k}**:")
                lines.append(_md(v, depth + 1))
            else:
                lines.append(f"{pad}- **{k}**: {_md(v, depth + 1)}")
        return "\n".join(lines) if lines else f"{pad}_(없음)_"
    if isinstance(obj, list):
        if depth >= _MD_MAX_DEPTH:
            return f"{pad}…({len(obj)}건)"
        lines = []
        for i, v in enumerate(obj[:_MD_MAX_LIST], 1):
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{i}.\n{_md(v, depth + 1)}")
            else:
                lines.append(f"{pad}{i}. {_md(v, depth + 1)}")
        if len(obj) > _MD_MAX_LIST:
            lines.append(f"{pad}…외 {len(obj) - _MD_MAX_LIST}건")
        return "\n".join(lines)
    return str(obj)


# A finished neoantigen job is ~18,000 characters, most of it internal
# bookkeeping the caller never needs: every peptide the pipeline scored and
# rejected (all_scored), the raw per-variant VEP annotations, the step-by-step
# timeline the web UI's progress bar reads, per-gene literature, non-fatal
# prediction warnings. A client with a result-size cap truncates that from the
# end, which threw away the actual answer — the selected candidate peptides —
# and left the model staring at a stray "all_scored" key, which is how PlayMCP
# came to report a COMPLETED job as "실패 사유는 'all_scored'". Dropping the
# bulky sections keeps the scientific payload (candidates, HLA context,
# interpretation) inside any sane budget. Same key set the local playground
# already prunes with; it belongs here, on the tool Kakao actually calls.
_JOB_VERBOSE_KEYS = {
    "timeline", "variant_annotations", "all_scored", "decision_log",
    "literature_by_gene", "prediction_errors", "references", "papers",
    "source_variant", "score_breakdown", "info", "samples",
    # Raw structural coordinates. A docking job's result carries the target's
    # full PDB text — 1.6 MILLION characters for the spike protein — plus a
    # docked-pose PDBQT per candidate. _md truncates a long string to 300 chars,
    # so this never actually blew the response, but those 300 chars are a
    # meaningless fragment of "HEADER AF-… COMPND 2 MOLECULE:…" that costs real
    # tokens and tells the model nothing. The numbers derived from the geometry
    # (affinity, pocket residues, confidence) are what belongs in an answer; the
    # coordinates themselves belong in the report file.
    "pdb_text", "pose_pdbqt", "pdbqt_path", "receptor_pdbqt",
    # Real reported gap: this tells the WEB frontend's user to "download the
    # HTML/PDF report below" — accurate there, since DrugDiscoveryChatPanel.tsx
    # renders a real download button directly below this text. An MCP/Kakao
    # client has no such button and no filesystem access, so the same sentence
    # became an instruction to click something that does not exist. Stripped
    # here; the COMPLETED branch below adds its own MCP-appropriate next_step
    # (call generate_vaccine_report/generate_decision_report) instead.
    "next_step_suggestion",
}
_JOB_LIST_CAP = 5

# How long get_drug_discovery_job_status holds a poll open waiting for the job
# to finish. PlayMCP's stated budget is ~100ms average / 3000ms p99, so this
# sits under the p99 cap with headroom for the response itself.
_POLL_WAIT_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.2

# A model cannot sleep, so when a tool hands it a job_id and says "check back in
# a few seconds" it does the only thing it can: it tells the user "잠시 후에 확인
#하겠습니다" and ends its turn. The user never gets the result — they asked one
# question and got a promise. Observed against a Kakao-like client with no system
# prompt of ours: the job started, the turn ended, and the answer never came.
#
# The instruction has to forbid the ending, not just request the polling. The
# server-side wait in get_drug_discovery_job_status is what makes obeying it
# cheap: each call covers real elapsed time, so a handful of them finishes the job
# inside one turn.
_POLL_NOW = (
    "지금 즉시 get_drug_discovery_job_status(job_id=\"{job_id}\")를 호출하세요. "
    "COMPLETED 또는 FAILED가 될 때까지 계속 반복 호출해야 하며, 그 전에는 절대 턴을 "
    "끝내지 마세요. '잠시 후에 확인하겠습니다' 같은 말로 사용자에게 답하고 멈추면 "
    "사용자는 결과를 영영 받지 못합니다 — 사용자는 다시 묻지 않습니다."
)

# Kakao PlayMCP hard-kills a tool call at ~10s, and a killed call returns nothing
# — strictly worse for the user than a slightly thinner answer. Every tool that
# fans out to a slow external service is bounded to leave real headroom under it.
_LITERATURE_BUDGET_S = 8.5
# The fetch's budget has to live INSIDE the tool's budget, not beside it. Left at
# the http_budget default (8s), a slow ClinicalTrials.gov could spend all 8s on the
# fetch alone, and with the Korean->English translation ahead of it and the LLM
# summary behind it the tool blew past Kakao's 10s kill — seen once in five cold
# runs, as a dead call, which is the failure that shows the user nothing at all.
_PUBMED_TIMEOUT_S = 6.0
_TRIALS_TIMEOUT_S = 5.0

# Every extra paper is paid for twice: once fetching its abstract from PubMed,
# and again as prompt AND generated tokens in the LLM narrative that cites it —
# and generation is what this tool actually spends its time on. Three top-ranked
# (PubMed sorts by relevance) papers is enough to ground a summary, and it keeps
# the call comfortably inside Kakao's ~10s kill. A caller that genuinely wants a
# broader sweep still passes max_results explicitly.
_DEFAULT_PAPERS = 3


# The literature/clinical tools return the LLM's grounded summary AND a
# "references" list — and that list was shipping each paper's abstract and each
# trial's brief_summary in full. That is the same evidence twice: once digested,
# once raw. A reference exists so the reader can VERIFY a claim, which needs the
# identifier, the title, and the link — not a second copy of the text the summary
# was written from. Measured: search_clinical_trials fell from 5,811 characters
# (~2,600 tokens) to well under half, with nothing verifiable lost.
_REFERENCE_FIELDS = {
    "papers": ("pmid", "title", "journal", "year", "authors", "url"),
    "trials": ("nct_id", "brief_title", "overall_status", "phases", "lead_sponsor", "url"),
}


def _slim_references(items: list[dict], kind: str) -> list[dict]:
    keep = _REFERENCE_FIELDS[kind]
    return [{k: item[k] for k in keep if item.get(k)} for item in items]


def _slim_job(obj):
    if isinstance(obj, dict):
        return {k: _slim_job(v) for k, v in obj.items() if k not in _JOB_VERBOSE_KEYS}
    if isinstance(obj, list):
        return [_slim_job(v) for v in obj[:_JOB_LIST_CAP]]
    return obj


# Real reported gap: passing a disease/drug NAME (e.g. "코로나 신약", "암
# 신약") instead of a real UniProt accession into the four target-lookup
# tools below produced a raw, confusing native error from whichever
# downstream API was hit first (UniProt 400 Bad Request / ChEMBL "No target
# found for 코로나 신약" / OpenTargets "no entry") instead of guidance toward
# the actual next step. Checked before the real network call so callers get
# a clear, actionable message immediately instead of a wasted round trip.
_UNIPROT_ID_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


# MCP spec: a tool that FAILED must say so — the result carries isError=true so
# the caller can tell a failure from an answer. FastMCP sets that flag when the
# tool raises, so a failure has to raise; returning an error string instead
# reports a successful call whose content happens to mention an error, which is
# how a ChEMBL timeout showed up in Kakao PlayMCP as a green "성공".
#
# The distinction that matters here is failure vs. honest emptiness. Zero papers,
# zero neoantigen candidates, a viral protein with no Reactome pathways, no
# ChEMBL target for an accession — those are real answers this project
# deliberately reports rather than hides, and they must keep returning normally
# with isError=false. Only a genuine failure (bad input, missing job, the
# upstream API erroring out) raises.
_FAILURE_MARKERS = ("request failed", "could not parse", "timed out", "timeout")


def _fail(message: str, **context) -> None:
    logger.info("[mcp] tool call rejected | %s | context=%s", message, context)
    raise ToolError(_md({"error": message, **context}))


def _raise_if_upstream_failed(result: dict) -> dict:
    """Raise when an engine reports that the external API call itself broke.
    An engine's "error" field is also used for honest absences ("No ChEMBL
    target found for ..."), which are NOT failures — hence matching on the
    request/parse failure wording rather than on the field being present."""
    err = (result or {}).get("error") or ""
    if err and any(m in err.lower() for m in _FAILURE_MARKERS):
        raise ToolError(err)
    return result


def _uniprot_guard(uniprot_id: str) -> None:
    """Raises when uniprot_id isn't shaped like a real accession (a caller
    mistake, so a real error), otherwise returns and the tool proceeds."""
    if _UNIPROT_ID_RE.match((uniprot_id or "").strip().upper()):
        return
    _fail(
        f"{uniprot_id!r} doesn't look like a real UniProt accession (e.g. P0DTC2, P04637).",
        next_step=(
            "This tool needs a specific protein's UniProt accession, not a disease/drug name "
            "or a Korean phrase. Resolve the real target protein first (e.g. the known causative "
            "protein/receptor for the disease or pathogen in question), then call this tool with "
            "that accession."
        ),
    )


# Real reported gap: without our own playground's job_id-hint injection (which
# only exists on the /playground/chat path, never on a raw MCP client like
# PlayMCP or Kakao), a model loses track of the literal job_id UUID a few
# turns into a conversation and substitutes a plausible-looking description
# instead (e.g. "KRAS G12D neoantigen vaccine"). The store lookup then just
# said "Job not found" — true, but it gives the model nothing to recover
# with, so it apologized and gave up instead of retrying with the real id.
# Checked before the store lookup so the caller gets a corrective next_step
# immediately, the same discipline as _uniprot_guard above.
_JOB_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# A well-formed job_id that is not in the store (the _job_id_guard above already
# ruled out a hallucinated id) means the job existed but is gone: the job store is
# in-process memory, so a server restart/redeploy — or, if the platform runs more
# than one replica, a poll routed to a different instance than the one that started
# the job — drops it. "Job not found" alone made the model apologize and stop; this
# tells it (and the user) the real, recoverable situation.
_JOB_LOST_MSG = (
    "이 job_id는 형식은 올바르지만 서버에 남아 있지 않습니다. 진행 중이던 작업이 "
    "유실된 것으로 보입니다(서버 재시작 등) — 분석 자체가 잘못된 것이 아닙니다."
)
_JOB_LOST_NEXT = (
    "처음 분석 요청(predict_neoantigen_candidates / predict_drug_binding 등)을 다시 "
    "호출해 새 작업을 시작하세요. 사용자에게는 '이전 작업이 만료되어 다시 시작한다'고 "
    "간단히 안내하면 됩니다."
)


def _job_id_guard(job_id: str, started_by: str) -> None:
    """Raises when job_id isn't shaped like a real UUID (a caller mistake —
    a description substituted for the id), otherwise returns and the tool
    proceeds to the real store lookup."""
    if _JOB_ID_RE.match((job_id or "").strip()):
        return
    _fail(
        f"{job_id!r} doesn't look like a real job_id (should be a UUID, e.g. "
        "'558c5427-7430-473e-8898-0347cd887993').",
        next_step=(
            f"job_id must be the exact id returned earlier by {started_by} — never a gene "
            "name, disease, or description. Look back through this conversation for that "
            "id, or start a new job."
        ),
    )


# Real reported gap: PubMed/ClinicalTrials.gov's own search indexes are
# effectively English-only, so a raw Korean query (e.g. "코로나 신약", "암
# 신약") silently returns zero real hits — a real, honest empty result, not
# a bug in the fetch itself, but with no hint about WHY, a caller has no way
# to know a better query would work. This is a real, code-computed check
# (Hangul character range), not a guess — appended only when the fetch
# genuinely found nothing.
_HANGUL_RE = re.compile(r"[가-힣]")


_MAX_RESULTS_CAP = 25


def _require_count(max_results: int) -> None:
    """A non-positive max_results made PubMed/ClinicalTrials return nothing, and the
    tool reported that as a clean "0건" — which a model reads as "no such research
    exists". A bad argument must never be able to manufacture a negative scientific
    finding. Caught by parameter fuzzing with max_results=-5."""
    if max_results < 1:
        _fail(f"max_results must be at least 1 (got {max_results}).")
    if max_results > _MAX_RESULTS_CAP:
        _fail(f"max_results must be at most {_MAX_RESULTS_CAP} (got {max_results}).")


def _require_query(query: str) -> None:
    """An empty/whitespace query is a caller mistake, not a search with no hits.
    Fuzzing caught both literature and trials answering "\\n\\t" with a cheerful
    0-result payload and isError=false — indistinguishable, to a model, from a
    real "nothing published on this" finding, which is a claim this project must
    never make falsely."""
    if not query or not query.strip():
        _fail("query is empty — pass a real search term (e.g. 'KRAS G12D vaccine').")


def _non_english_hint(query: str, total_count: int) -> str | None:
    if total_count > 0 or not _HANGUL_RE.search(query):
        return None
    return (
        "No results for this Korean-language query — PubMed/ClinicalTrials.gov's search "
        "indexes are effectively English-only. Try a well-formed English biomedical term "
        "instead (e.g. \"코로나 신약\" -> \"SARS-CoV-2 antiviral\" or \"COVID-19 therapeutics\"; "
        "\"암 신약\" is too broad on its own -> name the specific cancer type and/or target, "
        "e.g. \"colorectal cancer targeted therapy\" or \"EGFR inhibitor lung cancer\")."
    )


def _resolve_english_query(query: str) -> tuple[str, str | None]:
    """Real reported gap: search_literature/search_clinical_trials sent a
    raw Korean query straight to PubMed/ClinicalTrials.gov (both effectively
    English-only indexes), returning zero real hits with no attempt to
    translate first — even though drug_discovery_chat.py's conversational
    layer already has real, tested Korean->English translation (a curated
    common-term map, longest-match-first so "대장암" doesn't get shadowed by
    the substring "암", plus a real LLM fallback for anything not curated —
    see _translate_korean_query()'s own docstring for the "NONE" sentinel
    discipline that keeps it from guessing when there's genuinely no real
    topic). Reused here instead of reimplementing it. Returns
    (query_to_actually_search, original_query_or_None) — the second element
    is only set when translation actually happened, so callers can disclose
    it rather than silently substituting the term.
    """
    if not _HANGUL_RE.search(query):
        return query, None
    from services.drug_discovery_chat import _translate_korean_query
    translated = _translate_korean_query(query)
    if translated and not _HANGUL_RE.search(translated):
        return translated, query
    return query, None


# ── Tool 10: predict_drug_binding ───────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Predict Drug Binding",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def predict_drug_binding(
    uniprot_id:      str = "",
    target_sequence: str = "",
    ligand_smiles:   str = "",
    screen_library:  bool = False,
    goal_text:       str = "",
    max_candidates:  int = 3,
) -> str:
    """Starts AiRemedy(AI신약) SMALL-MOLECULE drug discovery as a background job, returning a job_id immediately (~30s-30min; RUNNING is normal, not a failure). Resolves the target structure via AlphaFold DB (uniprot_id) or ESMFold (target_sequence, <=400aa), then runs real AutoDock Vina docking. ligand_smiles docks one ligand; screen_library=true screens the curated ~22-drug library, capped by max_candidates (default 3; 0=full library; never faked). Give exactly one of uniprot_id/target_sequence — map the gene/disease yourself (KRAS/pancreatic=P01116, TP53=P04637, EGFR/lung=P00533, SARS-CoV-2 spike=P0DTC2, HER2=P04626, BRAF=P15056). MODALITY GUARD: this is protein inhibition, NOT mRNA-vaccine/neoantigen immunotherapy — a different therapy. Never call it to "evaluate" a vaccine candidate; that silently swaps the user's research topic. Mid-vaccine-study, ASK before running it. Poll get_drug_discovery_job_status(job_id) REPEATEDLY until COMPLETED/FAILED, then read "result" for ranked results."""
    from fastapi import HTTPException
    from api.drug_discovery_router import start_drug_discovery, DrugDiscoveryRequest

    logger.info("[mcp] predict_drug_binding called | uniprot_id=%s screen_library=%s max_candidates=%s",
                uniprot_id or "(none)", screen_library, max_candidates)
    if uniprot_id:
        _uniprot_guard(uniprot_id)
    if ligand_smiles:
        from rdkit import Chem
        if Chem.MolFromSmiles(ligand_smiles) is None:
            return _md({
                "available": False,
                "error": f"{ligand_smiles!r} is not a valid SMILES string.",
                "next_step": (
                    "ligand_smiles needs a real chemical structure in SMILES notation (e.g. "
                    "CC(=O)OC1=CC=CC=C1C(=O)O for aspirin). If you don't have one specific "
                    "ligand in mind, omit ligand_smiles and set screen_library=true instead to "
                    "screen the curated candidate library."
                ),
            })

    try:
        result = await start_drug_discovery(DrugDiscoveryRequest(
            uniprot_id=uniprot_id,
            target_sequence=target_sequence,
            ligand_smiles=ligand_smiles,
            screen_library=screen_library,
            goal_text=goal_text,
            max_candidates=max_candidates,
        ))
    except HTTPException as exc:
        _fail(exc.detail)
    return _md({**result, "다음_단계": _POLL_NOW.format(job_id=result["job_id"])})


# ── Tool 11: search_literature ──────────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Search Literature",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def search_literature(query: str, max_results: int = _DEFAULT_PAPERS) -> str:
    """Real live PubMed search + evidence summary for a drug discovery research question (e.g. "SARS-CoV-2 spike glycoprotein inhibitor") in AiRemedy(AI신약). Fetches real papers via NCBI E-utilities (title/abstract/journal/year/PMID/DOI) — never fabricates a paper or finding. When OPENAI_API_KEY is set, an LLM synthesizes an evidence_summary and per-finding PMID citations strictly grounded in the fetched abstracts (any citation to a PMID not in the real fetched set is discarded); otherwise returns a real extractive summary (first real sentence of each abstract). Always returns the real "references" list so callers can verify every claim."""
    from services.literature_engine import search_pubmed
    from services.drug_discovery_literature_agent import summarize_literature

    # Three sequential hops — translate, search PubMed, summarize — two of them
    # LLM calls. Left unbounded (PubMed alone allowed 15s) any one of them can
    # push the tool past Kakao's ~10s timeout, and a timeout there returns
    # nothing at all. The papers are the real evidence; the narrative is a
    # convenience on top. So the search gets a hard cap, and if the summary
    # can't be written in the time that's left, the real papers still go back
    # with the narrative honestly marked missing rather than the whole call dying.
    logger.info("[mcp] search_literature called | query=%r max_results=%s", query, max_results)
    _require_query(query)
    _require_count(max_results)
    deadline = time.monotonic() + _LITERATURE_BUDGET_S
    search_query, translated_from = await asyncio.to_thread(_resolve_english_query, query)
    fetched = _raise_if_upstream_failed(
        await asyncio.to_thread(search_pubmed, search_query, max_results, _PUBMED_TIMEOUT_S))
    if fetched.get("error"):
        return _md(fetched)

    papers = fetched["papers"]
    try:
        summary = await asyncio.wait_for(
            asyncio.to_thread(summarize_literature, search_query, papers),
            timeout=max(0.5, deadline - time.monotonic()),
        )
    except asyncio.TimeoutError:
        logger.warning("[mcp] literature summary timed out | query=%r", search_query)
        summary = {
            "references": papers,
            "limitations": "AI 요약 생성이 제한 시간을 초과해 논문 목록만 반환합니다 — 아래 논문은 실제 PubMed 검색 결과입니다.",
        }
    summary["references"] = _slim_references(summary.get("references") or papers, "papers")
    total_count = fetched.get("total_count", 0)
    hint = _non_english_hint(search_query, total_count)
    extra = {"next_step": hint} if hint else {}
    if translated_from:
        extra["translated_from"] = translated_from
    return _md({"query": search_query, "total_count": total_count, **summary, **extra})


# ── Tool 12: search_clinical_trials ─────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Search Clinical Trials",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def search_clinical_trials(query: str, max_results: int = 5) -> str:
    """Real live ClinicalTrials.gov v2 search + clinical landscape summary for a drug discovery research question (e.g. "KRAS G12D inhibitor") in AiRemedy(AI신약). Fetches real trials via the public ClinicalTrials.gov REST API (NCT ID/title/overall status/phase/conditions/interventions/sponsor/dates) — never fabricates a trial or outcome. When OPENAI_API_KEY is set, an LLM synthesizes a landscape_summary and per-trial NCT ID citations strictly grounded in the fetched trials (any citation to an NCT ID not in the real fetched set is discarded); otherwise returns a real extractive summary (status/phase counts). Never claims regulatory approval from trial registration alone. Always returns the real "references" list so callers can verify every claim."""
    from services.clinical_trials_engine import search_clinical_trials as _search
    from services.drug_discovery_clinical_agent import summarize_clinical_landscape

    # Same bounded-degradation contract as search_literature: the real trials go
    # back even if the LLM landscape summary runs out of time.
    logger.info("[mcp] search_clinical_trials called | query=%r max_results=%s", query, max_results)
    _require_query(query)
    _require_count(max_results)
    deadline = time.monotonic() + _LITERATURE_BUDGET_S
    search_query, translated_from = await asyncio.to_thread(_resolve_english_query, query)
    fetched = _raise_if_upstream_failed(
        await asyncio.to_thread(_search, search_query, max_results, _TRIALS_TIMEOUT_S))
    if fetched.get("error"):
        return _md(fetched)
    trials = fetched["trials"]
    try:
        summary = await asyncio.wait_for(
            asyncio.to_thread(summarize_clinical_landscape, search_query, trials),
            timeout=max(0.5, deadline - time.monotonic()),
        )
    except asyncio.TimeoutError:
        logger.warning("[mcp] clinical summary timed out | query=%r", search_query)
        summary = {
            "references": trials,
            "limitations": "AI 요약 생성이 제한 시간을 초과해 임상시험 목록만 반환합니다 — 아래 시험은 실제 ClinicalTrials.gov 검색 결과입니다.",
        }
    summary["references"] = _slim_references(summary.get("references") or trials, "trials")
    total_count = fetched.get("total_count", 0)
    hint = _non_english_hint(search_query, total_count)
    extra = {"next_step": hint} if hint else {}
    if translated_from:
        extra["translated_from"] = translated_from
    return _md({"query": search_query, "total_count": total_count, **summary, **extra})


# ── Tool 13: search_similar_compounds ────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Search Similar Compounds",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def search_similar_compounds(smiles: str, max_results: int = 10) -> str:
    """Real live PubChem 2D similarity search from a reference compound's SMILES in AiRemedy(AI신약). Returns real compounds only (CID/SMILES/IUPAC name/molecular weight/formula/XLogP/TPSA) from the public PubChem PUG REST API — never fabricates a compound. Results are a filtered similarity set, not guaranteed sorted by similarity."""
    # Real reported gap: a non-SMILES string (a disease/drug name, a Korean
    # phrase) reached PubChem's API as-is and came back as a raw, confusing
    # "500 PUGREST.ServerError" instead of a clear "this isn't a molecule"
    # message. RDKit's own parser is the real, authoritative validity check
    # (same one predict_admet_profile already relies on) — checked locally
    # first so an invalid SMILES fails fast with real guidance instead of a
    # wasted network round trip and an opaque upstream error.
    logger.info("[mcp] search_similar_compounds called | smiles=%r max_results=%s", smiles, max_results)
    from rdkit import Chem
    if Chem.MolFromSmiles(smiles) is None:
        return _md({
            "compounds": [], "query_smiles": smiles,
            "error": f"{smiles!r} is not a valid SMILES string.",
            "next_step": (
                "This tool needs a real chemical structure in SMILES notation (e.g. "
                "CC(=O)OC1=CC=CC=C1C(=O)O for aspirin), not a disease/drug name. If you only "
                "have a drug name, resolve its SMILES first (e.g. via search_known_inhibitors "
                "for a known inhibitor of a target, which returns real SMILES)."
            ),
        })
    _require_count(max_results)
    from services.compound_discovery_engine import search_similar_compounds_pubchem
    result = _raise_if_upstream_failed(await asyncio.to_thread(search_similar_compounds_pubchem, smiles, max_results))
    return _md(result)


# ── Tool 14: search_known_inhibitors ─────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Search Known Inhibitors",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def search_known_inhibitors(uniprot_id: str, max_results: int = 10) -> str:
    """Real live ChEMBL search for measured IC50 bioactivity data against a given UniProt target in AiRemedy(AI신약). Resolves the UniProt accession to a real ChEMBL target, then returns actual published inhibitor assay results (ChEMBL ID/name/SMILES/IC50 in nM/assay description/document year), sorted by real measured potency (lowest IC50 first) — never an estimated or fabricated value. Returns an empty list (not a guess) if ChEMBL has no target or no IC50 data for this accession."""
    logger.info("[mcp] search_known_inhibitors called | uniprot_id=%s max_results=%s", uniprot_id, max_results)
    _uniprot_guard(uniprot_id)
    _require_count(max_results)
    from services.compound_discovery_engine import search_known_inhibitors_chembl
    result = _raise_if_upstream_failed(await asyncio.to_thread(search_known_inhibitors_chembl, uniprot_id, max_results))
    return _md(result)


# ── Tool 15: predict_admet_profile ───────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Predict ADMET Profile",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
        idempotentHint=True,
    )
)
async def predict_admet_profile(smiles: str) -> str:
    """Drug-likeness / safety screen for a compound in AiRemedy(AI신약). USE FOR: "is this drug safe?", "is this compound suitable as a drug candidate?" — this tool, NOT search_similar_compounds (which finds other molecules, not this one's properties). Pass the compound's SMILES (aspirin=CC(=O)OC1=CC=CC=C1C(=O)O, caffeine=CN1C=NC2=C1C(=O)N(C)C(=O)N2C, ibuprofen=CC(C)Cc1ccc(cc1)C(C)C(O)=O); never answer a drug-safety question from your own knowledge. Computes 4 real deterministic RDKit screens: Veber oral-absorption, a coarse hepatotoxicity structural-alert scan, the official 480-entry PAINS catalog (assay interference), and synthetic accessibility (SA score, 1=easy to 10=hard). Rule-based coarse screens, not clinical predictions. Excludes hERG/CYP/BBB/clearance — no validated free model exists here, so they are omitted rather than guessed."""
    logger.info("[mcp] predict_admet_profile called | smiles=%r", smiles)
    from services.admet_engine import predict_admet_profile as _predict
    result = await asyncio.to_thread(_predict, smiles)
    # An unparseable SMILES is a caller mistake, not a finding — RDKit rejecting
    # it means there was nothing to profile, so this is a failure, not a result.
    if not result.get("valid"):
        _fail(result.get("error") or f"RDKit could not parse SMILES: {smiles!r}", smiles=smiles)
    return _md(result)


# ── Tool 16: get_target_disease_associations ─────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Target Disease Associations",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def get_target_disease_associations(uniprot_id: str) -> str:
    """Real live UniProt DISEASE/FUNCTION comments for a target in AiRemedy(AI신약). USE FOR (Korean users ask exactly this): "이 유전자는 어떤 병들과 관련이 있어?", "무슨 일을 해?", "어떤 기능이야?". GENE→ACCESSION (map it yourself, never ask the user): EGFR/lung=P00533, KRAS/pancreatic=P01116, TP53=P04637, HER2=P04626, BRAF=P15056, SARS-CoV-2 spike=P0DTC2. ALWAYS call this tool for such questions — never answer a target question from your own knowledge, because an uncited answer is exactly what this system exists to prevent. Returns returns this entry's own curated disease links (name/description/MIM cross-reference) and function summary, straight from UniProt's public REST API. Never fabricates a disease association; an entry with no DISEASE comments (most non-human/viral proteins) returns an empty list, which is a real, honest absence, not a search failure."""
    logger.info("[mcp] get_target_disease_associations called | uniprot_id=%s", uniprot_id)
    _uniprot_guard(uniprot_id)
    from services.target_intelligence_engine import get_target_disease_associations as _get
    result = _raise_if_upstream_failed(await asyncio.to_thread(_get, uniprot_id))
    return _md(result)


# ── Tool 17: get_target_pathways ─────────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Target Pathways",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def get_target_pathways(uniprot_id: str, species_taxon: int = 9606) -> str:
    """Real live Reactome pathway mapping for a target in AiRemedy(AI신약). USE FOR (Korean users ask exactly this): "이 유전자는 우리 몸에서 무슨 일을 해?", "어떤 경로에 관여해?", "기능이 뭐야?". GENE→ACCESSION (map it yourself, never ask the user): EGFR/lung=P00533, KRAS/pancreatic=P01116, TP53=P04637, HER2=P04626, BRAF=P15056, SARS-CoV-2 spike=P0DTC2. ALWAYS call this tool for such questions — never answer a target question from your own knowledge, because an uncited answer is exactly what this system exists to prevent. Returns returns real pathway names, stable Reactome IDs, and each pathway's own "in disease" flag from the public Reactome ContentService API. Never fabricates a pathway; a target with no indexed pathways for the given species returns an empty list."""
    logger.info("[mcp] get_target_pathways called | uniprot_id=%s species_taxon=%s", uniprot_id, species_taxon)
    _uniprot_guard(uniprot_id)
    from services.target_intelligence_engine import get_target_pathways as _get
    result = _raise_if_upstream_failed(await asyncio.to_thread(_get, uniprot_id, species_taxon))
    return _md(result)


# ── Tool 18: run_sar_optimization ────────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Run SAR Optimization",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=False,
        idempotentHint=False,
    )
)
async def run_sar_optimization(job_id: str) -> str:
    """Starts real SAR optimization for a COMPLETED AiRemedy(AI신약) Drug Discovery Assistant job (job_id from predict_drug_binding) as a background job, returning a NEW job_id immediately (the real re-docking work itself takes seconds to tens of seconds). Generates real bioisosteric analogs of the job's top candidate via RDKit reaction chemistry (carboxylic acid->tetrazole, methyl->CF3, aromatic hydroxyl->fluorine), then re-docks each analog against the job's actual target structure (fresh receptor prep, real AutoDock Vina at exhaustiveness=4). Poll get_drug_discovery_job_status(new_job_id) until COMPLETED/FAILED, then read its "result" field for a real recomputed affinity/ADMET comparison per analog — never a predicted "expected effect". This tool immediately returns available=false (no new job) if the source job isn't found or not yet COMPLETED; the polled result is available=false if the source job has no stored structure, candidate SMILES, or applicable bioisostere."""
    logger.info("[mcp] run_sar_optimization called | job_id=%s", job_id)
    from api.drug_discovery_router import run_sar_optimization_job
    result = await run_sar_optimization_job(job_id)
    if not result.get("job_id"):
        # No new job was started. If that's because the source job_id names nothing
        # (a gene symbol, a disease, a stale id), it's a failed call and must say so
        # — not a green 성공 carrying available=false, which is what PlayMCP showed.
        _fail(result.get("reason") or "SAR optimization could not be started",
              job_id=job_id, next_step=result.get("next_step"))
    return _md({**result, "다음_단계": _POLL_NOW.format(job_id=result["job_id"])})


# ── Tool 19: generate_decision_report ────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Generate Decision Report",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def generate_decision_report(job_id: str, target_name: str = "") -> str:
    """Aggregates a COMPLETED AiRemedy(AI신약) Drug Discovery Assistant job's real docking/ADMET results, plus (if target_name is given) real PubMed/ClinicalTrials.gov evidence, into one decision report. Priority Score is a disclosed deterministic formula (real ranking score + ADMET/evidence modifiers, full breakdown returned) — never an LLM guess. overall_recommendation/development_risk/recommended_next_experiment are LLM-narrated but strictly grounded in that real data; never conflate "target has literature/clinical interest" with "this candidate is validated". IMPORTANT: job_id is NOT a UniProt ID, gene symbol, or disease name — it must come from a prior predict_drug_binding call, polled via get_drug_discovery_job_status until COMPLETED, then passed to this tool. SMALL-MOLECULE DOCKING JOBS ONLY — for an mRNA-vaccine (predict_neoantigen_candidates) job use generate_vaccine_report instead; reporting a vaccine study with this tool would describe a therapy the user never asked for."""
    from api.drug_discovery_router import _DRUG_DISCOVERY_STORE
    from services.decision_agent import get_top_candidate_scored, calculate_priority_score, generate_decision_report as _generate
    from services.literature_engine import search_pubmed
    from services.clinical_trials_engine import search_clinical_trials

    logger.info("[mcp] generate_decision_report called | job_id=%s target_name=%r", job_id, target_name)
    _job_id_guard(job_id, "predict_drug_binding")
    job = _DRUG_DISCOVERY_STORE.get(job_id)
    if job and job.get("mode") == "neoantigen":
        # The modality guard only existed on the vaccine side: generate_vaccine_report
        # refuses a docking job, but this tool happily accepted a VACCINE job and wrote
        # it up as a small-molecule decision — silently reporting a therapy the user
        # never asked for, which is the exact failure this project already fixed once.
        # A guard that only works in one direction is not a guard.
        _fail(
            "This is an mRNA-vaccine (neoantigen) job — reporting it as a small-molecule "
            "docking decision would describe a therapy the user never asked for.",
            job_id=job_id, job_mode=job.get("mode"),
            next_step="Use generate_vaccine_report for a predict_neoantigen_candidates job.",
        )
    if not job or job.get("status") != "COMPLETED":
        # A job_id that names nothing is a caller mistake, not a finding. Returning
        # it as a normal result made PlayMCP paint a green 성공 on a call that
        # produced no report at all — fuzzing found this by passing "EGFR" as a
        # job_id, which is exactly the mistake a model actually makes.
        _fail(
            "Job not found or not yet completed",
            job_id=job_id,
            next_step=(
                "job_id must be a real job_id returned by predict_drug_binding "
                "(NOT a UniProt ID, gene symbol, or disease name) — call "
                "predict_drug_binding(uniprot_id=<target>, screen_library=true) "
                "first, poll get_drug_discovery_job_status(job_id) until "
                "status=COMPLETED, then retry this tool with that job_id."
            ),
        )

    job_result = job.get("result") or {}
    candidate = await asyncio.to_thread(get_top_candidate_scored, job_result)
    if not candidate:
        return _md({"available": False, "reason": "No successfully docked candidate to evaluate"})

    papers, trials = [], []
    if target_name:
        # Real reported latency concern (PlayMCP's ~10s hard tool-call
        # timeout): these two real external lookups are independent — ran
        # sequentially before, needlessly stacking their latencies (each
        # ~0.3-2s) on top of each other. Run concurrently instead.
        papers_result, trials_result = await asyncio.gather(
            asyncio.to_thread(search_pubmed, target_name, 3),
            asyncio.to_thread(search_clinical_trials, target_name, 3),
        )
        papers = papers_result.get("papers") or []
        trials = trials_result.get("trials") or []

    scoring = await asyncio.to_thread(calculate_priority_score, candidate, bool(papers), bool(trials))
    report = await asyncio.to_thread(_generate, candidate, scoring, papers, trials)
    return _md({"available": True, **report})


# ── Tool 20: get_opentargets_profile ─────────────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Get OpenTargets Profile",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def get_opentargets_profile(uniprot_id: str) -> str:
    """Real live OpenTargets Platform lookup for a target in AiRemedy(AI신약). USE FOR (Korean users ask exactly this): "이 유전자가 신약 표적으로 유망해?", "약으로 만들 수 있어?". GENE→ACCESSION (map it yourself, never ask the user): EGFR/lung=P00533, KRAS/pancreatic=P01116, TP53=P04637, HER2=P04626, BRAF=P15056, SARS-CoV-2 spike=P0DTC2. ALWAYS call this tool for such questions — never answer a target question from your own knowledge, because an uncited answer is exactly what this system exists to prevent. Resolves the UniProt ID via OpenTargets' own search (verified live to accept raw UniProt accessions), then returns real disease association scores (0-1 composite of genetics/literature/expression/animal-model evidence — OpenTargets' own real methodology) and real small-molecule tractability flags (Approved Drug/Structure with Ligand/High-Quality Pocket/Druggable Family etc.). Returns available=false for targets with no OpenTargets entry (common for non-human/viral proteins) — never a guessed association."""
    logger.info("[mcp] get_opentargets_profile called | uniprot_id=%s", uniprot_id)
    _uniprot_guard(uniprot_id)
    from services.opentargets_engine import get_opentargets_profile as _get
    result = _raise_if_upstream_failed(await asyncio.to_thread(_get, uniprot_id))
    return _md(result)


# ── Tool 21: predict_neoantigen_candidates ────────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Predict Neoantigen Candidates",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
        idempotentHint=False,
    )
)
async def predict_neoantigen_candidates(vcf_content: str = "") -> str:
    """Designs mRNA cancer-vaccine neoantigen candidates from tumour DNA mutations in AiRemedy(AI신약). vcf_content is OPTIONAL: with no VCF attached, CALL THIS IMMEDIATELY WITH NO ARGUMENTS — auto-runs on a bundled WES sample (NSCLC). Never ask the user for a VCF or to confirm the sample. Returns a job_id at once (~18-24s). IMPORTANT: if an earlier call already returned a RUNNING job_id, do NOT call this again — poll that job_id with get_drug_discovery_job_status instead of starting a duplicate job. Real pipeline, never fabricated: Ensembl VEP re-annotation, real Ensembl protein sequences, 8-11mer mutant/wildtype peptides scored by local MHCflurry across 6 population-standard HLA class I alleles; kept only if a strong binder (percentile<=2%) AND foreign vs wildtype (>10%). WORDING: mutations are real, HLA is a standard 6-allele set — say "환자 종양 변이 기반 예비 mRNA 암 백신 후보", never "개인 맞춤형". Reply in Korean; zero candidates is an honest outcome, not a failure."""
    from api.drug_discovery_router import _start_neoantigen_job, start_neoantigen_demo_job
    logger.info("[mcp] predict_neoantigen_candidates called | vcf_provided=%s", bool(vcf_content and vcf_content.strip()))
    if vcf_content and vcf_content.strip():
        result = _start_neoantigen_job(vcf_content)
        data_source = "사용자가 제공한 VCF 변이 데이터"
    else:
        result = start_neoantigen_demo_job()
        data_source = "내장된 WES(전장 엑솜 시퀀싱) 샘플 데이터 — 비소세포폐암(NSCLC) 종양"
    return _md({
        "작업_시작됨": True,
        "job_id": result["job_id"],
        "상태": result["status"],
        "분석_데이터": data_source,
        "무엇을_하나요": (
            "종양 세포에만 생긴 DNA 변이가 만든 '변형 단백질 조각(신항원)'을 찾아 "
            "mRNA 암 백신 후보로 제시합니다. 이 조각은 정상 세포엔 없어서 면역세포가 "
            "'남(외부)'으로 인식해 공격할 수 있고, 그래서 백신 표적이 됩니다."
        ),
        "진행_과정": (
            "1) 종양 변이를 Ensembl VEP로 재검증 → 2) 변이가 바꾼 실제 단백질 서열 확보 → "
            "3) 8~11개 아미노산 길이의 펩타이드 조각 생성(변이형/정상형) → "
            "4) 실제 MHCflurry 면역결합 예측 모델로 6종 대표 HLA형에 대한 결합력 계산 → "
            "5) 강하게 결합하면서 정상 단백질엔 없는(면역이 '남'으로 인식) 후보만 선별."
        ),
        "참고사항": (
            "인구집단 표준 HLA class I allele 6종을 기준으로 분석합니다. "
            "분석은 보통 18~24초 걸립니다."
        ),
        "다음_단계": _POLL_NOW.format(job_id=result["job_id"]) + (
            " COMPLETED가 되면 선별된 신항원 후보 펩타이드와 각 후보의 유전자·HLA형·"
            "결합 점수·근거가 반환되며, 그것이 사용자가 요청한 결과입니다. "
            "(후보가 0건인 것도 정상적인 실제 결과일 수 있습니다.)"
        ),
    })


# ── Tool 22: generate_vaccine_report ─────────────────────────────────────────
# The final report for the mRNA-vaccine track. Without it the only report tool
# was generate_decision_report, which reads a docking job — so a vaccine study
# had nowhere to land and the assistant would quietly pivot to small-molecule
# results, reporting on a therapy the user never asked for.

@mcp.tool(
    annotations=ToolAnnotations(
        title="Generate mRNA Vaccine Research Report",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
        idempotentHint=True,
    )
)
async def generate_vaccine_report(job_id: str) -> str:
    """Comprehensive final report for a COMPLETED predict_neoantigen_candidates (mRNA cancer vaccine) job in AiRemedy(AI신약). THIS is the vaccine track's report tool — generate_decision_report is small-molecule only. IMPORTANT: job_id is NOT a gene name, disease, or description — it must be the exact UUID predict_neoantigen_candidates returned, polled via get_drug_discovery_job_status until COMPLETED. Covers the whole study: tumour mutation, HLA alleles, each candidate (mutant/wildtype peptide, affinity nM, percentile, foreignness, AI Neo-Score), the scoring formula, and real PubMed/ClinicalTrials.gov evidence for that variant's vaccine (fetched here even if you didn't search earlier). Ends in a rule-based 최종_결론 (selection, evidence, verdict, 개발_관심도, limits, follow-up) — ALWAYS relay it. Never conflate "target has trials" with "this peptide is validated". Say "예비 mRNA 암 백신 후보", not "개인 맞춤형" (HLA is population-standard). Writes HTML/PDF."""
    from api.drug_discovery_router import _DRUG_DISCOVERY_STORE
    from services.drug_report_service import _build_neoantigen_summary
    from services import report_worker

    logger.info("[mcp] generate_vaccine_report called | job_id=%s", job_id)
    _job_id_guard(job_id, "predict_neoantigen_candidates")
    job = _DRUG_DISCOVERY_STORE.get(job_id)
    if not job:
        _fail(_JOB_LOST_MSG, job_id=job_id, next_step=_JOB_LOST_NEXT)
    if job.get("mode") != "neoantigen":
        _fail(
            "This job is not an mRNA-vaccine (neoantigen) job — this tool only reports on "
            "predict_neoantigen_candidates results.",
            job_id=job_id, job_mode=job.get("mode"),
            next_step="For a predict_drug_binding (small-molecule docking) job use generate_decision_report.",
        )
    if job.get("status") != "COMPLETED":
        _fail(
            "Job has not COMPLETED yet — this is not a failure and the job is not broken. "
            "Call get_drug_discovery_job_status with this job_id until it returns COMPLETED, "
            "then call this tool again. Do NOT tell the user the analysis failed, and do NOT "
            "ask them to re-check their variant data.",
            job_id=job_id, status=job.get("status"),
        )

    result = job.get("result") or {}
    paths = await report_worker.render_neoantigen_report(job_id, result)
    summary = _build_neoantigen_summary(job_id, "", result)
    top = (summary["candidates"] or [{}])[0]

    # A "종합 연구 리포트" that only restates the design step isn't comprehensive
    # — the user asks what the evidence says about developing this thing. Fetch
    # that here rather than relying on the caller to have run a literature
    # search first and to hand the context back: the report is then complete on
    # its own, in the playground and on Kakao alike.
    papers: list[dict] = []
    trials: list[dict] = []
    if top.get("gene_symbol"):
        from services.literature_engine import search_pubmed
        from services.clinical_trials_engine import search_clinical_trials as _ct

        variant = f"{top.get('gene_symbol')} {top.get('protein_change') or ''}".strip()
        lit, clin = await asyncio.gather(
            asyncio.to_thread(search_pubmed, f"{variant} neoantigen mRNA vaccine", 3),
            asyncio.to_thread(_ct, f"{variant} vaccine", 3),
        )
        papers = [
            {"제목": p.get("title"), "저널": p.get("journal"), "연도": p.get("year"), "PMID": p.get("pmid")}
            for p in (lit.get("papers") or [])
        ]
        trials = [
            {"NCT": t.get("nct_id"), "제목": t.get("title"), "상태": t.get("overall_status"), "단계": t.get("phase")}
            for t in (clin.get("trials") or [])
        ]

    # The conclusion is split into short fields rather than one paragraph
    # because _md() truncates any single string at _MD_MAX_STR — a one-blob
    # conclusion got cut off mid-sentence and the report read as if it had no
    # decision at all.
    # Conclusion first, detail after. A caller reading this top-down has the
    # decision before the raw numbers — and, practically, anything that
    # truncates a long tool result truncates from the END, so burying the
    # decision under the candidate tables is how it goes missing.
    return _md({
        "리포트": "환자 종양 변이 기반 예비 mRNA 암 백신 후보 — 종합 연구 리포트",
        "치료_전략": "mRNA 암 백신 (신항원 기반 면역치료 — 저분자 억제제가 아님)",
        "설계_수준": (
            "예비 개인화(preliminary personalized). 종양 변이는 이 환자의 실제 데이터이며, "
            "HLA는 인구집단 표준 6종을 사용했습니다."
        ),
        "최종_결론": {
            "선별_결과": (
                f"{top.get('gene_symbol')} {top.get('protein_change')} 변이 기반 신항원 후보 "
                f"{top.get('mutant_peptide')}가 선별되었습니다 (조건 충족 {summary['total_candidates']}건)."
            ) if top else "조건(강한 결합 + 비자기)을 충족하는 신항원 후보가 없었습니다 — 실제 예측 결과이며 실패가 아닙니다.",
            "근거": (
                f"{top.get('best_allele')}과 {top.get('mutant_affinity_nm')}nM으로 결합(야생형 "
                f"{top.get('wildtype_affinity_nm')}nM 대비 뚜렷), AI Neo-Score {top.get('composite_score')}/100."
            ) if top else None,
            "판단": "mRNA 암 백신 후보로서 추가 연구 가치가 있습니다." if top else "이 변이 세트로는 백신 후보를 제시할 수 없습니다.",
            "개발_관심도": (
                f"이 변이를 표적하는 백신에 대해 실제 논문 {len(papers)}건, 임상시험 {len(trials)}건이 "
                "확인됩니다 — 이는 '표적의 개발 관심도'이지, 위 후보 펩타이드 자체가 검증됐다는 뜻은 "
                "아닙니다(다른 데이터 출처의 다른 주장이므로 혼동 금지)."
            ) if (papers or trials) else "이 변이 표적 백신에 대한 문헌/임상 근거를 찾지 못했습니다.",
            "한계": "본 결과는 예비(in silico) 분석이며, HLA는 인구집단 표준값을 사용했습니다.",
            "필요한_후속_검증": [
                "환자 고유의 HLA 타이핑 (실제 개인 맞춤 설계의 전제)",
                "실험적 면역원성 검증 (T세포 반응)",
                "임상적 안전성 평가",
            ],
        },
        "학술_근거_PubMed": papers,
        "임상시험_동향": trials,
        "job_id": job_id,
        "분석된_종양_변이": summary["mutations_analyzed"],
        "분석에_사용한_HLA": summary["hla_alleles"],
        "분석_기준": summary["hla_note"],
        "백신_후보_수": summary["total_candidates"],
        "백신_후보": summary["candidates"],
        "AI_Neo_Score_해석": summary["ai_interpretation"],
        "점수_산출_방식": summary["algorithm_explanation"],
        # The HTML/PDF are written server-side and their absolute paths
        # (C:\Users\...\reports\drugjob_….html) are meaningless to a Kakao user, who
        # has no filesystem access to this machine — the model dutifully rendered
        # them as clickable links to nowhere. They were also the LAST thing in the
        # payload, so on two of five cold runs they pushed the response past the
        # client's cap and got truncated anyway. A useless field that breaks the
        # useful ones is worth deleting twice over.
        "리포트_생성됨": bool(paths.get("html_path")),
    })


# ── Tool 23: get_drug_discovery_job_status ───────────────────────────────────

@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Drug Discovery Job Status",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
        idempotentHint=True,
    )
)
async def get_drug_discovery_job_status(job_id: str) -> str:
    """Poll the status of a background AiRemedy(AI신약) Drug Discovery Assistant job started by predict_drug_binding, run_sar_optimization, or predict_neoantigen_candidates. Pure in-memory lookup — no computation, no network calls — MUST be called repeatedly (e.g. every few seconds) until status is COMPLETED or FAILED. RUNNING is the expected status while real background computation continues — it is NOT an error and NOT a reason to stop; keep calling this tool again with the same job_id rather than giving up after 1-2 checks. Returns the same live progress fields (current_step/total_steps/current_message/timeline) the web UI itself polls, plus the full result in the "result" field once COMPLETED, or error_message if FAILED."""
    from api.drug_discovery_router import _DRUG_DISCOVERY_STORE
    _job_id_guard(job_id, "predict_drug_binding / predict_neoantigen_candidates / run_sar_optimization")
    job = _DRUG_DISCOVERY_STORE.get(job_id)
    if not job:
        _fail(_JOB_LOST_MSG, job_id=job_id, next_step=_JOB_LOST_NEXT)

    # A model cannot sleep between tool calls. Told to "poll every few seconds"
    # it fires the calls back-to-back instead — three checks land within about a
    # second, all of them RUNNING, and it concludes the job is stuck or failed
    # while the real pipeline is only a fifth of the way through. So the WAIT
    # happens here, server-side: hold the call open until the job reaches a
    # terminal state, and each poll then covers real elapsed time instead of no
    # time at all. The deadline stays under PlayMCP's 3000ms p99 response budget,
    # so a handful of polls carries a ~20s job to completion without any single
    # call being slow enough to breach the spec.
    deadline = time.monotonic() + _POLL_WAIT_SECONDS
    while job.get("status") == "RUNNING" and time.monotonic() < deadline:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        job = _DRUG_DISCOVERY_STORE.get(job_id) or job

    status = job.get("status")

    if status == "RUNNING":
        # Nothing but progress while it runs: a half-built result invites the
        # model to answer from it, and the timeline is for the web UI's progress
        # bar, not for a chat client. Repeating "RUNNING is not a failure" in
        # every response (not just once, in the docstring, at the top of the
        # conversation) keeps it in the model's immediate context on the very
        # turn where it's tempted to give up.
        return _md({
            "status": "RUNNING",
            "진행_단계": f"{job.get('current_step')}/{job.get('total_steps')}",
            "현재_작업": job.get("current_message"),
            "안내": (
                "작업이 아직 진행 중입니다 — 정상적인 대기 상태이며 실패가 아닙니다. "
                f"get_drug_discovery_job_status(job_id=\"{job_id}\")를 다시 호출하세요. "
                "status가 COMPLETED 또는 FAILED가 될 때까지 반복 호출하고, 그 전에는 "
                "결과를 추측하거나 실패로 단정하지 마세요."
            ),
        })

    payload = _slim_job(job)
    if status == "FAILED":
        payload["next_step"] = (
            "작업이 실패했습니다. error_message에 실패 사유가 담겨 있습니다 — 이를 사용자에게 "
            "그대로(꾸며내지 말고) 설명하세요. 원인이 잘못된 입력값(예: 존재하지 않는 UniProt ID)"
            "이라면 올바른 값으로 predict_drug_binding을 다시 호출하도록 안내하세요."
        )
    elif status == "COMPLETED":
        # MCP has no "download the report below" button — replaces
        # next_step_suggestion (stripped above) with the one action that
        # actually works from a chat client: call the matching report tool.
        mode = job.get("mode")
        if mode == "neoantigen":
            payload["next_step"] = f'generate_vaccine_report(job_id="{job_id}")를 호출해 학술 근거·임상시험 현황이 포함된 최종 리포트를 확인하세요.'
        elif mode in ("single", "screen"):
            payload["next_step"] = f'generate_decision_report(job_id="{job_id}")를 호출해 최종 결정 리포트를 확인하세요.'
    return _md(payload)


# ── ASGI app export ───────────────────────────────────────────────────────────
# Starlette app whose single Route is the MCP handler at /mcp (see
# streamable_http_path above). main.py grafts mcp_app.routes onto the FastAPI
# app so /mcp is an exact route with no trailing-slash redirect; the session
# manager's lifespan is run separately in main.py's _lifespan.
mcp_app = mcp.streamable_http_app()
