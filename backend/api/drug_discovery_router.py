"""
Drug Discovery Assistant Router
/api/drug-discovery/*

Flow:
  POST /design           -> start pipeline job (async, returns job_id) — named pathogen/protein target ("Track A")
  POST /design-from-vcf  -> start pipeline job from a VCF file/text ("Track B", somatic-variant cancer targets)
  GET  /status/{id}      -> poll job status + result

Mirrors api/somatic_router.py's async job-store + background-task pattern.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import logging
import os
import re
import time
import uuid
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from services.drug_discovery_pipeline import run_drug_discovery_pipeline, run_drug_discovery_from_vcf
from services.neoantigen_pipeline import run_neoantigen_pipeline
from services.sar_optimization_service import run_sar_optimization as _run_sar_optimization_service
from services.drug_discovery_intent import parse_drug_discovery_intent, _is_cancer_topic
from services.drug_discovery_chat import (
    classify_drug_discovery_action,
    answer_completed_job_question,
    answer_running_job_question,
    _clean_external_search_query,
    is_neoantigen_query,
    is_neoantigen_demo_query,
    is_neoantigen_question,
    answer_neoantigen_question,
    is_general_explain_query,
    answer_general_explain_question,
    is_identity_query,
    answer_identity_question,
    is_literature_query,
    answer_literature_question,
    is_clinical_query,
    answer_clinical_question,
    is_compound_discovery_query,
    answer_compound_discovery_question,
    is_target_intelligence_query,
    answer_target_intelligence_question,
    is_sar_optimization_query,
    answer_sar_optimization_question,
    is_decision_report_query,
    answer_decision_report_question,
)
from services.fasta_engine import parse_fasta

_REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

logger = logging.getLogger("openbioshield.drug_discovery")
router = APIRouter(prefix="/api/drug-discovery", tags=["Drug Discovery Assistant"])

_DRUG_DISCOVERY_STORE: dict[str, dict] = {}

# The store had no eviction at all — every job ever run stayed in memory for the
# life of the process. That is not a small leak: a docking result carries the
# target's entire PDB text (1.6 MB for the SARS-CoV-2 spike) plus a docked pose per
# candidate, so a few hundred jobs is hundreds of megabytes of structures nobody
# will ever look at again. The MCP process is long-lived on Kakao's side, so it
# would simply grow until it died.
#
# Oldest-first eviction, and only ever of FINISHED jobs — a RUNNING job is still
# being polled and must never be evicted out from under its own status call. The
# cap is generous: the point is to bound the process, not to expire results a user
# might still be reading (Kakao's own conversation is much shorter-lived than this).
_MAX_STORED_JOBS = 50


def _evict_finished_jobs() -> None:
    """Called just BEFORE a new job is inserted, so it must make room for that job:
    trim to _MAX_STORED_JOBS - 1, leaving the store at exactly the cap once the
    caller inserts. Trimming to the cap itself settles one job above it."""
    overflow = len(_DRUG_DISCOVERY_STORE) - (_MAX_STORED_JOBS - 1)
    if overflow <= 0:
        return
    finished = [jid for jid, job in _DRUG_DISCOVERY_STORE.items()
                if job.get("status") in ("COMPLETED", "FAILED")]
    # dict preserves insertion order, so `finished` is already oldest-first.
    for jid in finished[:overflow]:
        _DRUG_DISCOVERY_STORE.pop(jid, None)
        logger.info("[drug_discovery] evicted finished job from store | job=%s", jid)

# Real asyncio Task handles, keyed by job_id — needed to actually cancel a
# running job (not just hide it in the UI). Populated at every job-creation
# call site (/design, /design-from-vcf, /design-from-file, /converse) right
# after asyncio.create_task(...), and popped by each background task
# function's own finally block once it finishes (success, failure, timeout,
# or cancellation).
_RUNNING_TASKS: dict[str, asyncio.Task] = {}

_STOP_PHRASES = ("중지", "멈춰", "멈춤", "그만", "stop", "cancel")


def stop_job(job_id: str, reason: str = "사용자 요청으로 중지되었습니다.") -> bool:
    """
    Real cancellation — calls Task.cancel() on the actual running asyncio
    task, not just a UI-side "pretend it's stopped". Sets the store status
    BEFORE cancelling so the background task's own CancelledError handler
    sees it's already been marked CANCELLED and doesn't overwrite it as a
    generic failure. Returns False if the job isn't found or already
    finished (nothing to stop).
    """
    task = _RUNNING_TASKS.get(job_id)
    store = _DRUG_DISCOVERY_STORE.get(job_id)
    if not task or task.done() or not store or store.get("status") != "RUNNING":
        return False
    store.update({"status": "CANCELLED", "error_message": reason})
    task.cancel()
    logger.info("[drug_discovery] job STOP requested | job=%s reason=%s", job_id, reason)
    return True


class ConversationState(str, Enum):
    """
    Explicit conversation FSM for the /converse session, replacing what was
    previously just inferred from scattered fields (active_job_id being set
    or not, job status, etc.). Simplified from a 7-state design doc to what
    actually has distinct, observable meaning for this two-track (pathogen /
    somatic-variant) assistant — REFLECTION isn't its own externally visible
    state since retries happen inside a single RUNNING pipeline call,
    already recorded in that job's own decision_log; FAILED is a real,
    necessary terminal state a spec sketch would omit but a real system
    can't.
    """
    START                      = "START"
    WAITING_FOR_REQUIRED_INPUT = "WAITING_FOR_REQUIRED_INPUT"
    DISEASE_SELECTED           = "DISEASE_SELECTED"
    WORKFLOW_PLANNED           = "WORKFLOW_PLANNED"
    RUNNING                    = "RUNNING"
    REPORT_READY               = "REPORT_READY"
    FAILED                     = "FAILED"


def _new_chat_session() -> dict:
    return {
        "messages": [], "active_job_id": None, "known_slots": {}, "processing": False,
        "state": ConversationState.START.value, "state_history": [],
    }


def _transition(session: dict, new_state: ConversationState, session_id: str = "") -> None:
    old_state = session.get("state")
    if old_state == new_state.value:
        return
    session["state"] = new_state.value
    session["state_history"].append({
        "from": old_state, "to": new_state.value, "at": datetime.utcnow().isoformat() + "Z",
    })
    logger.info("[drug_discovery_fsm] session=%s %s -> %s", session_id, old_state, new_state.value)

# Separate, independent session store for the conversational entry point —
# shares no state with agent_router.py's _CHAT_SESSIONS (isolation
# requirement for this feature).
_DRUG_DISCOVERY_CHAT_SESSIONS: dict[str, dict] = {}

# Screening docks every candidate in the curated library (up to 3 retry
# iterations each) instead of a single ligand, so it needs a much longer
# budget than single-ligand docking.
_TIMEOUT_SECONDS_SINGLE = 120
# Large targets (e.g. full-length Spike, ~1273 residues) blind-dock against a
# much bigger search box than a small protein — measured ~45s/candidate at
# exhaustiveness=8 (see drug_discovery_agent.py's Planner docstring), so a
# ~20-candidate screen needs real headroom even with the Planner's adaptive
# "fast" strategy for large targets.
_TIMEOUT_SECONDS_SCREEN = 1800
# No docking involved (MHCflurry inference + a handful of real network
# calls) — measured ~24s end-to-end against the sample VCF/BAM, so 5
# minutes is generous headroom, not a tight fit.
_TIMEOUT_SECONDS_NEOANTIGEN = 300
# Receptor re-prep (<=60s, one retry -> <=120s worst case) + up to 3 analogs
# re-docked at exhaustiveness=4 (<=120s each) with concurrency=2 (two
# batches worst case -> <=240s) = <=360s worst case if every subprocess call
# runs its full individual timeout without actually hanging. 420s gives
# headroom above that calculated ceiling.
_TIMEOUT_SECONDS_SAR = 420


class DrugDiscoveryRequest(BaseModel):
    uniprot_id:      str = ""
    target_sequence: str = ""
    ligand_smiles:   str = ""
    screen_library:  bool = False
    goal_text:       str = ""
    max_candidates:  int = 0  # 0 = screen the full curated library; screen mode only


class ConverseRequest(BaseModel):
    session_id: str
    message:    str


class VcfDesignRequest(BaseModel):
    vcf_text: str = ""
    vcf_path: str = ""  # server-side path under sample/ — demo/testing convenience, not a general file-read endpoint


class NeoantigenDesignRequest(BaseModel):
    vcf_text: str = ""
    vcf_path: str = ""  # server-side path under sample/ — demo/testing convenience, not a general file-read endpoint
    bam_path: str = ""  # server-side path under sample/ — same convenience/guard as vcf_path


def _new_job_store_entry(job_id: str, uniprot_id: str = "", mode: str = "screen") -> dict:
    return {
        "id":              job_id,
        "uniprot_id":      uniprot_id,
        "mode":            mode,
        "status":          "RUNNING",
        "current_step":    0,
        "total_steps":     4,
        "current_message": "대기 중",
        "timeline":        [],
        "result":          None,
        "error_message":   None,
        "created_at":      datetime.utcnow().isoformat() + "Z",
    }


@router.post("/design", status_code=202)
async def start_drug_discovery(req: DrugDiscoveryRequest) -> dict:
    """Start a structure -> [ligand docking | library screening] -> report job."""
    if not req.uniprot_id and not req.target_sequence:
        raise HTTPException(status_code=400, detail="One of uniprot_id or target_sequence is required")
    if not req.screen_library and not req.ligand_smiles:
        raise HTTPException(status_code=400, detail="ligand_smiles is required unless screen_library is true")

    job_id = str(uuid.uuid4())
    _evict_finished_jobs()
    _DRUG_DISCOVERY_STORE[job_id] = {
        "id":              job_id,
        "uniprot_id":      req.uniprot_id,
        "mode":            "screen" if req.screen_library else "single",
        "status":          "RUNNING",
        "current_step":    0,
        "total_steps":     4,
        "current_message": "대기 중",
        "timeline":        [],
        "result":          None,
        "error_message":   None,
        "created_at":      datetime.utcnow().isoformat() + "Z",
    }

    _RUNNING_TASKS[job_id] = asyncio.create_task(_run_drug_discovery_task(
        job_id=job_id,
        uniprot_id=req.uniprot_id,
        target_sequence=req.target_sequence,
        ligand_smiles=req.ligand_smiles,
        screen_library=req.screen_library,
        goal_text=req.goal_text,
        max_candidates=req.max_candidates,
    ))

    logger.info("[drug_discovery] job started | job=%s uniprot=%s screen=%s max_candidates=%s",
                job_id, req.uniprot_id, req.screen_library, req.max_candidates or "all")
    return {"job_id": job_id, "status": "RUNNING"}


@router.post("/design-from-vcf", status_code=202)
async def start_drug_discovery_from_vcf(
    req: VcfDesignRequest,
) -> dict:
    """
    "Track B" (somatic-variant cancer targets) job start — parses the given
    VCF, re-annotates every variant live via Ensembl VEP (never trusts the
    file's own GENE= label), resolves the confirmed gene to a real UniProt
    entry, then runs the exact same screening pipeline /design uses. See
    services/drug_discovery_pipeline.run_drug_discovery_from_vcf().
    """
    vcf_text = req.vcf_text
    if not vcf_text and req.vcf_path:
        # sample/ lives under backend/ (backend/sample/), not the repo root —
        # real reported bug: it used to be a repo-root sibling of backend/,
        # but Cloud Run's actual build (backend/Dockerfile, build context =
        # backend/ only) can never COPY anything outside that context, so a
        # repo-root sample/ silently never made it into the deployed image
        # no matter what the root Dockerfile's own COPY said. Keeping the
        # real files inside backend/ means every build path (local dev, the
        # repo-root Dockerfile via `COPY backend/ .`, and backend/Dockerfile
        # via `COPY . .`) includes them the same way, with no special-casing.
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        sample_dir = os.path.normpath(os.path.join(base_dir, "sample"))
        resolved = os.path.normpath(os.path.join(base_dir, req.vcf_path))
        if not resolved.startswith(sample_dir):
            raise HTTPException(status_code=400, detail="vcf_path must be a file under sample/")
        if not os.path.isfile(resolved):
            raise HTTPException(status_code=404, detail=f"File not found: {req.vcf_path}")
        with open(resolved, encoding="utf-8") as f:
            vcf_text = f.read()
    if not vcf_text:
        raise HTTPException(status_code=400, detail="One of vcf_text or vcf_path is required")

    job_id = str(uuid.uuid4())
    _evict_finished_jobs()
    _DRUG_DISCOVERY_STORE[job_id] = {
        "id":              job_id,
        "uniprot_id":      "",
        "mode":            "screen",
        "status":          "RUNNING",
        "current_step":    0,
        "total_steps":     4,
        "current_message": "대기 중",
        "timeline":        [],
        "result":          None,
        "error_message":   None,
        "created_at":      datetime.utcnow().isoformat() + "Z",
    }
    _RUNNING_TASKS[job_id] = asyncio.create_task(_run_drug_discovery_vcf_task(job_id=job_id, vcf_text=vcf_text))
    logger.info("[drug_discovery] VCF job started | job=%s vcf_path=%s", job_id, req.vcf_path or "(inline text)")
    return {"job_id": job_id, "status": "RUNNING"}


def _resolve_sample_path(relative_path: str, kind: str) -> str:
    """Same path-traversal guard as design-from-vcf's inline vcf_path
    resolution, factored out since /design-from-bam needs it twice
    (vcf_path + bam_path). sample/ lives under backend/ — see the comment
    on that other resolution for why."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    sample_dir = os.path.normpath(os.path.join(base_dir, "sample"))
    resolved = os.path.normpath(os.path.join(base_dir, relative_path))
    if not resolved.startswith(sample_dir):
        raise HTTPException(status_code=400, detail=f"{kind} must be a file under sample/")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")
    return resolved


@router.post("/design-from-bam", status_code=202)
async def start_neoantigen_prediction(req: NeoantigenDesignRequest) -> dict:
    """
    Neoantigen candidate identification — VCF variant calls (required) +
    an optional BAM (used only for a real read-count/coverage summary;
    real BAM-based HLA typing is not available on this deployment — see
    services/neoantigen_engine.py's module docstring for why, and
    services/neoantigen_pipeline.py for the full real pipeline this kicks
    off). Separate job mode ("neoantigen") from the docking/screening
    pipeline — the result shape is candidates/literature, not
    ranked_candidates/docking.
    """
    vcf_text = req.vcf_text
    if not vcf_text and req.vcf_path:
        resolved_vcf = _resolve_sample_path(req.vcf_path, "vcf_path")
        with open(resolved_vcf, encoding="utf-8") as f:
            vcf_text = f.read()
    if not vcf_text:
        raise HTTPException(status_code=400, detail="One of vcf_text or vcf_path is required")

    bam_path = _resolve_sample_path(req.bam_path, "bam_path") if req.bam_path else None

    result = _start_neoantigen_job(vcf_text, bam_path)
    logger.info("[neoantigen] job started | job=%s vcf_path=%s bam_path=%s",
                result["job_id"], req.vcf_path or "(inline text)", req.bam_path or "(none)")
    return result


def _start_neoantigen_job(vcf_text: str, bam_path: str | None = None) -> dict:
    """Shared job-creation logic for /design-from-bam and the MCP
    predict_neoantigen_candidates tool (inline VCF text only, no BAM) —
    single source of truth for the _DRUG_DISCOVERY_STORE entry shape, and
    for the MCP tool specifically, this is what lets it return a job_id in
    well under 100ms instead of blocking on the real ~18-24s MHCflurry
    pipeline (see get_drug_discovery_job_status for the polling side)."""
    job_id = str(uuid.uuid4())
    entry = _new_job_store_entry(job_id, mode="neoantigen")
    entry["total_steps"] = 5  # keep in sync with neoantigen_pipeline._MAX_STEPS
    _evict_finished_jobs()
    _DRUG_DISCOVERY_STORE[job_id] = entry
    _RUNNING_TASKS[job_id] = asyncio.create_task(
        _run_neoantigen_task(job_id=job_id, vcf_text=vcf_text, bam_path=bam_path),
    )
    return {"job_id": job_id, "status": "RUNNING"}


def start_neoantigen_demo_job() -> dict:
    """Start a neoantigen job on the bundled WES sample (sample/NSCLC_variants.vcf
    + sample/NSCLC.bam), so a caller with no VCF of its own can still run the
    real pipeline end-to-end. This is what lets the MCP
    predict_neoantigen_candidates tool auto-run when invoked with no
    vcf_content — the user types "암 백신" and the WES demo proceeds without any
    file upload. Same sample data the "mRNA 암 백신 데모" chat flow uses."""
    vcf_path = _resolve_sample_path("sample/NSCLC_variants.vcf", "vcf_path")
    bam_path = _resolve_sample_path("sample/NSCLC.bam", "bam_path")
    with open(vcf_path, encoding="utf-8") as f:
        vcf_text = f.read()
    return _start_neoantigen_job(vcf_text, bam_path)


# VCF files always contain a "##fileformat=VCF" meta-line and a "#CHROM"
# header line per the VCF spec — a real, reliable format signature, not a
# guess. Anything else starting with ">" is treated as FASTA.
def _sniff_bio_format(text: str) -> str | None:
    head = text.lstrip()[:2000]
    if "##fileformat=VCF" in head or head.startswith("#CHROM") or "\n#CHROM" in head:
        return "vcf"
    if head.startswith(">"):
        return "fasta"
    return None


@router.post("/design-from-file", status_code=202)
async def start_drug_discovery_from_file(file: UploadFile = File(...)) -> dict:
    """
    Unified upload gateway for both tracks — auto-detects VCF ("Track B")
    vs FASTA ("Track A") from the file's own content (not the filename,
    which can't be trusted) and routes to the matching pipeline. Real
    detection, real parsing; nothing here fabricates a target or result if
    the upload doesn't resolve to one (see run_drug_discovery_from_vcf() and
    protein_structure_engine.predict_structure_esmfold()'s existing
    fail-soft handling for e.g. a sequence exceeding ESMFold's real length
    limit).
    """
    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    fmt = _sniff_bio_format(text)

    if fmt == "vcf":
        job_id = str(uuid.uuid4())
        _evict_finished_jobs()
        _DRUG_DISCOVERY_STORE[job_id] = _new_job_store_entry(job_id)
        _RUNNING_TASKS[job_id] = asyncio.create_task(_run_drug_discovery_vcf_task(job_id=job_id, vcf_text=text))
        logger.info("[drug_discovery] file upload -> VCF job started | job=%s filename=%s", job_id, file.filename)
        return {"job_id": job_id, "status": "RUNNING", "detected_format": "vcf"}

    if fmt == "fasta":
        records = parse_fasta(text)
        if not records or not records[0]["sequence"]:
            raise HTTPException(status_code=400, detail="FASTA 파일에서 서열을 찾지 못했습니다.")
        sequence = records[0]["sequence"]
        job_id = str(uuid.uuid4())
        _evict_finished_jobs()
        _DRUG_DISCOVERY_STORE[job_id] = _new_job_store_entry(job_id)
        goal_text = f"업로드된 FASTA 서열({records[0]['id']})을 억제할 수 있는 승인된 약물을 찾아줘"
        _RUNNING_TASKS[job_id] = asyncio.create_task(
            _run_drug_discovery_fasta_task(job_id=job_id, sequence=sequence, goal_text=goal_text),
        )
        logger.info("[drug_discovery] file upload -> FASTA job started | job=%s filename=%s seq_len=%d",
                    job_id, file.filename, len(sequence))
        return {"job_id": job_id, "status": "RUNNING", "detected_format": "fasta"}

    raise HTTPException(
        status_code=400,
        detail="파일 형식을 인식하지 못했습니다 — .vcf(##fileformat=VCF 헤더 포함) 또는 .fasta/.fa('>'로 시작) 파일만 지원합니다.",
    )


@router.get("/debug/store-size")
async def get_store_size() -> dict:
    """What the process is actually holding. Exists so a leak can be OBSERVED from
    outside rather than argued about — scripts/mcp_leak.py samples this alongside
    the process RSS, and an unbounded job store shows up here first."""
    payload_bytes = sum(
        len((job.get("result") or {}).get("structure", {}).get("pdb_text") or "")
        for job in _DRUG_DISCOVERY_STORE.values()
    )
    return {
        "jobs": len(_DRUG_DISCOVERY_STORE),
        "running_tasks": len(_RUNNING_TASKS),
        "max_jobs": _MAX_STORED_JOBS,
        "retained_structure_mb": round(payload_bytes / 1024 / 1024, 1),
    }


@router.get("/status/{job_id}")
async def get_drug_discovery_status(job_id: str) -> dict:
    job = _DRUG_DISCOVERY_STORE.get(job_id)
    if not job:
        # Not logged at info level — the frontend polls this every second,
        # so a genuinely missing job (e.g. queried after a server restart
        # wiped the in-memory store) would otherwise flood the log; a 404
        # is already visible in uvicorn's own access log.
        raise HTTPException(status_code=404, detail="Drug discovery job not found")
    return job


@router.post("/stop/{job_id}")
async def stop_drug_discovery(job_id: str) -> dict:
    """
    Real cancellation of a running job (asyncio Task.cancel(), not just a
    UI-side flag) — see stop_job(). Note a real limitation: if the job is
    currently blocked inside a synchronous subprocess dispatched via
    asyncio.to_thread (mk_prepare_receptor or vina.exe), cancelling the Task
    stops the pipeline from proceeding to any further step, but that one
    already-launched subprocess keeps running in its OS thread until it
    finishes on its own — Task.cancel() has no way to reach into and kill a
    subprocess that isn't tracked as its own killable handle. Its result is
    simply discarded when it finishes.
    """
    job = _DRUG_DISCOVERY_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Drug discovery job not found")
    stopped = stop_job(job_id)
    if not stopped and job.get("status") == "RUNNING":
        # Job is marked RUNNING but has no live task (e.g. server restarted
        # since it started) — reflect reality rather than silently no-op.
        job.update({"status": "CANCELLED", "error_message": "사용자 요청으로 중지되었습니다."})
        stopped = True
    return {"job_id": job_id, "stopped": stopped, "status": job["status"]}


def _find_latest_report(job_id: str, extension: str) -> str | None:
    """Mirrors agent_router.py's _find_local_report — globs by filename
    prefix rather than trusting an in-memory path, so it still works after a
    server restart as long as the file is on disk."""
    files = sorted(_glob.glob(os.path.join(_REPORT_DIR, f"drugjob_{job_id}_*.{extension}")))
    return files[-1] if files else None


@router.get("/report/{job_id}")
async def get_drug_discovery_report(job_id: str):
    """HTML report (AI 해설 + 후보 목록 + 강점/한계점) — same pattern as
    GET /api/agent/report/{job_id} for the primer-design pipeline."""
    html_path = _find_latest_report(job_id, "html")
    if not html_path:
        logger.warning("[drug_discovery] HTML report requested but not found | job=%s", job_id)
        raise HTTPException(status_code=404, detail="리포트가 아직 생성되지 않았습니다.")
    logger.info("[drug_discovery] HTML report served | job=%s path=%s", job_id, html_path)
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get("/report/{job_id}/pdf")
async def get_drug_discovery_report_pdf(job_id: str):
    pdf_path = _find_latest_report(job_id, "pdf")
    if not pdf_path:
        logger.warning("[drug_discovery] PDF report requested but not found | job=%s", job_id)
        raise HTTPException(status_code=404, detail="PDF 리포트가 아직 생성되지 않았습니다.")
    logger.info("[drug_discovery] PDF report served | job=%s path=%s", job_id, pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"drug_discovery_{job_id[:8]}.pdf")


# Numbering/keywords must match the real menu text in drug_discovery_
# intent.py's mentions_research_topic branch (1=문헌검색, 2=신약스크리닝,
# 3=타겟정보) and DrugDiscoveryChatPanel.tsx's matching quick-pick buttons —
# all three surfaces were deliberately kept in sync when the menu order was
# changed (문헌 검색 first) so a typed "1"/"2"/"3" always means the same
# thing the numbered list on screen says it means.
_ORDINAL_CHOICE_RE = re.compile(r"^\s*([1-3])(?!\d)")


def _resolve_intent_clarification_choice(message: str, topic: str) -> str | None:
    """
    Real, reported bug (round 2): the exact-match tuples ("1", "1)", "1.")
    missed the extremely common Korean reply pattern "1번"/"1번이요"/"1 번"
    ("number 1") — any suffix after the leading digit fell through to
    fresh intent parsing and re-triggered the same menu forever, same
    failure mode as the originally-fixed loop bug. Now matches a leading
    digit 1-3 not immediately followed by another digit (so e.g. "12월"
    isn't mistaken for choice "1"), regardless of what follows it.
    """
    stripped = message.strip()
    low = stripped.lower()

    # Real reported bug: "mRNA 암 백신 데모" (an explicit demo request typed
    # in reply to the menu) was being rewritten into "mRNA 억제할 수 있는
    # 승인된 약물을 찾아줘" — a small-molecule SCREENING request — because
    # "mrna"/"백신" are ALSO in the choice-2 keyword list below, and that
    # match won before the demo trigger was ever considered. The rewritten
    # text no longer contained "데모", so is_neoantigen_demo_query()
    # downstream never got a chance to see the real, original message, and
    # "mRNA" (unresolvable as a UniProt target) silently fell back to the
    # default SARS-CoV-2 Spike screening target instead. Checked first so
    # an explicit demo/neoantigen mention always passes through unmodified,
    # regardless of which menu option's keywords it also happens to overlap
    # with.
    if is_neoantigen_demo_query(message) or is_neoantigen_query(message):
        return message

    choice_match = _ORDINAL_CHOICE_RE.match(stripped)
    choice = choice_match.group(1) if choice_match else None

    if choice == "1" or any(k in low for k in ("논문", "문헌", "paper", "literature")):
        return f"{topic} 관련 논문 찾아줘"
    if choice == "2" or any(k in low for k in ("스크리닝", "신약", "약물", "screen", "백신", "vaccine", "mrna")):
        if _is_cancer_topic(topic):
            # Real reported bug: choice 2 always composed a small-molecule
            # screening request, which for a cancer topic (no single
            # standardized target) ran against a guessed default target and
            # returned an irrelevant result. Re-sending the bare topic
            # re-triggers drug_discovery_intent.py's cancer/needs_vcf branch,
            # which now also sets neoantigen_mode so the frontend routes the
            # VCF/BAM upload to the real neoantigen pipeline instead.
            return topic
        return f"{topic} 억제할 수 있는 승인된 약물을 찾아줘"
    if choice == "3" or any(k in low for k in ("타겟", "질병", "target")):
        return f"{topic}의 질병 연관성과 관련 경로 분석해줘"
    return None


@router.post("/converse")
async def converse_drug_discovery(req: ConverseRequest) -> dict:
    """
    Free-text entry point for the Drug Discovery chat panel. Single
    request/response per turn (no SSE) — when the parsed intent is
    complete enough to start a design, fires the same background pipeline
    used by /design and returns a job_id for the caller to poll via
    /status/{job_id}.
    """
    logger.info("[drug_discovery] converse received | session=%s message=%r", req.session_id, req.message[:200])

    session = _DRUG_DISCOVERY_CHAT_SESSIONS.setdefault(req.session_id, _new_chat_session())
    session["messages"].append({"role": "user", "text": req.message})

    # Real reported bug: stop-phrase detection previously only ran deep
    # inside the RUNNING-job branch further below, AFTER several earlier
    # branches (pending intent-clarification, neoantigen-demo/-prompt,
    # literature/clinical/compound-discovery/target-intelligence queries)
    # that could each claim the message first if it happened to also match
    # their own keywords — so a stop word sent at the "wrong" moment in the
    # conversation silently did nothing. Checked here, first and
    # unconditionally: if the message contains a stop word and a job is
    # actually running, stop it before any other routing gets a chance to
    # misinterpret the message.
    lowered_msg = req.message.strip().lower()
    if any(p in lowered_msg for p in _STOP_PHRASES):
        stop_target_id = session.get("active_job_id")
        stop_target = _DRUG_DISCOVERY_STORE.get(stop_target_id or "")
        if stop_target and stop_target.get("status") == "RUNNING":
            stopped = stop_job(stop_target_id)
            reply = "진행 중이던 작업을 중지했습니다." if stopped else "이미 종료된 작업이라 중지할 게 없습니다."
            session["active_job_id"] = None
            _transition(session, ConversationState.FAILED, req.session_id)
            session["messages"].append({"role": "agent", "text": reply})
            logger.info("[drug_discovery] converse stopped job via top-level stop-phrase check | session=%s job=%s",
                        req.session_id, stop_target_id)
            return {"reply": reply, "action": "stopped", "job_id": stop_target_id, "state": session["state"]}

    # Real reported request: "너는 무슨 프로그램이니?"-style questions about
    # the assistant itself — checked early, unconditionally, so it answers
    # regardless of whatever else is going on in the conversation (a pending
    # clarification menu, an active job, etc.).
    if is_identity_query(req.message):
        result = answer_identity_question()
        reply = result["text"]
        session["messages"].append({"role": "agent", "text": reply})
        logger.info("[drug_discovery] converse answered (identity) | session=%s", req.session_id)
        return {"reply": reply, "action": "chat", "job_id": session.get("active_job_id"), "state": session["state"]}

    # Real, reported bug: a bare numeric reply ("2") to the intent-
    # clarification menu (see drug_discovery_intent.py's
    # mentions_research_topic branch) was never recognized as answering
    # that menu — it fell through to fresh intent parsing, which (since the
    # topic was still sitting in known_slots.goal_text) re-triggered the
    # SAME clarification menu, looping forever no matter what the user
    # typed. Resolved here, before any other routing, by rewriting the
    # message into the exact same fully-composed follow-up the matching
    # frontend quick-pick button would have sent (see FUNCTION_EXAMPLES-
    # style composition in DrugDiscoveryChatPanel.tsx) — this reuses every
    # real routing check below unchanged instead of duplicating logic.
    pending_clarification = session.get("known_slots", {}).get("needs_intent_clarification")
    if pending_clarification:
        topic = session["known_slots"].get("intent_topic")
        rewritten = _resolve_intent_clarification_choice(req.message, topic) if topic else None
        if rewritten:
            logger.info("[drug_discovery] resolved pending intent clarification | session=%s choice=%r -> %r",
                        req.session_id, req.message, rewritten)
            req.message = rewritten
        # Consumed either way — a reply that doesn't match 1/2/3 or a
        # recognizable keyword is treated as a brand-new message instead of
        # getting stuck (same "don't dead-end" discipline as the existing
        # pending_confirmation handling in parse_drug_discovery_intent()),
        # and the flag must not survive to silently re-fire on some later,
        # unrelated turn.
        session["known_slots"] = {
            **session["known_slots"], "needs_intent_clarification": False, "intent_topic": None,
        }

    # Real reported bug: "신항원 후보 찾기" (or "mRNA 백신 찾아줘" etc.) typed
    # as plain chat text had no route at all — the neoantigen pipeline
    # always needs an actual VCF file, which chat text can't provide, so
    # this always asks for the VCF/BAM upload (same UI as the cancer-topic
    # branch) rather than falling through to intent-parsing and getting a
    # confusing non-answer. Checked before the literature/clinical block
    # since "신항원" messages should never be misread as an ordinary
    # literature search.
    # Real reported gap: typing the demo button's own label text ("암 mRNA
    # 자가 백신 데모") instead of clicking it just re-showed the same "please
    # click the button" prompt below — checked first (a strict superset of
    # is_neoantigen_query) so an explicit demo request actually starts the
    # sample/NSCLC_variants.vcf + sample/NSCLC.bam job, mirroring exactly
    # what handleSampleNeoantigen() does client-side (DrugDiscoveryChatPanel.
    # tsx) via the shared _start_neoantigen_job() helper.
    if is_neoantigen_demo_query(req.message):
        vcf_path = _resolve_sample_path("sample/NSCLC_variants.vcf", "vcf_path")
        bam_path = _resolve_sample_path("sample/NSCLC.bam", "bam_path")
        with open(vcf_path, encoding="utf-8") as f:
            vcf_text = f.read()
        result = _start_neoantigen_job(vcf_text, bam_path)
        job_id = result["job_id"]
        session["active_job_id"] = job_id
        session["known_slots"] = {}
        _transition(session, ConversationState.RUNNING, req.session_id)
        reply = (
            "VCF 변이를 실시간 검증하고 실제 MHCflurry 모델로 신항원 후보를 예측합니다 "
            "(BAM 기반 실제 HLA 타이핑은 이 환경에서 미지원 — 표준 population allele 사용, "
            "리포트에서 명시됩니다)."
        )
        session["messages"].append({"role": "agent", "text": reply})
        logger.info("[neoantigen] converse started demo job via text | session=%s job=%s", req.session_id, job_id)
        return {
            "reply": reply, "action": "start_design", "job_id": job_id, "state": session["state"],
            "needs_vcf": False, "neoantigen_mode": True,
        }

    # Real reported bug: a genuine question ("mRNA 암백신이 뭐야?") also
    # matches is_neoantigen_query() below and got misrouted straight to the
    # "upload a VCF / click the demo" action prompt instead of an actual
    # answer. Checked here, after the demo-launch check (so an explicit demo
    # request still fires immediately) but before the generic action prompt.
    if is_neoantigen_question(req.message):
        result = await asyncio.to_thread(answer_neoantigen_question, req.message)
        reply = result["text"]
        session["messages"].append({"role": "agent", "text": reply})
        logger.info("[drug_discovery] converse answered (neoantigen_explainer) | session=%s", req.session_id)
        return {
            "reply": reply, "action": "chat", "job_id": session.get("active_job_id"), "state": session["state"],
        }

    if is_neoantigen_query(req.message):
        reply = (
            "신항원(neoantigen) 후보 식별을 위해서는 실제 VCF(체세포 변이) 파일이 필요합니다 "
            "(선택적으로 BAM 파일도 함께 사용할 수 있습니다 — 단, 이 환경에서는 BAM 기반 실제 HLA "
            "타이핑은 지원되지 않아 population-common allele을 대신 사용합니다). 📎 버튼으로 직접 "
            "업로드하시거나, 아래 '🧬 암 mRNA 자가 백신 데모' 버튼으로 실제 NSCLC 샘플(KRAS G12D 변이, "
            "실제 MHCflurry 예측으로 검증됨)을 먼저 체험해 보실 수 있습니다."
        )
        session["messages"].append({"role": "agent", "text": reply})
        logger.info("[drug_discovery] converse answered (neoantigen_prompt) | session=%s", req.session_id)
        return {
            "reply": reply, "action": "chat", "job_id": session.get("active_job_id"), "state": session["state"],
            "needs_vcf": True, "neoantigen_mode": True,
        }

    # Real reported bug: "결핵에 대해 설명해줘" fell through to the mentions_
    # research_topic 1/2/3 menu since no other classifier claimed it. Per
    # explicit follow-up feedback, this is answered as a direct general-AI
    # explanation (NOT a literature-search/panel result) — checked as its
    # own standalone branch, before the literature/clinical/etc block below,
    # so it never gets folded into that block's panel-rendering dispatch.
    if is_general_explain_query(req.message):
        result = await asyncio.to_thread(answer_general_explain_question, req.message)
        reply = result["text"]
        session["messages"].append({"role": "agent", "text": reply})
        logger.info("[drug_discovery] converse answered (general_explain) | session=%s", req.session_id)
        return {
            "reply": reply, "action": "chat", "job_id": session.get("active_job_id"), "state": session["state"],
        }

    # Literature/Clinical/Compound-discovery questions (real live PubMed /
    # ClinicalTrials.gov / PubChem / ChEMBL search) are meaningful regardless
    # of job state — checked first, before the RUNNING/COMPLETED branching
    # below, so asking about them while a design job happens to be running
    # gets a real search instead of being absorbed into that job's generic
    # status reply. known_slots is wiped once a design job actually starts
    # (see start_design branch below), so a target established in the turn
    # right before one of these questions can only still be found on the
    # active job's own stored target_name/uniprot_id — checked first since
    # they're the more recently resolved values in that case.
    if (is_literature_query(req.message) or is_clinical_query(req.message)
            or is_compound_discovery_query(req.message) or is_target_intelligence_query(req.message)):
        active_job_for_lookup = _DRUG_DISCOVERY_STORE.get(session.get("active_job_id") or "")
        known_slots = session.get("known_slots", {})
        locked_target_name = (active_job_for_lookup.get("target_name") if active_job_for_lookup else None) or known_slots.get("target_name")
        locked_uniprot_id = (active_job_for_lookup.get("uniprot_id") if active_job_for_lookup else None) or known_slots.get("uniprot_id")

        # Real reported bug: an active/completed job's own target used to win
        # UNCONDITIONALLY here, so typing a clearly different topic ("결핵
        # 관련 논문 찾아줘") right after a SARS-CoV-2 job completed still
        # searched SARS-CoV-2 literature — the message-level mention was
        # never even looked at once a job existed. Now the SAME fresh-
        # mention-first resolution applies whether or not a job is active:
        # an explicit mention in THIS message always takes priority; only
        # falls back to the locked job/known_slots target when the current
        # message doesn't name anything resolvable on its own (a bare "관련
        # 논문 찾아줘" follow-up still correctly stays on the locked target).
        # Reuses the same curated, UniProt-verified fuzzy-match
        # drug_discovery_intent.py already uses — real, deterministic,
        # never a guess.
        from services.drug_discovery_intent import _fuzzy_match_target
        fresh_match = _fuzzy_match_target(req.message.lower())
        if fresh_match:
            uniprot_id, target_name = fresh_match[0], fresh_match[1]
        else:
            # The curated table only covers ~7 SARS-CoV-2/Influenza
            # targets — a real topic outside it (a gene symbol like
            # "EGFR", or a pathogen only in the Korean term map like
            # "결핵") needs a broader check before falling back to the
            # locked target, otherwise it can never be recognized as a
            # fresh mention. Dry-runs the same real resolution
            # _clean_external_search_query would do for this message
            # alone (target_name=None so it can't just echo back the
            # locked value) — "" means the message truly has no topic of
            # its own (relies on _translate_korean_query's NONE-sentinel
            # fix, not a guess), so only THEN fall back.
            fresh_topic = _clean_external_search_query(req.message, None)
            if fresh_topic and fresh_topic != locked_target_name:
                target_name, uniprot_id = fresh_topic, None
            else:
                target_name = locked_target_name
                uniprot_id = locked_uniprot_id

        if is_literature_query(req.message):
            result = await asyncio.to_thread(answer_literature_question, req.message, target_name)
            action = "literature_search"
        elif is_clinical_query(req.message):
            result = await asyncio.to_thread(answer_clinical_question, req.message, target_name)
            action = "clinical_search"
        elif is_target_intelligence_query(req.message):
            result = await asyncio.to_thread(answer_target_intelligence_question, target_name, uniprot_id)
            action = "target_intelligence"
        else:
            result = await asyncio.to_thread(answer_compound_discovery_question, req.message, target_name, uniprot_id)
            action = "compound_discovery"

        # Real session-wide memory (reported request: "병원균이나 암에 대해서
        # 한번 물어보면 세션 내내 기억하게 해줘"): the best-known resolved
        # topic for THIS turn — target_name when already resolved, else
        # whatever result["query"] actually searched with (e.g. "EGFR"
        # resolved via _clean_external_search_query's raw-text fallback — a
        # real gene name the small curated pathogen table doesn't cover).
        resolved_topic = target_name or (result.get("available") and result.get("query")) or None
        if not active_job_for_lookup:
            # A fresh mention of a different pathogen/gene next turn
            # re-resolves and overwrites this (the "update" behavior
            # requested), since fresh_match/the raw message above always
            # takes priority over the remembered value.
            if resolved_topic and resolved_topic != known_slots.get("target_name"):
                session["known_slots"] = {**known_slots, "target_name": resolved_topic, "uniprot_id": uniprot_id}

        reply = result["text"]
        session["messages"].append({"role": "agent", "text": reply})
        logger.info("[drug_discovery] converse answered (%s) | session=%s", action, req.session_id)
        return {
            "reply": reply, "action": action, "job_id": session.get("active_job_id"), "state": session["state"],
            "panel": {"type": action, **{k: v for k, v in result.items() if k != "text"}},
            # Real reported gap: after picking "1) 관련 논문 검색" from the
            # bare-topic-mention menu, the follow-up quick-pick buttons reset
            # to the generic default examples instead of staying on-topic
            # (the other two menu options, "신약 스크리닝"/"타겟 정보만 확인",
            # disappeared) — the user has to retype the topic to see them
            # again. Returning the resolved topic here lets the frontend
            # re-show those SAME topic-scoped quick-pick buttons (see
            # DrugDiscoveryChatPanel.tsx's intentTopic-driven button row)
            # after ANY literature/clinical/target-intelligence/compound-
            # discovery answer, not just the original menu prompt.
            "intent_topic": resolved_topic,
        }

    active_job_id = session.get("active_job_id")
    if active_job_id:
        active_job = _DRUG_DISCOVERY_STORE.get(active_job_id)
        if active_job and active_job["status"] == "RUNNING":
            # Stop-phrase handling now happens unconditionally at the very
            # top of this function (before any other routing) — see that
            # check for why. Nothing left to do here for a stop word.
            # Rich, conversational reply using live job state — previously
            # every message while a job ran got the exact same canned
            # "이미 진행 중입니다" line regardless of what was actually
            # asked, unlike the primer-design agent's running_job_reply().
            reply = await asyncio.to_thread(answer_running_job_question, req.message, active_job)
            logger.info("[drug_discovery] converse answered (job running) | session=%s job=%s",
                        req.session_id, active_job_id)
            session["messages"].append({"role": "agent", "text": reply})
            return {"reply": reply, "action": "status", "job_id": active_job_id, "state": session["state"]}

        if active_job and active_job["status"] == "COMPLETED":
            # SAR optimization needs the job's full result (real candidate +
            # real target structure to re-dock analogs against) — checked
            # before the ask_question classifier below since it triggers
            # real new computation (Vina re-docking), not just Q&A over
            # already-existing data.
            if is_sar_optimization_query(req.message):
                result = await asyncio.to_thread(answer_sar_optimization_question, active_job.get("result") or {})
                logger.info("[drug_discovery] converse answered (sar_optimization) | session=%s job=%s",
                            req.session_id, active_job_id)
                session["messages"].append({"role": "agent", "text": result["text"]})
                return {
                    "reply": result["text"], "action": "sar_optimization", "job_id": active_job_id, "state": session["state"],
                    "panel": {"type": "sar_optimization", **{k: v for k, v in result.items() if k != "text"}},
                }

            # Decision report similarly triggers real new work (fresh
            # literature/clinical searches for the target), not just Q&A
            # over already-existing data.
            if is_decision_report_query(req.message):
                result = await asyncio.to_thread(
                    answer_decision_report_question, active_job.get("result") or {}, active_job.get("target_name"),
                )
                logger.info("[drug_discovery] converse answered (decision_report) | session=%s job=%s",
                            req.session_id, active_job_id)
                session["messages"].append({"role": "agent", "text": result["text"]})
                return {
                    "reply": result["text"], "action": "decision_report", "job_id": active_job_id, "state": session["state"],
                    "panel": {"type": "decision_report", **{k: v for k, v in result.items() if k != "text"}},
                }

            # Real gap, not hypothetical: once a job finished, there was no
            # way to ask questions about its actual results (e.g. "왜 1위야?")
            # — every subsequent message was parsed as if it were a
            # brand-new, unrelated design request. Mirrors agent/__init__.py's
            # classify_action()+ScientificChat.ask() pattern, own
            # implementation, real result data only (never fabricated).
            #
            # A completed session's target focus is never *silently* switched
            # (the original problem: a stray "chat" classification re-triggering
            # slot-filling for a brand-new target). Only an explicit new-target
            # design request — classified as "start_design" below — switches
            # targets, and only after a full session reset so nothing from the
            # old target leaks in. "ask_question"/"chat" turns stay locked onto
            # the current target. The "🆕 새 연구" button (handleNewResearch() in
            # DrugDiscoveryChatPanel.tsx) still exists as an explicit reset that
            # mints a fresh session_id, but is no longer the *only* way to move
            # on to a new target.
            action = await asyncio.to_thread(
                classify_drug_discovery_action, req.message, True, active_job["status"],
                active_job.get("target_name") or active_job.get("uniprot_id"),
            )
            if action == "start_design":
                # The user explicitly asked to research a genuinely new target.
                # Rather than dead-ending them into the "🆕 새 연구" button (the
                # real reported problem: "타겟 전환이 안돼"), switch targets
                # in-session by performing a FULL reset here and then falling
                # through to the normal fresh-intent handling below. Wiping
                # active_job_id AND known_slots AND resetting the FSM to START
                # is exactly what keeps the previous target from bleeding into
                # the new one (the original 대장암→결핵 contamination bug), so a
                # typed switch is now just as safe as the button — which resets
                # by minting a brand-new session_id client-side.
                logger.info("[drug_discovery] converse switching target in-session | session=%s old_job=%s",
                            req.session_id, active_job_id)
                session["active_job_id"] = None
                session["known_slots"] = {}
                _transition(session, ConversationState.START, req.session_id)
                # Deliberately no return — control continues to the intent
                # parsing / job-start block below with a clean session, so the
                # new target's design request is handled from scratch.
            else:
                # "ask_question" and "chat" are both answered the same way —
                # answer_completed_job_question() already gives a reasonable,
                # real-data-grounded answer to general chat, not just narrow
                # result lookups (falls back to ai_summary/an honest "not in
                # this data" rather than fabricating).
                reply = await asyncio.to_thread(
                    answer_completed_job_question, req.message, active_job.get("result") or {},
                )
                logger.info("[drug_discovery] converse answered (%s, focus retained) | session=%s job=%s",
                            action, req.session_id, active_job_id)
                session["messages"].append({"role": "agent", "text": reply})
                return {"reply": reply, "action": action, "job_id": active_job_id, "state": session["state"]}

    # Real race, not hypothetical: two /converse calls for the same session
    # arriving close together (double-click, a retried fetch, etc.) can both
    # pass the active_job_id check above and both reach here before either
    # one sets session["active_job_id"] below — the intent-parsing await
    # just below can take seconds (real LLM call + live UniProt search),
    # which is a wide-open window for that interleaving. Checking and
    # setting this flag here happens with no `await` in between, so it's
    # atomic with respect to any other coroutine on this single-threaded
    # event loop and closes the window — this is what was actually causing
    # duplicate jobs/reports, not the earlier (also real) client-side races.
    if session.get("processing"):
        logger.info("[drug_discovery] converse blocked | session=%s reason=concurrent_processing", req.session_id)
        reply = "이전 메시지를 아직 처리하고 있습니다. 잠시만 기다려 주세요."
        session["messages"].append({"role": "agent", "text": reply})
        return {"reply": reply, "action": "status", "job_id": session.get("active_job_id")}
    session["processing"] = True

    try:
        # parse_drug_discovery_intent can now make real blocking network calls
        # (live UniProt search/verification) in addition to the existing
        # synchronous LLM call, so it's dispatched off the event loop the same
        # way the pipeline orchestrators already do for blocking work.
        intent = await asyncio.to_thread(
            parse_drug_discovery_intent, req.message, known_slots=session["known_slots"],
        )

        # Real reported gap: the regex-based is_general_explain_query() catch
        # -all (services/drug_discovery_chat.py, checked earlier in this
        # function) only fires on the specific "~에 대해 설명/알려/소개"
        # phrasing — "코로나가 뭐야?" doesn't match it and fell all the way
        # through to here, where parse_drug_discovery_intent's
        # mentions_research_topic branch showed the 1/2/3 action menu instead
        # of just answering. Rather than keep adding regex variants, this
        # reuses the SAME LLM call parse_drug_discovery_intent already makes
        # (which now also judges "is this a general question about a topic")
        # instead of a second round-trip — the actual answer still comes from
        # answer_general_explain_question()'s own separate, real-knowledge-
        # only system prompt (parse_drug_discovery_intent's system prompt is
        # a strict structured extractor, not an explainer, so it never
        # generates the explanation text itself).
        if intent.get("is_general_question") and intent.get("question_topic"):
            result = await asyncio.to_thread(answer_general_explain_question, req.message)
            reply = result["text"]
            session["messages"].append({"role": "agent", "text": reply})
            logger.info("[drug_discovery] converse answered (general_explain, via intent parser) | session=%s",
                        req.session_id)
            return {"reply": reply, "action": "chat", "job_id": session.get("active_job_id"), "state": session["state"]}

        # Persist every known slot (even on "chat" turns) so the next turn keeps context.
        session["known_slots"] = {
            "mode":            intent.get("mode"),
            "uniprot_id":      intent.get("uniprot_id"),
            "target_sequence": intent.get("target_sequence"),
            "ligand_smiles":   intent.get("ligand_smiles"),
            "goal_text":       intent.get("goal_text"),
            "organism_query":  intent.get("organism_query"),
            "target_name":     intent.get("target_name"),
            "pending_confirmation": intent.get("pending_confirmation", False),
            # Real bug fix: previously never persisted, so a bare numeric
            # reply ("2") to the intent-clarification menu couldn't be
            # recognized as answering it on the next turn — see the
            # resolution logic at the top of this function.
            "needs_intent_clarification": bool(intent.get("needs_intent_clarification")),
            "intent_topic": intent.get("intent_topic"),
        }
        reply = intent["reply"]
        screen_library = intent.get("mode") == "screen"
        target_known = bool(intent.get("uniprot_id") or intent.get("target_sequence"))

        job_id: str | None = None
        if intent["action"] == "start_design":
            _transition(session, ConversationState.WORKFLOW_PLANNED, req.session_id)
            job_id = str(uuid.uuid4())
            _evict_finished_jobs()
            _DRUG_DISCOVERY_STORE[job_id] = {
                "id":              job_id,
                "uniprot_id":      intent.get("uniprot_id") or "",
                "target_name":     intent.get("target_name") or "",
                "mode":            intent.get("mode"),
                "status":          "RUNNING",
                "current_step":    0,
                "total_steps":     4,
                "current_message": "대기 중",
                "timeline":        [],
                "result":          None,
                "error_message":   None,
                "created_at":      datetime.utcnow().isoformat() + "Z",
            }
            session["active_job_id"] = job_id
            session["known_slots"] = {}  # reset for the next, unrelated request
            _transition(session, ConversationState.RUNNING, req.session_id)
            _RUNNING_TASKS[job_id] = asyncio.create_task(_run_drug_discovery_task(
                job_id=job_id,
                uniprot_id=intent.get("uniprot_id") or "",
                target_sequence=intent.get("target_sequence") or "",
                ligand_smiles=intent.get("ligand_smiles") or "",
                screen_library=screen_library,
                goal_text=intent.get("goal_text") or "",
                session_id=req.session_id,
            ))
            logger.info("[drug_discovery] converse started job | job=%s session=%s mode=%s", job_id, req.session_id, intent.get("mode"))
        elif target_known:
            # Target/disease identified, but not yet enough to plan a
            # workflow (e.g. mode or ligand still missing).
            _transition(session, ConversationState.DISEASE_SELECTED, req.session_id)
        else:
            _transition(session, ConversationState.WAITING_FOR_REQUIRED_INPUT, req.session_id)

        session["messages"].append({"role": "agent", "text": reply})
        return {
            "reply": reply, "action": intent["action"], "job_id": job_id, "state": session["state"],
            "needs_vcf": bool(intent.get("needs_vcf")),
            "needs_intent_clarification": bool(intent.get("needs_intent_clarification")),
            "intent_topic": intent.get("intent_topic"),
            # Real reported bug: cancer topics' VCF upload routed into the
            # small-molecule screening pipeline against a guessed default
            # target instead of the real neoantigen/mRNA vaccine pipeline —
            # tells the frontend to route the VCF/BAM upload to
            # /design-from-bam instead of /design-from-vcf when true (see
            # drug_discovery_intent.py's _is_cancer_topic).
            "neoantigen_mode": bool(intent.get("neoantigen_mode")),
        }
    finally:
        session["processing"] = False


# ── Background task ──────────────────────────────────────────────────────────

def _mark_session_terminal(session_id: str, state: ConversationState) -> None:
    """Only /converse-originated jobs have a session to update — /design,
    /design-from-vcf, /design-from-file called directly (e.g. from MCP)
    have no session concept, so session_id is "" there and this no-ops."""
    if not session_id:
        return
    session = _DRUG_DISCOVERY_CHAT_SESSIONS.get(session_id)
    if session is not None:
        _transition(session, state, session_id)


async def _run_drug_discovery_task(
    job_id:          str,
    uniprot_id:      str,
    target_sequence: str,
    ligand_smiles:   str,
    screen_library:  bool = False,
    goal_text:       str = "",
    session_id:      str = "",
    max_candidates:  int = 0,
) -> None:
    store = _DRUG_DISCOVERY_STORE[job_id]
    t0 = time.perf_counter()
    timeout_seconds = _TIMEOUT_SECONDS_SCREEN if screen_library else _TIMEOUT_SECONDS_SINGLE

    async def _progress_cb(iteration: int, max_iterations: int, message: str) -> None:
        store["current_step"]    = iteration
        store["total_steps"]     = max_iterations
        store["current_message"] = message
        store["timeline"].append({
            "step":    iteration,
            "message": message,
            "elapsed": round(time.perf_counter() - t0, 1),
        })

    try:
        result = await asyncio.wait_for(
            run_drug_discovery_pipeline(
                uniprot_id=uniprot_id,
                target_sequence=target_sequence,
                ligand_smiles=ligand_smiles,
                screen_library=screen_library,
                goal_text=goal_text,
                job_id=job_id,
                progress_cb=_progress_cb,
                max_candidates=max_candidates,
            ),
            timeout=timeout_seconds,
        )
        # A pipeline that returns an "error" key did NOT succeed — it gave up early
        # (structure could not be resolved, receptor prep failed, so docking never
        # ran). Storing that as COMPLETED is how a broken run reached the user as a
        # green success carrying an empty result, which the model then narrated as a
        # scientific finding ("no candidates survived filtering"). Only exceptions
        # were being treated as failures; a pipeline that fails *politely*, by
        # returning its reason instead of raising, was indistinguishable from one
        # that worked.
        if isinstance(result, dict) and result.get("error"):
            logger.error("[drug_discovery] job FAILED (파이프라인이 사유를 반환) | job=%s | %s",
                         job_id, result["error"])
            store.update({
                "status": "FAILED",
                "error_message": str(result["error"])[:500],
                "result": result,
            })
            _mark_session_terminal(session_id, ConversationState.FAILED)
            return

        store.update({"status": "COMPLETED", "result": result})
        logger.info("[drug_discovery] job COMPLETE | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
        _mark_session_terminal(session_id, ConversationState.REPORT_READY)
    except asyncio.CancelledError:
        # stop_job() already set status="CANCELLED" BEFORE calling
        # Task.cancel(), so this just logs — never overwrite that with a
        # generic failure. Must re-raise so asyncio correctly records the
        # task as cancelled rather than as having returned normally.
        logger.info("[drug_discovery] job CANCELLED | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
        _mark_session_terminal(session_id, ConversationState.FAILED)
        raise
    except asyncio.TimeoutError:
        logger.error("[drug_discovery] job TIMEOUT | job=%s limit=%ds", job_id, timeout_seconds)
        store.update({
            "status": "FAILED",
            "error_message": f"파이프라인이 제한 시간({timeout_seconds}초)을 초과했습니다.",
        })
        _mark_session_terminal(session_id, ConversationState.FAILED)
    except Exception as exc:
        logger.exception("[drug_discovery] job FAILED | job=%s", job_id)
        store.update({"status": "FAILED", "error_message": str(exc)[:500]})
        _mark_session_terminal(session_id, ConversationState.FAILED)
    finally:
        _RUNNING_TASKS.pop(job_id, None)


async def _run_drug_discovery_vcf_task(job_id: str, vcf_text: str) -> None:
    store = _DRUG_DISCOVERY_STORE[job_id]
    t0 = time.perf_counter()

    async def _progress_cb(iteration: int, max_iterations: int, message: str) -> None:
        store["current_step"]    = iteration
        store["total_steps"]     = max_iterations
        store["current_message"] = message
        store["timeline"].append({
            "step":    iteration,
            "message": message,
            "elapsed": round(time.perf_counter() - t0, 1),
        })

    try:
        result = await asyncio.wait_for(
            run_drug_discovery_from_vcf(vcf_text=vcf_text, job_id=job_id, progress_cb=_progress_cb),
            timeout=_TIMEOUT_SECONDS_SCREEN,
        )
        if result.get("error"):
            logger.error("[drug_discovery_vcf] job FAILED | job=%s reason=%s", job_id, result["error"])
            store.update({"status": "FAILED", "error_message": result["error"], "result": result})
        else:
            store.update({"status": "COMPLETED", "result": result})
            logger.info("[drug_discovery_vcf] job COMPLETE | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
    except asyncio.CancelledError:
        logger.info("[drug_discovery_vcf] job CANCELLED | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
        raise
    except asyncio.TimeoutError:
        logger.error("[drug_discovery_vcf] job TIMEOUT | job=%s limit=%ds", job_id, _TIMEOUT_SECONDS_SCREEN)
        store.update({
            "status": "FAILED",
            "error_message": f"파이프라인이 제한 시간({_TIMEOUT_SECONDS_SCREEN}초)을 초과했습니다.",
        })
    except Exception as exc:
        logger.exception("[drug_discovery_vcf] job FAILED | job=%s", job_id)
        store.update({"status": "FAILED", "error_message": str(exc)[:500]})
    finally:
        _RUNNING_TASKS.pop(job_id, None)


async def _run_neoantigen_task(job_id: str, vcf_text: str, bam_path: str | None) -> None:
    store = _DRUG_DISCOVERY_STORE[job_id]
    t0 = time.perf_counter()

    async def _progress_cb(step: int, total_steps: int, message: str) -> None:
        store["current_step"]    = step
        store["total_steps"]     = total_steps
        store["current_message"] = message
        store["timeline"].append({
            "step":    step,
            "message": message,
            "elapsed": round(time.perf_counter() - t0, 1),
        })

    try:
        result = await asyncio.wait_for(
            run_neoantigen_pipeline(vcf_text=vcf_text, bam_path=bam_path, job_id=job_id, progress_cb=_progress_cb),
            timeout=_TIMEOUT_SECONDS_NEOANTIGEN,
        )
        if result.get("error"):
            logger.error("[neoantigen] job FAILED | job=%s reason=%s", job_id, result["error"])
            store.update({"status": "FAILED", "error_message": result["error"], "result": result})
        else:
            store.update({"status": "COMPLETED", "result": result})
            logger.info("[neoantigen] job COMPLETE | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
    except asyncio.CancelledError:
        logger.info("[neoantigen] job CANCELLED | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
        raise
    except asyncio.TimeoutError:
        logger.error("[neoantigen] job TIMEOUT | job=%s limit=%ds", job_id, _TIMEOUT_SECONDS_NEOANTIGEN)
        store.update({
            "status": "FAILED",
            "error_message": f"파이프라인이 제한 시간({_TIMEOUT_SECONDS_NEOANTIGEN}초)을 초과했습니다.",
        })
    except Exception as exc:
        logger.exception("[neoantigen] job FAILED | job=%s", job_id)
        store.update({"status": "FAILED", "error_message": str(exc)[:500]})
    finally:
        _RUNNING_TASKS.pop(job_id, None)


async def run_sar_optimization_job(source_job_id: str) -> dict:
    """Starts a new background SAR-optimization job (real bioisosteric analog
    generation + real re-docking) for a completed drug-discovery job's top
    candidate, returning immediately with a NEW job_id to poll via
    get_drug_discovery_job_status/GET /status/{job_id} — the result appears
    in that job's "result" field once status is COMPLETED. Used by the MCP
    run_sar_optimization tool so it doesn't block on the real re-docking
    work (previously several seconds to tens of seconds, well over
    PlayMCP's 3000ms p99 budget)."""
    source_job = _DRUG_DISCOVERY_STORE.get(source_job_id)
    if not source_job or source_job.get("status") != "COMPLETED":
        return {
            "available": False,
            "reason": "Job not found or not yet completed",
            "next_step": (
                "job_id must be a real job_id returned by predict_drug_binding "
                "(NOT a UniProt ID, gene symbol, or disease name) — call "
                "predict_drug_binding(uniprot_id=<target>, screen_library=true) "
                "first, poll get_drug_discovery_job_status(job_id) until "
                "status=COMPLETED, then retry this tool with that job_id."
            ),
        }
    job_id = str(uuid.uuid4())
    _evict_finished_jobs()
    _DRUG_DISCOVERY_STORE[job_id] = _new_job_store_entry(job_id, mode="sar_optimization")
    _RUNNING_TASKS[job_id] = asyncio.create_task(_run_sar_task(job_id, source_job.get("result") or {}))
    return {"job_id": job_id, "status": "RUNNING"}


async def _run_sar_task(job_id: str, source_result: dict) -> None:
    store = _DRUG_DISCOVERY_STORE[job_id]
    t0 = time.perf_counter()
    try:
        # Real reported gap: unlike every other job runner in this file, this one
        # had no asyncio.wait_for ceiling. The subprocess calls inside (receptor
        # prep, each analog's docking) are each individually timeout-bounded, so
        # this never hung forever — but with no top-level cap and no TIMEOUT-
        # specific status, a legitimately slow run (retry + 3 analogs) could sit
        # at RUNNING for several minutes while get_drug_discovery_job_status's own
        # docstring tells the model that's normal and to keep polling.
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_sar_optimization_service, source_result),
            timeout=_TIMEOUT_SECONDS_SAR,
        )
        store.update({"status": "COMPLETED", "result": result})
        logger.info("[sar_optimization] job COMPLETE | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
    except asyncio.CancelledError:
        logger.info("[sar_optimization] job CANCELLED | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
        raise
    except asyncio.TimeoutError:
        logger.error("[sar_optimization] job TIMEOUT | job=%s limit=%ds", job_id, _TIMEOUT_SECONDS_SAR)
        store.update({
            "status": "FAILED",
            "error_message": f"SAR 최적화가 제한 시간({_TIMEOUT_SECONDS_SAR}초)을 초과했습니다.",
        })
    except Exception as exc:
        logger.exception("[sar_optimization] job FAILED | job=%s", job_id)
        store.update({"status": "FAILED", "error_message": str(exc)[:500]})
    finally:
        _RUNNING_TASKS.pop(job_id, None)


async def _run_drug_discovery_fasta_task(job_id: str, sequence: str, goal_text: str) -> None:
    """FASTA-upload ('Track A' file gateway) background task — reuses
    run_drug_discovery_pipeline's existing target_sequence path unmodified
    (real ESMFold prediction, including its existing real length-limit
    fail-soft behavior); no new pipeline logic, just a new front door."""
    store = _DRUG_DISCOVERY_STORE[job_id]
    t0 = time.perf_counter()

    async def _progress_cb(iteration: int, max_iterations: int, message: str) -> None:
        store["current_step"]    = iteration
        store["total_steps"]     = max_iterations
        store["current_message"] = message
        store["timeline"].append({
            "step":    iteration,
            "message": message,
            "elapsed": round(time.perf_counter() - t0, 1),
        })

    try:
        result = await asyncio.wait_for(
            run_drug_discovery_pipeline(
                target_sequence=sequence,
                screen_library=True,
                goal_text=goal_text,
                job_id=job_id,
                progress_cb=_progress_cb,
            ),
            timeout=_TIMEOUT_SECONDS_SCREEN,
        )
        store.update({"status": "COMPLETED", "result": result})
        logger.info("[drug_discovery_fasta] job COMPLETE | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
    except asyncio.CancelledError:
        logger.info("[drug_discovery_fasta] job CANCELLED | job=%s elapsed=%.1fs", job_id, time.perf_counter() - t0)
        raise
    except asyncio.TimeoutError:
        logger.error("[drug_discovery_fasta] job TIMEOUT | job=%s limit=%ds", job_id, _TIMEOUT_SECONDS_SCREEN)
        store.update({
            "status": "FAILED",
            "error_message": f"파이프라인이 제한 시간({_TIMEOUT_SECONDS_SCREEN}초)을 초과했습니다.",
        })
    except Exception as exc:
        logger.exception("[drug_discovery_fasta] job FAILED | job=%s", job_id)
        store.update({"status": "FAILED", "error_message": str(exc)[:500]})
    finally:
        _RUNNING_TASKS.pop(job_id, None)
