"""
Clinical Trials Engine — Drug Discovery Assistant.

Real, deterministic ClinicalTrials.gov v2 API integration (no key/auth
required, public REST endpoint) — same "never fabricate" discipline as
literature_engine.py: returns only real trials (NCT ID, title, real
overall status, real phase, real conditions/interventions, real sponsor
and dates) exactly as ClinicalTrials.gov reports them. Verified live
against the real API (response shape, field names, and the countTotal
query param were all confirmed via direct calls before writing this
module — not guessed from documentation).
"""
from __future__ import annotations

import logging
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
_FIELDS = (
    "NCTId,BriefTitle,OverallStatus,Phase,Condition,InterventionName,"
    "InterventionType,BriefSummary,LeadSponsorName,StartDate,CompletionDate"
)

# Same convention as literature_engine.py / protein_structure_engine.py —
# this environment sits behind a proxy/firewall that can break strict TLS
# verification.
_VERIFY_SSL = os.getenv("NCBI_VERIFY_SSL", "false").lower() == "true"


def _extract_trial(study: dict) -> dict | None:
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    status = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    conditions = protocol.get("conditionsModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    description = protocol.get("descriptionModule") or {}
    sponsor = ((protocol.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {})

    return {
        "nct_id": nct_id,
        "brief_title": ident.get("briefTitle"),
        "overall_status": status.get("overallStatus"),
        "phases": design.get("phases") or [],
        "conditions": conditions.get("conditions") or [],
        "interventions": [
            {"type": iv.get("type"), "name": iv.get("name")}
            for iv in (arms.get("interventions") or [])
        ],
        "brief_summary": description.get("briefSummary"),
        "lead_sponsor": sponsor.get("name"),
        "start_date": (status.get("startDateStruct") or {}).get("date"),
        "completion_date": (status.get("completionDateStruct") or {}).get("date"),
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


def search_clinical_trials(query: str, max_results: int = 5,
                           timeout: float = http_budget.DEFAULT_BUDGET_S) -> dict:
    """
    Real live ClinicalTrials.gov v2 search. Returns {"trials": [...],
    "query": ..., "total_count": ...} on success, or {"trials": [],
    "error": ...} on any failure — never raises, never fabricates a trial
    when the real API is unreachable or returns nothing.
    """
    if not query or not query.strip():
        return {"trials": [], "query": query, "total_count": 0, "error": "Empty query"}

    try:
        resp = http_budget.get(_STUDIES_URL, {
            "query.term": query, "pageSize": max_results,
            "fields": _FIELDS, "countTotal": "true",
        }, http_budget.Budget(timeout))
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("[clinical_trials] ClinicalTrials.gov request failed | query=%r error=%s", query, exc)
        return {"trials": [], "query": query, "total_count": 0, "error": f"ClinicalTrials.gov request failed: {exc}"}
    except ValueError as exc:  # JSON decode error
        logger.warning("[clinical_trials] Could not parse response | query=%r error=%s", query, exc)
        return {"trials": [], "query": query, "total_count": 0, "error": f"Could not parse response: {exc}"}

    trials = [t for t in (_extract_trial(s) for s in data.get("studies") or []) if t]
    total_count = data.get("totalCount", len(trials))

    logger.info("[clinical_trials] search | query=%r found=%d total_count=%d", query, len(trials), total_count)
    return {"trials": trials, "query": query, "total_count": total_count}
