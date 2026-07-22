"""
Target Intelligence Engine — Drug Discovery Assistant (Phase 3 of the
Agentic Drug Discovery AI Platform master plan).

Real, deterministic UniProt + Reactome REST API integration (both public,
key-free) — no LLM in this module, same discipline as
compound_discovery_engine.py: this data (disease names, function text,
pathway names) is already real factual text/citations straight from the
source databases, so a narrative layer would only risk paraphrasing away
precision for no benefit.

Two real capabilities:
  - get_target_disease_associations(): real UniProt DISEASE comments
    (disease name/description/MIM cross-reference/PubMed evidence) and the
    real curated FUNCTION text — never a fabricated disease link.
  - get_target_pathways(): real Reactome pathway mapping for a UniProt
    accession — real pathway names and stable IDs, with each pathway's
    real "isInDisease" flag from Reactome itself.

Both verified live against the real APIs before this module was written
(EGFR/P00533 correctly returned real DISEASE comments citing "Lung
cancer"/MIM:211980 and real Reactome pathways like "Signaling by ERBB2").
A target with genuinely no disease/pathway data (e.g. most viral proteins,
which aren't in Reactome's human-pathway index) returns an honest empty
list, never a guessed association.
"""
from __future__ import annotations

import logging
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

_UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
_REACTOME_PATHWAYS_URL = "https://reactome.org/ContentService/data/mapping/UniProt/{uniprot_id}/pathways"

# Same convention as the other *_engine.py modules — this environment sits
# behind a proxy/firewall that can break strict TLS verification.
_VERIFY_SSL = os.getenv("NCBI_VERIFY_SSL", "false").lower() == "true"

# Both lookups used to allow 15s — longer than Kakao PlayMCP waits before killing
# the tool call, so a slow Reactome (measured at 15.1s on a lookup that normally
# takes 1.4s) meant the user got nothing at all. See services/http_budget.py.
_BUDGET_S = http_budget.DEFAULT_BUDGET_S


def get_target_disease_associations(uniprot_id: str, timeout: float = _BUDGET_S) -> dict:
    """
    Real live UniProt fetch of this entry's own curated DISEASE and
    FUNCTION comments. Returns {"diseases": [...], "function_summary": str
    | None, "error": None} on success, {"diseases": [], "function_summary":
    None, "error": ...} on failure. An entry with no DISEASE comments (most
    non-human/viral proteins) returns an empty list — real absence of
    curated disease association, not a search failure.
    """
    if not uniprot_id or not uniprot_id.strip():
        return {"diseases": [], "function_summary": None, "error": "Empty uniprot_id"}

    try:
        resp = http_budget.get(
            _UNIPROT_ENTRY_URL.format(uniprot_id=uniprot_id.strip()),
            {"fields": "cc_disease,cc_function"},
            http_budget.Budget(timeout),
        )
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return {"diseases": [], "function_summary": None, "error": f"No UniProt entry for {uniprot_id}"}
        logger.warning("[target_intelligence] UniProt request failed | uniprot=%s error=%s", uniprot_id, exc)
        return {"diseases": [], "function_summary": None, "error": f"UniProt request failed: {exc}"}
    except httpx.HTTPError as exc:
        logger.warning("[target_intelligence] UniProt request failed | uniprot=%s error=%s", uniprot_id, exc)
        return {"diseases": [], "function_summary": None, "error": f"UniProt request failed: {exc}"}
    except ValueError as exc:
        logger.warning("[target_intelligence] Could not parse UniProt response | uniprot=%s error=%s", uniprot_id, exc)
        return {"diseases": [], "function_summary": None, "error": f"Could not parse response: {exc}"}

    diseases = []
    function_summary = None
    for comment in data.get("comments", []):
        if comment.get("commentType") == "DISEASE":
            disease = comment.get("disease") or {}
            cross_ref = disease.get("diseaseCrossReference") or {}
            diseases.append({
                "name": disease.get("diseaseId"),
                "acronym": disease.get("acronym"),
                "description": disease.get("description"),
                "mim_id": cross_ref.get("id") if cross_ref.get("database") == "MIM" else None,
            })
        elif comment.get("commentType") == "FUNCTION" and function_summary is None:
            texts = comment.get("texts") or []
            if texts:
                function_summary = texts[0].get("value")

    logger.info("[target_intelligence] UniProt disease lookup | uniprot=%s diseases=%d", uniprot_id, len(diseases))
    return {"diseases": diseases, "function_summary": function_summary, "error": None}


def get_target_pathways(uniprot_id: str, species_taxon: int = 9606, timeout: float = _BUDGET_S) -> dict:
    """
    Real live Reactome pathway mapping for a UniProt accession. Returns
    {"pathways": [...], "error": None} on success (empty list is a real,
    honest "not in Reactome's index for this species" result — common for
    non-human/viral targets, not a failure), or {"pathways": [], "error":
    ...} on an actual request failure.
    """
    if not uniprot_id or not uniprot_id.strip():
        return {"pathways": [], "error": "Empty uniprot_id"}

    try:
        resp = http_budget.get(
            _REACTOME_PATHWAYS_URL.format(uniprot_id=uniprot_id.strip()),
            {"species": species_taxon},
            http_budget.Budget(timeout),
        )
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        # Reactome answers 404 for an accession it has no pathways for — a real,
        # honest absence (common for viral targets), not a failure.
        if exc.response.status_code == 404:
            return {"pathways": [], "error": None}
        logger.warning("[target_intelligence] Reactome request failed | uniprot=%s error=%s", uniprot_id, exc)
        return {"pathways": [], "error": f"Reactome request failed: {exc}"}
    except httpx.HTTPError as exc:
        logger.warning("[target_intelligence] Reactome request failed | uniprot=%s error=%s", uniprot_id, exc)
        return {"pathways": [], "error": f"Reactome request failed: {exc}"}
    except ValueError as exc:
        logger.warning("[target_intelligence] Could not parse Reactome response | uniprot=%s error=%s", uniprot_id, exc)
        return {"pathways": [], "error": f"Could not parse response: {exc}"}

    pathways = [
        {
            "name": p.get("displayName"),
            "stable_id": p.get("stId"),
            "in_disease": p.get("isInDisease", False),
            "url": f"https://reactome.org/PathwayBrowser/#/{p.get('stId')}",
        }
        for p in (data or [])
    ]
    logger.info("[target_intelligence] Reactome pathway lookup | uniprot=%s pathways=%d", uniprot_id, len(pathways))
    return {"pathways": pathways, "error": None}


# ── Target prioritization scoring ("타겟 우선순위 스코어링" roadmap item) ──
#
# A real, transparent, deterministic score combining three already-real
# signals gathered by other agents (OpenTargets disease association +
# tractability, ChEMBL known-inhibitor count) into one "how promising is
# this target, before committing to a full screen" number — same
# disclosed-formula discipline as decision_agent.py's priority score, just
# applied one step earlier in the pipeline (target selection, not candidate
# ranking). No LLM, no new network calls (reuses data the caller already
# fetched); every component is real, none inferred/guessed.

_DISEASE_SCORE_WEIGHT = 40.0    # OpenTargets max disease association score (0-1) scaled to 0-40
_TRACTABILITY_WEIGHT = 30.0     # OpenTargets small-molecule tractability flag count (capped at 5) scaled to 0-30
_INHIBITOR_WEIGHT = 30.0        # Real ChEMBL known-inhibitor count (capped at 5) scaled to 0-30
_CAP_COUNT = 5


def calculate_target_priority_score(
    opentargets_result: dict | None, known_inhibitor_count: int,
) -> dict:
    """
    opentargets_result: get_opentargets_profile()'s real output (or None/
    unavailable). known_inhibitor_count: real len(inhibitors) from
    compound_discovery_engine.search_known_inhibitors_chembl(). Returns
    {"priority_score": float, "breakdown": {...}} — every term disclosed,
    same as decision_agent.calculate_priority_score().
    """
    breakdown: dict[str, float] = {}

    max_disease_score = 0.0
    if opentargets_result and opentargets_result.get("available"):
        diseases = opentargets_result.get("diseases") or []
        if diseases:
            max_disease_score = max(d["score"] for d in diseases)
    disease_component = round(max_disease_score * _DISEASE_SCORE_WEIGHT, 1)
    breakdown["disease_association_component"] = disease_component

    tractability_count = 0
    if opentargets_result and opentargets_result.get("available"):
        tractability_count = len(opentargets_result.get("tractability_small_molecule") or [])
    tractability_component = round(min(tractability_count, _CAP_COUNT) / _CAP_COUNT * _TRACTABILITY_WEIGHT, 1)
    breakdown["tractability_component"] = tractability_component

    inhibitor_component = round(min(known_inhibitor_count, _CAP_COUNT) / _CAP_COUNT * _INHIBITOR_WEIGHT, 1)
    breakdown["known_inhibitor_component"] = inhibitor_component

    total = round(disease_component + tractability_component + inhibitor_component, 1)
    breakdown["final_priority_score"] = total
    return {"priority_score": total, "breakdown": breakdown}
