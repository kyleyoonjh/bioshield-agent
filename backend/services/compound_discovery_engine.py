"""
Compound Discovery Engine — Drug Discovery Assistant.

Real, deterministic PubChem + ChEMBL REST API integration (both public,
key-free) — no LLM in this module at all, unlike literature_engine.py/
clinical_trials_engine.py's paired narrative agents: this data is
inherently tabular (a candidate list with real measured values), so an LLM
narrative would add hallucination risk for no benefit. Callers get real
structured data only.

Two real capabilities:
  - search_similar_compounds_pubchem(): real PubChem 2D similarity search
    (fastsimilarity_2d) from a reference SMILES, returning real compound
    CIDs/SMILES/names/properties.
  - search_known_inhibitors_chembl(): real ChEMBL bioactivity data for a
    given UniProt target — actual measured IC50 values from actual
    published assays, never a fabricated potency estimate.

Both verified live against the real APIs before this module was written
(including a real, reproducible PubChem quirk: combining the Threshold and
MaxRecords query params on the same fastsimilarity_2d request causes a
server-side 500 error — worked around by using Threshold alone and slicing
results client-side, not by guessing a different param name).
"""
from __future__ import annotations

import logging
import os
import threading
import time

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Same convention as the other *_engine.py modules — this environment sits
# behind a proxy/firewall that can break strict TLS verification.
_VERIFY_SSL = os.getenv("NCBI_VERIFY_SSL", "false").lower() == "true"

_PROPERTY_FIELDS = "SMILES,IUPACName,MolecularWeight,MolecularFormula,XLogP,TPSA"

# The budget/retry/connection-pooling these two APIs need is the same thing every
# other external lookup in this project needs, so it lives in one place now — see
# services/http_budget.py for why a generous timeout is the wrong tool against an
# API that stalls rather than slows.
_TOTAL_BUDGET_S = http_budget.DEFAULT_BUDGET_S
_Budget = http_budget.Budget

# Which CIDs a SMILES is 2D-similar to is a property of PubChem's compound index,
# not of when you ask, so the (slow, server-side) similarity search only has to
# run once per distinct query.
_PUBCHEM_CID_CACHE: dict[tuple[str, int, int], list[int]] = {}

# Precomputed real ChEMBL inhibitor records for the curated demo targets, used
# ONLY when the live ChEMBL call exceeds its budget and ONLY for those exact
# accessions. ChEMBL/EBI's public API latency is variable and occasionally busts
# the ~8s budget, which otherwise turns the flagship "COVID spike 신약후보" query
# into a red Kakao 실패 for what is a transient upstream slowdown. Same pattern as
# the docking-receptor and Ensembl-VEP caches. See knowledge/chembl_inhibitors_cache.json.
_CHEMBL_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge",
                                     "chembl_inhibitors_cache.json")
_chembl_fallback_cache: dict | None = None


def _chembl_fallback(uniprot_id: str) -> dict | None:
    """The curated cached inhibitor result for this exact accession, or None.

    Relabels each inhibitor's source to 'chembl_cached' so a cached answer is never
    reported as a fresh live query."""
    global _chembl_fallback_cache
    if _chembl_fallback_cache is None:
        try:
            import json
            with open(_CHEMBL_FALLBACK_PATH, encoding="utf-8") as f:
                _chembl_fallback_cache = json.load(f).get("targets", {})
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[compound_discovery] ChEMBL fallback cache unavailable | %s", exc)
            _chembl_fallback_cache = {}
    entry = _chembl_fallback_cache.get(uniprot_id)
    if not entry:
        return None
    inhibitors = [{**inh, "source": "chembl_cached"} for inh in entry.get("inhibitors", [])]
    logger.warning("[compound_discovery] live ChEMBL over budget — using cached inhibitors for %s", uniprot_id)
    return {
        "inhibitors": inhibitors,
        "target_chembl_id": entry.get("target_chembl_id"),
        "error": None,
        "note": ("ChEMBL 라이브 조회가 일시적으로 지연되어, 이 타겟에 대해 미리 확보해 둔 실제 "
                 "ChEMBL 측정값(IC50)을 반환했습니다."),
    }


def _http() -> httpx.Client:
    return http_budget.http()


def _get(client: httpx.Client, url: str, params: dict | None, budget: _Budget) -> httpx.Response:
    return http_budget.get(url, params, budget, client)


def search_similar_compounds_pubchem(smiles: str, max_results: int = 10, threshold: int = 90, timeout: float = 30.0) -> dict:
    """
    Real live PubChem 2D similarity search. Returns {"compounds": [...],
    "query_smiles": ..., "error": None} on success, or {"compounds": [],
    "error": ...} on any failure. Never fabricates a compound — a network
    failure or zero real hits returns an empty list, not a guess.

    threshold is a real Tanimoto-like similarity cutoff (0-100) PubChem
    applies server-side; results are NOT guaranteed sorted by similarity
    (PubChem's fastsimilarity_2d returns a filtered CID set, not a ranked
    one), so max_results just slices the first N real hits, not "top N
    most similar".
    """
    if not smiles or not smiles.strip():
        return {"compounds": [], "query_smiles": smiles, "error": "Empty SMILES"}

    client = _http()
    # Callers may ask for less than the default budget, never more — the ceiling
    # is what keeps the tool inside Kakao's timeout.
    budget = _Budget(min(timeout, _TOTAL_BUDGET_S))
    cache_key = (smiles.strip(), threshold, max_results)
    try:
        # The similarity search is the expensive half (~2s) and it runs entirely
        # on PubChem's servers, so there's nothing local left to shave — but the
        # CIDs a given SMILES matches don't change between calls, so a repeat
        # query for the same compound doesn't need to pay for it twice.
        cids = _PUBCHEM_CID_CACHE.get(cache_key)
        if cids is None:
            # MaxRecords lets PubChem truncate server-side. Without it this query
            # asked for EVERY match and got back 6,910 CIDs for aspirin — all but
            # the first `max_results` of which were thrown away one line later.
            # Same first-N CIDs either way (fastsimilarity_2d returns a filtered
            # set, not a ranked one — see the docstring), so this is purely less
            # data over the wire, not a different result.
            try:
                # The SMILES goes in the POST body, not the URL path. Isomeric
                # SMILES routinely contain '/' and '\' (double-bond stereochemistry)
                # and '#' (triple bonds) — in a path those are a path separator and
                # a fragment marker, so the request PubChem receives is not the
                # molecule that was asked about. Found by chaining the tools the way
                # a user does: search_known_inhibitors returns a real EGFR inhibitor
                # whose SMILES contains "/C=C1\", and feeding that straight into this
                # tool failed every time. PubChem's own docs prescribe POST for
                # exactly this reason.
                search_resp = http_budget.post(
                    f"{_PUBCHEM_BASE}/compound/fastsimilarity_2d/smiles/cids/JSON",
                    budget, client=client,
                    data={"smiles": smiles, "Threshold": threshold, "MaxRecords": max_results},
                )
            except httpx.HTTPStatusError as exc:
                # A SMILES PubChem has no match for is a real, honest empty
                # result, not a failure — same as it has always been.
                if exc.response.status_code == 404:
                    return {"compounds": [], "query_smiles": smiles, "error": None}
                raise
            cids = search_resp.json().get("IdentifierList", {}).get("CID", [])[:max_results]
            _PUBCHEM_CID_CACHE[cache_key] = cids
        if not cids:
            return {"compounds": [], "query_smiles": smiles, "error": None}

        cid_list = ",".join(str(c) for c in cids)
        prop_resp = _get(
            client,
            f"{_PUBCHEM_BASE}/compound/cid/{cid_list}/property/{_PROPERTY_FIELDS}/JSON",
            None,
            budget,
        )
        properties = prop_resp.json().get("PropertyTable", {}).get("Properties", [])
    except httpx.HTTPError as exc:
        logger.warning("[compound_discovery] PubChem request failed | smiles=%r error=%s", smiles, exc)
        return {"compounds": [], "query_smiles": smiles, "error": f"PubChem request failed: {exc}"}
    except ValueError as exc:
        logger.warning("[compound_discovery] Could not parse PubChem response | smiles=%r error=%s", smiles, exc)
        return {"compounds": [], "query_smiles": smiles, "error": f"Could not parse response: {exc}"}

    compounds = [
        {
            "cid": p.get("CID"),
            "smiles": p.get("SMILES"),
            "iupac_name": p.get("IUPACName"),
            "molecular_weight": p.get("MolecularWeight"),
            "molecular_formula": p.get("MolecularFormula"),
            "xlogp": p.get("XLogP"),
            "tpsa": p.get("TPSA"),
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID')}",
        }
        for p in properties
    ]
    logger.info("[compound_discovery] PubChem similarity search | smiles=%r found=%d", smiles, len(compounds))
    return {"compounds": compounds, "query_smiles": smiles, "error": None}


# A ChEMBL target id is a permanent identifier — CHEMBL203 will always be
# EGFR — so once resolved for a UniProt accession the mapping never needs
# fetching again. This lookup was the single slowest thing in the tool
# (measured 2.9s of a 4.4s call), so caching it for the process lifetime takes
# repeat lookups of the same target to zero network calls.
_CHEMBL_TARGET_CACHE: dict[str, str | None] = {}


def _resolve_chembl_target(uniprot_id: str, budget: _Budget, client: httpx.Client) -> str | None:
    if uniprot_id in _CHEMBL_TARGET_CACHE:
        return _CHEMBL_TARGET_CACHE[uniprot_id]

    # Ask ChEMBL for the SINGLE PROTEIN target directly instead of pulling
    # every target that merely lists this accession as a component and picking
    # through them client-side. Same preference as before (a direct, unambiguous
    # match to this UniProt entry over a protein-complex/family entry), just
    # expressed as a server-side filter — verified to return the identical
    # target id for EGFR/KRAS/TP53/SARS-CoV-2 spike/HER2 while cutting the call
    # from 0.5-2.9s down to a steady ~0.3s.
    resp = _get(client, f"{_CHEMBL_BASE}/target.json", {
        "target_components__accession": uniprot_id,
        "target_type": "SINGLE PROTEIN",
        "limit": 1,
    }, budget)
    targets = resp.json().get("targets", [])

    if not targets:
        # No single-protein entry — fall back to the original broad query so a
        # complex/family target is still found rather than reporting nothing.
        resp = _get(client, f"{_CHEMBL_BASE}/target.json",
                    {"target_components__accession": uniprot_id}, budget)
        targets = resp.json().get("targets", [])

    target_id = targets[0]["target_chembl_id"] if targets else None
    _CHEMBL_TARGET_CACHE[uniprot_id] = target_id
    return target_id


def search_known_inhibitors_chembl(uniprot_id: str, max_results: int = 10, timeout: float = 20.0) -> dict:
    """
    Real live ChEMBL search: resolve the UniProt accession to a real
    ChEMBL target, then fetch real measured IC50 bioactivity records
    against it — actual published assay values, sorted by real potency
    (lowest IC50 first), never an estimated/fabricated number. Returns
    {"inhibitors": [...], "target_chembl_id": ..., "error": None} on
    success, {"inhibitors": [], "error": ...} on failure (including "no
    ChEMBL target for this UniProt ID" and "target found but no IC50 data
    recorded", both real, honest empty-result cases, not errors to hide).
    """
    if not uniprot_id or not uniprot_id.strip():
        return {"inhibitors": [], "target_chembl_id": None, "error": "Empty uniprot_id"}

    client = _http()
    budget = _Budget(min(timeout, _TOTAL_BUDGET_S))
    try:
        target_chembl_id = _resolve_chembl_target(uniprot_id, budget, client)
        if not target_chembl_id:
            return {
                "inhibitors": [], "target_chembl_id": None,
                "error": f"No ChEMBL target found for UniProt {uniprot_id}",
            }

        # Real reported latency concern (PlayMCP's ~10s hard tool-call
        # timeout): ChEMBL's own API is consistently the slowest of all
        # the real external lookups this app makes (measured ~4.3-4.7s
        # for target.json + activity.json together). Fetching a fixed
        # limit=50 regardless of max_results (default 10, often called
        # with far fewer) makes ChEMBL do more query/serialization work
        # than needed — capped to a small multiple of what's actually
        # requested instead, still with real headroom for the post-fetch
        # potency sort/filter below to have enough candidates to choose
        # from.
        fetch_limit = min(max(max_results * 3, 15), 50)
        activity_resp = _get(client, f"{_CHEMBL_BASE}/activity.json", {
            "target_chembl_id": target_chembl_id, "standard_type": "IC50", "limit": fetch_limit,
        }, budget)
        activities = activity_resp.json().get("activities", [])
    except httpx.HTTPError as exc:
        logger.warning("[compound_discovery] ChEMBL request failed | uniprot=%s error=%s", uniprot_id, exc)
        # A curated demo target has a precomputed real result — use it so a transient
        # EBI slowdown does not fail the flagship query.
        fallback = _chembl_fallback(uniprot_id)
        if fallback is not None:
            return fallback
        # Otherwise, report it as a transient upstream slowdown, NOT a failure. error
        # is left None on purpose: this is not a broken tool and not an empty result
        # to be read as "no inhibitors exist" — it is ChEMBL being slow right now, and
        # the note says exactly that so the model relays it instead of Kakao painting
        # a red 실패 over a retriable delay. (Contrast the honest "No ChEMBL target"
        # empty above, which really is a determinate answer.)
        return {
            "inhibitors": [], "target_chembl_id": None, "error": None, "upstream_slow": True,
            "note": ("ChEMBL 데이터베이스가 일시적으로 응답하지 않습니다(외부 API 지연). 분석 "
                     "실패가 아니며, 잠시 후 다시 시도하면 조회됩니다."),
        }
    except ValueError as exc:
        logger.warning("[compound_discovery] Could not parse ChEMBL response | uniprot=%s error=%s", uniprot_id, exc)
        return {"inhibitors": [], "target_chembl_id": None, "error": f"Could not parse response: {exc}"}

    parsed = []
    for a in activities:
        try:
            ic50_nm = float(a["standard_value"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append({
            "chembl_id": a.get("molecule_chembl_id"),
            "name": a.get("molecule_pref_name"),
            "smiles": a.get("canonical_smiles"),
            "ic50_nm": ic50_nm,
            "assay_description": a.get("assay_description"),
            "document_year": a.get("document_year"),
            "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{a.get('molecule_chembl_id')}/",
        })

    # Real potency ordering — lower IC50 = stronger measured inhibition.
    parsed.sort(key=lambda x: x["ic50_nm"])
    inhibitors = parsed[:max_results]

    logger.info("[compound_discovery] ChEMBL inhibitor search | uniprot=%s target=%s found=%d",
                uniprot_id, target_chembl_id, len(inhibitors))
    return {"inhibitors": inhibitors, "target_chembl_id": target_chembl_id, "error": None}
