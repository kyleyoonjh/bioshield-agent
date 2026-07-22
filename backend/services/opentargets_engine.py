"""
OpenTargets Engine — Drug Discovery Assistant (Target Intelligence Agent
extension, "OpenTargets 실연동" roadmap item).

Real, deterministic OpenTargets Platform GraphQL API integration (public,
key-free) — verified live before this module was written: the search
endpoint accepts a raw UniProt accession directly (e.g. "P00533") and
resolves it to the correct real Ensembl gene ID, associatedDiseases returns
real composite association scores (0-1, aggregated from genetics/
literature/expression/animal-model evidence — OpenTargets' own real
methodology, not computed here), and tractability returns real per-modality
druggability buckets (only the "SM" = Small Molecule modality is used here,
matching this project's actual small-molecule-only docking scope).

Never fabricates: a target with no OpenTargets entry (e.g. most viral
proteins, which aren't human Ensembl genes) returns an honest empty
result, not a guessed association.
"""
from __future__ import annotations

import logging
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

_OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Same convention as the other *_engine.py modules — this environment sits
# behind a proxy/firewall that can break strict TLS verification.
_VERIFY_SSL = os.getenv("NCBI_VERIFY_SSL", "false").lower() == "true"

_SEARCH_QUERY = """
query TargetSearch($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {size: 1, index: 0}) {
    hits { id entity name }
  }
}
"""

_PROFILE_QUERY = """
query TargetProfile($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {size: 5, index: 0}) {
      count
      rows { score disease { id name } }
    }
    tractability { label modality value }
  }
}
"""


def _graphql(query: str, variables: dict, budget: http_budget.Budget) -> dict | None:
    try:
        resp = http_budget.post(
            _OPENTARGETS_URL, budget, json={"query": query, "variables": variables})
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("[opentargets] request failed | error=%s", exc)
        return None
    except ValueError as exc:
        logger.warning("[opentargets] could not parse response | error=%s", exc)
        return None
    if data.get("errors"):
        logger.warning("[opentargets] GraphQL errors | %s", data["errors"])
        return None
    return data.get("data")


def get_opentargets_profile(uniprot_id: str, timeout: float = http_budget.DEFAULT_BUDGET_S) -> dict:
    """
    Real live OpenTargets lookup: UniProt accession -> real Ensembl target
    ID (via OpenTargets' own search) -> real disease associations + real
    small-molecule tractability. Returns {"available": bool, "ensembl_id",
    "target_symbol", "diseases": [{"name","score"}],
    "tractability_small_molecule": [str, ...], "error": str|None}.
    An empty "diseases"/"tractability_small_molecule" list with
    available=True is a real, honest "OpenTargets has this target but no
    hits for that field" result — never a guess.
    """
    if not uniprot_id or not uniprot_id.strip():
        return {"available": False, "error": "Empty uniprot_id"}

    # This profile costs TWO sequential GraphQL round trips (accession -> Ensembl
    # id, then id -> profile). Each used to allow 15s on its own, so a slow day at
    # OpenTargets could spend 30s on a call Kakao abandons after 10. One budget
    # now spans both hops.
    budget = http_budget.Budget(timeout)

    search_data = _graphql(_SEARCH_QUERY, {"q": uniprot_id.strip()}, budget)
    if search_data is None:
        return {"available": False, "error": "OpenTargets search request failed"}
    hits = (search_data.get("search") or {}).get("hits") or []
    if not hits:
        return {"available": False, "error": f"No OpenTargets target entry for {uniprot_id}"}
    ensembl_id = hits[0]["id"]

    profile_data = _graphql(_PROFILE_QUERY, {"ensemblId": ensembl_id}, budget)
    if profile_data is None or not profile_data.get("target"):
        return {"available": False, "error": "OpenTargets profile request failed"}
    target = profile_data["target"]

    diseases = [
        {"name": row["disease"]["name"], "score": round(row["score"], 3)}
        for row in ((target.get("associatedDiseases") or {}).get("rows") or [])
    ]
    sm_tractability = [
        t["label"] for t in (target.get("tractability") or [])
        if t.get("modality") == "SM" and t.get("value")
    ]

    logger.info("[opentargets] profile | uniprot=%s ensembl=%s diseases=%d sm_tractability=%d",
                uniprot_id, ensembl_id, len(diseases), len(sm_tractability))
    return {
        "available": True,
        "ensembl_id": ensembl_id,
        "target_symbol": target.get("approvedSymbol"),
        "diseases": diseases,
        "tractability_small_molecule": sm_tractability,
        "error": None,
    }
