"""
UniProt live search/verification — Drug Discovery Assistant.

Lets the assistant go beyond the small curated knowledge/target_synonyms.json
list (originally just 6 hand-verified SARS-CoV-2 proteins) to essentially
any pathogen/organism with a UniProt entry, without hardcoding a bigger
synonym dictionary: a real, live REST query against UniProtKB's public
search API (https://rest.uniprot.org), same key-free-API convention as
protein_structure_engine.py's AlphaFold DB / ESM Atlas calls.

Two entry points:
  - search_reviewed_proteins(): free-text search for candidate target
    proteins when the user only named a disease/pathogen, not a specific
    protein or UniProt ID. Returns real, verifiable candidates (never a
    single silently-guessed "best" answer) — the caller decides whether to
    present choices or take the top hit.
  - verify_uniprot_id_exists(): confirms a UniProt ID some other step
    (regex extraction, LLM extraction) claims is real actually exists,
    before the pipeline trusts it — the same "never trust an unvalidated
    LLM claim" discipline drug_discovery_intent.py already applies to
    LLM-supplied SMILES via RDKit validation.
"""
from __future__ import annotations

import logging
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

# Same convention as protein_structure_engine.py's STRUCTURE_VERIFY_SSL —
# this environment sits behind a proxy/firewall that breaks strict TLS.
_VERIFY_SSL = os.getenv("STRUCTURE_API_VERIFY_SSL", "false").lower() == "true"

_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
_FIELDS = "accession,protein_name,organism_name,gene_names,annotation_score"


def _extract_protein_name(entry: dict) -> str | None:
    desc = entry.get("proteinDescription") or {}
    recommended = desc.get("recommendedName") or {}
    full_name = (recommended.get("fullName") or {}).get("value")
    if full_name:
        return full_name
    submitted = desc.get("submissionNames") or []
    if submitted:
        return (submitted[0].get("fullName") or {}).get("value")
    return None


def search_reviewed_proteins(query_text: str, limit: int = 5, timeout: float = 15.0) -> list[dict]:
    """
    Free-text search against UniProtKB, restricted to reviewed (Swiss-Prot)
    entries and sorted by UniProt's own annotation_score (data-quality/
    completeness signal, not a relevance guess we invent). query_text is
    typically a pathogen/organism name (e.g. "Mycobacterium tuberculosis"
    or "influenza A virus") extracted from the user's message.

    Returns a list of real candidates (never fabricated):
    [{"uniprot_id", "protein_name", "organism", "gene_name", "annotation_score"}, ...]
    or [] on no matches / API failure — callers must not silently default
    a target when this returns empty; that decision belongs to the caller.
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return []

    params = {
        "query": f"{query_text} AND reviewed:true",
        "fields": _FIELDS,
        "format": "json",
        "size": limit,
        "sort": "annotation_score desc",
    }
    try:
        resp = http_budget.get(_SEARCH_URL, params, budget=http_budget.Budget(timeout))
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("[uniprot_search] search failed | query=%r error=%s", query_text, exc)
        return []

    results = []
    for entry in data.get("results", []):
        organism = (entry.get("organism") or {}).get("scientificName")
        gene_names = entry.get("genes") or []
        gene_name = None
        if gene_names:
            gene_name = (gene_names[0].get("geneName") or {}).get("value")
        results.append({
            "uniprot_id":       entry.get("primaryAccession"),
            "protein_name":     _extract_protein_name(entry),
            "organism":         organism,
            "gene_name":        gene_name,
            "annotation_score": entry.get("annotationScore"),
        })
    logger.info("[uniprot_search] search | query=%r -> %d candidate(s)", query_text, len(results))
    return results


def verify_uniprot_id_exists(uniprot_id: str, timeout: float = 10.0) -> dict | None:
    """
    Confirms uniprot_id is a real, resolvable UniProtKB entry — used to
    validate any ID a less-reliable step (regex match, LLM extraction)
    claims is real before the pipeline trusts it. Returns
    {"uniprot_id", "protein_name", "organism"} if real, else None (either
    a genuine 404 or a network failure — both mean "cannot confirm this
    ID is real", so callers should treat them the same: don't proceed as
    if it were validated).
    """
    uniprot_id = (uniprot_id or "").strip()
    if not uniprot_id:
        return None

    try:
        with httpx.Client(timeout=timeout, verify=_VERIFY_SSL) as client:
            resp = client.get(_ENTRY_URL.format(uniprot_id=uniprot_id))
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            entry = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("[uniprot_search] ID verification failed | uniprot_id=%s error=%s", uniprot_id, exc)
        return None

    return {
        "uniprot_id":   uniprot_id,
        "protein_name": _extract_protein_name(entry),
        "organism":     (entry.get("organism") or {}).get("scientificName"),
    }
