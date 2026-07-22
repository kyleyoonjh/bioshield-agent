"""
Protein Structure Engine — Drug Discovery Assistant.

Two public, key-free APIs, no local model weights (no GPU in this
environment, matching the constraint documented in MASTER_PLAN.md):

  - EBI AlphaFold DB: pre-computed structures for known UniProt entries.
  - ESM Atlas (esmatlas.com): on-demand folding for arbitrary sequences
    via the ESMFold model, hosted by Meta.

Both return the same normalized envelope so callers never need to branch
on which source served the structure. Network/API failures are returned
as a result dict (source="unavailable"), never raised — this mirrors
ncbi_service.py's fallback convention so the pipeline can decide what to
do next instead of crashing.

Synchronous (httpx.Client, not AsyncClient) to match every other
registry/tool_registry.py entry — router_core.route() is a plain sync
call, dispatched via asyncio.to_thread by the pipeline orchestrators when
blocking work needs to happen off the event loop.
"""
from __future__ import annotations

import logging
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

# Same convention as ncbi_service.py's NCBI_VERIFY_SSL — this environment
# sits behind a proxy/firewall that can break strict TLS verification.
STRUCTURE_VERIFY_SSL = os.getenv("STRUCTURE_API_VERIFY_SSL", "false").lower() == "true"

_ALPHAFOLD_DB_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
_ESMFOLD_URL      = "https://api.esmatlas.com/foldSequence/v1/pdb/"
_ESMFOLD_MAX_AA   = 400  # ESM Atlas's own single-sequence length limit
_UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"


def fetch_uniprot_sequence(uniprot_id: str, timeout: float = 15.0) -> str | None:
    """
    Real live fetch of a UniProt entry's amino-acid sequence — used when a
    curated/requested uniprot_id has no AlphaFold DB structure (confirmed
    real case: Influenza M2 proton channel, P06821) so the caller can fall
    back to real ESMFold prediction instead of just giving up. Returns None
    on any failure (never raises).
    """
    try:
        resp = http_budget.get(_UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id.strip()),
                               budget=http_budget.Budget(timeout))
        fasta_text = resp.text
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        logger.warning("[protein_structure] UniProt sequence fetch failed | uniprot=%s error=%s", uniprot_id, exc)
        return None
    except httpx.HTTPError as exc:
        logger.warning("[protein_structure] UniProt sequence fetch failed | uniprot=%s error=%s", uniprot_id, exc)
        return None

    lines = [l for l in fasta_text.splitlines() if l and not l.startswith(">")]
    sequence = "".join(lines)
    return sequence or None


def _empty_result(source: str, reason: str) -> dict:
    return {"pdb_text": None, "source": source, "confidence": None, "reason": reason}


def fetch_known_structure(uniprot_id: str, timeout: float = 15.0) -> dict:
    """Look up a pre-computed AlphaFold structure for a known UniProt ID."""
    url = _ALPHAFOLD_DB_URL.format(uniprot_id=uniprot_id.strip())
    try:
        resp = http_budget.get(url, budget=http_budget.Budget(timeout))
        entries = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            # Not an error — this target simply has no AlphaFold entry; the caller
            # falls back to ESMFold from the UniProt sequence.
            return _empty_result("unavailable", f"No AlphaFold DB entry for {uniprot_id!r}")
        logger.warning("[protein_structure] AlphaFold DB lookup failed | uniprot=%s error=%s", uniprot_id, exc)
        return _empty_result("unavailable", f"AlphaFold DB request failed: {exc}")
    except httpx.HTTPError as exc:
        logger.warning("[protein_structure] AlphaFold DB lookup failed | uniprot=%s error=%s", uniprot_id, exc)
        return _empty_result("unavailable", f"AlphaFold DB request failed: {exc}")

    if not entries:
        return _empty_result("unavailable", f"Empty AlphaFold DB response for {uniprot_id!r}")

    entry = entries[0]
    pdb_url = entry.get("pdbUrl")
    if not pdb_url:
        return _empty_result("unavailable", "AlphaFold DB entry missing pdbUrl")

    try:
        pdb_resp = http_budget.get(pdb_url, budget=http_budget.Budget(timeout))
        pdb_text = pdb_resp.text
    except httpx.HTTPError as exc:
        logger.warning("[protein_structure] AlphaFold PDB download failed | url=%s error=%s", pdb_url, exc)
        return _empty_result("unavailable", f"PDB file download failed: {exc}")

    return {
        "pdb_text":   pdb_text,
        "source":     "alphafold_db",
        "confidence": entry.get("globalMetricValue"),  # mean pLDDT
        "reason":     None,
        "uniprot_id": uniprot_id,
        "model_created_date": entry.get("modelCreatedDate"),
        # Real per-entry PAE (Predicted Aligned Error) JSON URL, straight
        # from this same AlphaFold DB response — never guessed from a URL
        # pattern (see structural_analysis_engine.fetch_pae_matrix()).
        # None for ESMFold-predicted structures, which never go through
        # this function at all.
        "pae_doc_url": entry.get("paeDocUrl"),
    }


def predict_structure_esmfold(sequence: str, timeout: float = 60.0) -> dict:
    """Fold an arbitrary protein sequence via the public ESM Atlas API."""
    seq = "".join(sequence.split()).upper()
    if len(seq) > _ESMFOLD_MAX_AA:
        return _empty_result(
            "unavailable",
            f"Sequence length {len(seq)}aa exceeds ESM Atlas's {_ESMFOLD_MAX_AA}aa single-sequence limit",
        )
    if not seq:
        return _empty_result("unavailable", "Empty sequence")

    try:
        with httpx.Client(timeout=timeout, verify=STRUCTURE_VERIFY_SSL) as client:
            resp = client.post(_ESMFOLD_URL, content=seq)
            resp.raise_for_status()
            pdb_text = resp.text
    except httpx.HTTPError as exc:
        logger.warning("[protein_structure] ESMFold request failed | len=%d error=%s", len(seq), exc)
        return _empty_result("unavailable", f"ESMFold request failed: {exc}")

    if not pdb_text or not pdb_text.strip().startswith(("ATOM", "HEADER", "MODEL")):
        return _empty_result("unavailable", "ESMFold returned a non-PDB response")

    return {
        "pdb_text":   pdb_text,
        "source":     "esmfold_api",
        "confidence": None,  # per-residue pLDDT is embedded in the B-factor column, not summarized here
        "reason":     None,
        "sequence_length": len(seq),
    }
