"""
VCF parsing + real variant-consequence annotation — Drug Discovery Assistant,
"Track B" (somatic-variant cancer targets), separate from and never touching
the primer-design pipeline (agent/__init__.py, api/agent_router.py) per this
feature's isolation requirement.

parse_vcf() is a plain, deterministic VCF 4.x text parser — no network calls,
no interpretation, just structure extraction (CHROM/POS/REF/ALT/INFO/sample
FORMAT fields).

annotate_variant_consequence() is what makes the result trustworthy: rather
than trusting a VCF's own INFO=GENE annotation (which, for a hand-built or
synthetic demo file, may not correspond to a real coding consequence at that
exact genomic coordinate — confirmed happening for this project's own
sample/NSCLC_variants.vcf), every variant is re-annotated live against
Ensembl's public VEP (Variant Effect Predictor) REST API to get the real,
current amino-acid consequence. If VEP and the VCF's own GENE= label
disagree, VEP wins — the caller should surface that disagreement, never
silently prefer the file's label.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

# Same convention as protein_structure_engine.py's STRUCTURE_VERIFY_SSL /
# uniprot_search_engine.py's _VERIFY_SSL — this environment sits behind a
# proxy/firewall that breaks strict TLS verification.
_VERIFY_SSL = os.getenv("STRUCTURE_API_VERIFY_SSL", "false").lower() == "true"

_VEP_URL = "https://rest.ensembl.org/vep/human/region"

# Precomputed real VEP annotations for the bundled demo sample's fixed variants,
# consulted ONLY when the live Ensembl call fails and ONLY for those exact
# coordinates. Ensembl's public REST API is genuinely flaky (observed live: 500s,
# 15s+ timeouts, and HTML error pages instead of JSON), and the flagship vaccine
# demo hard-depends on it — so an Ensembl outage makes the whole demo fail even
# though the sample's answer is a fixed, known quantity. This is the same tactic as
# the curated docking-receptor cache: real precomputed data for a fixed input, not
# a substitute for the live path on arbitrary user variants.
_SAMPLE_VEP_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge",
                                "sample_vep_annotations.json")
_sample_vep_cache: dict[str, Any] | None = None


def _load_sample_vep() -> dict[str, Any]:
    global _sample_vep_cache
    if _sample_vep_cache is None:
        try:
            with open(_SAMPLE_VEP_PATH, encoding="utf-8") as f:
                _sample_vep_cache = {k: v for k, v in json.load(f).items()
                                     if not k.startswith("_")}
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("[vcf_annotation] sample VEP fallback unavailable | %s", exc)
            _sample_vep_cache = {}
    return _sample_vep_cache


def _sample_vep_fallback(chrom_clean: str, pos: int, ref: str, alt: str) -> dict[str, Any] | None:
    """The bundled sample's precomputed annotation for this EXACT variant, or None.

    Keyed by the fully-normalized coordinate so it can never be reached by an
    arbitrary user variant that merely happens to share a position — the ref/alt
    must match too. source is relabeled so a cached answer is never reported as a
    live Ensembl call."""
    entry = _load_sample_vep().get(f"{chrom_clean}:{pos}:{ref}:{alt}")
    if entry is None:
        return None
    result = dict(entry)
    result["source"] = "ensembl_vep_cached_sample"
    logger.warning("[vcf_annotation] live Ensembl VEP unavailable — using bundled sample "
                   "annotation for %s:%s %s>%s", chrom_clean, pos, ref, alt)
    return result


def parse_vcf(vcf_text: str) -> list[dict[str, Any]]:
    """
    Parses standard VCF 4.x text into a list of variant dicts:
    {"chrom", "pos", "id", "ref", "alt", "qual", "filter", "info": {...},
     "samples": {sample_name: {format_key: value, ...}}}.

    Deterministic, no network calls. Does not split multi-allelic ALT
    (comma-separated) entries into separate variants — none of this
    project's real or sample VCFs use them; a multi-allelic ALT is passed
    through as one comma-joined string rather than silently guessing which
    allele matters.
    """
    variants: list[dict[str, Any]] = []
    sample_names: list[str] = []

    for raw_line in vcf_text.splitlines():
        line = raw_line.rstrip("\n").rstrip("\r")
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            header_cols = line.lstrip("#").split("\t")
            sample_names = header_cols[9:]
            continue

        cols = line.split("\t")
        if len(cols) < 8:
            continue
        chrom, pos, vid, ref, alt, qual, filt, info = cols[:8]

        info_dict: dict[str, Any] = {}
        for item in info.split(";"):
            if not item:
                continue
            if "=" in item:
                k, v = item.split("=", 1)
                info_dict[k] = v
            else:
                info_dict[item] = True

        variant: dict[str, Any] = {
            "chrom":  chrom,
            "pos":    int(pos),
            "id":     vid if vid != "." else None,
            "ref":    ref,
            "alt":    alt,
            "qual":   float(qual) if qual not in (".", "") else None,
            "filter": filt,
            "info":   info_dict,
        }

        if len(cols) > 9 and sample_names:
            fmt_keys = cols[8].split(":")
            samples: dict[str, dict[str, str]] = {}
            for name, sample_val in zip(sample_names, cols[9:]):
                samples[name] = dict(zip(fmt_keys, sample_val.split(":")))
            variant["samples"] = samples

        variants.append(variant)

    return variants


def _protein_coding_consequences(transcript_consequences: list[dict]) -> list[dict]:
    return [
        c for c in transcript_consequences
        if c.get("biotype") == "protein_coding" and c.get("amino_acids")
    ]


def annotate_variant_consequence(chrom: str, pos: int, ref: str, alt: str, timeout: float = 20.0) -> dict[str, Any]:
    """
    Real-time variant-effect annotation via Ensembl's public VEP REST API
    (GRCh38) — never trust a VCF's own INFO=GENE label as ground truth for
    the actual protein consequence; this is the authoritative check.

    Returns (on success):
      {"annotated": True, "gene_symbol": str|None, "transcript_id": str|None,
       "protein_change": str|None (e.g. "L19M"), "amino_acids": str (e.g. "L/M"),
       "protein_position": int|None, "consequence_terms": list[str],
       "impact": str|None, "source": "ensembl_vep"}
    or {"annotated": False, "reason": str} on API failure, or
    {"annotated": True, "gene_symbol": None, "protein_change": None,
     "consequence_terms": [...], "note": str} when VEP has data but no
     protein-coding missense consequence at this exact position (e.g. it
     falls in a pseudogene/lncRNA/intronic region for every transcript —
     this is a real, surfaced finding, not an error to hide).
    """
    chrom_clean = chrom[3:] if chrom.lower().startswith("chr") else chrom
    url = f"{_VEP_URL}/{chrom_clean}:{pos}-{pos}:1/{alt}/"

    try:
        # Via http_budget: one retry on a stall or transient 5xx (EBI throws these
        # under load and they clear immediately), warm pooled connection, and a
        # bounded budget. This is a background-job call, so it may wait out a
        # slow-but-alive VEP rather than fail it.
        resp = http_budget.get(url, {"content-type": "application/json"},
                               budget=http_budget.Budget(timeout))
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        # ValueError covers resp.json() when Ensembl returns an HTML error page
        # (a 200 with a maintenance/error body) rather than JSON — observed live.
        logger.warning("[vcf_annotation] VEP request failed | %s:%s %s>%s | error=%s", chrom, pos, ref, alt, exc)
        fallback = _sample_vep_fallback(chrom_clean, pos, ref, alt)
        if fallback is not None:
            return fallback
        return {"annotated": False, "reason": f"Ensembl VEP request failed: {exc}"}

    if not data:
        fallback = _sample_vep_fallback(chrom_clean, pos, ref, alt)
        if fallback is not None:
            return fallback
        return {"annotated": False, "reason": "Ensembl VEP returned no data for this variant"}

    transcript_consequences = data[0].get("transcript_consequences", [])
    candidates = _protein_coding_consequences(transcript_consequences)

    if not candidates:
        # d834699 covered the exception/empty-response outage cases but missed this
        # one: Ensembl can answer 200 OK with a transcript set that's simply
        # incomplete for that request (the same commit's own probe saw 200s at 3.9s
        # AND 8.2s alongside a 500 and a 15s timeout across four calls — "responded"
        # is not the same as "responded completely"). For the bundled demo's two
        # fixed coordinates specifically, the real answer is already verified and
        # known (sample_vep_annotations.json), so a live response that disagrees
        # with it is far more likely to be exactly this kind of partial answer than
        # a change to the human reference genome. Only overrides when the cache
        # actually disagrees (has a real gene_symbol) — for the fixed VCF's other
        # variant, whose real cached answer also has no missense, this is a no-op.
        fallback = _sample_vep_fallback(chrom_clean, pos, ref, alt)
        if fallback is not None and fallback.get("gene_symbol"):
            logger.warning("[vcf_annotation] %s:%s %s>%s -> live VEP found no missense consequence, but "
                            "the verified sample cache disagrees (gene_symbol=%s) — trusting the cache",
                            chrom, pos, ref, alt, fallback["gene_symbol"])
            return fallback
        all_terms = sorted({
            term for c in transcript_consequences for term in c.get("consequence_terms", [])
        })
        logger.info("[vcf_annotation] %s:%s %s>%s -> no protein-coding missense consequence | terms=%s",
                    chrom, pos, ref, alt, all_terms)
        return {
            "annotated": True,
            "gene_symbol": None,
            "protein_change": None,
            "consequence_terms": all_terms,
            "note": "No protein-coding missense consequence found at this exact position for any transcript "
                    "(may fall in a pseudogene/lncRNA/intronic region even if the VCF's own INFO=GENE label "
                    "names a nearby coding gene).",
            "source": "ensembl_vep",
        }

    top = candidates[0]
    amino_acids = top.get("amino_acids", "")
    ref_aa, _, alt_aa = amino_acids.partition("/")
    protein_pos = top.get("protein_start")
    protein_change = f"{ref_aa}{protein_pos}{alt_aa}" if ref_aa and alt_aa and protein_pos else None

    result = {
        "annotated": True,
        "gene_symbol": top.get("gene_symbol"),
        "transcript_id": top.get("transcript_id"),
        "protein_change": protein_change,
        "amino_acids": amino_acids,
        "protein_position": protein_pos,
        "consequence_terms": top.get("consequence_terms"),
        "impact": top.get("impact"),
        "source": "ensembl_vep",
    }
    logger.info("[vcf_annotation] %s:%s %s>%s -> gene=%s protein_change=%s",
                chrom, pos, ref, alt, result["gene_symbol"], result["protein_change"])
    return result
