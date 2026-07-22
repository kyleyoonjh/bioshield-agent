"""
Literature Engine — Drug Discovery Assistant.

Real, deterministic PubMed E-utilities integration (esearch -> efetch),
same base API/conventions already used by services/ncbi_service.py (the
primer-design pipeline's NCBI integration) but for literature search
(db=pubmed) rather than sequence fetch (db=nuccore) — separate module, no
shared code, since the response shape (article metadata/abstracts vs. raw
FASTA) is entirely different.

Returns real papers only: title, abstract, journal, year, authors, and PMID
exactly as PubMed reports them. Never invents a paper, PMID, or finding —
any summarization/narrative built on top of this (see
drug_discovery_literature_agent.py) is grounded only in what this module
actually fetched.
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_NCBI_TOOL = "OpenBioShield"
_NCBI_EMAIL = os.getenv("NCBI_EMAIL", "research@openbioshield.ai")
_NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Same convention as ncbi_service.py / protein_structure_engine.py — this
# environment sits behind a proxy/firewall that can break strict TLS
# verification.
_VERIFY_SSL = os.getenv("NCBI_VERIFY_SSL", "false").lower() == "true"


def _text(elem: ET.Element | None, path: str) -> str | None:
    found = elem.find(path) if elem is not None else None
    return found.text.strip() if found is not None and found.text else None


def _parse_article(article_elem: ET.Element) -> dict | None:
    citation = article_elem.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation, "PMID")
    article = citation.find("Article")
    if article is None or not pmid:
        return None

    title = _text(article, "ArticleTitle") or ""

    # A structured abstract (BACKGROUND/METHODS/RESULTS/...) has multiple
    # <AbstractText Label="..."> elements — real papers usually do; joining
    # them with their real labels preserves that structure instead of
    # silently concatenating unrelated sections.
    abstract_parts = []
    for abstract_text in article.findall("Abstract/AbstractText"):
        label = abstract_text.get("Label")
        text = "".join(abstract_text.itertext()).strip()
        if not text:
            continue
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts) if abstract_parts else None

    journal = _text(article, "Journal/Title")
    year = _text(article, "Journal/JournalIssue/PubDate/Year")
    if not year:
        # Some entries only carry a free-text MedlineDate (e.g. "2024 Jan-Feb")
        # instead of a structured Year — real PubMed data, just a different
        # real field, not something to guess a year for.
        medline_date = _text(article, "Journal/JournalIssue/PubDate/MedlineDate")
        year = (medline_date or "").split()[0] if medline_date else None

    authors = []
    for author in article.findall("AuthorList/Author"):
        last = _text(author, "LastName")
        collective = _text(author, "CollectiveName")
        if last:
            authors.append(last)
        elif collective:
            authors.append(collective)
    author_str = authors[0] + " et al." if len(authors) > 1 else (authors[0] if authors else None)

    doi = None
    for eloc in article.findall("ELocationID"):
        if eloc.get("EIdType") == "doi":
            doi = eloc.text
            break

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": author_str,
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def search_pubmed(query: str, max_results: int = 5,
                  timeout: float = http_budget.DEFAULT_BUDGET_S) -> dict:
    """
    Real live PubMed search: esearch (query -> PMIDs) then efetch (PMIDs ->
    full article metadata + abstracts). Returns {"papers": [...], "query":
    ..., "total_count": ...} on success, or {"papers": [], "error": ...} on
    any failure — never raises, never fabricates a paper when the real API
    is unreachable or returns nothing.
    """
    if not query or not query.strip():
        return {"papers": [], "query": query, "total_count": 0, "error": "Empty query"}

    common_params = {"tool": _NCBI_TOOL, "email": _NCBI_EMAIL}
    if _NCBI_API_KEY:
        common_params["api_key"] = _NCBI_API_KEY

    # esearch then efetch: two sequential round trips under one budget, so a
    # stalled NCBI can't outlast the client waiting on the tool call.
    budget = http_budget.Budget(timeout)
    try:
        search_resp = http_budget.get(_ESEARCH_URL, {
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance", **common_params,
        }, budget)
        search_data = search_resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        total_count = int(search_data.get("esearchresult", {}).get("count", 0))

        if not pmids:
            return {"papers": [], "query": query, "total_count": total_count}

        fetch_resp = http_budget.get(_EFETCH_URL, {
            "db": "pubmed", "id": ",".join(pmids), "rettype": "abstract",
            "retmode": "xml", **common_params,
        }, budget)
        root = ET.fromstring(fetch_resp.text)
    except httpx.HTTPError as exc:
        logger.warning("[literature] PubMed request failed | query=%r error=%s", query, exc)
        return {"papers": [], "query": query, "total_count": 0, "error": f"PubMed request failed: {exc}"}
    except ET.ParseError as exc:
        logger.warning("[literature] PubMed XML parse failed | query=%r error=%s", query, exc)
        return {"papers": [], "query": query, "total_count": 0, "error": f"Could not parse PubMed response: {exc}"}

    papers = []
    for article_elem in root.findall("PubmedArticle"):
        parsed = _parse_article(article_elem)
        if parsed:
            papers.append(parsed)

    logger.info("[literature] PubMed search | query=%r found=%d total_count=%d", query, len(papers), total_count)
    return {"papers": papers, "query": query, "total_count": total_count}
