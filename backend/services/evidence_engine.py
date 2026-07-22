"""
Evidence & Confidence Engine — Phase 1 interface scaffold (MASTER_PLAN v2.0).

Development stub: fetch_chembl_evidence() / fetch_drugbank_evidence() /
fetch_pubmed_evidence() below return mocked placeholder data, not real API
responses — this file exists to define the shape the real integrations
will fill in later (ChEMBL REST API, DrugBank, PubMed E-utilities). Not
wired into drug_discovery_pipeline.py or drug_report_service.py yet; no
mocked value from here reaches a user-facing report today.

ConfidenceEngine's scoring math is real and works today against whatever
these functions return — once the fetch_* functions are pointed at real
APIs, the confidence numbers become real without any change to the
combination logic.
"""
from __future__ import annotations

from typing import Any


def fetch_chembl_evidence(drug_name: str) -> dict[str, Any]:
    """Stub — TODO: replace with real ChEMBL REST API calls
    (https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name})."""
    return {
        "source": "chembl",
        "drug_name": drug_name,
        "chembl_id": None,
        "known_targets": [],
        "max_phase": None,
        "bioactivities_count": 0,
    }


def fetch_drugbank_evidence(drug_name: str) -> dict[str, Any]:
    """Stub — TODO: replace with real DrugBank API/data lookups."""
    return {
        "source": "drugbank",
        "drug_name": drug_name,
        "drugbank_id": None,
        "approval_status": None,
        "known_interactions": [],
    }


def fetch_pubmed_evidence(drug_name: str, target_name: str = "") -> dict[str, Any]:
    """Stub — TODO: replace with real PubMed E-utilities search
    (https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi)."""
    return {
        "source": "pubmed",
        "query": f"{drug_name} {target_name}".strip(),
        "article_count": 0,
        "top_articles": [],
    }


def gather_evidence(drug_name: str, target_name: str = "") -> dict[str, Any]:
    """Convenience aggregator — calls all three source stubs and returns
    them keyed by source, matching the shape ConfidenceEngine expects."""
    return {
        "chembl":   fetch_chembl_evidence(drug_name),
        "drugbank": fetch_drugbank_evidence(drug_name),
        "pubmed":   fetch_pubmed_evidence(drug_name, target_name),
    }


class ConfidenceEngine:
    """
    Combines a docking result, ADMET profile, and evidence-source lookups
    into one quantitative confidence percentage. Weights are an explicit,
    inspectable engineering choice (not learned) — same transparency
    convention as drug_ranking_engine.py's weighted affinity/drug-likeness
    formula.
    """
    _W_DOCKING  = 0.5
    _W_ADMET    = 0.3
    _W_EVIDENCE = 0.2

    def compute_confidence(self, docking_result: dict, admet_profile: dict, evidence: dict) -> dict[str, Any]:
        docking_score  = self._docking_component(docking_result)
        admet_score    = self._admet_component(admet_profile)
        evidence_score = self._evidence_component(evidence)

        confidence_pct = (
            self._W_DOCKING  * docking_score
            + self._W_ADMET    * admet_score
            + self._W_EVIDENCE * evidence_score
        )
        return {
            "confidence_pct": round(max(0.0, min(100.0, confidence_pct)), 1),
            "components": {
                "docking_score":  round(docking_score, 1),
                "admet_score":    round(admet_score, 1),
                "evidence_score": round(evidence_score, 1),
            },
            "weights": {"docking": self._W_DOCKING, "admet": self._W_ADMET, "evidence": self._W_EVIDENCE},
        }

    @staticmethod
    def _docking_component(docking_result: dict) -> float:
        affinity = (docking_result or {}).get("best_affinity_kcal_mol")
        if affinity is None:
            return 0.0
        # Same -4..-12 kcal/mol normalization drug_ranking_engine.py uses
        # for its affinity_score, so this stays consistent with the ranking
        # the researcher already saw for this candidate.
        span = 8.0
        score = (-4.0 - affinity) / span * 100.0
        return max(0.0, min(100.0, score))

    @staticmethod
    def _admet_component(admet_profile: dict) -> float:
        admet_profile = admet_profile or {}
        if not admet_profile.get("valid"):
            return 0.0
        absorption_score = (admet_profile.get("oral_absorption") or {}).get("score", 0.0)
        risk = (admet_profile.get("hepatotoxicity") or {}).get("risk", "flagged")
        risk_penalty = {"low": 0.0, "moderate": 20.0, "flagged": 45.0}.get(risk, 45.0)
        return max(0.0, min(100.0, absorption_score - risk_penalty))

    @staticmethod
    def _evidence_component(evidence: dict) -> float:
        """
        Scores against gather_evidence()'s shape. Always near-zero today
        since fetch_*_evidence() are unwired stubs (chembl_id/max_phase are
        always None) — this will start reflecting real literature/DB signal
        the moment those functions are pointed at live APIs, with zero
        change needed here.
        """
        evidence = evidence or {}
        chembl = evidence.get("chembl") or {}
        pubmed = evidence.get("pubmed") or {}
        score = 0.0
        if chembl.get("chembl_id"):
            score += 40.0
        if chembl.get("max_phase"):
            score += chembl["max_phase"] * 10.0  # e.g. phase 4 (approved) -> +40
        score += min(20.0, (pubmed.get("article_count") or 0) * 2.0)
        return max(0.0, min(100.0, score))
