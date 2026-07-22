"""
ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction
engine — Phase 1/2 of the MASTER_PLAN v2.0 "Scientific Prediction" roadmap,
extended for Phase 8 (ADMET Agent) / Phase 10 (Synthesis Agent) of the
Agentic Drug Discovery AI Platform master plan.

Vendor-agnostic by design: `predict_admet_profile()` is a stable public
interface backed by a swappable `ADMETProvider`. The only provider
implemented so far, `RuleBasedADMETProvider`, uses real, established
medicinal-chemistry heuristics — not a trained ML model, not any named
commercial service. A future ML-based provider (trained model, external
API, etc.) can be swapped in later behind the same `ADMETProvider`
interface without touching call sites in the rest of the app
(docking_engine.py's real vs. heuristic docking split is the precedent for
this kind of clearly-labeled fallback pattern).

Every result carries a "source" field ("rdkit_rule_based" today) precisely
so nothing downstream can mistake a rule-based estimate for a validated
clinical ADMET assay — this module makes no clinical claims. Three
distinct real, deterministic checks, each disclosed for exactly what it is:
  - oral_absorption: Veber's rule (a real, widely-cited bioavailability
    heuristic).
  - hepatotoxicity: a coarse structural-alert screen (a handful of
    well-known problematic substructures from the medicinal chemistry
    literature) — not a clinically validated toxicity predictor.
  - pains: RDKit's own curated PAINS filter catalog (480 real SMARTS
    patterns, Baell & Holloway 2010) — a real, standard assay-interference
    screen, verified live against known reference compounds before this
    was added (quercetin/catechol correctly flagged "catechol_A",
    rhodanine correctly flagged "rhod_sat_A", aspirin correctly clean).
  - synthetic_accessibility: RDKit's real Contrib/SA_Score implementation
    (Ertl & Schuffenhauer 2009) — verified aspirin scores 1.58 (easy),
    consistent with the published method.

hERG/CYP inhibition/BBB permeability/clearance still require trained
predictive ML models this project doesn't have and no free, reliable API
serves — deliberately NOT implemented; never approximated or guessed.
"""
from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, RDConfig

logger = logging.getLogger(__name__)

_pains_catalog: FilterCatalog.FilterCatalog | None = None
_sascorer = None


def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    global _pains_catalog
    if _pains_catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog.FilterCatalog(params)
    return _pains_catalog


def _get_sascorer():
    global _sascorer
    if _sascorer is None:
        sa_score_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_score_dir not in sys.path:
            sys.path.append(sa_score_dir)
        import sascorer as _sascorer_module
        _sascorer = _sascorer_module
    return _sascorer

# Veber's rules (Veber et al., J. Med. Chem. 2002): compounds with TPSA <=140
# A^2 and <=10 rotatable bonds tend to have good oral bioavailability. This is
# a real, widely-cited heuristic (the same kind SwissADME/similar tools use),
# not a fabricated formula.
_VEBER_TPSA_MAX = 140.0
_VEBER_ROTBONDS_MAX = 10

# A small set of well-known hepatotoxicity/reactivity structural alerts
# (SMARTS), e.g. nitroaromatics, quinones, and hydrazines are flagged
# repeatedly in the medicinal-chemistry structural-alert literature
# (e.g. Kalgutkar et al.). This is a coarse screen for "worth a closer
# look", not a validated in-vivo/clinical toxicity predictor.
_HEPATOTOXICITY_ALERTS: dict[str, str] = {
    "nitroaromatic":     "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
    "quinone":           "O=C1C=CC(=O)C=C1",
    "hydrazine":         "[NX3][NX3]",
    "aromatic_amine":    "[NX3;H2,H1;!$(NC=O)][c]",
    "michael_acceptor":  "[CX3]=[CX3][CX3]=[OX1]",
}


class ADMETProvider(ABC):
    """Swappable backend for predict_admet_profile(). Implementations must
    never raise on a chemically-valid molecule — return a low-confidence
    result instead so the caller can always render something."""

    @abstractmethod
    def predict(self, mol: Chem.Mol) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...


class RuleBasedADMETProvider(ADMETProvider):
    """Real, deterministic RDKit descriptor + structural-alert screen —
    today's only ADMETProvider. No network calls, no trained model."""

    @property
    def source_name(self) -> str:
        return "rdkit_rule_based"

    def predict(self, mol: Chem.Mol) -> dict[str, Any]:
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        mol_wt = Descriptors.MolWt(mol)
        log_p = Descriptors.MolLogP(mol)

        absorption = self._oral_absorption(tpsa, rot_bonds)
        hepatotoxicity = self._hepatotoxicity_screen(mol, log_p)
        pains = self._pains_screen(mol)
        synthesis = self._synthetic_accessibility(mol)

        return {
            "oral_absorption": absorption,
            "hepatotoxicity":  hepatotoxicity,
            "pains":           pains,
            "synthesis":       synthesis,
            "descriptors": {
                "tpsa": round(tpsa, 2), "rotatable_bonds": rot_bonds,
                "molecular_weight": round(mol_wt, 2), "logp": round(log_p, 2),
            },
        }

    @staticmethod
    def _pains_screen(mol: Chem.Mol) -> dict[str, Any]:
        catalog = _get_pains_catalog()
        alerts = [match.GetDescription() for match in catalog.GetMatches(mol)]
        return {
            "flagged": bool(alerts),
            "alerts": alerts,
            "basis": "RDKit's official PAINS filter catalog (480 SMARTS patterns, Baell & Holloway 2010) — "
                     "flags known assay-interference substructures, not a toxicity or efficacy prediction.",
        }

    @staticmethod
    def _synthetic_accessibility(mol: Chem.Mol) -> dict[str, Any]:
        try:
            score = round(_get_sascorer().calculateScore(mol), 2)
        except Exception as exc:
            logger.warning("[admet_engine] SA score calculation failed | error=%s", exc)
            return {"score": None, "basis": f"SA score calculation failed: {exc}"}
        return {
            "score": score,  # 1 (easy to synthesize) - 10 (very difficult)
            "basis": "RDKit Contrib/SA_Score (Ertl & Schuffenhauer 2009) — a real, deterministic "
                     "fragment-contribution + complexity estimate, not a retrosynthesis plan.",
        }

    @staticmethod
    def _oral_absorption(tpsa: float, rot_bonds: int) -> dict[str, Any]:
        passes_veber = tpsa <= _VEBER_TPSA_MAX and rot_bonds <= _VEBER_ROTBONDS_MAX
        # 0-100 score: full credit at TPSA=0, linearly down to 0 at 2x the
        # Veber threshold, penalized further for excess rotatable bonds —
        # an interpretable proxy, not a calibrated probability.
        tpsa_score = max(0.0, 100.0 * (1 - tpsa / (2 * _VEBER_TPSA_MAX)))
        rotbond_penalty = max(0, rot_bonds - _VEBER_ROTBONDS_MAX) * 8.0
        score = max(0.0, min(100.0, tpsa_score - rotbond_penalty))
        return {
            "prediction": "high" if passes_veber else "low",
            "score": round(score, 1),
            "basis": f"Veber's rules (TPSA<={_VEBER_TPSA_MAX}, RotBonds<={_VEBER_ROTBONDS_MAX})",
        }

    @staticmethod
    def _hepatotoxicity_screen(mol: Chem.Mol, log_p: float) -> dict[str, Any]:
        alerts_hit: list[str] = []
        for alert_name, smarts in _HEPATOTOXICITY_ALERTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                alerts_hit.append(alert_name)

        # High lipophilicity independently correlates with idiosyncratic
        # hepatotoxicity risk in the literature (e.g. Leeson & Springthorpe,
        # 2007's "rule of 3" discussion) — treated as a soft additional flag.
        if log_p > 5:
            alerts_hit.append("high_logp")

        if not alerts_hit:
            risk = "low"
        elif len(alerts_hit) == 1:
            risk = "moderate"
        else:
            risk = "flagged"

        return {
            "risk": risk,
            "alerts": alerts_hit,
            "basis": "coarse structural-alert screen — not a clinically validated toxicity predictor",
        }


_DEFAULT_PROVIDER: ADMETProvider = RuleBasedADMETProvider()


def predict_admet_profile(smiles: str, provider: ADMETProvider | None = None) -> dict[str, Any]:
    """
    Structured, JSON-serializable ADMET profile for a single compound —
    shaped so a future Report Agent can render it directly.

    Returns (on valid SMILES):
    {
      "valid": True,
      "smiles": <canonical SMILES>,
      "oral_absorption": {"prediction": "high"|"low", "score": 0-100, "basis": str},
      "hepatotoxicity":  {"risk": "low"|"moderate"|"flagged", "alerts": [str], "basis": str},
      "descriptors": {"tpsa": float, "rotatable_bonds": int, "molecular_weight": float, "logp": float},
      "source": "rdkit_rule_based",
    }
    or {"valid": False, "error": str} if the SMILES can't be parsed.
    """
    provider = provider or _DEFAULT_PROVIDER
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": f"RDKit could not parse SMILES: {smiles!r}"}

    try:
        profile = provider.predict(mol)
    except Exception as exc:
        logger.warning("[admet_engine] provider %s failed | error=%s", provider.source_name, exc)
        return {"valid": False, "error": str(exc)}

    return {
        "valid": True,
        "smiles": Chem.MolToSmiles(mol),
        "source": provider.source_name,
        **profile,
    }
