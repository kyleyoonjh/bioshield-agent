"""
SAR Optimization Engine — Drug Discovery Assistant (Phase 9 of the
Agentic Drug Discovery AI Platform master plan).

Real, deterministic bioisosteric replacement via RDKit reaction SMARTS —
each transformation below is a well-established medicinal-chemistry
bioisostere, not an LLM guessing where to add a fluorine. Every generated
analog is a real, RDKit-sanitized molecule.

Deliberately does NOT predict an "expected effect" for any analog here —
that would just be a fabricated number wearing a scientific costume.
services/sar_optimization_service.py actually RE-DOCKS every generated
analog against the real target receptor and reports the real recomputed
affinity/ADMET, so "does this change help" is answered by real
computation, never a guess.

Each reaction SMARTS was verified against real reference molecules before
being added here (aspirin/ibuprofen -> tetrazole; ibuprofen -> CF3;
acetaminophen -> fluorophenol; caffeine correctly produces ZERO CF3
matches since its methyls are N-attached, not C-attached — confirming the
pattern discriminates correctly rather than over-matching). This also
caught a real RDKit reaction-SMARTS pitfall during development: an
unmapped context atom is silently DELETED from the product, not
preserved — the first draft of the methyl->CF3 transformation got this
wrong (produced an isolated "FC(F)F" fragment, discarding the rest of the
molecule) until both the transformed atom and its neighbor were
explicitly atom-mapped.
"""
from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

_TRANSFORMATIONS = [
    {
        "name": "carboxylic_acid_to_tetrazole",
        "reaction_smarts": "[C:1](=[O:2])[OX2H1]>>[C:1]1=[N][N]=[N][NH]1",
        "rationale": "카르복실산 -> 테트라졸: 유사한 산성도(pKa)를 유지하면서 대사 안정성과 막 투과성을 "
                     "개선하는 경우가 많은 고전적 생물학적 등가체 치환 (예: losartan의 비페닐테트라졸).",
    },
    {
        "name": "methyl_to_trifluoromethyl",
        "reaction_smarts": "[CH3;X4:1]-[#6:2]>>[C:1](F)(F)(F)[#6:2]",
        "rationale": "메틸 -> 트리플루오로메틸: 해당 위치의 산화적 대사(CYP450)를 차단해 대사 안정성을 "
                     "높이는 데 흔히 쓰이는 치환 ('fluorine scanning').",
    },
    {
        "name": "phenol_to_fluorine",
        "reaction_smarts": "[OX2H1][c:1]>>[F][c:1]",
        "rationale": "방향족 하이드록실 -> 불소: 대사되기 쉬운 페놀을 제거하고 입체적 크기는 "
                     "유지하면서 전자적 성질을 조정하는 치환.",
    },
]


def generate_analogs(smiles: str, max_per_transformation: int = 2) -> list[dict]:
    """
    Applies each real bioisosteric transformation to the given SMILES.
    Returns [{"transformation", "smiles", "rationale"}, ...] — only real,
    RDKit-sanitizable products that differ from the input. A
    transformation that doesn't structurally apply to this molecule (e.g.
    no carboxylic acid present) simply contributes no entries, never a
    placeholder or guessed structure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    original_canonical = Chem.MolToSmiles(mol)

    analogs: list[dict] = []
    for transform in _TRANSFORMATIONS:
        try:
            rxn = AllChem.ReactionFromSmarts(transform["reaction_smarts"])
            products = rxn.RunReactants((mol,))
        except Exception as exc:
            logger.warning("[sar_optimization] reaction %s failed to run | error=%s", transform["name"], exc)
            continue

        seen_smiles: set[str] = set()
        for product_tuple in products:
            product = product_tuple[0]
            try:
                Chem.SanitizeMol(product)
                product_smiles = Chem.MolToSmiles(product)
            except Exception:
                continue
            if product_smiles == original_canonical or product_smiles in seen_smiles:
                continue
            seen_smiles.add(product_smiles)
            analogs.append({
                "transformation": transform["name"],
                "smiles": product_smiles,
                "rationale": transform["rationale"],
            })
            if len(seen_smiles) >= max_per_transformation:
                break

    return analogs
