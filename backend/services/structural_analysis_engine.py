"""
Structural Analysis Engine — Drug Discovery Assistant.

Two real, deterministic analyses, both computed from data the pipeline
already produces/fetches (never fabricated):

  - analyze_binding_pocket(): parses the REAL docked-pose PDBQT coordinates
    (AutoDock Vina's best mode, kept via docking_engine.dock_ligand(...,
    keep_pose=True)) and the REAL prepared receptor PDBQT, computes actual
    pairwise atom-atom distances, and reports which receptor residues are
    in contact plus distance-based hydrogen-bond/hydrophobic-contact
    candidates. This is a distance-cutoff heuristic, not a full donor-H...
    acceptor angle geometry check — disclosed explicitly via "method",
    since PDBQT's atom typing doesn't make a rigorous angle check free to
    get right, and a disclosed heuristic beats a fabricated-looking
    "3 hydrogen bonds found" number with no basis.

  - fetch_pae_matrix() / summarize_pae_for_residues(): AlphaFold DB's own
    prediction API response (already fetched by
    protein_structure_engine.fetch_known_structure()) includes a real
    "paeDocUrl" pointing at that entry's actual PAE (Predicted Aligned
    Error) JSON — this module fetches that real file and computes a real
    mean PAE, both overall and restricted to a given set of residue
    numbers (e.g. the binding-pocket residues found above). Only available
    for AlphaFold DB-sourced structures; ESMFold predictions carry no PAE
    data at all from the API used here (only per-residue pLDDT via the
    B-factor column), so callers must treat a None return as "not
    available for this target", never approximate/guess a number.
"""
from __future__ import annotations

import logging
import math
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

STRUCTURE_VERIFY_SSL = os.getenv("STRUCTURE_API_VERIFY_SSL", "false").lower() == "true"

# Distance cutoffs are standard structural-biology rules of thumb (heavy-atom
# distances, since PDBQT here has no guaranteed explicit hydrogens on every
# atom) — not tuned/fitted, and disclosed as heuristics in the returned
# "method" field rather than presented as a rigorous QM-level determination.
_HBOND_CUTOFF_A = 3.5
_HYDROPHOBIC_CUTOFF_A = 4.5
_CONTACT_CUTOFF_A = 4.5  # any-residue-in-contact reporting radius

_POLAR_ELEMENTS = {"N", "O"}
_HYDROPHOBIC_ELEMENTS = {"C"}


def _parse_pdbqt_atoms(pdbqt_text: str, only_first_model: bool = True) -> list[dict]:
    """
    Fixed-width PDB/PDBQT ATOM/HETATM column parsing (same column offsets
    already used elsewhere in this codebase for real PDB coordinate
    scanning, e.g. receptor_prep_engine's blind-box computation). Returns
    only the first MODEL block by default — for a multi-pose docked-ligand
    PDBQT, MODEL 1 is Vina's best-scoring pose, which is the only one this
    pipeline ever reports a headline affinity for.
    """
    atoms = []
    in_first_model_only = only_first_model
    seen_first_model_end = False
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            if in_first_model_only and atoms:
                seen_first_model_end = True
        if seen_first_model_end and in_first_model_only:
            break
        if line.startswith(("ATOM", "HETATM")):
            try:
                atoms.append({
                    "name":    line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain":   line[21].strip(),
                    "resnum":  int(line[22:26].strip()),
                    "x":       float(line[30:38]),
                    "y":       float(line[38:46]),
                    "z":       float(line[46:54]),
                    "element": (line[76:78].strip() or line[12:14].strip())[:1].upper(),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def _distance(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def analyze_binding_pocket(
    receptor_pdbqt_text: str,
    pose_pdbqt_text: str,
    contact_cutoff: float = _CONTACT_CUTOFF_A,
    hbond_cutoff: float = _HBOND_CUTOFF_A,
    hydrophobic_cutoff: float = _HYDROPHOBIC_CUTOFF_A,
) -> dict:
    """
    Real geometry-based contact analysis between the actual docked ligand
    pose and the actual prepared receptor. Returns {"available": False,
    "reason": ...} if either PDBQT can't be parsed into atoms (never
    fabricates a pocket for an empty/malformed input).
    """
    receptor_atoms = _parse_pdbqt_atoms(receptor_pdbqt_text, only_first_model=False)
    ligand_atoms = _parse_pdbqt_atoms(pose_pdbqt_text, only_first_model=True)
    if not receptor_atoms or not ligand_atoms:
        return {
            "available": False,
            "reason": f"Could not parse atoms (receptor={len(receptor_atoms)}, ligand_pose={len(ligand_atoms)})",
        }

    residues_in_contact: dict[tuple[str, int], float] = {}
    hydrogen_bond_candidates = []
    hydrophobic_contacts = []

    for lig_atom in ligand_atoms:
        for rec_atom in receptor_atoms:
            dist = _distance(lig_atom, rec_atom)
            if dist > contact_cutoff:
                continue
            key = (rec_atom["chain"], rec_atom["resnum"])
            residues_in_contact[key] = min(residues_in_contact.get(key, dist), dist)

            if (
                dist <= hbond_cutoff
                and lig_atom["element"] in _POLAR_ELEMENTS
                and rec_atom["element"] in _POLAR_ELEMENTS
            ):
                hydrogen_bond_candidates.append({
                    "ligand_atom": lig_atom["name"],
                    "residue": f"{rec_atom['resname']}{rec_atom['resnum']}",
                    "receptor_atom": rec_atom["name"],
                    "distance_angstrom": round(dist, 2),
                })
            elif (
                dist <= hydrophobic_cutoff
                and lig_atom["element"] in _HYDROPHOBIC_ELEMENTS
                and rec_atom["element"] in _HYDROPHOBIC_ELEMENTS
            ):
                hydrophobic_contacts.append({
                    "ligand_atom": lig_atom["name"],
                    "residue": f"{rec_atom['resname']}{rec_atom['resnum']}",
                    "receptor_atom": rec_atom["name"],
                    "distance_angstrom": round(dist, 2),
                })

    residue_list = [
        {"chain": chain, "resnum": resnum, "min_distance_angstrom": round(dist, 2)}
        for (chain, resnum), dist in sorted(residues_in_contact.items(), key=lambda kv: kv[1])
    ]

    return {
        "available": True,
        "residues_in_contact": residue_list,
        "hydrogen_bond_candidates": hydrogen_bond_candidates,
        "hydrophobic_contacts": hydrophobic_contacts,
        "method": (
            f"Distance-cutoff heuristic on real AutoDock Vina best-pose coordinates "
            f"(contact<={contact_cutoff}A, H-bond candidate: both atoms N/O and <={hbond_cutoff}A, "
            f"hydrophobic: both atoms C and <={hydrophobic_cutoff}A) — not a donor-H...acceptor angle "
            f"geometry check, so 'candidate' bonds are distance-only evidence, not confirmed H-bonds."
        ),
    }


def fetch_pae_matrix(pae_doc_url: str, timeout: float = 15.0) -> list[list[float]] | None:
    """
    Real live fetch of an AlphaFold DB entry's own PAE JSON file (the URL
    comes from that same entry's prediction-API response — never guessed/
    constructed from a URL pattern, since AlphaFold DB's versioning suffix
    has changed before and a wrong guess would silently 404 or fetch a
    stale entry). Returns None on any failure — never fabricates a matrix.
    """
    if not pae_doc_url:
        return None
    try:
        resp = http_budget.get(pae_doc_url, budget=http_budget.Budget(timeout))
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("[structural_analysis] PAE fetch failed | url=%s error=%s", pae_doc_url, exc)
        return None

    try:
        payload = data[0] if isinstance(data, list) else data
        matrix = payload["predicted_aligned_error"]
        if not matrix or not isinstance(matrix, list):
            return None
        return matrix
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("[structural_analysis] Unexpected PAE JSON shape | url=%s error=%s", pae_doc_url, exc)
        return None


def summarize_pae_for_residues(pae_matrix: list[list[float]], residue_numbers: list[int]) -> dict:
    """
    Real mean-PAE computation — overall matrix mean, plus the mean confined
    to the given residue numbers' rows/columns (e.g. binding-pocket
    residues from analyze_binding_pocket()). residue_numbers are expected
    1-indexed (AlphaFold's own per-residue convention); out-of-range
    indices are skipped rather than raising, since the pocket-residue list
    comes from a receptor that may have had residues dropped by
    --allow_bad_res (see receptor_prep_engine.py).
    """
    n = len(pae_matrix)
    all_values = [v for row in pae_matrix for v in row]
    overall_mean = round(sum(all_values) / len(all_values), 2) if all_values else None

    valid_idx = [r - 1 for r in residue_numbers if 0 <= r - 1 < n]
    pocket_values = [pae_matrix[i][j] for i in valid_idx for j in valid_idx if i != j]
    pocket_mean = round(sum(pocket_values) / len(pocket_values), 2) if pocket_values else None

    return {
        "available": True,
        "overall_mean_pae": overall_mean,
        "pocket_residue_mean_pae": pocket_mean,
        "pocket_residues_used": [r for r in residue_numbers if 0 <= r - 1 < n],
        "method": "Real AlphaFold DB PAE matrix (Angstroms), mean over all residue pairs vs. mean restricted "
                  "to the reported binding-pocket residue pairs — lower values indicate higher relative "
                  "positional confidence between those residues.",
    }
