"""
Docking Engine — Drug Discovery Assistant.

analyze_ligand() is real, deterministic cheminformatics via RDKit (SMILES
parsing, 3D conformer embedding, standard descriptors, Lipinski Rule-of-Five).

dock_ligand() runs real AutoDock Vina docking via the official compiled
Windows binary (bin/vina.exe, from ccsb-scripps/AutoDock-Vina's GitHub
releases — NOT the `vina` PyPI package, which has no Windows wheel and
fails to build from source without Boost, confirmed on this dev machine).
This mirrors the project's existing BLAST+/MAFFT subprocess-binary pattern
rather than a Python-bindings import. Ligand PDBQT prep uses `meeko`
(pure Python + RDKit). When the exe is missing, or the docking subprocess
fails for any reason, this falls back to a deterministic heuristic score
derived from molecular weight, LogP, and rotatable-bond count — the two
paths are never conflated: every result carries an explicit source field
("vina" | "heuristic") so nothing downstream can mistake one for the
other.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import re
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)

def _find_vina() -> str:
    """Same platform trap as receptor_prep_engine's: bin/vina.exe is the Windows
    development layout. A Linux deployment that ships a vina binary puts it on PATH
    under the name `vina`, so look there too rather than reporting "no Vina" on a
    machine that has one."""
    override = os.getenv("VINA_EXE_PATH")
    if override:
        return override
    on_path = shutil.which("vina")
    if on_path:
        return on_path
    return os.path.join(os.path.dirname(__file__), "..", "bin", "vina.exe")


_VINA_EXE_PATH = _find_vina()
_VINA_TIMEOUT_SECONDS = 120

# Vina defaults to using every available CPU core per invocation. That's fine
# for a single docking run, but this app can have several jobs in flight at
# once (e.g. a library screen loops through 20+ candidates one dock_ligand
# call at a time) — without a cap, two concurrent jobs fully starve each
# other instead of sharing the machine, which looked like a hang rather than
# a slowdown (observed directly: two real jobs both pinning all 8 cores).
# Capping each call to half the logical cores lets at least two jobs make
# simultaneous progress.
_VINA_CPU_LIMIT = max(1, (os.cpu_count() or 4) // 2)

_AFFINITY_ROW_RE = re.compile(r"^\s*\d+\s+(-?\d+\.?\d*)\s+")


def _vina_available() -> bool:
    return os.path.isfile(_VINA_EXE_PATH)


def analyze_ligand(smiles: str) -> dict:
    """Parse a SMILES string and compute standard drug-likeness descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "error": f"RDKit could not parse SMILES: {smiles!r}"}

    mol_h = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
    has_3d = embed_status == 0
    if has_3d:
        AllChem.MMFFOptimizeMolecule(mol_h)

    mol_wt   = Descriptors.MolWt(mol)
    log_p    = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_accept = Descriptors.NumHAcceptors(mol)
    tpsa     = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    lipinski_violations = sum([
        mol_wt   > 500,
        log_p    > 5,
        h_donors > 5,
        h_accept > 10,
    ])

    return {
        "valid":               True,
        "canonical_smiles":    Chem.MolToSmiles(mol),
        "molecular_weight":    round(mol_wt, 2),
        "logp":                round(log_p, 2),
        "h_bond_donors":       h_donors,
        "h_bond_acceptors":    h_accept,
        "tpsa":                round(tpsa, 2),
        "rotatable_bonds":     rot_bonds,
        "lipinski_violations": lipinski_violations,
        "drug_like":           lipinski_violations <= 1,
        "has_3d_conformer":    has_3d,
        "source":              "rdkit",
    }


def _heuristic_binding_score(mol_wt: float, log_p: float, rot_bonds: int) -> float:
    """
    Deterministic molecular-property proxy for binding affinity, roughly
    scaled to look like a Vina kcal/mol output (-4 to -9 range for typical
    drug-like molecules) so downstream ranking code doesn't need two
    different scales. This is NOT a physical docking simulation — no pose,
    no receptor geometry is considered. It rewards moderate size/LogP
    (favoring "drug-like" space around MW~300, LogP~2.5) and penalizes
    excess rotatable bonds (entropic cost proxy).
    """
    size_term    = -0.5 * math.log(max(mol_wt, 1.0) / 300.0 + 1.0)
    logp_term    = -0.3 * math.exp(-((log_p - 2.5) ** 2) / 8.0)
    entropy_term = 0.15 * rot_bonds
    return round(-6.0 + size_term + logp_term + entropy_term, 2)


def _prepare_ligand_pdbqt(smiles: str) -> str:
    """Embed a 3D conformer and write a PDBQT file via meeko. Returns the file path."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol_h)

    preparator = MoleculePreparation()
    setups = preparator.prepare(mol_h)
    ligand_pdbqt, _, _ = PDBQTWriterLegacy.write_string(setups[0])

    fd, path = tempfile.mkstemp(suffix=".pdbqt")
    with os.fdopen(fd, "w") as f:
        f.write(ligand_pdbqt)
    return path


def _parse_best_affinity(vina_stdout: str) -> float | None:
    """Vina prints a 'mode | affinity | rmsd...' table; mode 1 is always the best pose."""
    for line in vina_stdout.splitlines():
        match = _AFFINITY_ROW_RE.match(line)
        if match:
            return float(match.group(1))
    return None


# Real Vina table row shape (confirmed live against actual vina.exe stdout
# before writing this): "   1       -8.184          0          0" — mode,
# affinity, rmsd lower-bound, rmsd upper-bound relative to mode 1 (the best
# pose). _AFFINITY_ROW_RE only ever captured the affinity column; this
# captures all four so docking confidence can be computed from Vina's own
# already-produced output, not a fabricated number, and at zero extra cost
# (same stdout, no extra Vina call).
_MODE_ROW_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*$")


def _parse_all_modes(vina_stdout: str) -> list[dict]:
    modes = []
    for line in vina_stdout.splitlines():
        match = _MODE_ROW_RE.match(line)
        if match:
            modes.append({
                "mode": int(match.group(1)),
                "affinity_kcal_mol": float(match.group(2)),
                "rmsd_lower_bound": float(match.group(3)),
                "rmsd_upper_bound": float(match.group(4)),
            })
    return modes


def _compute_docking_confidence(modes: list[dict]) -> dict | None:
    """
    Real confidence proxy from Vina's own reported alternate binding
    modes — specific to this project's blind-docking methodology (whole-
    protein search box, no known pocket), where the real failure mode is
    Vina finding several energetically comparable poses scattered across
    completely different regions of the protein surface (confirmed live:
    a weak/promiscuous ligand against a blind box produced a top-3 mean
    RMSD of ~24 Angstroms — poses on opposite sides of the protein, not
    just conformational jitter within one site). Tight RMSD clustering
    among the top few modes despite the wide-open search box is real
    evidence of a consistently-preferred site; wide scatter is real
    evidence of ambiguity. Returns None when Vina reported fewer than 2
    modes (nothing to compare).
    """
    if len(modes) < 2:
        return None
    top_k = min(3, len(modes))
    rmsds = [modes[i]["rmsd_lower_bound"] for i in range(1, top_k)]
    mean_rmsd = sum(rmsds) / len(rmsds)
    affinity_spread = round(modes[top_k - 1]["affinity_kcal_mol"] - modes[0]["affinity_kcal_mol"], 2)
    # Disclosed linear scale, not fitted/tuned: 0A (identical pose) = 100,
    # 10A+ (a different region of the protein) = 0.
    pose_consistency_score = round(max(0.0, min(100.0, 100.0 * (1 - mean_rmsd / 10.0))), 1)
    return {
        "pose_count": len(modes),
        "mean_rmsd_top_poses_angstrom": round(mean_rmsd, 2),
        "affinity_spread_top_poses_kcal_mol": affinity_spread,
        "pose_consistency_score": pose_consistency_score,
        "method": (
            "Vina가 보고한 상위 모드 간 RMSD 기반 (0A=동일 포즈=100점, 10A+=단백질의 다른 위치=0점, "
            "선형 보간). 블라인드 도킹 특유의 '여러 후보 결합 부위' 모호성을 실측한 값이며, "
            "결합친화도 자체의 정확도를 보장하지는 않습니다."
        ),
    }


def _run_vina(
    receptor_pdbqt_path: str,
    ligand_pdbqt_path: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int = 8,
    keep_pose: bool = False,
) -> dict:
    fd, out_path = tempfile.mkstemp(suffix=".pdbqt")
    os.close(fd)
    try:
        cmd = [
            _VINA_EXE_PATH,
            "--receptor", receptor_pdbqt_path,
            "--ligand", ligand_pdbqt_path,
            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
            "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--cpu", str(_VINA_CPU_LIMIT),
            "--out", out_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_VINA_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"vina.exe exited {proc.returncode}: {proc.stderr[:300]}")
        affinity = _parse_best_affinity(proc.stdout)
        if affinity is None:
            raise RuntimeError("Could not parse affinity from vina.exe output")
        modes = _parse_all_modes(proc.stdout)
        result = {
            "best_affinity_kcal_mol": round(affinity, 3),
            "docking_confidence": _compute_docking_confidence(modes),
        }
        if keep_pose:
            # Real docked-pose atom coordinates (MODEL 1 = Vina's best pose),
            # normally discarded right after the affinity number is parsed —
            # kept only when a caller explicitly asks (e.g. for real
            # structural_analysis_engine.analyze_binding_pocket() geometry),
            # since retaining/parsing this for every one of a ~20-candidate
            # screen would be wasted cost the common ranking path doesn't need.
            with open(out_path, encoding="utf-8", errors="replace") as f:
                result["pose_pdbqt"] = f.read()
        return result
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def dock_ligand(
    receptor_pdbqt_path: str,
    ligand_smiles: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int = 8,
    keep_pose: bool = False,
) -> dict:
    """
    Dock a ligand against a prepared receptor. Runs real AutoDock Vina
    (bin/vina.exe) when present and the receptor file exists; falls back
    to a heuristic score (clearly labeled) otherwise.

    keep_pose=True additionally returns the real docked-pose PDBQT text
    (best mode) under "pose_pdbqt" — only when source=="vina" (the
    heuristic path has no real 3D pose to return). Used by
    structural_analysis_engine.analyze_binding_pocket() for real
    geometry-based contact analysis; left False by default since a
    library screen docking dozens of candidates has no use for every
    candidate's raw pose text.
    """
    ligand_info = analyze_ligand(ligand_smiles)
    if not ligand_info["valid"]:
        return {"docked": False, "error": ligand_info["error"], "source": "none"}

    if _vina_available() and os.path.isfile(receptor_pdbqt_path):
        ligand_pdbqt_path = None
        try:
            ligand_pdbqt_path = _prepare_ligand_pdbqt(ligand_smiles)
            vina_result = _run_vina(
                receptor_pdbqt_path, ligand_pdbqt_path, center, box_size, exhaustiveness, keep_pose=keep_pose,
            )
            result = {
                "docked":                 True,
                "source":                 "vina",
                "best_affinity_kcal_mol": vina_result["best_affinity_kcal_mol"],
                "docking_confidence":     vina_result.get("docking_confidence"),
                "ligand_analysis":        ligand_info,
            }
            if keep_pose:
                result["pose_pdbqt"] = vina_result.get("pose_pdbqt")
            return result
        except Exception as exc:
            logger.warning("[docking] Vina run failed, falling back to heuristic | error=%s", exc)
        finally:
            if ligand_pdbqt_path:
                try:
                    os.unlink(ligand_pdbqt_path)
                except OSError:
                    pass

    score = _heuristic_binding_score(
        ligand_info["molecular_weight"], ligand_info["logp"], ligand_info["rotatable_bonds"],
    )
    return {
        "docked":                 True,
        "source":                 "heuristic",
        "best_affinity_kcal_mol": score,
        "note":                   "AutoDock Vina unavailable (binary missing or receptor not prepared) — "
                                   "this is a deterministic molecular-property proxy, "
                                   "not a physical docking simulation.",
        "ligand_analysis":        ligand_info,
    }
