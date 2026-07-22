"""
Receptor preparation for blind AutoDock Vina docking against an arbitrary
resolved target structure (AlphaFold DB / ESMFold PDB text).

Fixes a real bug in the earlier Drug Discovery MVP: dock_ligand() always
docked against the fixed refs/docking/1HVR.pdbqt (HIV-1 protease) file
regardless of what target the user actually asked about. This module
prepares a receptor from the target structure that was actually resolved
in step 1 of the pipeline, so docking results reflect the real target.

Uses meeko's mk_prepare_receptor CLI (same tool already used for the 1HVR
reference receptor's one-time prep, see refs/docking/README.md) with its
built-in --box_enveloping/--padding blind-docking box computation — the
search box is real geometry from the actual fetched/predicted structure,
not a guessed placeholder (see refs/docking/README.md for how a guessed
box previously produced meaningless near-zero affinities).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

def _find_mk_prepare_receptor() -> str:
    """
    Find meeko's receptor-prep CLI, whose name on PATH is NOT the same on every
    platform — which is the whole trap here.

    meeko (0.7.1) declares its console script as literally "mk_prepare_receptor.py",
    dot-py and all. pip honours that name verbatim on Linux, so the container gets
    /usr/local/bin/mk_prepare_receptor.py. On Windows, pip generates an .exe launcher
    and drops the suffix, so the dev machine gets Scripts/mk_prepare_receptor.exe.

    So shutil.which("mk_prepare_receptor") succeeds on Windows and returns None on
    Linux. An earlier "fix" searched only that one name and looked correct on this dev
    box while still failing in production — where prepare_receptor_from_pdb() caught
    the resulting FileNotFoundError in its own broad `except Exception`, returned
    {"prepared": False}, and left every docking job to degrade to the heuristic scorer
    with no receptor and no binding-pocket analysis. It never crashed and never logged.
    Only the container check (scripts/container_checks.py, run against the real image
    in CI) caught it.

    Search BOTH names, PATH first. .py before the bare name: that is the name meeko
    actually declares, so it is the one that exists on the deployed platform.
    """
    override = os.getenv("MK_PREPARE_RECEPTOR_PATH")
    if override:
        return override
    for name in ("mk_prepare_receptor.py", "mk_prepare_receptor"):
        on_path = shutil.which(name)
        if on_path:
            return on_path
    return os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts",
                        "mk_prepare_receptor.exe")


_MK_PREPARE_RECEPTOR_PATH = _find_mk_prepare_receptor()
_RECEPTOR_PREP_TIMEOUT_SECONDS = 60
_BOX_PADDING_ANGSTROM = 8.0

_BOX_LINE_RE = re.compile(r"^(center_x|center_y|center_z|size_x|size_y|size_z)\s*=\s*(-?\d+\.?\d*)")


def prepare_receptor_from_pdb(pdb_text: str, padding: float = _BOX_PADDING_ANGSTROM) -> dict:
    """
    Prepares a real PDBQT receptor + blind-docking box from arbitrary PDB
    text. AlphaFold DB / ESMFold predictions are apo structures (no
    co-crystallized ligand), so no HETATM stripping is needed here (unlike
    the one-time 1HVR reference receptor prep).

    Returns {"prepared": True, "pdbqt_path", "center", "box_size", "tmp_dir",
    "source": "meeko_blind_envelope"} on success, or {"prepared": False,
    "reason": ...} on any failure (never raises). Caller is responsible for
    removing "tmp_dir" once docking against this receptor is complete.
    """
    tmp_dir = tempfile.mkdtemp(prefix="receptor_prep_")
    pdb_path = os.path.join(tmp_dir, "target.pdb")
    out_basename = os.path.join(tmp_dir, "target")
    try:
        with open(pdb_path, "w") as f:
            f.write(pdb_text)

        base_cmd = [
            _MK_PREPARE_RECEPTOR_PATH,
            "--read_pdb", pdb_path,
            "-o", out_basename,
            "-p",
            "-v",
            "--box_enveloping", pdb_path,
            "--padding", str(padding),
        ]
        proc = subprocess.run(
            base_cmd, capture_output=True, text=True, timeout=_RECEPTOR_PREP_TIMEOUT_SECONDS,
        )
        pdbqt_path = out_basename + ".pdbqt"
        box_path = out_basename + ".box.txt"
        used_allow_bad_res = False
        if not os.path.isfile(pdbqt_path) or not os.path.isfile(box_path):
            # Real failure mode confirmed on ESMFold-predicted structures with
            # unusual local geometry (e.g. Influenza M2's cysteines): meeko's
            # bond/padding inference can raise on residues it can't cleanly
            # process. --allow_bad_res drops just those residues instead of
            # failing the whole receptor — retried only on failure (not by
            # default) and disclosed via "source", not silently applied.
            retry_cmd = base_cmd + ["-a"]
            proc = subprocess.run(
                retry_cmd, capture_output=True, text=True, timeout=_RECEPTOR_PREP_TIMEOUT_SECONDS,
            )
            used_allow_bad_res = True
        if not os.path.isfile(pdbqt_path) or not os.path.isfile(box_path):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {
                "prepared": False,
                "reason": f"mk_prepare_receptor did not produce expected output "
                          f"(exit {proc.returncode}): {proc.stderr[-300:]}",
            }

        box_values: dict[str, float] = {}
        with open(box_path) as f:
            for line in f:
                match = _BOX_LINE_RE.match(line.strip())
                if match:
                    box_values[match.group(1)] = float(match.group(2))
        required = {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}
        if not required.issubset(box_values):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {"prepared": False, "reason": f"Could not parse box dimensions from {box_path}"}

        if used_allow_bad_res:
            logger.warning("[receptor_prep] required --allow_bad_res (dropped residue(s) mk_prepare_receptor "
                            "could not process) — receptor is missing those residues' atoms")
        return {
            "prepared": True,
            "pdbqt_path": pdbqt_path,
            "center":    (box_values["center_x"], box_values["center_y"], box_values["center_z"]),
            "box_size":  (box_values["size_x"], box_values["size_y"], box_values["size_z"]),
            "tmp_dir":   tmp_dir,
            "source":    "meeko_blind_envelope_allow_bad_res" if used_allow_bad_res else "meeko_blind_envelope",
        }
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"prepared": False, "reason": "mk_prepare_receptor timed out"}
    except Exception as exc:
        logger.warning("[receptor_prep] failed | error=%s", exc)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"prepared": False, "reason": str(exc)}


def cleanup_receptor(prep_result: dict) -> None:
    tmp_dir = prep_result.get("tmp_dir")
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)
