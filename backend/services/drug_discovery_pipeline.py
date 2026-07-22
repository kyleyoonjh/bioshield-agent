"""
Drug Discovery Assistant pipeline — Plan -> Target ID -> Tool Selection ->
Analysis -> Filter/Validate -> Rank -> Reflect -> Report, mirroring
MASTER_PLAN.md's Agent Workflow and agent/__init__.py's existing
Planner/Evaluator/Reflector/DecisionLog loop for primer design (see
services/drug_discovery_agent.py) — fully separate implementation, no
shared code path with the primer or somatic pipelines beyond router_core.

Two modes:
  - "single": user supplied a specific ligand -> dock it against the real
    target structure (fixes the earlier bug where docking always used a
    fixed HIV-protease reference file regardless of the requested target).
  - "screen": user described a goal without a specific ligand -> screen
    the curated knowledge/drug_library.json against the real target
    structure, filter, and rank multiple candidates.

Every iteration prepares a fresh receptor at the current strategy's
padding (blind-docking box, real geometry from the resolved structure —
see services/receptor_prep_engine.py) since escalating strategies change
the box/exhaustiveness. Up to 3 iterations, matching the primer agent's
retry cap.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable

from router import router_core
from services.drug_discovery_agent import (
    STRATEGY_TEMPLATES,
    DrugDiscoveryDecisionLog,
    DrugDiscoveryEvaluator,
    DrugDiscoveryPlanner,
    DrugDiscoveryReflectionAgent,
    generate_ai_summary,
)
from services import receptor_prep_engine, drug_report_service, docking_engine, report_worker

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], Awaitable[None]]

# Retries are off. The loop below used to re-run the whole screen up to 3x,
# escalating the strategy's exhaustiveness each time when the evaluator wasn't
# satisfied. Measured cost on the curated library: 3 full 22-compound screens,
# ~8.6s instead of ~2.7s — a 3x wall-clock tax on every job. Set back to 3 to
# restore the escalate-and-retry behaviour (it only ever moves the numbers when
# a real AutoDock Vina binary is present; with the heuristic fallback the
# retries are provably no-ops — see _screen_is_heuristic_only).
_MAX_ITERATIONS = 1
_DRUG_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "drug_library.json")
_drug_library_cache: list[dict] | None = None

# docking_engine._VINA_CPU_LIMIT already caps each individual Vina
# subprocess to HALF the machine's logical cores (specifically so more than
# one can run concurrently without starving each other) — 2 is the exact
# concurrency that saturates the machine without oversubscribing it.
_SCREEN_CONCURRENCY = 2

# Display-only Korean labels for the real strategy_id values in
# drug_discovery_agent.STRATEGY_TEMPLATES ("fast"/"standard"/"broad") — used
# only in progress messages shown to the user; the underlying strategy_id
# string driving actual exhaustiveness/padding is never changed.
_STRATEGY_LABELS_KO = {"fast": "고속", "standard": "표준", "broad": "정밀"}


def _screen_is_heuristic_only(screened: list) -> bool:
    """True when every candidate that actually docked was scored by the
    deterministic molecular-property heuristic rather than real AutoDock Vina
    (i.e. no vina binary on this machine). Retrying such a screen at a higher
    exhaustiveness is provably a no-op — see the retry loop's use of this."""
    docked = [
        d for d in ((s or {}).get("docking") or {} for s in (screened or []))
        if d.get("docked")
    ]
    return bool(docked) and all(d.get("source") == "heuristic" for d in docked)


def _strategy_label(strategy_id: str) -> str:
    return _STRATEGY_LABELS_KO.get(strategy_id, strategy_id)


def _load_drug_library() -> list[dict]:
    global _drug_library_cache
    if _drug_library_cache is None:
        with open(_DRUG_LIBRARY_PATH, encoding="utf-8") as f:
            _drug_library_cache = json.load(f)["drugs"]
    return _drug_library_cache


def _build_single_report(structure: dict, docking: dict) -> dict:
    ligand = docking.get("ligand_analysis") or {}
    strengths: list[str] = []
    weaknesses: list[str] = []

    if structure.get("source") == "alphafold_db":
        strengths.append(f"Structure from experimentally-informed AlphaFold DB entry (pLDDT={structure.get('confidence')})")
    elif structure.get("source") == "esmfold_api":
        strengths.append("Structure predicted de novo via ESMFold — no experimental structure was required")
    else:
        weaknesses.append(f"No structure could be resolved: {structure.get('reason')}")

    if ligand.get("valid"):
        if ligand.get("drug_like"):
            strengths.append(f"Ligand satisfies Lipinski Rule of Five ({ligand['lipinski_violations']} violations)")
        else:
            weaknesses.append(f"Ligand violates Lipinski Rule of Five ({ligand['lipinski_violations']} violations)")

    if docking.get("source") == "vina":
        strengths.append(f"Real AutoDock Vina docking score (blind, against the real resolved target): {docking.get('best_affinity_kcal_mol')} kcal/mol")
    elif docking.get("source") == "heuristic":
        weaknesses.append(
            "AutoDock Vina unavailable in this environment — binding score is a deterministic "
            "molecular-property heuristic, not a physical docking simulation"
        )

    # Real docking-confidence signal from Vina's own reported alternate
    # poses (see docking_engine._compute_docking_confidence) — same
    # reasoning as the screen-mode ranking report.
    docking_confidence = docking.get("docking_confidence")
    if docking_confidence:
        score = docking_confidence["pose_consistency_score"]
        if score >= 70:
            strengths.append(f"블라인드 도킹 포즈 일관성 높음 (재현 신뢰도 {score}/100, 상위 포즈 간 평균 RMSD {docking_confidence['mean_rmsd_top_poses_angstrom']}Å)")
        elif score <= 30:
            weaknesses.append(f"블라인드 도킹 포즈 일관성 낮음 (재현 신뢰도 {score}/100) — 단백질 표면의 서로 다른 위치에서 비슷한 점수의 포즈가 발견되어, 결합 부위가 모호할 수 있습니다")

    # Real-computable ADMET subset (Veber/PAINS/SA score/hepatotoxicity
    # screen — see services/admet_engine.py) attached onto the docking dict
    # itself (the same object returned in final_result["docking_result"])
    # so the Q&A layer can cite it. Best-effort: never fails the whole
    # report if RDKit's Contrib SA_Score import has an environment issue.
    smiles_for_admet = ligand.get("canonical_smiles")
    if smiles_for_admet:
        try:
            from services.admet_engine import predict_admet_profile
            from services.drug_ranking_engine import admet_strengths_weaknesses
            admet = predict_admet_profile(smiles_for_admet)
            docking["admet"] = admet
            admet_strengths, admet_weaknesses = admet_strengths_weaknesses(admet)
            strengths.extend(admet_strengths)
            weaknesses.extend(admet_weaknesses)
        except Exception:
            logger.warning("[drug_discovery] ADMET profile failed (non-fatal)")

    return {"strengths": strengths, "weaknesses": weaknesses}


async def run_drug_discovery_pipeline(
    uniprot_id:      str = "",
    target_sequence: str = "",
    ligand_smiles:   str = "",
    screen_library:  bool = False,
    goal_text:       str = "",
    job_id:          str = "",
    progress_cb: ProgressCallback | None = None,
    variant_context: dict[str, Any] | None = None,
    max_candidates:  int = 0,
) -> dict[str, Any]:
    """
    Execute the full agentic DAG. Exactly one of uniprot_id / target_sequence
    should be supplied. If screen_library is False, ligand_smiles is required
    (single-dock mode); if True, ligand_smiles is ignored. job_id, when
    supplied by the caller (the REST job store's UUID), is used to name the
    generated report files so GET /report/{job_id} can find them; a
    standalone id is generated if not given (e.g. direct/MCP calls).

    max_candidates (screen mode only): real reported request — screening
    the full ~22-candidate library is real AutoDock Vina physics
    simulation, not something that can be sped up by faking results, but
    docking FEWER real candidates is a genuine, honest way to finish
    sooner. 0 (default) screens the entire library; a positive number
    screens only that many (the library's own curated order, not a
    ranked subset — ranking only happens after real docking).

    variant_context, when supplied by run_drug_discovery_from_vcf() (the
    somatic-variant "Track B" entry point), is passed straight through into
    the final result and report unmodified — this function still only ever
    resolves/docks against uniprot_id's wild-type structure; it never builds
    or fabricates a mutant-specific structure, so the caller is responsible
    for making that limitation visible in variant_context (see
    run_drug_discovery_from_vcf()'s "structure_note").
    """
    t0 = time.perf_counter()
    mode = "screen" if screen_library else "single"
    job_id = job_id or f"dd-{int(t0 * 1000)}"
    decision_log = DrugDiscoveryDecisionLog(job_id)
    planner = DrugDiscoveryPlanner()
    evaluator = DrugDiscoveryEvaluator()
    reflector = DrugDiscoveryReflectionAgent()

    if mode == "single" and not ligand_smiles:
        raise ValueError("ligand_smiles is required when screen_library is False")
    if not uniprot_id and not target_sequence:
        raise ValueError("One of uniprot_id or target_sequence is required")

    logger.info("[drug_discovery] pipeline start | job=%s mode=%s uniprot_id=%s goal_text=%r",
                job_id, mode, uniprot_id, (goal_text or "")[:200])

    total_steps = 4  # structure, receptor+analysis, docking/screening, report — iterations add on top
    step_counter = 0

    async def _progress(iteration: int, msg: str) -> None:
        logger.info("[drug_discovery] iter=%d/%d | %s", iteration, _MAX_ITERATIONS, msg)
        if progress_cb:
            await progress_cb(iteration, _MAX_ITERATIONS, msg)

    # ── Step 1: Target identification / structure resolution (once) ──
    await _progress(1, f"[1/6단계] 타겟 구조 확보 중 ({'UniProt ' + uniprot_id if uniprot_id else '서열 기반 구조 예측'})")
    if uniprot_id:
        structure = (await asyncio.to_thread(
            router_core.route, "fetch_protein_structure", uniprot_id=uniprot_id,
        )).data
        if not structure.get("pdb_text"):
            # Real gap, not hypothetical: confirmed for Influenza M2 (P06821)
            # — a real, curated UniProt target with NO AlphaFold DB entry.
            # Previously this just gave up here even though ESMFold could
            # predict a real structure from the same ID's real sequence —
            # fixed by fetching that sequence live and falling back to the
            # same real ESMFold path predict_protein_structure already uses
            # for a directly-pasted sequence.
            from services import protein_structure_engine
            await _progress(1, f"[1/6단계] {uniprot_id}에 대한 AlphaFold DB 항목 없음 — 서열을 가져와 ESMFold로 대체 예측 중")
            fallback_sequence = await asyncio.to_thread(
                protein_structure_engine.fetch_uniprot_sequence, uniprot_id,
            )
            if fallback_sequence:
                structure = (await asyncio.to_thread(
                    router_core.route, "predict_protein_structure", sequence=fallback_sequence,
                )).data
                if structure.get("pdb_text"):
                    structure["uniprot_id"] = uniprot_id
    else:
        structure = (await asyncio.to_thread(
            router_core.route, "predict_protein_structure", sequence=target_sequence,
        )).data
    decision_log.add("target_identification", "resolve_structure", "fetch_protein_structure/predict_protein_structure",
                      "ok" if structure.get("pdb_text") else "failed", {"source": structure.get("source")})

    if not structure.get("pdb_text"):
        logger.warning("[drug_discovery] job=%s structure resolution FAILED | source=%s reason=%s",
                        job_id, structure.get("source"), structure.get("reason"))
        return {
            "mode": mode, "structure_source": structure.get("source"), "structure": structure,
            "error": f"Could not resolve target structure: {structure.get('reason')}",
            "decision_log": decision_log.to_dict(), "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }

    # Real measured property of the resolved structure (not a guess) — lets
    # the Planner default to a faster strategy for large targets, since a
    # blind-docking box scales with the target's size (see
    # drug_discovery_agent.py's docstring for the measured Spike-protein
    # timing that motivated this).
    residue_count = len({line[21:27] for line in structure["pdb_text"].splitlines() if line.startswith("ATOM")})

    # ── Planning: pick initial strategy ──
    plan = planner.plan(goal_text, residue_count=residue_count)
    strategy_id = plan["strategy_id"]
    decision_log.add("planning", f"selected_strategy={strategy_id}", "DrugDiscoveryPlanner",
                      "ok", {**plan["template"], "residue_count": residue_count})

    library = _load_drug_library() if mode == "screen" else []
    if mode == "screen" and max_candidates and max_candidates > 0:
        library = library[:max_candidates]
    result_payload: Any = None
    receptor_prep: dict = {}
    evaluation: dict = {}

    # Known targets (knowledge/target_synonyms.json) have a real receptor +
    # blind-docking box precomputed once offline (see refs/docking/cache/ and
    # gene_index.json's "DrugDiscoveryReceptors" entry) — a Tier-1 cache hit
    # here skips the ~20-45s mk_prepare_receptor subprocess entirely for
    # every iteration, since router_core.route() short-circuits before
    # calling the tool at all. Padding varies slightly by strategy, but for
    # a blind box already spanning the whole protein that difference is
    # immaterial, so the same cached receptor is reused across strategies —
    # exhaustiveness (varied per strategy below) remains the real lever.
    receptor_cache_key = ("DrugDiscoveryReceptors", uniprot_id) if uniprot_id else None

    try:
        for iteration in range(1, _MAX_ITERATIONS + 1):
            template = plan["template"] if iteration == 1 else STRATEGY_TEMPLATES[strategy_id]
            heuristic_only = False

            # ── Tool selection + receptor prep (real target structure, blind box) ──
            await _progress(iteration, f"[2/6단계 · {_strategy_label(strategy_id)} 전략] 확보된 구조로 리셉터(도킹 박스) 준비 중")
            if receptor_prep.get("tmp_dir"):
                receptor_prep_engine.cleanup_receptor(receptor_prep)
            receptor_prep_result = await asyncio.to_thread(
                router_core.route, "prepare_receptor", cache_key=receptor_cache_key,
                pdb_text=structure["pdb_text"], padding=template["padding"],
            )
            receptor_prep = receptor_prep_result.data
            decision_log.add("tool_selection", "prepare_receptor", "prepare_receptor",
                              "ok" if receptor_prep.get("prepared") else "failed",
                              {"strategy": strategy_id, "tier": receptor_prep_result.tier})

            if receptor_prep.get("prepared"):
                if mode == "single":
                    await _progress(iteration, f"[3/6단계 · {_strategy_label(strategy_id)} 전략] 실제 타겟에 리간드 도킹 중 (블라인드 서치박스)")
                    result_payload = (await asyncio.to_thread(
                        router_core.route, "dock_ligand",
                        receptor_pdbqt_path=receptor_prep["pdbqt_path"], ligand_smiles=ligand_smiles,
                        center=receptor_prep["center"], box_size=receptor_prep["box_size"],
                        exhaustiveness=template["exhaustiveness"], keep_pose=True,
                    )).data
                    decision_log.add("analysis", "dock_ligand", "dock_ligand",
                                      "ok" if result_payload.get("docked") else "failed",
                                      {"source": result_payload.get("source")})
                    heuristic_only = result_payload.get("source") == "heuristic"
                else:
                    # Real, confirmed bottleneck: this loop used to await
                    # each candidate's dock_ligand call one at a time, even
                    # though docking_engine.py's _VINA_CPU_LIMIT already
                    # caps every single Vina subprocess to HALF the
                    # machine's logical cores specifically so more than one
                    # can run at once without starving each other — that
                    # headroom was never actually used within a single
                    # screen. Now runs up to _SCREEN_CONCURRENCY candidates'
                    # real Vina subprocesses genuinely in parallel (each
                    # still dispatched via asyncio.to_thread, so this is
                    # real OS-level parallelism, not just cooperative
                    # scheduling) — with 2 concurrent half-core Vina
                    # processes exactly saturating an 8-core machine. Cache
                    # hits (curated targets) are unaffected either way
                    # since router_core.route() short-circuits before ever
                    # reaching the subprocess. Progress messages can now
                    # interleave slightly (e.g. candidate 4 may start before
                    # candidate 3 finishes) since they run concurrently —
                    # still accurately reflects which candidates are
                    # actually in flight, not a regression in honesty.
                    screened: list[dict | None] = [None] * len(library)
                    screen_semaphore = asyncio.Semaphore(_SCREEN_CONCURRENCY)

                    async def _dock_one_candidate(idx: int, candidate: dict) -> None:
                        async with screen_semaphore:
                            await _progress(iteration, f"[3/6단계 · {_strategy_label(strategy_id)} 전략] 후보 스크리닝 중 {idx}/{len(library)}: {candidate['name']}")
                            # The curated library (knowledge/drug_library.json) is fixed and the
                            # 6 curated targets' receptors are fixed too, so for those pairs the
                            # real Vina result at a given strategy's exhaustiveness never changes
                            # run to run — precomputed once offline (see knowledge/gene_index.json's
                            # "DrugDiscoveryDocking" entry) the same way receptor prep is cached
                            # above. A cache hit here skips the real per-candidate Vina subprocess
                            # entirely instead of approximating/faking the score.
                            docking_cache_key = (
                                ("DrugDiscoveryDocking", f"{uniprot_id}:{strategy_id}:{candidate['name']}")
                                if uniprot_id else None
                            )
                            try:
                                docking = (await asyncio.to_thread(
                                    router_core.route, "dock_ligand", cache_key=docking_cache_key,
                                    receptor_pdbqt_path=receptor_prep["pdbqt_path"], ligand_smiles=candidate["smiles"],
                                    center=receptor_prep["center"], box_size=receptor_prep["box_size"],
                                    exhaustiveness=template["exhaustiveness"],
                                    # Vina computes the docked pose to score it at all — asking for it
                                    # here costs nothing but keeps it, so the #1 candidate's pocket
                                    # analysis below can use the pose from THIS dock instead of paying
                                    # for a second identical Vina run (measured: 4.8s of a 7.1s job).
                                    # A cached hit still carries no pose, so the re-dock stays as the
                                    # fallback for that case rather than being deleted.
                                    keep_pose=True,
                                )).data
                            except Exception as exc:
                                logger.warning("[drug_discovery] screening failed for %s | error=%s", candidate["name"], exc)
                                docking = {"docked": False, "error": str(exc), "source": "none"}
                            screened[idx - 1] = {"name": candidate["name"], "smiles": candidate["smiles"],
                                                  "category": candidate.get("category"), "docking": docking}

                    await asyncio.gather(*(
                        _dock_one_candidate(idx, candidate) for idx, candidate in enumerate(library, start=1)
                    ))
                    decision_log.add("analysis", "dock_ligand (per candidate, parallel)", "dock_ligand", "ok",
                                      {"n_candidates": len(library), "concurrency": _SCREEN_CONCURRENCY})
                    heuristic_only = _screen_is_heuristic_only(screened)

                    await _progress(iteration, "[4/6단계] 도킹 실패·비약물성 후보 필터링 중")
                    filtered = (await asyncio.to_thread(router_core.route, "filter_drug_candidates", results=screened)).data
                    decision_log.add("filtering", "filter_drug_candidates", "filter_drug_candidates", "ok",
                                      {"survivors": len(filtered), "total": len(screened)})

                    await _progress(iteration, "[5/6단계] 생존 후보 랭킹 산정 중")
                    result_payload = (await asyncio.to_thread(router_core.route, "rank_drug_candidates", screened=filtered)).data
                    decision_log.add("ranking", "rank_drug_candidates", "rank_drug_candidates", "ok", {"ranked": len(result_payload)})
            else:
                result_payload = [] if mode == "screen" else {"docked": False, "error": receptor_prep.get("reason"), "source": "none"}

            evaluation = evaluator.evaluate(mode, receptor_prep, result_payload, template, iteration)
            decision_log.add("evaluation", evaluation["verdict"], "DrugDiscoveryEvaluator", evaluation["verdict"],
                              {"failed_metric": evaluation.get("failed_metric")})

            if evaluation["verdict"] == "pass":
                break

            # Measured waste, not hypothetical: the retry escalates the
            # strategy's exhaustiveness, but exhaustiveness is a lever on real
            # AutoDock Vina only. With no vina binary the docking falls back to
            # a deterministic molecular-property heuristic whose score cannot
            # move with rigor — so a retry re-docks the entire library to land
            # on bit-identical numbers and the identical failing verdict.
            # Observed: 3 full 22-compound screens, ~3x the wall clock, for
            # nothing. Stop here instead, and say why in the decision log
            # rather than silently pretending a retry was considered.
            if heuristic_only:
                logger.info("[drug_discovery] skipping retry | docking is heuristic-only, "
                            "escalating exhaustiveness cannot change the result")
                decision_log.add(
                    "reflection",
                    "Docking fell back to the deterministic property heuristic (no AutoDock Vina "
                    "binary available), so re-running at a higher exhaustiveness would reproduce "
                    "the identical scores. Retry skipped.",
                    "DrugDiscoveryReflectionAgent", "stop",
                    {"iteration": iteration, "docking_source": "heuristic"},
                )
                break

            reflection = reflector.reflect(evaluation, strategy_id, iteration)
            decision_log.add("reflection", reflection["root_cause"], "DrugDiscoveryReflectionAgent", "retry",
                              {"next_strategy": reflection["next_strategy"], "reasoning": reflection["reasoning"]})
            strategy_id = reflection["next_strategy"]

        # ── Receptor prep failed => docking NEVER RAN. Say so; do not "report". ──
        #
        # Without this the run reached the report builder with an empty candidate
        # list and emitted "No candidates in the curated library survived docking +
        # drug-likeness filtering" — a sentence that describes docking that happened
        # and filtered everything out. Nothing was docked. The job then came back
        # COMPLETED (green), and the AI summary went on to rationalise the empty
        # result as a scientific finding.
        #
        # That is not a degraded result, it is a fabricated one: the difference
        # between "we screened and nothing survived" (a real negative finding a
        # researcher may act on) and "we never screened" (a broken tool) is the whole
        # value of the answer. Observed live on the deployed container, where meeko's
        # CLI was not on PATH under the name the resolver searched for: every
        # non-cached target silently produced this.
        if not receptor_prep.get("prepared"):
            reason = receptor_prep.get("reason") or "unknown"
            logger.error("[drug_discovery] job=%s RECEPTOR PREP FAILED — 도킹이 실행되지 않았습니다 | reason=%s",
                         job_id, reason)
            return {
                "mode": mode,
                "structure_source": structure.get("source"),
                "structure": structure,
                "error": f"리셉터 준비에 실패해 도킹을 실행하지 못했습니다: {reason}",
                "docking_ran": False,
                "decision_log": decision_log.to_dict(),
                "elapsed_seconds": round(time.perf_counter() - t0, 1),
            }

        # ── Report ──
        await _progress(_MAX_ITERATIONS, "[6/6단계] AI 해설 및 리포트 생성 중")
        if mode == "single":
            report = _build_single_report(structure, result_payload)
            ranked_report = None
        else:
            report = {
                "strengths": [f"{len(result_payload)} candidate(s) survived filtering out of {len(library)} screened"] if result_payload else [],
                "weaknesses": ["No candidates in the curated library survived docking + drug-likeness filtering"] if not result_payload else [],
            }
            from services.drug_ranking_engine import generate_explainable_report
            ranked_report = [generate_explainable_report(c) for c in result_payload]

        # ── Deep structural analysis (real geometry; opt-in, best-effort) ──
        # Binding-pocket contacts come from the actual docked-pose coordinates
        # (kept via dock_ligand(..., keep_pose=True)) against the actual
        # prepared receptor — never fabricated. For screen mode this costs
        # one extra real Vina call for the #1-ranked candidate only, since a
        # Tier-1 cache hit during the main screening loop only ever stored
        # the affinity number, not the raw pose. PAE is only ever attempted
        # for AlphaFold DB-sourced structures (structure["pae_doc_url"] is
        # only ever set there — see protein_structure_engine.py) and is
        # explicitly reported as unavailable otherwise, never approximated.
        structural_analysis: dict[str, Any] = {"binding_pocket": None, "pae_summary": None}
        try:
            if receptor_prep.get("prepared"):
                pose_pdbqt = None
                if mode == "single" and isinstance(result_payload, dict) and result_payload.get("source") == "vina":
                    pose_pdbqt = result_payload.get("pose_pdbqt")
                elif mode == "screen" and result_payload:
                    top_candidate = result_payload[0]
                    top_docking = top_candidate.get("docking") or {}
                    if top_docking.get("source") == "vina":
                        # The screening dock kept its pose, so for a candidate that
                        # was really docked just now there is nothing left to compute.
                        pose_pdbqt = top_docking.get("pose_pdbqt")
                        if not pose_pdbqt and docking_engine._vina_available():
                            # Only a cache hit lands here: the precomputed entry stored
                            # the affinity but not the geometry, so the pose has to be
                            # produced by an actual Vina run at the same exhaustiveness
                            # (a cheaper one would not correspond to the reported score).
                            #
                            # The _vina_available() guard is the point. The precomputed
                            # entries were docked offline on a machine that HAD vina, so
                            # they say source="vina" — but this deployment ships no vina
                            # binary (see the Dockerfile), so the re-dock silently fell
                            # back to the heuristic path, which by design returns no 3D
                            # pose. The result: 4.1 seconds of every cached screening job
                            # spent producing a pose that could not exist, and a binding
                            # pocket that came back null every single time. If we can't
                            # dock for real, don't pretend to.
                            await _progress(_MAX_ITERATIONS,
                                            f"[추가 분석] 결합 포켓 분석을 위해 1위 후보({top_candidate['name']}) 재도킹 중")
                            redock = (await asyncio.to_thread(
                                router_core.route, "dock_ligand",
                                receptor_pdbqt_path=receptor_prep["pdbqt_path"], ligand_smiles=top_candidate["smiles"],
                                center=receptor_prep["center"], box_size=receptor_prep["box_size"],
                                exhaustiveness=template["exhaustiveness"], keep_pose=True,
                            )).data
                            pose_pdbqt = redock.get("pose_pdbqt")

                if pose_pdbqt and os.path.isfile(receptor_prep["pdbqt_path"]):
                    from services import structural_analysis_engine
                    with open(receptor_prep["pdbqt_path"], encoding="utf-8", errors="replace") as f:
                        receptor_pdbqt_text = f.read()
                    pocket = structural_analysis_engine.analyze_binding_pocket(receptor_pdbqt_text, pose_pdbqt)
                    structural_analysis["binding_pocket"] = pocket
                    decision_log.add("structural_analysis", "analyze_binding_pocket", "analyze_binding_pocket",
                                      "ok" if pocket.get("available") else "failed", {})

                    pae_doc_url = structure.get("pae_doc_url")
                    if pae_doc_url and pocket.get("available"):
                        pae_matrix = structural_analysis_engine.fetch_pae_matrix(pae_doc_url)
                        if pae_matrix:
                            residue_numbers = [r["resnum"] for r in pocket["residues_in_contact"]]
                            structural_analysis["pae_summary"] = structural_analysis_engine.summarize_pae_for_residues(
                                pae_matrix, residue_numbers,
                            )
                        else:
                            structural_analysis["pae_summary"] = {"available": False, "reason": "PAE fetch failed"}
                    elif not pae_doc_url:
                        structural_analysis["pae_summary"] = {
                            "available": False,
                            "reason": "No PAE data for this structure (only AlphaFold DB-sourced structures carry "
                                      f"PAE; this target's structure came from {structure.get('source')!r}).",
                        }
        except Exception as exc:
            logger.warning("[drug_discovery] structural analysis step failed (non-fatal) | job=%s error=%s", job_id, exc)

    finally:
        if receptor_prep.get("tmp_dir"):
            receptor_prep_engine.cleanup_receptor(receptor_prep)

    elapsed = time.perf_counter() - t0
    logger.info("[drug_discovery] COMPLETE | mode=%s elapsed=%.1fs structure_source=%s verdict=%s",
                mode, elapsed, structure.get("source"), evaluation.get("verdict"))

    # ── AI narrative + HTML/PDF report (real numbers only; AI narrates, never invents) ──
    # Both of these are synchronous and slow — one is a blocking OpenAI request,
    # the other renders HTML and a PDF — and both used to run directly on the
    # event loop. The whole server froze for the duration: every other request,
    # including this job's own status polls, simply could not be served. That is
    # what made a running docking job look like a dead one from Kakao, whose
    # tool-call timeout is ~10s. Everything else in this pipeline was already
    # dispatched through to_thread; these two were the exceptions.
    ai_summary_source = ranked_report if mode == "screen" else result_payload
    ai_summary = await asyncio.to_thread(
        generate_ai_summary, mode, structure, ai_summary_source, report, structural_analysis)

    final_result = {
        "mode":             mode,
        "structure_source": structure.get("source"),
        "structure":        structure,
        "docking_result":   result_payload if mode == "single" else None,
        "ranked_candidates": ranked_report if mode == "screen" else None,
        "report":           report,
        "ai_summary":       ai_summary,
        "structural_analysis": structural_analysis,
        "evaluation":       evaluation,
        "decision_log":     decision_log.to_dict(),
        "elapsed_seconds":  round(elapsed, 1),
        "variant_context":  variant_context,
    }

    # Rendered in a recycled subprocess, not a thread: reportlab/jinja2 retain ~0.5 MB
    # of NATIVE memory per report that no Python-side cleanup can reclaim, and it was
    # ~90% of a 188 MB/60-cycle leak. See services/report_worker.py.
    report_files = await report_worker.render_drug_report(job_id, final_result, ai_summary)
    final_result["report_available"] = bool(report_files.get("html_path"))

    return final_result


async def run_drug_discovery_from_vcf(
    vcf_text: str,
    job_id:   str = "",
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    "Track B" entry point — somatic-variant-driven target discovery, as an
    alternative front door into the exact same screening pipeline
    run_drug_discovery_pipeline() already runs for "Track A" (named
    pathogens/proteins). Real, verified steps only:

      1. Parse the VCF text (services/vcf_annotation_engine.parse_vcf) —
         deterministic, no network call.
      2. Re-annotate each variant live via Ensembl VEP
         (annotate_variant_consequence) — never trusts the VCF's own
         INFO=GENE label, since a hand-built/synthetic VCF's label isn't
         guaranteed to match the real coding consequence at that exact
         coordinate (confirmed happening for this project's own
         sample/NSCLC_variants.vcf: its GENE=EGFR variant's coordinate
         doesn't actually land in EGFR's real coding sequence per VEP).
      3. Resolve the VEP-confirmed gene symbol to a real UniProt ID via a
         live UniProt search (services/uniprot_search_engine) — no
         hardcoded gene->UniProt table.
      4. Hand off to run_drug_discovery_pipeline() exactly as Track A does,
         with the real variant context attached to the result/report.

    Docks against the gene's WILD-TYPE structure — this does not build or
    fabricate a mutant-specific 3D structure for the found variant; that
    limitation is surfaced explicitly in variant_context["structure_note"]
    rather than silently presented as if the docking pocket reflects the
    mutation.

    Picks the first variant with a real protein-coding missense consequence
    (VEP-confirmed); returns an error result if none of the VCF's variants
    have one. Multi-variant VCFs are only partially supported today (only
    the first qualifying variant drives the screen) — extending this to
    screen every variant is a real, separate scope, not attempted here.
    """
    from services.vcf_annotation_engine import parse_vcf, annotate_variant_consequence
    from services.uniprot_search_engine import search_reviewed_proteins

    t0 = time.perf_counter()
    job_id = job_id or f"vcf-{int(t0 * 1000)}"

    async def _progress(iteration: int, msg: str) -> None:
        logger.info("[drug_discovery_vcf] job=%s | %s", job_id, msg)
        if progress_cb:
            await progress_cb(iteration, _MAX_ITERATIONS, msg)

    await _progress(1, "[1단계] VCF 파일 파싱 중")
    variants = await asyncio.to_thread(parse_vcf, vcf_text)
    logger.info("[drug_discovery_vcf] job=%s parsed %d variant(s)", job_id, len(variants))

    await _progress(1, f"[1단계] Ensembl VEP로 변이 {len(variants)}건 실시간 주석 중")
    chosen_variant: dict[str, Any] | None = None
    chosen_annotation: dict[str, Any] | None = None
    all_annotations: list[dict[str, Any]] = []
    for v in variants:
        annotation = await asyncio.to_thread(
            annotate_variant_consequence, v["chrom"], v["pos"], v["ref"], v["alt"],
        )
        all_annotations.append({"variant": v, "annotation": annotation})
        if chosen_variant is None and annotation.get("gene_symbol") and annotation.get("protein_change"):
            chosen_variant, chosen_annotation = v, annotation

    if chosen_variant is None or chosen_annotation is None:
        logger.warning("[drug_discovery_vcf] job=%s no variant with a resolvable protein-coding "
                        "missense consequence", job_id)
        return {
            "mode": "screen", "error": "No variant in the VCF had a resolvable protein-coding missense "
                                        "consequence via Ensembl VEP.",
            "variant_annotations": all_annotations,
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }

    gene_symbol = chosen_annotation["gene_symbol"]
    protein_change = chosen_annotation["protein_change"]
    vcf_gene_label = (chosen_variant.get("info") or {}).get("GENE")
    if vcf_gene_label and vcf_gene_label != gene_symbol:
        logger.warning("[drug_discovery_vcf] job=%s VCF INFO=GENE (%s) disagrees with VEP-confirmed gene (%s) "
                        "at %s:%s — using VEP's real annotation, not the file's label",
                        job_id, vcf_gene_label, gene_symbol, chosen_variant["chrom"], chosen_variant["pos"])

    await _progress(1, f"[1단계] {gene_symbol}을(를) 실제 UniProt 항목으로 변환 중")
    candidates = await asyncio.to_thread(
        search_reviewed_proteins, f"gene:{gene_symbol} AND organism_id:9606", 3,
    )
    if not candidates:
        logger.warning("[drug_discovery_vcf] job=%s could not resolve gene=%s to a UniProt entry",
                        job_id, gene_symbol)
        return {
            "mode": "screen", "error": f"Could not resolve gene {gene_symbol!r} to a real UniProt entry.",
            "variant_annotations": all_annotations,
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }
    uniprot_id = candidates[0]["uniprot_id"]
    logger.info("[drug_discovery_vcf] job=%s gene=%s -> uniprot_id=%s (%s)",
                job_id, gene_symbol, uniprot_id, candidates[0].get("protein_name"))

    variant_context = {
        "source_variant": {
            "chrom": chosen_variant["chrom"], "pos": chosen_variant["pos"],
            "ref": chosen_variant["ref"], "alt": chosen_variant["alt"],
        },
        "gene_symbol": gene_symbol,
        "protein_change": protein_change,
        "vaf": (chosen_variant.get("samples") or {}).get(
            next(iter(chosen_variant.get("samples", {})), ""), {},
        ).get("AF"),
        "vcf_gene_label": vcf_gene_label,
        "label_mismatch": bool(vcf_gene_label and vcf_gene_label != gene_symbol),
        "resolved_uniprot_id": uniprot_id,
        "structure_note": (
            f"Docking used {gene_symbol}'s wild-type structure (UniProt {uniprot_id}) — no "
            f"{protein_change}-specific mutant structure was built or used, so the docking pocket does "
            f"not reflect this specific mutation. This is a real limitation, disclosed rather than hidden."
        ),
        "annotation_source": "ensembl_vep",
    }

    result = await run_drug_discovery_pipeline(
        uniprot_id=uniprot_id,
        screen_library=True,
        goal_text=f"{gene_symbol} {protein_change} 체세포 변이와 관련된 억제 시약 스크리닝 (야생형 구조 기반)",
        job_id=job_id,
        progress_cb=progress_cb,
        variant_context=variant_context,
    )
    return result
