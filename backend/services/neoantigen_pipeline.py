"""
Neoantigen Candidate Pipeline — orchestrates VCF -> VEP annotation -> (real,
scoped-down) HLA context -> MHCflurry binding/foreignness -> real PubMed/LLM
literature grounding, into a single job result. Fully separate
implementation from drug_discovery_pipeline.py's docking/screening
pipeline — this produces mRNA-neoantigen-vaccine candidates, not
docked/ranked small molecules, so it does not reuse that pipeline's
report/ranking machinery.

See services/neoantigen_engine.py's module docstring for the real
scope boundary: BAM-based HLA typing (OptiType) is not implemented on this
Windows dev environment (no Docker/WSL, no pysam wheel), so a supplied BAM
file is only used for an honest real read-count/coverage summary — the
actual HLA alleles used for MHCflurry come from neoantigen_engine.
COMMON_HLA_ALLELES, always disclosed as such in the result, never
presented as this patient's real genotype.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], Awaitable[None]]
# 5 steps since PubMed grounding was dropped from the pipeline (see below).
_MAX_STEPS = 5


async def run_neoantigen_pipeline(
    vcf_text: str,
    bam_path: str | None = None,
    job_id: str = "",
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    from services.vcf_annotation_engine import parse_vcf, annotate_variant_consequence
    from services.neoantigen_engine import (
        predict_neoantigen_candidates, COMMON_HLA_ALLELES,
        calculate_neoantigen_composite_score, generate_ai_interpretation, ALGORITHM_EXPLANATION,
    )
    from services.bam_reader import read_bam_summary

    t0 = time.perf_counter()
    job_id = job_id or f"neoantigen-{int(t0 * 1000)}"

    async def _progress(step: int, msg: str) -> None:
        logger.info("[neoantigen_pipeline] job=%s | %s", job_id, msg)
        if progress_cb:
            await progress_cb(step, _MAX_STEPS, msg)

    await _progress(1, "[1단계] VCF 파일 파싱 중")
    variants = await asyncio.to_thread(parse_vcf, vcf_text)
    logger.info("[neoantigen_pipeline] job=%s parsed %d variant(s)", job_id, len(variants))

    await _progress(2, f"[2단계] Ensembl VEP로 변이 {len(variants)}건 실시간 주석 중")
    mutations: list[dict[str, Any]] = []
    all_annotations: list[dict[str, Any]] = []
    for v in variants:
        annotation = await asyncio.to_thread(
            annotate_variant_consequence, v["chrom"], v["pos"], v["ref"], v["alt"],
        )
        all_annotations.append({"variant": v, "annotation": annotation})
        if annotation.get("gene_symbol") and annotation.get("protein_change") and annotation.get("protein_position"):
            ref_aa, _, alt_aa = (annotation.get("amino_acids") or "").partition("/")
            if ref_aa and alt_aa:
                mutations.append({
                    "gene_symbol": annotation["gene_symbol"],
                    "transcript_id": annotation["transcript_id"],
                    "protein_position": annotation["protein_position"],
                    "ref_aa": ref_aa,
                    "alt_aa": alt_aa,
                    "protein_change": annotation["protein_change"],
                    "source_variant": {"chrom": v["chrom"], "pos": v["pos"], "ref": v["ref"], "alt": v["alt"]},
                    "vaf": (v.get("samples") or {}).get(next(iter(v.get("samples", {})), ""), {}).get("AF"),
                })

    if not mutations:
        # Two very different reasons produce zero mutations, and collapsing them
        # into one message is the same failure-wearing-a-success-label problem the
        # docking pipeline had. If every variant was actually ANNOTATED (annotated:
        # True) and simply none were protein-coding missense, that is a real, honest
        # negative finding a user can trust. But if the annotations FAILED at the
        # infrastructure level (annotated: False — Ensembl timed out, 500'd, or
        # returned an HTML error page, all observed live on its public REST API),
        # then we did NOT determine there is no missense variant; we could not check.
        # Saying "no variant had a resolvable missense consequence via Ensembl VEP"
        # in that case reads as a scientific result when it is an outage.
        any_infra_failure = any(
            not a["annotation"].get("annotated") for a in all_annotations
        )
        if any_infra_failure:
            logger.error("[neoantigen_pipeline] job=%s Ensembl VEP unreachable — 주석 실패", job_id)
            return {
                "mode": "neoantigen",
                "error": ("변이 주석 서비스(Ensembl VEP)에 연결하지 못했습니다. 외부 유전체 "
                          "데이터베이스가 일시적으로 불안정한 상태이며, 이는 분석 결과가 아니라 "
                          "일시적 장애입니다. 잠시 후 다시 시도해 주세요."),
                "external_dependency_down": True,
                "variant_annotations": all_annotations,
                "elapsed_seconds": round(time.perf_counter() - t0, 1),
            }
        logger.warning("[neoantigen_pipeline] job=%s no variant with a resolvable protein-coding "
                        "missense consequence", job_id)
        return {
            "mode": "neoantigen",
            "error": "No variant in the VCF had a resolvable protein-coding missense consequence via Ensembl VEP.",
            "variant_annotations": all_annotations,
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
        }

    bam_summary: dict[str, Any] | None = None
    hla_note: str
    if bam_path:
        await _progress(3, "[3단계] BAM 파일 헤더/리드 확인 중 (표준 HLA 6종 기준 분석)")
        try:
            raw_summary = await asyncio.to_thread(read_bam_summary, bam_path)
            bam_summary = {
                "ref_name": raw_summary["ref_name"],
                "ref_length": raw_summary["ref_length"],
                "read_count": raw_summary["read_count"],
            }
        except Exception as exc:
            logger.warning("[neoantigen_pipeline] job=%s BAM read failed | error=%s", job_id, exc)
            bam_summary = {"error": str(exc)}
        # Softened wording, same fact: the scoring really does run against
        # population-standard alleles, so that has to stay stated — but as a
        # neutral description of the analysis basis rather than a warning.
        # The full caveat (and the "type the patient's own HLA before real
        # use" instruction) still lives in the detailed report — see
        # drug_report_service.py.
        hla_note = (
            f"인구집단 표준 HLA class I allele {len(COMMON_HLA_ALLELES)}종"
            f"({', '.join(COMMON_HLA_ALLELES)}) 기준으로 분석했습니다."
        )
    else:
        hla_note = (
            f"BAM 파일이 제공되지 않아 인구집단 고빈도 HLA class I allele "
            f"{len(COMMON_HLA_ALLELES)}종을 기본값으로 사용했습니다."
        )

    await _progress(4, f"[4단계] MHCflurry로 {len(mutations)}개 변이의 실제 MHC 결합/제시 예측 중")
    prediction = await asyncio.to_thread(predict_neoantigen_candidates, mutations)

    # PubMed grounding removed from this pipeline. It was a real PubMed search
    # plus an LLM summary, and once MHCflurry warmed up it was the single
    # biggest cost left in the job (~4.2s of ~9.5s) while contributing nothing
    # to the neoantigen prediction itself. The same evidence is still available
    # on demand through the dedicated search_literature MCP tool, so nothing is
    # lost — it just no longer taxes every vaccine design. Key kept (empty) so
    # existing consumers (drug_report_service, drug_discovery_chat) stay happy.
    literature_by_gene: dict[str, Any] = {}

    await _progress(5, "[5단계] AI Neo-Score 산출 및 해석 생성 중")
    scored_candidates = []
    for c in prediction["candidates"]:
        score = await asyncio.to_thread(calculate_neoantigen_composite_score, c)
        scored_candidates.append({**c, "composite_score": score["composite_score"], "score_breakdown": score["breakdown"]})
    ai_interpretation = await asyncio.to_thread(generate_ai_interpretation, scored_candidates, hla_note)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("[neoantigen_pipeline] job=%s done in %.1fs | mutations=%d candidates=%d",
                job_id, elapsed, len(mutations), len(prediction["candidates"]))

    result = {
        "mode": "neoantigen",
        "mutations_analyzed": mutations,
        "variant_annotations": all_annotations,
        "hla_alleles": prediction["hla_alleles"],
        "hla_note": hla_note,
        "bam_summary": bam_summary,
        "candidates": scored_candidates,
        "ai_interpretation": ai_interpretation,
        "algorithm_explanation": ALGORITHM_EXPLANATION,
        "all_scored": prediction["all_scored"],
        "prediction_errors": prediction["errors"],
        "literature_by_gene": literature_by_gene,
        "elapsed_seconds": elapsed,
    }

    # Real HTML/PDF report (same drugjob_ filename prefix + REPORT_DIR as
    # the docking pipeline's report — see drug_report_service.generate_
    # neoantigen_report()'s docstring for why sharing the prefix is safe),
    # served by the existing GET /api/drug-discovery/report/{job_id}(/pdf)
    # endpoints unmodified. Deterministic next_step_suggestion (no LLM) —
    # mirrors the "다음 단계" pattern already used elsewhere in
    # drug_discovery_chat.py, scoped to what's actually actionable for a
    # completed neoantigen result (literature lookup / report download /
    # re-run on a different VCF) rather than the docking-only SAR/binding-
    # pocket suggestions that don't apply to this pipeline.
    from services import drug_report_service
    report_files = await asyncio.to_thread(drug_report_service.generate_neoantigen_report, job_id, result)
    result["report_available"] = bool(report_files.get("html_path"))
    if scored_candidates:
        top_gene = scored_candidates[0].get("gene_symbol")
        result["next_step_suggestion"] = (
            f"다음 단계: 아래 HTML/PDF 리포트를 다운로드해 알고리즘 설명과 최종 결론을 확인하시거나, "
            f"'{top_gene} 관련 논문 찾아줘'처럼 최상위 후보 유전자의 관련 문헌을 더 찾아볼 수 있습니다."
        )
    else:
        result["next_step_suggestion"] = (
            "다음 단계: 이 변이는 조건을 만족하는 신항원 후보가 없는 실제 결과입니다 — 다른 VCF 파일을 "
            "업로드해 재시도하거나, 분석된 변이의 관련 문헌을 찾아볼 수 있습니다."
        )

    return result
