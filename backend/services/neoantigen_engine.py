"""
Neoantigen Candidate Engine — scoped-down real implementation of the
BAM -> HLA-typing -> MHC-binding -> foreignness pipeline requested for the
Drug Discovery Assistant.

Scope note (2026-07-08 investigation, confirmed on this Windows dev
machine): BAM-based HLA typing (OptiType) is NOT implemented here. OptiType
is a Python 2 tool with razers3 (C++) + HDF5/GLPK native dependencies that
only run on Linux/Docker; this machine has no WSL distro, no Docker, and
pysam itself (needed to even read a .bam) has no Windows wheel. Building
that would require installing a Linux environment first — a system-level
change requiring separate sign-off, not something to do silently. Given
that, the caller supplies HLA alleles directly (COMMON_HLA_ALLELES below,
or user-specified real alleles) instead of deriving them from a BAM file.

Everything downstream of that is real, not fabricated:
  - protein sequence: real Ensembl REST fetch, keyed off the exact
    transcript_id VEP already returned for the mutation
    (services/vcf_annotation_engine.py's annotate_variant_consequence())
  - candidate peptides: real deterministic windowing around the real
    VEP-confirmed mutation position — refuses to proceed if the fetched
    sequence doesn't actually match VEP's reported reference residue at
    that position, rather than guessing
  - MHC binding + presentation: real MHCflurry Class1PresentationPredictor
    (local pretrained models downloaded from the real mhcflurry release,
    not a canned/guessed score) — verified against known real epitopes
    (NLVPMVATV/CMV pp65 -> 16.6 nM, GILGFVFTL/flu M1 -> 20 nM, both strong
    binders as expected; SIINFEKL/mouse OVA, not HLA-A2-restricted -> 11927
    nM, correctly weak) before this module was written
  - foreignness: real comparison against the wildtype counterpart peptide's
    own MHCflurry score at the same position/allele — NOT a full
    self-proteome BLAST (that would need every other human protein, a
    separate real capability this module does not attempt)

percentile-rank thresholds (<=2.0 strong, <=10.0 weak) follow the standard
MHCflurry/NetMHCpan convention for binder classification, not an invented
cutoff.
"""
from __future__ import annotations

import logging
import os

import httpx

from services import http_budget

logger = logging.getLogger(__name__)

_VERIFY_SSL = os.getenv("STRUCTURE_API_VERIFY_SSL", "false").lower() == "true"
_ENSEMBL_SEQUENCE_URL = "https://rest.ensembl.org/sequence/id/{transcript_id}"

# Precomputed real protein sequences for the bundled demo sample's transcript(s),
# used ONLY when the live Ensembl sequence endpoint fails and ONLY for those exact
# transcript IDs. The vaccine pipeline needs TWO independent Ensembl calls (VEP for
# the annotation, then this for the protein sequence to build peptides); an outage
# in either one breaks the demo, so both have a sample fallback. See
# knowledge/sample_vep_annotations.json.
_SAMPLE_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge",
                                    "sample_vep_annotations.json")
_sample_sequences: dict[str, str] | None = None


def _sample_protein_sequence(transcript_id: str) -> str | None:
    global _sample_sequences
    if _sample_sequences is None:
        try:
            import json
            with open(_SAMPLE_FIXTURE_PATH, encoding="utf-8") as f:
                _sample_sequences = json.load(f).get("protein_sequences", {})
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[neoantigen] sample sequence fallback unavailable | %s", exc)
            _sample_sequences = {}
    return _sample_sequences.get(transcript_id)

# Real, well-characterized common HLA class I alleles (high population
# frequency per IEDB/literature) — used only as a stand-in for real
# patient-specific HLA typing, which this module cannot perform (see
# module docstring). Never presented to the user as this patient's actual
# genotype. Capped at 6 (2 per A/B/C locus) — MHCflurry's genotype-style
# prediction treats the allele list as one individual's real HLA class I
# genotype and rejects more than 6.
COMMON_HLA_ALLELES = [
    "HLA-A*02:01", "HLA-A*01:01",
    "HLA-B*07:02", "HLA-B*08:01",
    "HLA-C*07:01", "HLA-C*07:02",
]

_PEPTIDE_LENGTHS = (8, 9, 10, 11)
_STRONG_PERCENTILE = 2.0
_WEAK_PERCENTILE = 10.0

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from mhcflurry import Class1PresentationPredictor
        _predictor = Class1PresentationPredictor.load()
    return _predictor


def fetch_ensembl_protein_sequence(transcript_id: str, timeout: float = 15.0) -> str | None:
    """Real live Ensembl REST fetch of a transcript's translated protein
    sequence. Returns None on any failure (network error, unknown ID) —
    never a guessed/placeholder sequence."""
    try:
        # http_budget: one retry on a stall or transient 5xx, warm pooled connection,
        # bounded budget. Background-job call, so it can wait out a slow-but-alive
        # Ensembl instead of failing it.
        resp = http_budget.get(
            _ENSEMBL_SEQUENCE_URL.format(transcript_id=transcript_id),
            {"type": "protein", "content-type": "text/x-fasta"},
            budget=http_budget.Budget(timeout),
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None  # a genuine "no such transcript" — do NOT paper over with a fallback
        logger.warning("[neoantigen] Ensembl protein sequence fetch failed | transcript=%s error=%s", transcript_id, exc)
        cached = _sample_protein_sequence(transcript_id)
        if cached:
            logger.warning("[neoantigen] live Ensembl unavailable — using bundled sample protein "
                           "sequence for %s", transcript_id)
        return cached or None
    except httpx.HTTPError as exc:
        logger.warning("[neoantigen] Ensembl protein sequence fetch failed | transcript=%s error=%s", transcript_id, exc)
        cached = _sample_protein_sequence(transcript_id)
        if cached:
            logger.warning("[neoantigen] live Ensembl unavailable — using bundled sample protein "
                           "sequence for %s", transcript_id)
        return cached or None
    lines = resp.text.strip().splitlines()
    seq = "".join(line for line in lines if not line.startswith(">"))
    return seq or None


def generate_peptide_windows(protein_seq: str, protein_position: int, ref_aa: str, alt_aa: str) -> list[dict]:
    """
    protein_position is 1-based (VEP convention). Returns
    [{"length", "start", "mutant_peptide", "wildtype_peptide"}, ...] for
    every window length in _PEPTIDE_LENGTHS that fully contains the mutated
    residue and fits inside the real protein sequence bounds. Pure real
    sequence slicing — no fabricated residues. Refuses (returns []) if the
    fetched sequence doesn't actually have ref_aa at this position, since
    that means VEP's transcript version and the fetched sequence disagree
    and trusting the position further would be a guess.
    """
    idx = protein_position - 1
    if idx < 0 or idx >= len(protein_seq):
        logger.warning("[neoantigen] protein_position %d out of bounds for sequence of length %d", protein_position, len(protein_seq))
        return []
    if protein_seq[idx] != ref_aa:
        logger.warning(
            "[neoantigen] sequence mismatch at position %d: VEP says %s, fetched sequence has %s — skipping",
            protein_position, ref_aa, protein_seq[idx],
        )
        return []

    windows = []
    for length in _PEPTIDE_LENGTHS:
        for start in range(max(0, idx - length + 1), min(idx, len(protein_seq) - length) + 1):
            end = start + length
            wildtype_peptide = protein_seq[start:end]
            offset = idx - start
            mutant_peptide = wildtype_peptide[:offset] + alt_aa + wildtype_peptide[offset + 1:]
            windows.append({
                "length": length, "start": start,
                "mutant_peptide": mutant_peptide, "wildtype_peptide": wildtype_peptide,
            })
    return windows


def predict_neoantigen_candidates(mutations: list[dict], hla_alleles: list[str] | None = None) -> dict:
    """
    mutations: [{"gene_symbol", "transcript_id", "protein_position",
    "ref_aa", "alt_aa", "protein_change"}, ...] — pre-filtered to
    protein-coding missense variants already annotated by
    vcf_annotation_engine.annotate_variant_consequence().

    Returns {"candidates": [...], "errors": [...], "hla_alleles": [...]}.
    "candidates" is ranked strongest-and-most-foreign first:
      {"gene_symbol", "protein_change", "peptide_length", "mutant_peptide",
       "wildtype_peptide", "mutant_affinity_nm", "mutant_percentile",
       "wildtype_affinity_nm", "wildtype_percentile", "best_allele",
       "foreignness", "is_strong_binder", "is_self_similar"}
    Every affinity/percentile value is a real MHCflurry model output against
    a real peptide sequence — nothing here is inferred or fabricated.
    """
    hla_alleles = hla_alleles or COMMON_HLA_ALLELES
    errors: list[str] = []
    seq_cache: dict[str, str | None] = {}
    peptide_meta: list[dict] = []

    for mut in mutations:
        transcript_id = mut.get("transcript_id")
        protein_position = mut.get("protein_position")
        ref_aa = mut.get("ref_aa")
        alt_aa = mut.get("alt_aa")
        label = mut.get("gene_symbol") or transcript_id or "?"
        if not (transcript_id and protein_position and ref_aa and alt_aa):
            errors.append(f"{label}: 미완성 변이 주석(transcript/position/residue 정보 부족) — 건너뜀")
            continue

        if transcript_id not in seq_cache:
            seq_cache[transcript_id] = fetch_ensembl_protein_sequence(transcript_id)
        seq = seq_cache[transcript_id]
        if not seq:
            errors.append(f"{label} ({transcript_id}): Ensembl에서 단백질 서열을 가져오지 못했습니다")
            continue

        windows = generate_peptide_windows(seq, protein_position, ref_aa, alt_aa)
        if not windows:
            errors.append(f"{label} {mut.get('protein_change') or ''}: 유효한 펩타이드 윈도우를 생성하지 못했습니다 (서열 위치 불일치 또는 범위 초과)")
            continue

        for w in windows:
            peptide_meta.append({**mut, **w})

    if not peptide_meta:
        # Must match the full return's shape below (candidates/all_scored/errors/
        # hla_alleles) - the caller (neoantigen_pipeline.py) indexes "all_scored"
        # unconditionally. Omitting it here raised a real KeyError whenever every
        # mutation failed upstream (e.g. an Ensembl fetch timeout), turning a
        # legitimate "no candidates" result into a job that FAILED outright.
        return {"candidates": [], "all_scored": [], "errors": errors, "hla_alleles": hla_alleles}

    predictor = _get_predictor()
    mutant_peptides = sorted({m["mutant_peptide"] for m in peptide_meta})
    wildtype_peptides = sorted({m["wildtype_peptide"] for m in peptide_meta})

    mutant_df = predictor.predict(peptides=mutant_peptides, alleles=hla_alleles)
    wildtype_df = predictor.predict(peptides=wildtype_peptides, alleles=hla_alleles)

    mutant_by_pep = {
        row["peptide"]: row for row in mutant_df.to_dict("records")
    }
    wildtype_by_pep = {
        row["peptide"]: row for row in wildtype_df.to_dict("records")
    }

    candidates = []
    for meta in peptide_meta:
        m_row = mutant_by_pep.get(meta["mutant_peptide"])
        w_row = wildtype_by_pep.get(meta["wildtype_peptide"])
        if m_row is None or w_row is None:
            continue
        foreignness = round(float(w_row["presentation_percentile"]) - float(m_row["presentation_percentile"]), 3)
        candidates.append({
            "gene_symbol": meta.get("gene_symbol"),
            "protein_change": meta.get("protein_change"),
            "peptide_length": meta["length"],
            "mutant_peptide": meta["mutant_peptide"],
            "wildtype_peptide": meta["wildtype_peptide"],
            "mutant_affinity_nm": round(float(m_row["affinity"]), 1),
            "mutant_percentile": round(float(m_row["presentation_percentile"]), 3),
            "wildtype_affinity_nm": round(float(w_row["affinity"]), 1),
            "wildtype_percentile": round(float(w_row["presentation_percentile"]), 3),
            "best_allele": m_row["best_allele"],
            "foreignness": foreignness,
            "is_strong_binder": float(m_row["presentation_percentile"]) <= _STRONG_PERCENTILE,
            "is_self_similar": float(w_row["presentation_percentile"]) <= _WEAK_PERCENTILE,
        })

    # Real candidates only: strong mutant binder AND not self-similar
    # (wildtype counterpart would also have been presented, so the immune
    # system likely already tolerates it).
    real_candidates = [c for c in candidates if c["is_strong_binder"] and not c["is_self_similar"]]
    real_candidates.sort(key=lambda c: (c["mutant_percentile"], -c["foreignness"]))

    return {"candidates": real_candidates, "all_scored": candidates, "errors": errors, "hla_alleles": hla_alleles}


# ── Composite decision-support score ("AI Neo-Score") ──────────────────────
#
# A transparent, deterministic score over exactly three real, already-
# computed signals — same disclosed-formula discipline as
# target_intelligence_engine.calculate_target_priority_score() and
# decision_agent.calculate_priority_score() elsewhere in this codebase: no
# LLM, no invented inputs. Deliberately does NOT include a "clinical success
# probability" term — there is no real data source for that (no trial
# outcome data exists for an unvalidated candidate from sample/synthetic
# input), and fabricating one would violate this project's real-data-only
# discipline. This is a heuristic ranking aid for comparing candidates
# against each other, not a validated/peer-reviewed clinical scoring
# system, and must never be presented as a clinical adoption verdict.
_AFFINITY_WEIGHT = 30.0        # real MHCflurry binding affinity (nM), log-scaled
_PRESENTATION_WEIGHT = 40.0    # real MHCflurry presentation percentile
_FOREIGNNESS_WEIGHT = 30.0     # real percentile differential vs the wildtype counterpart
_AFFINITY_FLOOR_NM = 50.0      # nM at/below which affinity_component saturates at max
_AFFINITY_CEILING_NM = 500.0   # nM at/above which affinity_component is 0 (standard "strong binder" cutoff)


def calculate_neoantigen_composite_score(candidate: dict) -> dict:
    """
    candidate: one entry from predict_neoantigen_candidates()'s "candidates"
    or "all_scored" list. Returns {"composite_score": float (0-100),
    "breakdown": {...}} — every term traces back to a real MHCflurry output
    already computed for this exact candidate, nothing inferred.
    """
    import math

    affinity_nm = candidate["mutant_affinity_nm"]
    ratio = max(affinity_nm, _AFFINITY_FLOOR_NM) / _AFFINITY_FLOOR_NM
    ceiling_ratio = _AFFINITY_CEILING_NM / _AFFINITY_FLOOR_NM
    affinity_component = round(
        max(0.0, 1.0 - math.log10(ratio) / math.log10(ceiling_ratio)) * _AFFINITY_WEIGHT, 1,
    )

    presentation_component = round(
        max(0.0, 1.0 - candidate["mutant_percentile"] / _STRONG_PERCENTILE) * _PRESENTATION_WEIGHT, 1,
    )

    foreignness_component = round(
        min(max(candidate["foreignness"], 0.0), _FOREIGNNESS_WEIGHT), 1,
    )

    total = round(affinity_component + presentation_component + foreignness_component, 1)
    breakdown = {
        "affinity_component": affinity_component,
        "presentation_component": presentation_component,
        "foreignness_component": foreignness_component,
        "affinity_nm": affinity_nm,
        "mutant_percentile": candidate["mutant_percentile"],
        "foreignness": candidate["foreignness"],
    }
    return {"composite_score": total, "breakdown": breakdown}


# Real, condensed formula disclosure shown directly in the UI/job result —
# same content as docs/neoantigen_analysis_report_KRAS_G12D.md's Section 3,
# kept in sync manually since that doc is a one-off narrative report, not
# generated from this constant.
ALGORITHM_EXPLANATION = (
    "AI Neo-Score = 결합친화도(30점, MHCflurry IC50를 50~500nM 구간에서 로그 스케일로 정규화) + "
    "제시확률(40점, MHCflurry presentation percentile을 0~2.0 구간에서 선형 정규화) + "
    "비자기 신선도(30점, 야생형 대비 percentile 차이를 0~30 구간으로 캡핑) — 100점 만점. "
    "'임상 성공 가능성'은 이 변이/환자에 대한 실제 데이터가 없어 포함하지 않았습니다. "
    "이 점수는 후보 간 상대 비교를 돕는 휴리스틱이며, 검증된 임상 스코어링 시스템이 아닙니다."
)


def generate_ai_interpretation(candidates: list[dict], hla_note: str) -> str:
    """
    Real, LLM-narrated interpretation of the top real candidate's
    composite-score breakdown — strictly grounded in already-computed real
    numbers (calculate_neoantigen_composite_score's output plus the
    candidate's own MHCflurry fields); the LLM narrates, it never invents
    new values. Mirrors drug_discovery_agent.generate_ai_summary()'s
    real-data-only discipline. Explicitly instructed to never render a
    clinical-success-probability claim or an adoption verdict (recommend/
    hold/reject) — no real data source for either; see this module's
    calculate_neoantigen_composite_score() docstring for why those are
    excluded from the score itself.
    """
    from services.drug_discovery_chat import _chat, _use_demo

    if not candidates:
        return "강한 결합 + 비자기 조건을 모두 만족하는 신항원 후보가 없어 AI 해석을 생략합니다."

    top = candidates[0]
    score = calculate_neoantigen_composite_score(top)
    context = {
        "gene_symbol": top.get("gene_symbol"), "protein_change": top.get("protein_change"),
        "mutant_peptide": top.get("mutant_peptide"), "wildtype_peptide": top.get("wildtype_peptide"),
        "mutant_affinity_nm": top.get("mutant_affinity_nm"), "mutant_percentile": top.get("mutant_percentile"),
        "foreignness": top.get("foreignness"), "best_allele": top.get("best_allele"),
        "composite_score": score["composite_score"], "score_breakdown": score["breakdown"],
        "hla_note": hla_note,
    }
    def _factual_summary() -> str:
        """The same real numbers, stated plainly, with no LLM involved. Used when
        there is no API key AND when the LLM call fails — the narration is a garnish
        on a computation that already happened, so losing it must never mean losing
        the result."""
        return (
            f"{context['gene_symbol']} {context['protein_change']} 후보({context['mutant_peptide']})의 "
            f"AI Neo-Score는 {score['composite_score']}/100입니다 "
            f"(결합친화도 {score['breakdown']['affinity_component']}/30, "
            f"제시확률 {score['breakdown']['presentation_component']}/40, "
            f"foreignness {score['breakdown']['foreignness_component']}/30)."
        )

    if _use_demo():
        return _factual_summary()
    system = (
        "당신은 신항원 후보 데이터를 연구자에게 설명하는 과학 커뮤니케이터입니다. 아래 실제 계산된 "
        "수치(composite_score와 그 breakdown, 결합친화도, percentile, foreignness)만 근거로 2-4문장 "
        "해석을 작성하세요. 절대 새로운 수치를 지어내지 마세요. '임상 성공 가능성'이나 '채택/보류/기각' "
        "같은 임상적 판단은 절대 내리지 마세요 — 이 점수는 후보 비교용 휴리스틱일 뿐, 검증된 임상 "
        "스코어가 아닙니다. 분석 기준(hla_note의 인구집단 표준 HLA 6종)은 담백하게 사실만 언급하고, "
        "'실제 환자 유전형이 아니다' 같은 경고조 표현은 쓰지 마세요. 한국어로 작성하세요."
    )
    try:
        return _chat(system, f"실제 데이터:\n{context}", temperature=0.3, max_tokens=350) or ""
    except Exception as exc:
        logger.warning("[neoantigen] AI interpretation failed, using factual summary: %s", exc)
        return _factual_summary()
