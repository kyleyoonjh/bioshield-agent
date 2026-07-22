"""
Drug Discovery Assistant report generator — HTML (Jinja2) + PDF (ReportLab).

Mirrors services/report_service.py's approach for the primer-design pipeline
(same libraries, same backend/reports/ output directory, same
JSON+HTML+PDF triple) but is a fully separate implementation with its own
drug-domain template (ranked candidates / single docking result, not
primer/oligo fields) — no shared code, no shared filename prefix
("drugjob_" vs "assay_") so the two domains' report files never collide.

All numeric content here is copied verbatim from the pipeline's own
deterministic output (docking_engine, drug_ranking_engine) or the
already-generated ai_summary narrative — this module only formats, it
never computes or invents anything.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
try:
    os.makedirs(_REPORT_DIR, exist_ok=True)
except (PermissionError, OSError):
    _REPORT_DIR = "/tmp"

# Static, always-present reference glossary — real, established domain
# conventions (Vina's own scoring convention, drug_ranking_engine.py's
# documented formula, AlphaFold's own published pLDDT bands), not AI
# narration. Guarantees every report explains what each number means and
# which direction is better even if the AI summary above it is terse or
# unavailable.
_METRIC_GLOSSARY: list[dict[str, str]] = [
    {
        "label": "결합친화도 (Binding Affinity, kcal/mol)",
        "meaning": "AutoDock Vina가 예측한 리간드-단백질 결합 에너지.",
        "direction": "더 음수(낮을수록)일수록 강한 결합 — 대략 -6 이하 유의미, -8 이하 강한 결합.",
    },
    {
        "label": "종합 점수 (Final Score, 0-100)",
        "meaning": "결합친화도 점수(70%) + drug-likeness 점수(30%)의 가중합 (스크리닝 모드에서만 계산).",
        "direction": "100에 가까울수록 좋은 후보.",
    },
    {
        "label": "Drug-likeness 점수 (0-100) / Lipinski 위반",
        "meaning": "Lipinski's Rule of Five(분자량≤500, LogP≤5, 수소공여체≤5, 수소수용체≤10) 위반 횟수 기반.",
        "direction": "위반 횟수는 0에 가까울수록, 점수는 100에 가까울수록 경구 약물로 개발 가능성이 높음.",
    },
    {
        "label": "구조 신뢰도 (pLDDT, 0-100)",
        "meaning": "AlphaFold DB/ESMFold가 예측한 3D 구조의 신뢰도 (AlphaFold 공식 기준).",
        "direction": "90 이상 매우 높음 · 70-90 신뢰할 만함 · 50-70 낮음 · 50 미만 매우 낮음 — 100에 가까울수록 좋음.",
    },
    {
        "label": "데이터 소스 (실제 Vina / 휴리스틱)",
        "meaning": "'실제 Vina'는 물리 기반 도킹 시뮬레이션, '휴리스틱'은 Vina를 쓸 수 없을 때의 분자 특성 기반 근사치.",
        "direction": "'실제 Vina' 결과가 더 신뢰도 높음 — 휴리스틱 값은 참고용으로만 사용.",
    },
]


def _build_summary(job_id: str, generated_at: str, result: dict, ai_summary: str) -> dict:
    mode = result.get("mode")
    structure = result.get("structure") or {}
    report = result.get("report") or {}

    candidates: list[dict] = []
    if mode == "single":
        docking = result.get("docking_result") or {}
        ligand = docking.get("ligand_analysis") or {}
        candidates.append({
            "rank": 1,
            "name": "요청된 리간드",
            "category": None,
            "smiles": ligand.get("canonical_smiles"),
            "affinity": docking.get("best_affinity_kcal_mol"),
            "source": docking.get("source"),
            "lipinski_violations": ligand.get("lipinski_violations"),
            "score": None,
            "strength": [],
            "weakness": [],
        })
    else:
        for c in (result.get("ranked_candidates") or []):
            candidates.append({
                "rank": c.get("rank"), "name": c.get("name"), "category": c.get("category"),
                "smiles": c.get("smiles"), "affinity": c.get("best_affinity_kcal_mol"),
                "source": c.get("docking_source"), "lipinski_violations": None,
                "score": c.get("score"),
                "strength": c.get("strength") or [], "weakness": c.get("weakness") or [],
            })

    return {
        "job_id":              job_id,
        "generated_at":        generated_at,
        "mode":                mode,
        "uniprot_id":          structure.get("uniprot_id"),
        "structure_source":    result.get("structure_source"),
        "structure_confidence": structure.get("confidence"),
        "candidates":          candidates,
        "total_candidates":    len(candidates),
        "strengths":           report.get("strengths") or [],
        "weaknesses":          report.get("weaknesses") or [],
        "ai_summary":          ai_summary,
        "evaluation":          result.get("evaluation") or {},
        "metric_glossary":     _METRIC_GLOSSARY,
        "variant_context":     result.get("variant_context"),
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>Drug Discovery Report — {{ job_id }}</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; color: #1e293b; max-width: 860px; margin: 32px auto; padding: 0 16px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .meta { color: #64748b; font-size: 12px; margin-bottom: 20px; }
  .ai-summary { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 10px; padding: 14px 16px; font-size: 13px; line-height: 1.6; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 20px; }
  th, td { border-bottom: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; }
  th { background: #f8fafc; color: #475569; }
  .badge { font-size: 10px; padding: 1px 6px; border-radius: 8px; }
  .badge-vina { background: #dbeafe; color: #1d4ed8; }
  .badge-heuristic { background: #fef3c7; color: #92400e; }
  .section { font-size: 14px; font-weight: 700; margin: 18px 0 8px; }
  ul { font-size: 12px; margin: 4px 0; padding-left: 20px; }
  .strength { color: #166534; }
  .weakness { color: #92400e; }
  .candidate-detail { margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed #e2e8f0; }
  .candidate-detail-header { font-size: 12.5px; font-weight: 700; margin-bottom: 3px; }
  .candidate-detail ul { margin: 2px 0; }
  .variant-context { background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 12px 16px; font-size: 12px; margin-bottom: 20px; }
  .variant-context .mismatch { color: #b91c1c; font-weight: 600; }
  .variant-context .caveat { color: #92400e; margin-top: 6px; }
  .glossary { font-size: 11px; }
  .glossary dt { font-weight: 700; margin-top: 8px; color: #334155; }
  .glossary dd { margin: 2px 0 0 0; color: #475569; }
  .glossary dd.direction { color: #2563eb; }
</style></head>
<body>
  <h1>🔬 AiRemedy — Drug Discovery Report</h1>
  <p class="meta">Job {{ job_id }} · Generated {{ generated_at }} · Target {{ uniprot_id or "N/A" }}
    ({{ structure_source }}{% if structure_confidence %}, pLDDT {{ structure_confidence }}{% endif %}) ·
    Mode: {{ "단일 도킹" if mode == "single" else "라이브러리 스크리닝" }}</p>

  {% if variant_context %}
  <div class="variant-context">
    🧬 <b>체세포 변이 기반 타겟 (Track B)</b><br>
    유전자: <b>{{ variant_context.gene_symbol }}</b> · 단백질 변이: <b>{{ variant_context.protein_change }}</b>
    · 위치: {{ variant_context.source_variant.chrom }}:{{ variant_context.source_variant.pos }}
    {{ variant_context.source_variant.ref }}&gt;{{ variant_context.source_variant.alt }}
    {% if variant_context.vaf %} · VAF: {{ variant_context.vaf }}{% endif %}
    <br>주석 출처: Ensembl VEP (실시간 검증)
    {% if variant_context.label_mismatch %}
    <div class="mismatch">⚠ VCF 파일의 GENE={{ variant_context.vcf_gene_label }} 라벨과 실제 검증된 유전자({{ variant_context.gene_symbol }})가 다릅니다 — 실제 검증 결과를 사용했습니다.</div>
    {% endif %}
    <div class="caveat">⚠ {{ variant_context.structure_note }}</div>
  </div>
  {% endif %}

  {% if ai_summary %}
  <div class="ai-summary">🤖 <b>AI 해설</b><br>{{ ai_summary }}</div>
  {% endif %}

  <table>
    <tr><th>#</th><th>이름</th><th>분류</th><th>SMILES</th><th>친화도 (kcal/mol)</th><th>소스</th>{% if mode != "single" %}<th>종합 점수</th>{% endif %}</tr>
    {% for c in candidates %}
    <tr>
      <td>{{ c.rank }}</td><td>{{ c.name }}</td><td>{{ c.category or "-" }}</td>
      <td style="font-family: monospace; font-size: 10px;">{{ c.smiles or "-" }}</td>
      <td>{{ c.affinity if c.affinity is not none else "N/A" }}</td>
      <td><span class="badge badge-{{ c.source }}">{{ "실제 Vina" if c.source == "vina" else "휴리스틱" }}</span></td>
      {% if mode != "single" %}<td>{{ c.score if c.score is not none else "-" }}</td>{% endif %}
    </tr>
    {% endfor %}
  </table>

  {% if mode != "single" and candidates %}
  <div class="section">📋 후보별 상세 강점/약점 (전체 {{ candidates|length }}건)</div>
  {% for c in candidates %}
  <div class="candidate-detail">
    <div class="candidate-detail-header">#{{ c.rank }} {{ c.name }}{% if c.category %} ({{ c.category }}){% endif %} — 종합 점수 {{ c.score if c.score is not none else "-" }}</div>
    {% if c.strength %}<ul>{% for s in c.strength %}<li class="strength">✓ {{ s }}</li>{% endfor %}</ul>{% endif %}
    {% if c.weakness %}<ul>{% for w in c.weakness %}<li class="weakness">⚠ {{ w }}</li>{% endfor %}</ul>{% endif %}
    {% if not c.strength and not c.weakness %}<p style="font-size: 11px; color: #94a3b8; margin: 2px 0;">특이사항 없음</p>{% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if strengths %}
  <div class="section">✓ 전체 요약 강점</div>
  <ul>{% for s in strengths %}<li class="strength">{{ s }}</li>{% endfor %}</ul>
  {% endif %}
  {% if weaknesses %}
  <div class="section">⚠ 전체 요약 한계점</div>
  <ul>{% for w in weaknesses %}<li class="weakness">{{ w }}</li>{% endfor %}</ul>
  {% endif %}

  {% if metric_glossary %}
  <div class="section">📖 지표 해설 (각 수치의 의미와 좋은 방향)</div>
  <dl class="glossary">
    {% for m in metric_glossary %}
    <dt>{{ m.label }}</dt>
    <dd>{{ m.meaning }}</dd>
    <dd class="direction">→ {{ m.direction }}</dd>
    {% endfor %}
  </dl>
  {% endif %}

  <p class="meta">모든 구조/리간드/도킹 점수는 결정론적 엔진(RDKit, AlphaFold DB, ESMFold, AutoDock Vina) 전담 — AI는 해설만 담당.</p>
</body></html>"""


def _render_html(summary: dict) -> str:
    from jinja2 import Environment, Undefined  # type: ignore
    env = Environment(undefined=Undefined)
    tmpl = env.from_string(_HTML_TEMPLATE)
    return tmpl.render(**summary)


def _render_pdf(summary: dict, pdf_path: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    h1    = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=4)
    h2    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4)
    body  = styles["Normal"]
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.HexColor("#64748b"))
    ai    = ParagraphStyle("AI", parent=body, fontSize=9, textColor=colors.HexColor("#3730a3"), leading=13)
    green = ParagraphStyle("Green", parent=body, fontSize=9, textColor=colors.HexColor("#166534"))
    amber = ParagraphStyle("Amber", parent=body, fontSize=9, textColor=colors.HexColor("#854d0e"))

    story = []
    story.append(Paragraph("AiRemedy — Drug Discovery Report", h1))
    story.append(Paragraph(
        f"Job: <b>{summary['job_id']}</b> | Generated: {summary['generated_at']} | "
        f"Target: {summary.get('uniprot_id') or 'N/A'} ({summary.get('structure_source')}) | "
        f"Mode: {'단일 도킹' if summary['mode'] == 'single' else '라이브러리 스크리닝'}",
        small,
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 4 * mm))

    vc = summary.get("variant_context")
    if vc:
        red     = ParagraphStyle("Red", parent=body, fontSize=9, textColor=colors.HexColor("#b91c1c"))
        caveat  = ParagraphStyle("Caveat", parent=body, fontSize=9, textColor=colors.HexColor("#92400e"))
        sv = vc["source_variant"]
        story.append(Paragraph("체세포 변이 기반 타겟 (Track B)", h2))
        vaf_txt = f" | VAF: {vc['vaf']}" if vc.get("vaf") else ""
        story.append(Paragraph(
            f"유전자: <b>{vc['gene_symbol']}</b> | 단백질 변이: <b>{vc['protein_change']}</b> | "
            f"위치: {sv['chrom']}:{sv['pos']} {sv['ref']}&gt;{sv['alt']}{vaf_txt} | 주석 출처: Ensembl VEP",
            body,
        ))
        if vc.get("label_mismatch"):
            story.append(Paragraph(
                f"&#9888; VCF 파일의 GENE={vc.get('vcf_gene_label')} 라벨과 실제 검증된 유전자가 다릅니다 — "
                f"실제 검증 결과를 사용했습니다.", red,
            ))
        story.append(Paragraph(f"&#9888; {vc['structure_note']}", caveat))
        story.append(Spacer(1, 5 * mm))

    if summary.get("ai_summary"):
        story.append(Paragraph("AI 해설", h2))
        story.append(Paragraph(summary["ai_summary"], ai))
        story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("후보 목록", h2))
    header = ["#", "이름", "친화도 (kcal/mol)", "소스"] + (["점수"] if summary["mode"] != "single" else [])
    rows = [header]
    for c in summary["candidates"]:
        row = [str(c["rank"]), c["name"], str(c["affinity"]) if c["affinity"] is not None else "N/A",
               "실제 Vina" if c["source"] == "vina" else "휴리스틱"]
        if summary["mode"] != "single":
            row.append(str(c["score"]) if c["score"] is not None else "-")
        rows.append(row)
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 5 * mm))

    if summary["mode"] != "single" and summary["candidates"]:
        story.append(Paragraph(f"후보별 상세 강점/약점 (전체 {len(summary['candidates'])}건)", h2))
        cand_header = ParagraphStyle("CandHeader", parent=body, fontSize=9, textColor=colors.HexColor("#1e293b"), spaceBefore=5)
        no_note = ParagraphStyle("NoNote", parent=body, fontSize=8, textColor=colors.HexColor("#94a3b8"))
        for c in summary["candidates"]:
            score_label = c["score"] if c["score"] is not None else "-"
            story.append(Paragraph(
                f"<b>#{c['rank']} {c['name']}</b>" + (f" ({c['category']})" if c.get("category") else "") +
                f" — 종합 점수 {score_label}",
                cand_header,
            ))
            for s in (c.get("strength") or []):
                story.append(Paragraph(f"&#10003; {s}", green))
            for w in (c.get("weakness") or []):
                story.append(Paragraph(f"&#9888; {w}", amber))
            if not c.get("strength") and not c.get("weakness"):
                story.append(Paragraph("특이사항 없음", no_note))
        story.append(Spacer(1, 4 * mm))

    if summary.get("strengths"):
        story.append(Paragraph("전체 요약 강점", h2))
        for s in summary["strengths"]:
            story.append(Paragraph(f"&#10003; {s}", green))
        story.append(Spacer(1, 3 * mm))
    if summary.get("weaknesses"):
        story.append(Paragraph("전체 요약 한계점", h2))
        for w in summary["weaknesses"]:
            story.append(Paragraph(f"&#9888; {w}", amber))
        story.append(Spacer(1, 3 * mm))

    if summary.get("metric_glossary"):
        story.append(Paragraph("지표 해설 (각 수치의 의미와 좋은 방향)", h2))
        glossary_label = ParagraphStyle("GlossaryLabel", parent=body, fontSize=9, textColor=colors.HexColor("#334155"), spaceBefore=4)
        glossary_meaning = ParagraphStyle("GlossaryMeaning", parent=body, fontSize=8.5, textColor=colors.HexColor("#475569"))
        glossary_direction = ParagraphStyle("GlossaryDirection", parent=body, fontSize=8.5, textColor=colors.HexColor("#2563eb"))
        for m in summary["metric_glossary"]:
            story.append(Paragraph(f"<b>{m['label']}</b>", glossary_label))
            story.append(Paragraph(m["meaning"], glossary_meaning))
            story.append(Paragraph(f"&#8594; {m['direction']}", glossary_direction))

    doc.build(story)


# ── Neoantigen (mRNA vaccine) report — separate template/summary builder,
# same _REPORT_DIR + "drugjob_" filename prefix so the existing
# GET /api/drug-discovery/report/{job_id}(/pdf) endpoints and
# _find_latest_report() glob work unmodified (job_id is a fresh UUID per
# job regardless of pipeline, so there's no real collision risk sharing the
# prefix with docking-pipeline reports). All numeric content here is copied
# verbatim from neoantigen_pipeline.py's own MHCflurry/composite-score
# output — this module only formats.

_NEOANTIGEN_METRIC_GLOSSARY: list[dict[str, str]] = [
    {
        "label": "결합친화도 (Mutant Affinity, nM)",
        "meaning": "MHCflurry가 예측한 변이 펩타이드-MHC 결합 친화도(IC50).",
        "direction": "낮을수록 강한 결합 — 500nM 미만이 표준 'binder' 기준, 50nM 이하면 매우 강한 결합.",
    },
    {
        "label": "제시확률 (Presentation Percentile, %)",
        "meaning": "MHCflurry가 예측한, 이 펩타이드가 세포 표면 MHC에 실제로 제시될 상대적 순위(낮을수록 상위 %).",
        "direction": "2.0% 이하면 강한 바인더 — NetMHCpan/MHCflurry의 표준 판정 기준.",
    },
    {
        "label": "비자기 신선도 (Foreignness)",
        "meaning": "야생형(정상) 대응 펩타이드 대비 변이 펩타이드의 제시확률 차이 — 클수록 면역계가 기존에 접해보지 못한 '새로운' 항원.",
        "direction": "야생형 percentile이 10% 초과(자기 유사도 낮음)일 때만 후보로 채택 — foreignness가 클수록 좋음.",
    },
    {
        "label": "AI Neo-Score (0-100)",
        "meaning": "결합친화도(30점)+제시확률(40점)+비자기 신선도(30점)의 가중합 — 후보 간 상대 비교용 휴리스틱.",
        "direction": "100에 가까울수록 유망한 후보 — 단, 검증된 임상 스코어가 아니며 '임상 성공 가능성'을 의미하지 않음.",
    },
    {
        "label": "HLA 대립유전자 (population-common)",
        "meaning": "MHCflurry 예측에 사용된 HLA class I allele. 이 환경은 BAM 기반 실제 환자 HLA 타이핑을 지원하지 않아 인구집단 고빈도 allele을 대신 사용.",
        "direction": "이 환자의 실제 유전형이 아니므로 참고용으로만 사용 — 실제 적용 전 환자별 실제 HLA 타이핑이 필요.",
    },
]


def _neoantigen_conclusion(result: dict) -> str:
    """Deterministic, rule-based conclusion built only from real fields
    already in the pipeline result — same real-data-only discipline as
    neoantigen_engine.generate_ai_interpretation() (never a clinical
    verdict/adoption recommendation, since no real data source for that
    exists here)."""
    candidates = result.get("candidates") or []
    if not candidates:
        return (
            "실제 MHCflurry 예측 결과, 강한 결합(제시확률 ≤2%)과 비자기 조건(야생형 대비 foreignness "
            "충분)을 모두 만족하는 신항원 후보가 발견되지 않았습니다. 이는 실제 예측 결과이며(예: 보수적인 "
            "아미노산 치환에서 흔히 나타남), 알고리즘 실패가 아닙니다."
        )
    top = candidates[0]
    score = top.get("composite_score")
    score_txt = f", AI Neo-Score {score}/100" if score is not None else ""
    aff = top.get("mutant_affinity_nm")
    aff_txt = f" {aff}nM의 결합 친화도" if aff is not None else " 양호한 결합 친화도"
    allele = top.get("best_allele") or "표준 HLA"
    return (
        f"{top.get('gene_symbol')} {top.get('protein_change')} 변이를 기반으로 한 신항원 후보"
        f"({top.get('mutant_peptide')})가 선별되었으며, 표준 HLA 분석에서 {allele}과"
        f"{aff_txt}를 보여{score_txt} mRNA 암 백신 후보로서 추가 연구 가치가 있습니다"
        f" (조건 충족 후보 총 {len(candidates)}건). "
        "다만 본 결과는 예비(in silico) 분석입니다. 종양 변이는 이 환자의 실제 데이터이지만 HLA는 "
        "인구집단 표준값을 사용했으므로, 실제 개인 맞춤형 백신 개발을 위해서는 ①환자 고유의 HLA 타이핑, "
        "②실험적 면역원성 검증, ③임상적 안전성 평가가 필요합니다."
    )


def _build_neoantigen_summary(job_id: str, generated_at: str, result: dict) -> dict:
    candidates = result.get("candidates") or []
    return {
        "job_id":              job_id,
        "generated_at":        generated_at,
        "mutations_analyzed":  result.get("mutations_analyzed") or [],
        "hla_alleles":         result.get("hla_alleles") or [],
        "hla_note":            result.get("hla_note"),
        "bam_summary":         result.get("bam_summary"),
        "candidates":          candidates,
        "total_candidates":    len(candidates),
        "all_scored_count":    len(result.get("all_scored") or []),
        "prediction_errors":   result.get("prediction_errors") or [],
        "ai_interpretation":   result.get("ai_interpretation") or "",
        "algorithm_explanation": result.get("algorithm_explanation") or "",
        "literature_by_gene":  result.get("literature_by_gene") or {},
        "elapsed_seconds":     result.get("elapsed_seconds"),
        "conclusion":          _neoantigen_conclusion(result),
        "metric_glossary":     _NEOANTIGEN_METRIC_GLOSSARY,
    }


_NEOANTIGEN_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>Neoantigen / mRNA Vaccine Report — {{ job_id }}</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; color: #1e293b; max-width: 860px; margin: 32px auto; padding: 0 16px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .meta { color: #64748b; font-size: 12px; margin-bottom: 20px; }
  .hla-note { background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 12px 16px; font-size: 12px; margin-bottom: 20px; color: #92400e; }
  .ai-summary { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 10px; padding: 14px 16px; font-size: 13px; line-height: 1.6; margin-bottom: 20px; }
  .conclusion { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 16px; font-size: 13px; line-height: 1.6; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 20px; }
  th, td { border-bottom: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; }
  th { background: #f8fafc; color: #475569; }
  .section { font-size: 14px; font-weight: 700; margin: 18px 0 8px; }
  .algorithm { font-size: 12px; background: #f8fafc; border-radius: 10px; padding: 12px 16px; margin-bottom: 20px; }
  .glossary { font-size: 11px; }
  .glossary dt { font-weight: 700; margin-top: 8px; color: #334155; }
  .glossary dd { margin: 2px 0 0 0; color: #475569; }
  .glossary dd.direction { color: #2563eb; }
  .lit-gene { font-size: 12px; margin-bottom: 10px; }
  .lit-gene b { color: #334155; }
  .mono { font-family: monospace; font-size: 10px; }
</style></head>
<body>
  <h1>🧬 AiRemedy — Neoantigen / mRNA Vaccine Report</h1>
  <p class="meta">Job {{ job_id }} · Generated {{ generated_at }}
    {% if elapsed_seconds %} · 소요 시간 {{ elapsed_seconds }}초{% endif %}
    · 분석된 변이 {{ mutations_analyzed|length }}건 · 발견된 후보 {{ total_candidates }}건 (전체 스코어링 {{ all_scored_count }}건 중)</p>

  <div class="hla-note">⚠ <b>HLA 타이핑 관련 중요 제한사항</b><br>{{ hla_note }}</div>

  {% if ai_interpretation %}
  <div class="ai-summary">🤖 <b>AI 해석</b><br>{{ ai_interpretation }}</div>
  {% endif %}

  <div class="conclusion">✅ <b>결론</b><br>{{ conclusion }}</div>

  <div class="section">🧬 분석된 변이</div>
  <table>
    <tr><th>유전자</th><th>단백질 변이</th><th>위치</th><th>VAF</th></tr>
    {% for m in mutations_analyzed %}
    <tr>
      <td>{{ m.gene_symbol }}</td><td>{{ m.protein_change }}</td>
      <td>{{ m.source_variant.chrom }}:{{ m.source_variant.pos }} {{ m.source_variant.ref }}&gt;{{ m.source_variant.alt }}</td>
      <td>{{ m.vaf if m.vaf is not none else "-" }}</td>
    </tr>
    {% endfor %}
  </table>

  {% if candidates %}
  <div class="section">🎯 신항원 후보 (강한 결합 + 비자기 조건 모두 만족, AI Neo-Score 순)</div>
  <table>
    <tr><th>#</th><th>유전자</th><th>변이</th><th>펩타이드</th><th>HLA</th><th>친화도(nM)</th><th>제시확률(%)</th><th>Foreignness</th><th>AI Neo-Score</th></tr>
    {% for c in candidates %}
    <tr>
      <td>{{ loop.index }}</td><td>{{ c.gene_symbol }}</td><td>{{ c.protein_change }}</td>
      <td class="mono">{{ c.mutant_peptide }}</td><td>{{ c.best_allele }}</td>
      <td>{{ c.mutant_affinity_nm }}</td><td>{{ c.mutant_percentile }}</td><td>{{ c.foreignness }}</td>
      <td>{{ c.composite_score if c.composite_score is not none else "-" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if prediction_errors %}
  <div class="section">⚠ 예측 제외 항목</div>
  <ul>{% for e in prediction_errors %}<li>{{ e }}</li>{% endfor %}</ul>
  {% endif %}

  {% if literature_by_gene %}
  <div class="section">📚 관련 문헌 (실시간 PubMed 검증)</div>
  {% for gene, lit in literature_by_gene.items() %}
  <div class="lit-gene">
    <b>{{ gene }}</b> —
    {% if lit.available %}{{ lit.evidence_summary }}{% else %}검색된 관련 문헌 없음{% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if algorithm_explanation %}
  <div class="section">📐 알고리즘 설명</div>
  <div class="algorithm">{{ algorithm_explanation }}</div>
  {% endif %}

  {% if metric_glossary %}
  <div class="section">📖 지표 해설 (각 수치의 의미와 좋은 방향)</div>
  <dl class="glossary">
    {% for m in metric_glossary %}
    <dt>{{ m.label }}</dt>
    <dd>{{ m.meaning }}</dd>
    <dd class="direction">→ {{ m.direction }}</dd>
    {% endfor %}
  </dl>
  {% endif %}

  <p class="meta">모든 결합친화도/제시확률/foreignness 값은 실제 로컬 MHCflurry 모델(Class1PresentationPredictor) 출력 — AI는 해설만 담당.</p>
</body></html>"""


def _render_neoantigen_html(summary: dict) -> str:
    from jinja2 import Environment, Undefined  # type: ignore
    env = Environment(undefined=Undefined)
    tmpl = env.from_string(_NEOANTIGEN_HTML_TEMPLATE)
    return tmpl.render(**summary)


def _render_neoantigen_pdf(summary: dict, pdf_path: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    h1    = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=4)
    h2    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4)
    body  = styles["Normal"]
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.HexColor("#64748b"))
    ai    = ParagraphStyle("AI", parent=body, fontSize=9, textColor=colors.HexColor("#3730a3"), leading=13)
    amber = ParagraphStyle("Amber", parent=body, fontSize=9, textColor=colors.HexColor("#854d0e"))
    green_text = ParagraphStyle("GreenText", parent=body, fontSize=9, textColor=colors.HexColor("#166534"), leading=13)

    story = []
    story.append(Paragraph("AiRemedy — Neoantigen / mRNA Vaccine Report", h1))
    story.append(Paragraph(
        f"Job: <b>{summary['job_id']}</b> | Generated: {summary['generated_at']} | "
        f"변이 {len(summary['mutations_analyzed'])}건 | 후보 {summary['total_candidates']}건",
        small,
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("HLA 타이핑 관련 중요 제한사항", h2))
    story.append(Paragraph(f"&#9888; {summary.get('hla_note') or ''}", amber))
    story.append(Spacer(1, 4 * mm))

    if summary.get("ai_interpretation"):
        story.append(Paragraph("AI 해석", h2))
        story.append(Paragraph(summary["ai_interpretation"], ai))
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("결론", h2))
    story.append(Paragraph(summary["conclusion"], green_text))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("분석된 변이", h2))
    mut_rows = [["유전자", "단백질 변이", "위치", "VAF"]]
    for m in summary["mutations_analyzed"]:
        sv = m.get("source_variant") or {}
        mut_rows.append([
            m.get("gene_symbol") or "-", m.get("protein_change") or "-",
            f"{sv.get('chrom')}:{sv.get('pos')} {sv.get('ref')}>{sv.get('alt')}" if sv else "-",
            str(m.get("vaf")) if m.get("vaf") is not None else "-",
        ])
    mut_table = Table(mut_rows, hAlign="LEFT")
    mut_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(mut_table)
    story.append(Spacer(1, 5 * mm))

    if summary["candidates"]:
        story.append(Paragraph("신항원 후보 (AI Neo-Score 순)", h2))
        cand_rows = [["#", "유전자", "변이", "펩타이드", "친화도(nM)", "제시확률(%)", "Foreignness", "Neo-Score"]]
        for i, c in enumerate(summary["candidates"], 1):
            cand_rows.append([
                str(i), c.get("gene_symbol") or "-", c.get("protein_change") or "-",
                c.get("mutant_peptide") or "-", str(c.get("mutant_affinity_nm")),
                str(c.get("mutant_percentile")), str(c.get("foreignness")),
                str(c.get("composite_score")) if c.get("composite_score") is not None else "-",
            ])
        cand_table = Table(cand_rows, hAlign="LEFT")
        cand_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(cand_table)
        story.append(Spacer(1, 5 * mm))

    if summary.get("algorithm_explanation"):
        story.append(Paragraph("알고리즘 설명", h2))
        story.append(Paragraph(summary["algorithm_explanation"], body))
        story.append(Spacer(1, 5 * mm))

    if summary.get("metric_glossary"):
        story.append(Paragraph("지표 해설 (각 수치의 의미와 좋은 방향)", h2))
        glossary_label = ParagraphStyle("GlossaryLabel", parent=body, fontSize=9, textColor=colors.HexColor("#334155"), spaceBefore=4)
        glossary_meaning = ParagraphStyle("GlossaryMeaning", parent=body, fontSize=8.5, textColor=colors.HexColor("#475569"))
        glossary_direction = ParagraphStyle("GlossaryDirection", parent=body, fontSize=8.5, textColor=colors.HexColor("#2563eb"))
        for m in summary["metric_glossary"]:
            story.append(Paragraph(f"<b>{m['label']}</b>", glossary_label))
            story.append(Paragraph(m["meaning"], glossary_meaning))
            story.append(Paragraph(f"&#8594; {m['direction']}", glossary_direction))

    doc.build(story)


def generate_neoantigen_report(job_id: str, pipeline_result: dict) -> dict:
    """
    Writes {reports/}drugjob_{job_id}_{timestamp}.{json,html,pdf} for a
    completed neoantigen/mRNA-vaccine job — same filename prefix as
    generate_drug_report() (safe: job_id is a fresh UUID per job regardless
    of which pipeline produced it) so the existing report REST endpoints
    work for both without any router changes. Never raises — a failure on
    one format is logged and that path is set to None.
    """
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    iso_str = datetime.utcnow().isoformat() + "Z"
    base = f"drugjob_{job_id}_{now_str}"

    summary = _build_neoantigen_summary(job_id, iso_str, pipeline_result)

    json_path = os.path.join(_REPORT_DIR, f"{base}.json")
    html_path = os.path.join(_REPORT_DIR, f"{base}.html")
    pdf_path  = os.path.join(_REPORT_DIR, f"{base}.pdf")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[neoantigen_report] JSON write failed: %s", exc)
        json_path = None

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(_render_neoantigen_html(summary))
    except Exception as exc:
        logger.warning("[neoantigen_report] HTML render/write failed: %s", exc)
        html_path = None

    try:
        _render_neoantigen_pdf(summary, pdf_path)
    except Exception as exc:
        logger.warning("[neoantigen_report] PDF render failed: %s", exc)
        pdf_path = None

    return {"json_path": json_path, "html_path": html_path, "pdf_path": pdf_path, "base": base}


def generate_drug_report(job_id: str, pipeline_result: dict, ai_summary: str = "") -> dict:
    """
    Writes {reports/}drugjob_{job_id}_{timestamp}.{json,html,pdf} and returns
    the file paths. Never raises — a failure on one format is logged and that
    path is set to None so the pipeline result is still usable.
    """
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    iso_str = datetime.utcnow().isoformat() + "Z"
    base = f"drugjob_{job_id}_{now_str}"

    summary = _build_summary(job_id, iso_str, pipeline_result, ai_summary)

    json_path = os.path.join(_REPORT_DIR, f"{base}.json")
    html_path = os.path.join(_REPORT_DIR, f"{base}.html")
    pdf_path  = os.path.join(_REPORT_DIR, f"{base}.pdf")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("[drug_report] JSON write failed: %s", exc)
        json_path = None

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(_render_html(summary))
    except Exception as exc:
        logger.warning("[drug_report] HTML render/write failed: %s", exc)
        html_path = None

    try:
        _render_pdf(summary, pdf_path)
    except Exception as exc:
        logger.warning("[drug_report] PDF render failed: %s", exc)
        pdf_path = None

    return {"json_path": json_path, "html_path": html_path, "pdf_path": pdf_path, "base": base}
