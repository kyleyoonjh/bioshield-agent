import { useState, useRef, useEffect, useCallback } from "react";
import { logFetch } from "../utils/apiLogger";

// ── Types ────────────────────────────────────────────────────────────────────
// Standalone types for this panel — no imports from the primer-design agent's
// types/components, per this feature's isolation requirement.

interface ChatMessage {
  role: "user" | "agent" | "progress" | "results" | "panel";
  text?: string;
  result?: DrugDiscoveryResult;
  panel?: AgentPanel;
  jobId?: string;
  elapsedMs?: number;
}

// ── Agent panel types ────────────────────────────────────────────────────────
// Structured payloads from the real Literature/Clinical/Target-Intelligence/
// Compound-Discovery/SAR-Optimization/Decision agents (see backend/services/
// drug_discovery_chat.py's answer_*_question() functions and
// drug_discovery_router.py's /converse "panel" field). Every field here
// mirrors real data those functions already fetched/computed — nothing is
// rendered here that wasn't already real on the backend.

interface LiteraturePaper {
  pmid: string;
  title: string;
  abstract?: string;
  journal?: string;
  year?: string;
  authors?: string;
  doi?: string;
  url: string;
}

interface LiteraturePanelData {
  type: "literature_search";
  available: boolean;
  query?: string;
  evidence_summary?: string;
  key_findings?: { finding: string; pmid: string }[];
  limitations?: string;
  papers?: LiteraturePaper[];
}

interface ClinicalTrialRecord {
  nct_id: string;
  brief_title?: string;
  overall_status?: string;
  phases?: string[];
  conditions?: string[];
  interventions?: { type?: string; name?: string }[];
  lead_sponsor?: string;
  url: string;
}

interface ClinicalPanelData {
  type: "clinical_search";
  available: boolean;
  query?: string;
  landscape_summary?: string;
  key_trials?: { note: string; nct_id: string }[];
  development_stage_assessment?: string;
  trials?: ClinicalTrialRecord[];
}

interface TargetDisease {
  name: string;
  acronym?: string;
  description?: string;
  mim_id?: string | null;
}

interface TargetPathway {
  name: string;
  stable_id: string;
  in_disease: boolean;
  url: string;
}

interface OpenTargetsDisease {
  name: string;
  score: number;
}

interface TargetIntelligencePanelData {
  type: "target_intelligence";
  available: boolean;
  target_name?: string;
  uniprot_id?: string;
  function_summary?: string;
  diseases?: TargetDisease[];
  pathways?: TargetPathway[];
  opentargets_diseases?: OpenTargetsDisease[];
  opentargets_tractability?: string[];
  priority_score?: number;
  priority_breakdown?: Record<string, number>;
  known_inhibitor_count?: number;
}

interface AdmetProfile {
  valid: boolean;
  oral_absorption?: { prediction: "high" | "low"; score: number; basis: string };
  hepatotoxicity?: { risk: "low" | "moderate" | "flagged"; alerts: string[]; basis: string };
  pains?: { flagged: boolean; alerts: string[]; basis: string };
  synthesis?: { score: number | null; basis: string };
}

interface CompoundDiscoveryItem {
  // ChEMBL known-inhibitor shape
  chembl_id?: string;
  name?: string;
  ic50_nm?: number;
  document_year?: number;
  // PubChem similar-compound shape
  cid?: number;
  iupac_name?: string;
  molecular_weight?: string | number;
  smiles?: string;
  url: string;
}

interface CompoundDiscoveryPanelData {
  type: "compound_discovery";
  available: boolean;
  kind: "known_inhibitors" | "similar_compounds";
  label?: string;
  items?: CompoundDiscoveryItem[];
}

interface SarAnalog {
  transformation: string;
  rationale: string;
  smiles: string;
  docked: boolean;
  source?: string;
  best_affinity_kcal_mol?: number;
  delta_kcal_mol?: number | null;
  improved?: boolean | null;
  admet?: AdmetProfile | null;
}

interface SarOptimizationPanelData {
  type: "sar_optimization";
  available: boolean;
  reason?: string;
  base_name?: string;
  base_smiles?: string;
  base_affinity_kcal_mol?: number;
  analogs?: SarAnalog[];
  note?: string;
}

interface DecisionReportPanelData {
  type: "decision_report";
  available: boolean;
  candidate_name?: string;
  priority_score?: number;
  breakdown?: Record<string, number>;
  development_risk?: "low" | "moderate" | "high";
  risk_rationale?: string;
  overall_recommendation?: string;
  recommended_next_experiment?: string;
  target_name?: string;
  has_target_context?: boolean;
}

type AgentPanel =
  | LiteraturePanelData
  | ClinicalPanelData
  | TargetIntelligencePanelData
  | CompoundDiscoveryPanelData
  | SarOptimizationPanelData
  | DecisionReportPanelData;

interface LigandAnalysis {
  valid: boolean;
  molecular_weight?: number;
  logp?: number;
  h_bond_donors?: number;
  h_bond_acceptors?: number;
  tpsa?: number;
  rotatable_bonds?: number;
  lipinski_violations?: number;
  drug_like?: boolean;
  error?: string;
}

interface DockingConfidence {
  pose_count: number;
  mean_rmsd_top_poses_angstrom: number;
  affinity_spread_top_poses_kcal_mol: number;
  pose_consistency_score: number;
  method: string;
}

interface DockingResult {
  docked: boolean;
  source: "vina" | "heuristic" | "none";
  best_affinity_kcal_mol?: number;
  docking_confidence?: DockingConfidence | null;
  note?: string;
  error?: string;
  ligand_analysis?: LigandAnalysis;
}

interface RankedCandidate {
  rank: number;
  name: string;
  category?: string;
  smiles: string;
  score: number;
  score_breakdown: { affinity: number; drug_likeness: number };
  docking_source: "vina" | "heuristic" | "none";
  best_affinity_kcal_mol?: number;
  docking_confidence?: DockingConfidence | null;
  strength: string[];
  weakness: string[];
}

interface DrugDiscoveryResult {
  mode: "single" | "screen" | "neoantigen";
  structure_source: string;
  docking_result: DockingResult | null;
  ranked_candidates: RankedCandidate[] | null;
  report: { strengths: string[]; weaknesses: string[] };
  ai_summary?: string;
  report_available?: boolean;
  evaluation?: { verdict: string; failed_metric: string | null };
  error?: string;
  // mode: "neoantigen" fields — see services/neoantigen_pipeline.py's
  // run_neoantigen_pipeline() return shape. Real MHCflurry-derived
  // predictions only; hla_note discloses that BAM-based HLA typing itself
  // isn't performed (see services/neoantigen_engine.py's module docstring).
  mutations_analyzed?: { gene_symbol: string; protein_change: string }[];
  hla_alleles?: string[];
  hla_note?: string;
  bam_summary?: { ref_name?: string; ref_length?: number; read_count?: number; error?: string } | null;
  candidates?: NeoantigenCandidate[];
  all_scored?: NeoantigenCandidate[];
  prediction_errors?: string[];
  // Real, code-computed composite score (services/neoantigen_engine.py's
  // calculate_neoantigen_composite_score) + LLM narration strictly grounded
  // in it (generate_ai_interpretation) — deliberately excludes a "clinical
  // success probability" term and any adoption verdict, since no real data
  // source exists for either (see docs/neoantigen_analysis_report_KRAS_G12D.md).
  ai_interpretation?: string;
  algorithm_explanation?: string;
  literature_by_gene?: Record<string, {
    available: boolean; query: string; evidence_summary?: string;
    papers?: LiteraturePaper[];
  }>;
  // Deterministic (no LLM), computed in neoantigen_pipeline.py based on
  // whether any real candidate passed the strong-binder + foreign filter.
  next_step_suggestion?: string;
}

interface NeoantigenCandidate {
  gene_symbol: string;
  protein_change: string;
  peptide_length: number;
  mutant_peptide: string;
  wildtype_peptide: string;
  mutant_affinity_nm: number;
  mutant_percentile: number;
  wildtype_affinity_nm: number;
  wildtype_percentile: number;
  best_allele: string;
  foreignness: number;
  is_strong_binder: boolean;
  is_self_similar: boolean;
  composite_score?: number;
  score_breakdown?: {
    affinity_component: number; presentation_component: number; foreignness_component: number;
  };
}

interface DrugDiscoveryJob {
  status: "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_step: number;
  total_steps: number;
  current_message: string;
  result: DrugDiscoveryResult | null;
  error_message: string | null;
}

const POLL_INTERVAL_MS = 1000;

// Replaces the earlier per-pathogen target list (스파이크/뉴클레오캡시드/...)
// with one example per real FUNCTION/CAPABILITY the assistant has, on
// explicit user request — the old list only demonstrated "pick a target to
// screen," which undersold the Literature/Clinical/Target-Intelligence/
// Compound-Discovery agents built alongside it. Each message below is a
// real, complete example (not a partial template) using SARS-CoV-2 Spike
// (P0DTC2, curated + cached) as the running example so results come back
// fast; a user can send it as-is or edit the target/drug name before
// sending. Every message is verified real routing text against the
// backend's keyword-based intent detectors (see drug_discovery_chat.py's
// is_literature_query/is_clinical_query/is_target_intelligence_query) and,
// for 타겟 분석 specifically, against a real fix: standalone
// target-intelligence messages (no prior job/known_slots) now resolve a
// curated target straight from the message text itself (see
// drug_discovery_router.py's fuzzy-match fallback), verified live to
// return real UniProt/Reactome data with zero prior context.
const FUNCTION_EXAMPLES: { label: string; message: string }[] = [
  { label: "🧬 신약 스크리닝", message: "SARS-CoV-2 스파이크 단백질을 억제할 수 있는 승인된 약물을 찾아줘" },
  { label: "📚 문헌 검색", message: "SARS-CoV-2 스파이크 단백질 관련 논문 찾아줘" },
  { label: "🏥 임상시험 조회", message: "SARS-CoV-2 스파이크 단백질 관련 임상시험 현황 알려줘" },
  { label: "🎯 타겟 분석", message: "SARS-CoV-2 스파이크 단백질의 질병 연관성과 관련 경로 분석해줘" },
  { label: "🧫 화합물 검색", message: "아스피린과 유사한 화합물 찾아줘" },
];

// Shown only right after a screening/docking job completes (see
// lastCompletedJobId) — an "Elicitation" affordance for the conversational
// agent's newer capabilities (Literature/Clinical/Target-Intelligence/
// Compound-Discovery/SAR/Decision agents, all built on real PubMed/
// ClinicalTrials.gov/UniProt/Reactome/PubChem/ChEMBL data + real re-docking
// for SAR), which are otherwise only discoverable by already knowing the
// right thing to type. Every button's message is real routing text
// verified against the backend's keyword-based intent detectors (see
// drug_discovery_chat.py's is_literature_query/is_clinical_query/
// is_target_intelligence_query/is_sar_optimization_query/
// is_decision_report_query).
const POST_RESULT_QUICK_ACTIONS: { label: string; message: string }[] = [
  { label: "📚 관련 논문", message: "관련 논문 찾아줘" },
  { label: "🏥 임상시험 현황", message: "임상시험 현황 알려줘" },
  { label: "🎯 타겟 질병 연관성/경로", message: "이 타겟의 질병 연관성과 관련 경로 분석해줘" },
  { label: "🧬 결합 포켓 분석", message: "결합 포켓 주변 아미노산 잔기와 상호작용을 분석해줘" },
  { label: "🧪 구조 개선 (SAR)", message: "이 후보의 구조를 개선할 수 있는 유사체 찾아줘" },
  { label: "📊 종합 평가 리포트", message: "종합 평가 리포트 만들어줘 (우선순위 점수, 개발 위험도)" },
];

// Mirrors drug_discovery_intent.py's _is_cancer_topic — cosmetic only (the
// backend is the real source of truth for routing, since neoantigen_mode
// comes from its own /converse response), used here just so this quick-pick
// button's label doesn't say "신약 스크리닝" for a topic the backend will
// actually route to the neoantigen pipeline.
const CANCER_TOPIC_RE = /[가-힣]{2,}암|cancer|tumor|종양|carcinoma|sarcoma/i;

function makeSessionId(): string {
  return `dd-sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// Persisted across page refreshes (localStorage, not just React state) —
// otherwise F5 wipes the session id, chat history, and the pointer to any
// still-running background job. Since jobs run as backend background tasks
// independent of the browser connection, a refresh alone never stops one;
// without this persistence a user who refreshes mid-job loses all visible
// context and, not realizing the first request is still in flight, can
// resend the same request and end up with two genuinely separate completed
// jobs/reports (reported as "리포트가 2개 생성됨").
const STORAGE_KEY_SESSION = "dd-session-id";
const STORAGE_KEY_MESSAGES = "dd-messages";
const STORAGE_KEY_ACTIVE_JOB = "dd-active-job-id";

function getOrCreateSessionId(): string {
  const existing = localStorage.getItem(STORAGE_KEY_SESSION);
  if (existing) return existing;
  const fresh = makeSessionId();
  localStorage.setItem(STORAGE_KEY_SESSION, fresh);
  return fresh;
}

function loadStoredMessages(): ChatMessage[] | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_MESSAGES);
    return raw ? (JSON.parse(raw) as ChatMessage[]) : null;
  } catch {
    return null;
  }
}

// ── Result cards ─────────────────────────────────────────────────────────────

function SourceBadge({ source }: { source: "vina" | "heuristic" | "none" }) {
  return (
    <span
      className={`ml-1 text-[9px] px-1 py-0.5 rounded ${
        source === "vina" ? "bg-blue-100 text-blue-700" : "bg-amber-100 text-amber-700"
      }`}
    >
      {source === "vina" ? "실제 Vina" : "휴리스틱"}
    </span>
  );
}

function ReportLinks({ jobId, available }: { jobId?: string; available?: boolean }) {
  if (!jobId || !available) return null;
  return (
    <div className="flex gap-2 mt-2 pt-2 border-t border-slate-100">
      <a
        href={`/api/drug-discovery/report/${jobId}`}
        target="_blank"
        rel="noreferrer"
        className="text-[10px] px-2 py-1 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
      >
        📄 HTML 리포트
      </a>
      <a
        href={`/api/drug-discovery/report/${jobId}/pdf`}
        target="_blank"
        rel="noreferrer"
        className="text-[10px] px-2 py-1 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
      >
        ⬇ PDF 다운로드
      </a>
    </div>
  );
}

function AiSummary({ text }: { text?: string }) {
  if (!text) return null;
  return (
    // whitespace-pre-wrap: the AI summary now ends with a real, multi-line
    // "추천 다음 행동" (recommended next actions) section (see
    // drug_discovery_agent.py's generate_ai_summary()) — without this, the
    // "\n" line breaks between recommended actions collapsed into one
    // run-on paragraph.
    <div className="text-[10px] text-indigo-700 bg-indigo-50 rounded-lg px-2.5 py-1.5 mb-2 leading-relaxed whitespace-pre-wrap">
      🤖 {text}
    </div>
  );
}

function ResultCard({ result, jobId }: { result: DrugDiscoveryResult; jobId?: string }) {
  const dock = result.docking_result;
  const lig = dock?.ligand_analysis;
  const { report } = result;
  return (
    <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 px-3.5 py-3 max-w-[85%]">
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-sm">🧪</span>
        <span className="text-xs font-bold text-emerald-700">신약개발 분석 완료 (단일 도킹)</span>
      </div>

      <AiSummary text={result.ai_summary} />

      <div className="grid grid-cols-2 gap-1.5 text-[10px] mb-2">
        <div className="bg-white rounded-lg px-2 py-1.5 border border-emerald-100">
          <p className="text-slate-400">구조 출처</p>
          <p className="font-semibold text-slate-700">{result.structure_source}</p>
        </div>
        <div className="bg-white rounded-lg px-2 py-1.5 border border-emerald-100">
          <p className="text-slate-400">도킹 점수 (블라인드)</p>
          <p className="font-semibold text-slate-700">
            {dock?.best_affinity_kcal_mol ?? "N/A"} kcal/mol
            {dock && <SourceBadge source={dock.source} />}
          </p>
        </div>
      </div>

      {lig?.valid && (
        <div className="text-[10px] text-slate-500 mb-2">
          MW {lig.molecular_weight} · LogP {lig.logp} · Lipinski 위반 {lig.lipinski_violations}개
          {lig.drug_like ? " · drug-like ✓" : ""}
        </div>
      )}

      {report.strengths.length > 0 && (
        <ul className="text-[10px] text-emerald-700 space-y-0.5 mb-1">
          {report.strengths.map((s, i) => (
            <li key={i}>✓ {s}</li>
          ))}
        </ul>
      )}
      {report.weaknesses.length > 0 && (
        <ul className="text-[10px] text-amber-700 space-y-0.5">
          {report.weaknesses.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}
      <ReportLinks jobId={jobId} available={result.report_available} />
    </div>
  );
}

function RankedResultsCard({ result, jobId }: { result: DrugDiscoveryResult; jobId?: string }) {
  const candidates = result.ranked_candidates ?? [];
  return (
    <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 px-3.5 py-3 max-w-[92%]">
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-sm">🔬</span>
        <span className="text-xs font-bold text-indigo-700">약물 후보 스크리닝 완료 ({result.structure_source})</span>
      </div>

      <AiSummary text={result.ai_summary} />

      {candidates.length === 0 ? (
        <p className="text-[11px] text-amber-700">
          큐레이션된 약물 라이브러리에서 필터를 통과한 후보가 없었습니다. {result.report.weaknesses[0] ?? ""}
        </p>
      ) : (
        <div className="space-y-1.5">
          {candidates.map(c => (
            <div key={c.rank} className="bg-white rounded-lg px-2.5 py-1.5 border border-indigo-100">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-slate-700">
                  #{c.rank} {c.name}
                  {c.category && <span className="ml-1 text-[9px] text-slate-400">({c.category})</span>}
                </span>
                <span className="text-[10px] font-bold text-indigo-600">score {c.score}</span>
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {c.best_affinity_kcal_mol ?? "N/A"} kcal/mol
                <SourceBadge source={c.docking_source} />
                <span className="ml-1.5">· affinity {c.score_breakdown.affinity} / drug-likeness {c.score_breakdown.drug_likeness}</span>
              </div>
              {c.strength.length > 0 && (
                <ul className="mt-0.5 space-y-0.5">
                  {c.strength.map((s, i) => (
                    <li key={i} className="text-[9px] text-emerald-600">✓ {s}</li>
                  ))}
                </ul>
              )}
              {c.weakness.length > 0 && (
                <ul className="mt-0.5 space-y-0.5">
                  {c.weakness.map((w, i) => (
                    <li key={i} className="text-[9px] text-amber-600">⚠ {w}</li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}

      {result.evaluation && result.evaluation.verdict !== "pass" && (
        <p className="text-[9px] text-slate-400 mt-2">
          품질 기준 미달 상태로 최대 재시도 후 종료됨 ({result.evaluation.failed_metric})
        </p>
      )}
      <ReportLinks jobId={jobId} available={result.report_available} />
    </div>
  );
}

function NeoantigenResultsCard({ result, jobId }: { result: DrugDiscoveryResult; jobId?: string }) {
  const candidates = result.candidates ?? [];
  const scoredCount = result.all_scored?.length ?? 0;
  return (
    <div className="rounded-xl border border-purple-200 bg-purple-50/40 px-3.5 py-3 max-w-[92%]">
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-sm">🧬</span>
        <span className="text-xs font-bold text-purple-700">신항원 후보 식별 완료</span>
      </div>

      {result.hla_note && (
        <div className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5 mb-2 leading-relaxed">
          ⚠ {result.hla_note}
        </div>
      )}

      {result.bam_summary && !result.bam_summary.error && (
        <div className="text-[10px] text-slate-500 mb-2">
          BAM: {result.bam_summary.ref_name} 참조 · 리드 {result.bam_summary.read_count}개 (실제 파싱 결과, 헤더/리드 수 확인용 — HLA 타이핑에는 미사용)
        </div>
      )}

      {result.mutations_analyzed && result.mutations_analyzed.length > 0 && (
        <div className="text-[10px] text-slate-600 mb-2">
          분석된 변이: {result.mutations_analyzed.map(m => `${m.gene_symbol} ${m.protein_change}`).join(", ")}
        </div>
      )}

      {candidates.length === 0 ? (
        <p className="text-[11px] text-amber-700 mb-1">
          강한 결합 + 비자기(non-self) 조건을 모두 만족하는 신항원 후보가 없었습니다
          {scoredCount > 0 && ` (실제 MHCflurry로 평가한 ${scoredCount}개 펩타이드 윈도우 중 0건 통과).`}
          {" "}이는 정상적인 결과일 수 있습니다 — 변이가 MHC 결합에 미치는 영향이 미미하면(예: 유사한 성질의 아미노산 치환) 실제로 면역원성이 낮습니다.
        </p>
      ) : (
        <div className="space-y-1.5">
          {candidates.map((c, i) => (
            <div key={i} className="bg-white rounded-lg px-2.5 py-1.5 border border-purple-100">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-slate-700">
                  {c.gene_symbol} {c.protein_change} <span className="font-mono">{c.mutant_peptide}</span>
                </span>
                {c.composite_score !== undefined ? (
                  <span className="text-[10px] font-bold text-purple-600">AI Neo-Score {c.composite_score}/100</span>
                ) : (
                  <span className="text-[10px] font-bold text-purple-600">percentile {c.mutant_percentile}</span>
                )}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                결합 친화도 {c.mutant_affinity_nm} nM · percentile {c.mutant_percentile} · allele {c.best_allele} · foreignness {c.foreignness}
              </div>
              {c.score_breakdown && (
                <div className="text-[9px] text-purple-500 mt-0.5">
                  결합친화도 {c.score_breakdown.affinity_component}/30 · 제시확률 {c.score_breakdown.presentation_component}/40 · foreignness {c.score_breakdown.foreignness_component}/30
                </div>
              )}
              <div className="text-[9px] text-slate-400 mt-0.5">
                야생형 대응 펩타이드 {c.wildtype_peptide} — percentile {c.wildtype_percentile} (self, 낮은 제시 가능성)
              </div>
            </div>
          ))}
        </div>
      )}

      {result.ai_interpretation && (
        <div className="text-[10px] text-purple-700 bg-purple-50 rounded-lg px-2.5 py-1.5 mt-2 leading-relaxed whitespace-pre-wrap">
          🤖 {result.ai_interpretation}
        </div>
      )}

      {result.algorithm_explanation && (
        <details className="mt-2">
          <summary className="text-[9px] text-slate-400 cursor-pointer hover:text-slate-600">AI Neo-Score 산출 방식 (알고리즘 설명)</summary>
          <p className="text-[9px] text-slate-500 mt-1 leading-relaxed">{result.algorithm_explanation}</p>
        </details>
      )}

      {result.literature_by_gene && Object.entries(result.literature_by_gene).map(([gene, lit]) => (
        lit.available ? (
          <div key={gene} className="text-[10px] text-indigo-700 bg-indigo-50 rounded-lg px-2.5 py-1.5 mt-2 leading-relaxed">
            📚 {gene} 관련 문헌: {lit.evidence_summary}
          </div>
        ) : null
      ))}

      {result.prediction_errors && result.prediction_errors.length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {result.prediction_errors.map((e, i) => (
            <li key={i} className="text-[9px] text-amber-600">⚠ {e}</li>
          ))}
        </ul>
      )}

      {result.next_step_suggestion && (
        <p className="text-[10px] text-slate-500 mt-2">{result.next_step_suggestion}</p>
      )}

      <ReportLinks jobId={jobId} available={result.report_available} />
    </div>
  );
}

// ── Agent panel cards ────────────────────────────────────────────────────────
// One dedicated card per real agent (Literature/Clinical/Target-Intelligence/
// Compound-Discovery/SAR-Optimization/Decision) instead of a plain text
// bubble — every value rendered here is a field already computed/fetched by
// the backend (see the AgentPanel types above), never derived/guessed here.

function PanelShell({
  icon, title, colorClass, borderClass, bgClass, unavailableText, children,
}: {
  icon: string; title: string; colorClass: string; borderClass: string; bgClass: string;
  unavailableText?: string; children?: React.ReactNode;
}) {
  return (
    <div className={`rounded-xl border ${borderClass} ${bgClass} px-3.5 py-3 max-w-[92%]`}>
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-sm">{icon}</span>
        <span className={`text-xs font-bold ${colorClass}`}>{title}</span>
      </div>
      {unavailableText ? (
        <p className="text-[11px] text-amber-700">{unavailableText}</p>
      ) : (
        children
      )}
    </div>
  );
}

function LiteraturePanelCard({ data }: { data: LiteraturePanelData }) {
  if (!data.available) {
    return (
      <PanelShell icon="📚" title="관련 문헌" colorClass="text-sky-700" borderClass="border-sky-200" bgClass="bg-sky-50/40"
        unavailableText={data.query ? `'${data.query}'에 대해 PubMed에서 관련 논문을 찾지 못했습니다.` : "관련 논문을 찾지 못했습니다."} />
    );
  }
  return (
    <PanelShell icon="📚" title={`관련 문헌 (PubMed 실시간 검색${data.query ? `: ${data.query}` : ""})`}
      colorClass="text-sky-700" borderClass="border-sky-200" bgClass="bg-sky-50/40">
      {data.evidence_summary && (
        <p className="text-[10px] text-sky-900 bg-white/70 rounded-lg px-2.5 py-1.5 mb-2 leading-relaxed">{data.evidence_summary}</p>
      )}
      <div className="space-y-1.5">
        {(data.papers ?? []).map(p => (
          <div key={p.pmid} className="bg-white rounded-lg px-2.5 py-1.5 border border-sky-100">
            <a href={p.url} target="_blank" rel="noreferrer" className="text-[11px] font-semibold text-sky-700 hover:underline">
              {p.title}
            </a>
            <div className="text-[9px] text-slate-400 mt-0.5">
              {p.authors ? `${p.authors} · ` : ""}{p.journal ?? "?"} ({p.year ?? "?"}) · PMID {p.pmid}
            </div>
          </div>
        ))}
      </div>
      {data.limitations && <p className="text-[9px] text-slate-400 mt-2">한계: {data.limitations}</p>}
    </PanelShell>
  );
}

function ClinicalPanelCard({ data }: { data: ClinicalPanelData }) {
  if (!data.available) {
    return (
      <PanelShell icon="🏥" title="임상시험 현황" colorClass="text-teal-700" borderClass="border-teal-200" bgClass="bg-teal-50/40"
        unavailableText={data.query ? `'${data.query}'에 대해 ClinicalTrials.gov에서 관련 임상시험을 찾지 못했습니다.` : "관련 임상시험을 찾지 못했습니다."} />
    );
  }
  return (
    <PanelShell icon="🏥" title={`임상시험 현황 (ClinicalTrials.gov 실시간 검색${data.query ? `: ${data.query}` : ""})`}
      colorClass="text-teal-700" borderClass="border-teal-200" bgClass="bg-teal-50/40">
      {data.landscape_summary && (
        <p className="text-[10px] text-teal-900 bg-white/70 rounded-lg px-2.5 py-1.5 mb-2 leading-relaxed">{data.landscape_summary}</p>
      )}
      <div className="space-y-1.5">
        {(data.trials ?? []).map(t => (
          <div key={t.nct_id} className="bg-white rounded-lg px-2.5 py-1.5 border border-teal-100">
            <div className="flex items-center justify-between gap-2">
              <a href={t.url} target="_blank" rel="noreferrer" className="text-[11px] font-semibold text-teal-700 hover:underline">
                {t.brief_title ?? t.nct_id}
              </a>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-teal-100 text-teal-700 flex-shrink-0">{t.overall_status ?? "?"}</span>
            </div>
            <div className="text-[9px] text-slate-400 mt-0.5">
              {(t.phases ?? []).join(", ") || "단계 정보 없음"} · {t.nct_id}
            </div>
          </div>
        ))}
      </div>
      {data.development_stage_assessment && (
        <p className="text-[9px] text-slate-400 mt-2">개발 단계 평가: {data.development_stage_assessment}</p>
      )}
    </PanelShell>
  );
}

function TargetIntelligencePanelCard({ data }: { data: TargetIntelligencePanelData }) {
  if (!data.available) {
    return (
      <PanelShell icon="🎯" title="타겟 인텔리전스" colorClass="text-violet-700" borderClass="border-violet-200" bgClass="bg-violet-50/40"
        unavailableText="타겟이 식별되지 않아 조회할 수 없습니다." />
    );
  }
  return (
    <PanelShell icon="🎯" title={`타겟 인텔리전스: ${data.target_name} (UniProt ${data.uniprot_id})`}
      colorClass="text-violet-700" borderClass="border-violet-200" bgClass="bg-violet-50/40">
      {data.priority_score !== undefined && (
        <div className="bg-white rounded-lg px-2.5 py-1.5 mb-2 border border-violet-100">
          <div className="flex items-center justify-between">
            <p className="text-[9px] text-slate-400">타겟 우선순위 점수 (도킹 전 사전 평가)</p>
            <p className="text-sm font-bold text-violet-700">{data.priority_score}/100</p>
          </div>
          {data.priority_breakdown && (
            <p className="text-[9px] text-slate-400 mt-0.5">
              {Object.entries(data.priority_breakdown).filter(([k]) => k !== "final_priority_score")
                .map(([k, v]) => `${k}: ${v}`).join(" · ")}
              {data.known_inhibitor_count !== undefined && ` (ChEMBL 실측 억제제 ${data.known_inhibitor_count}건)`}
            </p>
          )}
        </div>
      )}
      {data.function_summary && (
        <p className="text-[10px] text-violet-900 bg-white/70 rounded-lg px-2.5 py-1.5 mb-2 leading-relaxed line-clamp-4">
          {data.function_summary}
        </p>
      )}
      <div className="grid grid-cols-1 gap-1.5">
        <div className="bg-white rounded-lg px-2.5 py-1.5 border border-violet-100">
          <p className="text-[9px] text-slate-400 mb-0.5">질병 연관성 (UniProt)</p>
          {(data.diseases ?? []).length > 0 ? (
            <ul className="space-y-0.5">
              {(data.diseases ?? []).map(d => (
                <li key={d.name} className="text-[10px] text-slate-700">
                  {d.name}{d.mim_id ? ` (MIM:${d.mim_id})` : ""}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[10px] text-slate-400">등재된 DISEASE 코멘트 없음 (비-인간/바이러스 단백질에 흔함)</p>
          )}
        </div>
        <div className="bg-white rounded-lg px-2.5 py-1.5 border border-violet-100">
          <p className="text-[9px] text-slate-400 mb-0.5">관련 경로 (Reactome)</p>
          {(data.pathways ?? []).length > 0 ? (
            <ul className="space-y-0.5">
              {(data.pathways ?? []).map(p => (
                <li key={p.stable_id} className="text-[10px]">
                  <a href={p.url} target="_blank" rel="noreferrer" className="text-violet-700 hover:underline">{p.name}</a>
                  {p.in_disease && <span className="ml-1 text-[9px] text-rose-500">[질병 관련]</span>}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[10px] text-slate-400">색인된 경로 없음</p>
          )}
        </div>
        <div className="bg-white rounded-lg px-2.5 py-1.5 border border-violet-100">
          <p className="text-[9px] text-slate-400 mb-0.5">질병 연관성 점수 (OpenTargets, 0~1 종합)</p>
          {(data.opentargets_diseases ?? []).length > 0 ? (
            <ul className="space-y-0.5">
              {(data.opentargets_diseases ?? []).map(d => (
                <li key={d.name} className="text-[10px] text-slate-700 flex justify-between">
                  <span>{d.name}</span>
                  <span className="font-semibold text-violet-600">{d.score}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[10px] text-slate-400">데이터 없음 (인간 유전자가 아닌 경우 흔함)</p>
          )}
        </div>
        {(data.opentargets_tractability ?? []).length > 0 && (
          <div className="bg-white rounded-lg px-2.5 py-1.5 border border-violet-100">
            <p className="text-[9px] text-slate-400 mb-0.5">소분자 약물 가능성 (OpenTargets Tractability)</p>
            <div className="flex flex-wrap gap-1">
              {(data.opentargets_tractability ?? []).map(t => (
                <span key={t} className="text-[9px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-700">{t}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </PanelShell>
  );
}

function CompoundDiscoveryPanelCard({ data }: { data: CompoundDiscoveryPanelData }) {
  const isInhibitors = data.kind === "known_inhibitors";
  if (!data.available) {
    return (
      <PanelShell icon="🧫" title={isInhibitors ? "알려진 억제제 (ChEMBL)" : "유사 화합물 (PubChem)"}
        colorClass="text-fuchsia-700" borderClass="border-fuchsia-200" bgClass="bg-fuchsia-50/40"
        unavailableText={data.label ? `'${data.label}'에 대한 데이터를 찾지 못했습니다.` : "데이터를 찾지 못했습니다."} />
    );
  }
  return (
    <PanelShell
      icon="🧫"
      title={isInhibitors ? `알려진 억제제 (ChEMBL 실측 IC50, ${data.label})` : `유사 화합물 (PubChem, ${data.label} 기준)`}
      colorClass="text-fuchsia-700" borderClass="border-fuchsia-200" bgClass="bg-fuchsia-50/40"
    >
      <div className="space-y-1.5">
        {(data.items ?? []).map((item, i) => (
          <div key={i} className="bg-white rounded-lg px-2.5 py-1.5 border border-fuchsia-100">
            <div className="flex items-center justify-between gap-2">
              <a href={item.url} target="_blank" rel="noreferrer" className="text-[11px] font-semibold text-fuchsia-700 hover:underline">
                {isInhibitors ? (item.name ?? item.chembl_id) : (item.iupac_name ?? item.smiles)}
              </a>
              {isInhibitors && item.ic50_nm !== undefined && (
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-fuchsia-100 text-fuchsia-700 flex-shrink-0">
                  IC50 {item.ic50_nm.toFixed(1)} nM
                </span>
              )}
            </div>
            <div className="text-[9px] text-slate-400 mt-0.5">
              {isInhibitors
                ? `${item.chembl_id ?? "?"}${item.document_year ? ` · ${item.document_year}` : ""}`
                : `CID ${item.cid} · MW ${item.molecular_weight}`}
            </div>
          </div>
        ))}
        {!isInhibitors && (
          <p className="text-[9px] text-slate-400">유사도 순 정렬은 아님 (PubChem 필터 결과)</p>
        )}
      </div>
    </PanelShell>
  );
}

function SarOptimizationPanelCard({ data }: { data: SarOptimizationPanelData }) {
  if (!data.available) {
    return (
      <PanelShell icon="🧪" title="SAR 최적화" colorClass="text-orange-700" borderClass="border-orange-200" bgClass="bg-orange-50/40"
        unavailableText={data.reason ?? "SAR 최적화를 수행할 수 없습니다."} />
    );
  }
  return (
    <PanelShell icon="🧪" title={`SAR 최적화: ${data.base_name} (원본 ${data.base_affinity_kcal_mol} kcal/mol)`}
      colorClass="text-orange-700" borderClass="border-orange-200" bgClass="bg-orange-50/40">
      <div className="space-y-1.5">
        {(data.analogs ?? []).map((a, i) => (
          <div key={i} className="bg-white rounded-lg px-2.5 py-1.5 border border-orange-100">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[10px] font-semibold text-slate-700">{a.transformation}</span>
              {a.docked && a.improved !== null && a.improved !== undefined && (
                <span className={`text-[9px] px-1.5 py-0.5 rounded flex-shrink-0 ${
                  a.improved ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700"
                }`}>
                  {a.improved ? "개선" : "악화"} Δ{a.delta_kcal_mol?.toFixed(2)}
                </span>
              )}
            </div>
            <p className="text-[9px] text-slate-400 mt-0.5">{a.rationale}</p>
            <p className="text-[9px] font-mono text-slate-500 mt-0.5 break-all">{a.smiles}</p>
            {a.docked ? (
              <div className="text-[10px] text-slate-600 mt-1">
                실제 재도킹: {a.best_affinity_kcal_mol} kcal/mol
                <SourceBadge source={(a.source as "vina" | "heuristic" | "none") ?? "none"} />
                {a.admet?.valid && (
                  <span className="ml-1.5 text-[9px] text-slate-400">
                    PAINS {a.admet.pains?.flagged ? "있음" : "없음"} · SA {a.admet.synthesis?.score}
                  </span>
                )}
              </div>
            ) : (
              <p className="text-[10px] text-rose-500 mt-1">재도킹 실패</p>
            )}
          </div>
        ))}
      </div>
      {data.note && <p className="text-[9px] text-slate-400 mt-2">{data.note}</p>}
    </PanelShell>
  );
}

function DecisionReportPanelCard({ data }: { data: DecisionReportPanelData }) {
  if (!data.available) {
    return (
      <PanelShell icon="📊" title="종합 의사결정 리포트" colorClass="text-blue-800" borderClass="border-blue-200" bgClass="bg-blue-50/40"
        unavailableText="종합 평가를 수행할 후보가 없습니다." />
    );
  }
  const riskColor = data.development_risk === "low" ? "bg-emerald-100 text-emerald-700"
    : data.development_risk === "high" ? "bg-rose-100 text-rose-700" : "bg-amber-100 text-amber-700";
  return (
    <PanelShell icon="📊" title={`종합 의사결정 리포트: ${data.candidate_name}`}
      colorClass="text-blue-800" borderClass="border-blue-200" bgClass="bg-blue-50/40">
      <div className="flex items-center gap-2 mb-2">
        <div className="bg-white rounded-lg px-2.5 py-1.5 border border-blue-100 flex-1">
          <p className="text-[9px] text-slate-400">우선순위 점수</p>
          <p className="text-sm font-bold text-blue-700">{data.priority_score}/100</p>
        </div>
        <div className="bg-white rounded-lg px-2.5 py-1.5 border border-blue-100 flex-1">
          <p className="text-[9px] text-slate-400">개발 위험도</p>
          <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${riskColor}`}>{data.development_risk}</span>
        </div>
      </div>
      {data.breakdown && (
        <p className="text-[9px] text-slate-400 mb-2">
          {Object.entries(data.breakdown).filter(([k]) => k !== "final_priority_score")
            .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`).join(" · ")}
        </p>
      )}
      <p className="text-[10px] text-slate-700 mb-1.5"><span className="font-semibold">종합 추천:</span> {data.overall_recommendation}</p>
      <p className="text-[10px] text-slate-700 mb-1.5"><span className="font-semibold">위험도 근거:</span> {data.risk_rationale}</p>
      <p className="text-[10px] text-slate-700"><span className="font-semibold">권장 다음 실험:</span> {data.recommended_next_experiment}</p>
      <p className="text-[9px] text-slate-400 mt-2">
        {data.has_target_context
          ? `※ 문헌/임상 근거는 타겟 '${data.target_name}'의 기존 연구 현황이며, 이 후보 화합물 자체의 검증을 의미하지 않습니다.`
          : "타겟이 식별되지 않아 문헌/임상 근거 없이 후보 자체의 실측 도킹/ADMET 데이터만 반영되었습니다."}
      </p>
    </PanelShell>
  );
}

function AgentPanelCard({ panel }: { panel: AgentPanel }) {
  switch (panel.type) {
    case "literature_search": return <LiteraturePanelCard data={panel} />;
    case "clinical_search": return <ClinicalPanelCard data={panel} />;
    case "target_intelligence": return <TargetIntelligencePanelCard data={panel} />;
    case "compound_discovery": return <CompoundDiscoveryPanelCard data={panel} />;
    case "sar_optimization": return <SarOptimizationPanelCard data={panel} />;
    case "decision_report": return <DecisionReportPanelCard data={panel} />;
    default: return null;
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

// Rewritten after a real reported mismatch: an earlier version presented
// the 6-step docking pipeline as if it were the immediate result of ANY
// query, but several other real conversational branches exist and don't
// lead there at all — a bare topic mention (no stated goal) triggers an
// intent menu first (see drug_discovery_intent.py's mentions_research_
// topic), and a cancer/disease mention (no single standard target exists
// per patient) routes to the VCF upload flow instead of docking. Leads with
// those real branches, then lists the assistant's other real capabilities
// (literature/clinical/target/compound agents + decision report — all real
// PubMed/ClinicalTrials.gov/UniProt/Reactome/PubChem/ChEMBL data, see
// drug_discovery_chat.py) instead of the step-by-step docking breakdown,
// which only applies to the screening path and duplicated the live
// progress feed's own step labels for no benefit.
const GREETING_MESSAGE: ChatMessage = {
  role: "agent",
  text: "안녕하세요! 신약개발 어시스턴트입니다. 원하시는 신약 개발 내용을 말씀해주세요:\n\n" +
    "🧬 mRNA 암 백신(신항원 후보 식별)을 원하시면 → 실제 VCF(유전체 변이) 파일 업로드를 안내해 드립니다\n" +
    "예: \"mRNA 암 백신 데모 보여줘\" 또는 📎 버튼으로 VCF/BAM 직접 업로드\n\n" +
    "🎯 구체적 목표를 말씀하시면 → 바로 스크리닝 시작\n" +
    "예: \"SARS-CoV-2 스파이크 단백질을 억제할 수 있는 승인된 약물을 찾아줘\"\n\n" +
    "🔎 병원체/타겟 이름만 말씀하시면 (예: \"결핵\") → 무엇을 원하시는지 먼저 여쭤봅니다\n\n" +
    "💡 이런 것도 할 수 있어요:\n" +
    "📚 문헌 검색 — 관련 최신 논문 실시간 조회 (PubMed)\n" +
    "🧬 신약 후보 탐색 — AlphaFold/ESMFold 구조 예측 기반 실제 도킹·스크리닝\n" +
    "🏥 임상시험 조회 — 진행 중/완료된 임상시험 현황 (ClinicalTrials.gov)\n" +
    "🎯 타겟 분석 — 질병 연관성과 관련 생물학적 경로 확인\n" +
    "🧫 화합물 검색 — 유사 화합물/알려진 억제제 탐색 (PubChem/ChEMBL)\n" +
    "📊 최종 레포트 — 종합 평가 리포트 (우선순위 점수, 개발 위험도)",
};

export default function DrugDiscoveryChatPanel() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadStoredMessages() ?? [GREETING_MESSAGE]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  // Which quick-pick set is contextually relevant right now — set from the
  // last /converse response's needs_vcf flag, not left static. Previously
  // the same 8 SARS-CoV-2/Influenza pathogen buttons showed up after every
  // "chat" reply regardless of context, including right after the agent
  // said "이 질환은 VCF가 필요합니다" — showing irrelevant buttons right
  // next to that message.
  const [showVcfPrompt, setShowVcfPrompt] = useState(false);
  // Set from the last /converse response's neoantigen_mode flag (see
  // drug_discovery_intent.py's _is_cancer_topic) — a cancer topic has no
  // single standardized target, so its VCF/BAM upload must route to the
  // real neoantigen (mRNA vaccine candidate) pipeline instead of the
  // small-molecule screening pipeline. Real reported bug: without this,
  // the sample-VCF button under a cancer prompt ran docking screening
  // against a guessed default target and returned an irrelevant result.
  const [neoantigenMode, setNeoantigenMode] = useState(false);
  // Set from the last /converse response's needs_intent_clarification flag
  // (see drug_discovery_intent.py's mentions_research_topic detection) — a
  // bare topic mention ("결핵") with no stated goal now asks explicitly
  // what the user wants instead of silently assuming screening or falling
  // back to a generic "more info needed" reply. intentTopic is the raw
  // topic text the backend extracted, reused to compose each button's
  // full follow-up message.
  const [intentTopic, setIntentTopic] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(() => localStorage.getItem(STORAGE_KEY_ACTIVE_JOB));
  const [jobRunning, setJobRunning] = useState(false);
  const [showQuickPicks, setShowQuickPicks] = useState(true);
  // Tracks "a job just completed and its session is still the active
  // context" independent of activeJobId (which the polling loop resets to
  // null the moment COMPLETED is detected, since it's only meant to gate
  // polling) — drives POST_RESULT_QUICK_ACTIONS visibility. Cleared when a
  // new job starts or the conversation is cleared, so stale post-result
  // actions don't linger across an unrelated new design. Initialized from
  // restored message history (not just left null) so a page refresh right
  // after a completed job doesn't lose the post-result quick actions until
  // the user types something new.
  const [lastCompletedJobId, setLastCompletedJobId] = useState<string | null>(() => {
    const restored = loadStoredMessages();
    if (!restored) return null;
    for (let i = restored.length - 1; i >= 0; i--) {
      if (restored[i].role === "results") return restored[i].jobId ?? null;
    }
    return null;
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastStepRef = useRef<number>(-1);
  const messagesRef = useRef<ChatMessage[]>(messages);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    messagesRef.current = messages;
    try {
      localStorage.setItem(STORAGE_KEY_MESSAGES, JSON.stringify(messages));
    } catch {
      // storage full/unavailable — chat still works, just won't survive a refresh
    }
  }, [messages]);

  useEffect(() => {
    if (activeJobId) localStorage.setItem(STORAGE_KEY_ACTIVE_JOB, activeJobId);
    else localStorage.removeItem(STORAGE_KEY_ACTIVE_JOB);
  }, [activeJobId]);

  // ── Polling ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!activeJobId) return;

    // Guards a persisted-but-stale job id (e.g. the page was refreshed right
    // after completion): if restored history already shows this job's
    // result, don't poll again — just drop the stale pointer instead of
    // risking a duplicate "results" card.
    const alreadyShown = messagesRef.current.some(m => m.role === "results" && m.jobId === activeJobId);
    if (alreadyShown) {
      setActiveJobId(null);
      return;
    }

    setJobRunning(true);
    lastStepRef.current = -1;

    pollRef.current = setInterval(async () => {
      try {
        const res = await logFetch(`/api/drug-discovery/status/${activeJobId}`);
        if (res.status === 404) {
          // The job is permanently gone (e.g. the backend restarted since
          // it started, wiping its in-memory store) — retrying forever
          // here was a real confirmed bug: jobRunning stayed true forever,
          // which left the "지우기" (clear) button permanently disabled
          // with no way to recover short of clearing localStorage by hand.
          if (pollRef.current) clearInterval(pollRef.current);
          setJobRunning(false);
          setActiveJobId(null);
          setMessages(prev =>
            prev.some(m => m.role === "agent" && m.jobId === activeJobId)
              ? prev
              : [...prev, {
                  role: "agent",
                  text: "⚠ 이전 작업을 더 이상 추적할 수 없습니다 (서버가 재시작되었을 수 있습니다). 다시 요청해 주세요.",
                  jobId: activeJobId ?? undefined,
                }],
          );
          return;
        }
        if (!res.ok) return; // other transient errors — keep polling
        const job: DrugDiscoveryJob = await res.json();

        if (job.current_step !== lastStepRef.current) {
          lastStepRef.current = job.current_step;
          setMessages(prev => [
            ...prev,
            { role: "progress", text: `${job.current_message} (${job.current_step}/${job.total_steps})` },
          ]);
        }

        if (job.status === "COMPLETED" && job.result) {
          // Stop polling synchronously, right here — setActiveJobId(null)
          // alone only stops future polls once React re-runs the effect's
          // cleanup on the next render, which happens asynchronously.
          if (pollRef.current) clearInterval(pollRef.current);
          setJobRunning(false);
          setActiveJobId(null);
          // Idempotent insert, keyed off the true latest state (not a ref
          // snapshot) — this is the last line of defense against duplicate
          // "results" cards no matter how many overlapping pollers end up
          // running for the same job_id (e.g. several stale intervals left
          // over from Vite Fast Refresh not cleanly tearing down an old
          // effect instance across hot-reloads during development). Even if
          // N stale pollers all fire this branch for the same job, only the
          // first state update actually adds the card; the rest see it
          // already present and no-op.
          setMessages(prev =>
            prev.some(m => m.role === "results" && m.jobId === activeJobId)
              ? prev
              : [...prev, { role: "results", result: job.result!, jobId: activeJobId ?? undefined }],
          );
          setLastCompletedJobId(activeJobId);
        } else if (job.status === "FAILED") {
          if (pollRef.current) clearInterval(pollRef.current);
          setJobRunning(false);
          setActiveJobId(null);
          setMessages(prev =>
            prev.some(m => m.role === "agent" && m.jobId === activeJobId)
              ? prev
              : [...prev, { role: "agent", text: `❌ ${job.error_message ?? "알 수 없는 오류"}`, jobId: activeJobId ?? undefined }],
          );
        } else if (job.status === "CANCELLED") {
          if (pollRef.current) clearInterval(pollRef.current);
          setJobRunning(false);
          setActiveJobId(null);
          setMessages(prev =>
            prev.some(m => m.role === "agent" && m.jobId === activeJobId)
              ? prev
              : [...prev, { role: "agent", text: `⏹ ${job.error_message ?? "작업이 중지되었습니다."}`, jobId: activeJobId ?? undefined }],
          );
        }
      } catch {
        // transient network error — keep polling
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [activeJobId]);

  // ── Send ───────────────────────────────────────────────────────────────────

  const handleSend = useCallback(async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || sending) return;

    setMessages(prev => [...prev, { role: "user", text }]);
    setInput("");
    setSending(true);
    setShowQuickPicks(false);

    try {
      const res = await logFetch("/api/drug-discovery/converse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data: {
        reply: string; action: string; job_id: string | null; needs_vcf?: boolean; panel?: AgentPanel;
        needs_intent_clarification?: boolean; intent_topic?: string | null; neoantigen_mode?: boolean;
      } = await res.json();
      // A structured panel (Literature/Clinical/Target-Intelligence/
      // Compound-Discovery/SAR/Decision) replaces the plain text bubble
      // with its dedicated card — the same real reply text is still kept
      // on the message for accessibility/history purposes even though the
      // card renders from the structured fields, not by parsing "text".
      if (data.panel) {
        setMessages(prev => [...prev, { role: "panel", text: data.reply, panel: data.panel }]);
      } else {
        setMessages(prev => [...prev, { role: "agent", text: data.reply }]);
      }
      if (data.action === "start_design" && data.job_id) {
        // A genuinely new design just started — any post-result quick
        // actions from a prior completed job are no longer the relevant
        // context.
        setLastCompletedJobId(null);
        setIntentTopic(null);
        setActiveJobId(data.job_id);
      } else {
        // Every other action (ask_question, literature_search,
        // clinical_search, target_intelligence, compound_discovery,
        // sar_optimization, decision_report, status, chat, ...) leaves
        // lastCompletedJobId untouched — a follow-up question about a
        // completed job shouldn't hide the very buttons that suggested it.
        setShowVcfPrompt(!!data.needs_vcf);
        setNeoantigenMode(!!data.neoantigen_mode);
        // Real reported gap: previously only synced from the bare-topic-
        // mention menu response (needs_intent_clarification) — a literature/
        // clinical/target-intelligence/compound-discovery answer's resolved
        // topic (see drug_discovery_router.py's intent_topic field on that
        // response) now ALSO re-shows the same topic-scoped quick-pick
        // buttons, instead of falling back to the generic default examples
        // right after answering one of the menu's own options.
        setIntentTopic(data.intent_topic ?? null);
        setShowQuickPicks(true);
      }
    } catch (err) {
      console.error("[DrugDiscoveryChatPanel] handleSend failed", err);
      setMessages(prev => [...prev, { role: "agent", text: "❌ 요청을 처리하지 못했습니다. 다시 시도해 주세요." }]);
      setShowVcfPrompt(false);
      setNeoantigenMode(false);
      setIntentTopic(null);
      setShowQuickPicks(true);
    } finally {
      setSending(false);
    }
  }, [input, sending, sessionId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Clear conversation ──────────────────────────────────────────────────────
  // "새 연구" (New Research) — an explicit, one-click way to switch the
  // session's target focus. The backend keeps answering follow-ups in the
  // current target's context (literature/clinical/SAR/decision, or
  // answer_completed_job_question()) so the focus never drifts silently, but
  // an explicit new-target design request now switches targets in-session
  // after a full server-side reset (see drug_discovery_router.py's /converse
  // start_design branch). This button remains as a guaranteed hard reset —
  // it mints a fresh session_id, so even stale/unresolvable client state is
  // cleared — but is no longer the ONLY way to move on to a new target.
  //
  // Also issues a fresh session_id (not just wiping local chat history) —
  // the backend's /converse session dict (known_slots, goal_text, FSM state)
  // is keyed by session_id and only ever resets here, so reusing the old id
  // would leave old context (e.g. a prior unsupported-disease mention) able
  // to bleed into the new conversation exactly like the 대장암→결핵
  // contamination bug.
  const handleNewResearch = useCallback(() => {
    // Deliberately never hard-disabled (unlike before) — a job stuck
    // "running" client-side (e.g. polling a job_id that no longer exists
    // after a backend restart) previously left this button permanently
    // unusable with no way to recover short of clearing localStorage by
    // hand. Confirmed real bug, not hypothetical.
    const confirmMsg = jobRunning
      ? "진행 중인 작업이 있습니다. 중지하고 새 연구를 시작할까요? 현재 대화 기록과 타겟 정보가 모두 초기화됩니다."
      : "새 연구를 시작할까요? 현재 대화 기록과 타겟 정보가 모두 초기화됩니다.";
    if (!window.confirm(confirmMsg)) return;
    if (activeJobId) {
      logFetch(`/api/drug-discovery/stop/${activeJobId}`, { method: "POST" }).catch(err =>
        console.error("[DrugDiscoveryChatPanel] handleNewResearch stop failed", err),
      );
    }
    if (pollRef.current) clearInterval(pollRef.current);
    const freshSessionId = makeSessionId();
    localStorage.setItem(STORAGE_KEY_SESSION, freshSessionId);
    localStorage.removeItem(STORAGE_KEY_ACTIVE_JOB);
    setSessionId(freshSessionId);
    setMessages([GREETING_MESSAGE]);
    setActiveJobId(null);
    setLastCompletedJobId(null);
    setJobRunning(false);
    setShowVcfPrompt(false);
    setNeoantigenMode(false);
    setIntentTopic(null);
    setShowQuickPicks(true);
    setInput("");
  }, [jobRunning, activeJobId]);

  // ── Stop the running job ────────────────────────────────────────────────────
  // Calls the real cancellation endpoint (asyncio Task.cancel() server-side,
  // not just a client-side flag) — see backend stop_job(). Note: if the job
  // is currently blocked inside a real subprocess (mk_prepare_receptor /
  // vina.exe), that specific already-launched subprocess keeps running in
  // the background until it finishes on its own; this stops the pipeline
  // from proceeding any further and marks the job CANCELLED immediately.
  const handleStop = useCallback(async () => {
    if (!activeJobId) return;
    try {
      await logFetch(`/api/drug-discovery/stop/${activeJobId}`, { method: "POST" });
    } catch (err) {
      // ignore — the poll loop will keep trying and surface a real error if the job is still alive
      console.error("[DrugDiscoveryChatPanel] handleStop failed", err);
    }
  }, [activeJobId]);

  // ── File upload (VCF "Track B" / FASTA "Track A" gateway) ──────────────────

  const handleFileSelected = useCallback(async (file: File) => {
    setUploading(true);
    setMessages(prev => [...prev, { role: "user", text: `📎 파일 업로드: ${file.name}` }]);
    setShowQuickPicks(false);
    setShowVcfPrompt(false);
    const wasNeoantigenMode = neoantigenMode;
    setNeoantigenMode(false);
    setIntentTopic(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      // Real reported bug: a cancer topic's VCF upload always ran the
      // small-molecule screening pipeline (design-from-file), even though
      // that mode has no single standardized target — route to the real
      // neoantigen/mRNA vaccine pipeline instead when the last /converse
      // reply flagged this as a cancer topic (neoantigenMode).
      if (wasNeoantigenMode) {
        const text = await file.text();
        const res = await logFetch("/api/drug-discovery/design-from-bam", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vcf_text: text }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "파일을 처리하지 못했습니다." }));
          setMessages(prev => [...prev, { role: "agent", text: `❌ ${err.detail ?? "파일을 처리하지 못했습니다."}` }]);
          setShowQuickPicks(true);
          return;
        }
        const data: { job_id: string } = await res.json();
        setMessages(prev => [...prev, {
          role: "agent",
          text: "VCF 변이를 실시간 검증하고 실제 MHCflurry 모델로 신항원(neoantigen) 후보를 예측합니다 (BAM 기반 실제 HLA 타이핑은 이 환경에서 미지원 — 표준 population allele 사용).",
        }]);
        setLastCompletedJobId(null);
        setActiveJobId(data.job_id);
        return;
      }
      const res = await logFetch("/api/drug-discovery/design-from-file", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "파일을 처리하지 못했습니다." }));
        setMessages(prev => [...prev, { role: "agent", text: `❌ ${err.detail ?? "파일을 처리하지 못했습니다."}` }]);
        setShowQuickPicks(true);
        return;
      }
      const data: { job_id: string; detected_format: "vcf" | "fasta" } = await res.json();
      const label = data.detected_format === "vcf" ? "VCF(체세포 변이)" : "FASTA(단백질 서열)";
      setMessages(prev => [...prev, { role: "agent", text: `${label} 파일로 인식하여 분석을 시작합니다.` }]);
      setLastCompletedJobId(null);
      setActiveJobId(data.job_id);
    } catch (err) {
      console.error("[DrugDiscoveryChatPanel] handleFileSelected failed", err);
      setMessages(prev => [...prev, { role: "agent", text: "❌ 파일 업로드에 실패했습니다. 다시 시도해 주세요." }]);
      setShowQuickPicks(true);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file next time
    if (file) handleFileSelected(file);
  };

  const handleSampleVcf = useCallback(async () => {
    setUploading(true);
    setMessages(prev => [...prev, { role: "user", text: "📎 샘플 VCF 사용 (sample/NSCLC_variants.vcf)" }]);
    setShowQuickPicks(false);
    setShowVcfPrompt(false);
    setNeoantigenMode(false);
    setIntentTopic(null);
    try {
      const res = await logFetch("/api/drug-discovery/design-from-vcf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vcf_path: "sample/NSCLC_variants.vcf" }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "샘플 VCF 처리에 실패했습니다." }));
        setMessages(prev => [...prev, { role: "agent", text: `❌ ${err.detail ?? "샘플 VCF 처리에 실패했습니다."}` }]);
        setShowQuickPicks(true);
        return;
      }
      const data: { job_id: string } = await res.json();
      // Deliberately doesn't name a specific variant here — the real
      // variant found is whatever Ensembl VEP actually confirms for this
      // file (currently real KRAS G12D, chr12:25245350 C>T), shown in the
      // report once the job completes rather than pre-announced.
      setMessages(prev => [...prev, {
        role: "agent",
        text: "샘플 VCF를 파싱하고 변이를 실시간으로 검증하여 분석을 시작합니다. 실제로 발견된 변이는 리포트에서 확인하실 수 있습니다.",
      }]);
      setLastCompletedJobId(null);
      setActiveJobId(data.job_id);
    } catch (err) {
      console.error("[DrugDiscoveryChatPanel] handleSampleVcf failed", err);
      setMessages(prev => [...prev, { role: "agent", text: "❌ 샘플 VCF 처리에 실패했습니다. 다시 시도해 주세요." }]);
      setShowQuickPicks(true);
    } finally {
      setUploading(false);
    }
  }, []);

  // Neoantigen candidate identification ("mRNA 백신" flow) — VCF (variant
  // calls) + BAM (read-count/coverage context only; real BAM-based HLA
  // typing needs Docker/WSL and isn't available on this deployment, see
  // services/neoantigen_engine.py) sample data, /design-from-bam.
  const handleSampleNeoantigen = useCallback(async () => {
    setUploading(true);
    setMessages(prev => [...prev, { role: "user", text: "📎 암 mRNA 자가 백신 데모 (신항원 후보 식별, sample/NSCLC_variants.vcf + sample/NSCLC.bam)" }]);
    setShowQuickPicks(false);
    setShowVcfPrompt(false);
    setNeoantigenMode(false);
    setIntentTopic(null);
    try {
      const res = await logFetch("/api/drug-discovery/design-from-bam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vcf_path: "sample/NSCLC_variants.vcf", bam_path: "sample/NSCLC.bam" }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "샘플 VCF/BAM 처리에 실패했습니다." }));
        setMessages(prev => [...prev, { role: "agent", text: `❌ ${err.detail ?? "샘플 VCF/BAM 처리에 실패했습니다."}` }]);
        setShowQuickPicks(true);
        return;
      }
      const data: { job_id: string } = await res.json();
      setMessages(prev => [...prev, {
        role: "agent",
        text: "VCF 변이를 실시간 검증하고 실제 MHCflurry 모델로 신항원 후보를 예측합니다 (BAM 기반 실제 HLA 타이핑은 이 환경에서 미지원 — 표준 population allele 사용, 리포트에서 명시됩니다).",
      }]);
      setLastCompletedJobId(null);
      setActiveJobId(data.job_id);
    } catch (err) {
      console.error("[DrugDiscoveryChatPanel] handleSampleNeoantigen failed", err);
      setMessages(prev => [...prev, { role: "agent", text: "❌ 샘플 VCF/BAM 처리에 실패했습니다. 다시 시도해 주세요." }]);
      setShowQuickPicks(true);
    } finally {
      setUploading(false);
    }
  }, []);

  // ── JSX ────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex-shrink-0 px-5 pt-5 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-xl">🧪</span>
          <h2 className="text-sm font-bold text-slate-800">Drug Discovery Assistant</h2>
          {jobRunning && (
            <span className="ml-auto flex items-center gap-1 text-[10px] text-blue-600 font-semibold">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              Running
            </span>
          )}
          {jobRunning && (
            <button
              onClick={handleStop}
              title="진행 중인 작업 중지"
              className="text-[10px] px-2 py-1 rounded-lg border border-red-200 text-red-500 hover:bg-red-50 transition-colors"
            >
              ⏹ 중지
            </button>
          )}
          <button
            onClick={handleNewResearch}
            title={jobRunning ? "진행 중인 작업을 중지하고 새 연구 시작 (타겟 초기화)" : "새 연구 시작 (현재 타겟/대화 기록 초기화)"}
            className={`text-[10px] px-2 py-1 rounded-lg border border-slate-200 text-slate-500 hover:border-blue-300 hover:text-blue-600 transition-colors ${
              jobRunning ? "" : "ml-auto"
            }`}
          >
            🆕 새 연구
          </button>
        </div>
        <p className="text-[11px] text-slate-400">단백질 구조 조회(AlphaFold DB / ESMFold) · 리간드 분석(RDKit) · 도킹(AutoDock Vina / 휴리스틱)</p>
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2 max-w-3xl mx-auto w-full">
        {messages.map((m, i) => {
          if (m.role === "progress") {
            return (
              <div key={i} className="flex items-center gap-1.5 text-[10px] text-slate-400 px-1">
                <span className="w-1 h-1 rounded-full bg-slate-300" />
                <span className="truncate">{m.text}</span>
              </div>
            );
          }
          if (m.role === "results" && m.result) {
            return (
              <div key={i} className="flex justify-start">
                {m.result.mode === "neoantigen" ? (
                  <NeoantigenResultsCard result={m.result} jobId={m.jobId} />
                ) : m.result.mode === "screen" ? (
                  <RankedResultsCard result={m.result} jobId={m.jobId} />
                ) : (
                  <ResultCard result={m.result} jobId={m.jobId} />
                )}
              </div>
            );
          }
          if (m.role === "panel" && m.panel) {
            return (
              <div key={i} className="flex justify-start">
                <AgentPanelCard panel={m.panel} />
              </div>
            );
          }
          const isUser = m.role === "user";
          return (
            <div key={i} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-xs whitespace-pre-wrap ${
                  isUser
                    ? "bg-blue-600 text-white rounded-br-sm"
                    : "bg-slate-100 text-slate-700 rounded-bl-sm"
                }`}
              >
                {m.text}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick-pick chips — contextual, not static: showVcfPrompt (set from
          the last /converse response's needs_vcf flag) means the agent
          just asked for a VCF specifically (e.g. an unsupported-cancer
          reply), so only the VCF action is relevant here — showing the
          function-example buttons in that context was the reported
          confusion ("이건 covid용인데 왜 맨날 떠있어", back when these were
          per-pathogen buttons). Once a job has completed (lastCompletedJobId
          set), the post-result follow-up actions (Literature/Clinical/
          Target-Intelligence/Binding-pocket/SAR/Decision agents, now with
          real target context auto-applied) are the relevant next step, not
          "here's what this assistant can do" — those two contexts are
          mutually exclusive. Otherwise (idle / generic clarification, no
          completed job yet) FUNCTION_EXAMPLES apply — one real, complete
          example message per capability (screening/literature/clinical/
          target-analysis/compound-search), replacing the earlier per-
          pathogen target list on explicit request. */}
      {showQuickPicks && !sending && !jobRunning && !uploading && (
        <div className="flex-shrink-0 px-4 pb-2 max-w-3xl mx-auto w-full flex flex-wrap gap-1.5">
          {showVcfPrompt && neoantigenMode ? (
            // Cancer topic (no single standardized target) — routes to the
            // real neoantigen/mRNA vaccine pipeline, not small-molecule
            // screening. See handleSampleNeoantigen/neoantigenMode.
            <button
              onClick={handleSampleNeoantigen}
              className="text-[10px] px-2.5 py-1 rounded-full border border-purple-200 text-purple-500 hover:border-purple-400 hover:text-purple-700 transition-colors"
            >
              🧬 암 mRNA 자가 백신 데모 (NSCLC)
            </button>
          ) : showVcfPrompt ? (
            <button
              onClick={handleSampleVcf}
              className="text-[10px] px-2.5 py-1 rounded-full border border-rose-200 text-rose-500 hover:border-rose-400 hover:text-rose-700 transition-colors"
            >
              🧬 샘플 VCF 사용 (NSCLC)
            </button>
          ) : intentTopic ? (
            // A bare topic mention ("결핵") with no stated goal — never
            // silently assume screening (see drug_discovery_intent.py's
            // mentions_research_topic detection); let the user pick, each
            // button composing a real, complete follow-up message that
            // combines the remembered topic with the chosen action.
            [
              // 문헌 검색을 1순위로: 아직 정보가 없는 주제라면 비용이 드는
              // 스크리닝보다 먼저 관련 연구를 파악하는 게 자연스러운 순서.
              { label: "📚 관련 논문 검색", message: `${intentTopic} 관련 논문 찾아줘` },
              // 암은 코로나와 달리 단일 표준 타겟이 없어 (drug_discovery_intent.py의
              // _is_cancer_topic) — 이 라벨/메시지의 "암"이라는 글자 자체가 백엔드의
              // 같은 판정을 다시 통과시켜 VCF/BAM 기반 신항원 파이프라인으로 안내한다
              // (신약 스크리닝으로 잘못 빠지던 실제 리포트된 버그 수정).
              CANCER_TOPIC_RE.test(intentTopic)
                ? { label: "🧬 mRNA 자가 백신 후보 찾기", message: intentTopic }
                : { label: "🧬 신약 스크리닝", message: `${intentTopic} 억제할 수 있는 승인된 약물을 찾아줘` },
              { label: "🎯 타겟 정보만 확인", message: `${intentTopic}의 질병 연관성과 관련 경로 분석해줘` },
            ].map(opt => (
              <button
                key={opt.label}
                onClick={() => handleSend(opt.message)}
                className="text-[10px] px-2.5 py-1 rounded-full border border-amber-200 text-amber-600 hover:border-amber-400 hover:text-amber-800 transition-colors"
              >
                {opt.label}
              </button>
            ))
          ) : lastCompletedJobId ? (
            POST_RESULT_QUICK_ACTIONS.map(qa => (
              <button
                key={qa.label}
                onClick={() => handleSend(qa.message)}
                className="text-[10px] px-2.5 py-1 rounded-full border border-indigo-200 text-indigo-500 hover:border-indigo-400 hover:text-indigo-700 transition-colors"
              >
                {qa.label}
              </button>
            ))
          ) : (
            <>
              {FUNCTION_EXAMPLES.map(fe => (
                <button
                  key={fe.label}
                  onClick={() => handleSend(fe.message)}
                  className="text-[10px] px-2.5 py-1 rounded-full border border-slate-200 text-slate-500 hover:border-blue-300 hover:text-blue-600 transition-colors"
                >
                  {fe.label}
                </button>
              ))}
              <button
                onClick={handleSampleNeoantigen}
                title="VCF(변이) + BAM(리드 컨텍스트, 실제 HLA 타이핑은 미지원) 샘플로 실제 MHCflurry 신항원 후보 예측 실행"
                className="text-[10px] px-2.5 py-1 rounded-full border border-purple-200 text-purple-500 hover:border-purple-400 hover:text-purple-700 transition-colors"
              >
                🧬 mRNA 암 백신
              </button>
            </>
          )}
        </div>
      )}

      {/* Input area */}
      <div className="flex-shrink-0 px-4 py-3 border-t border-slate-100 max-w-3xl mx-auto w-full">
        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".vcf,.fasta,.fa"
            className="hidden"
            onChange={handleFileInputChange}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || sending || jobRunning}
            title="VCF(.vcf) 또는 FASTA(.fasta/.fa) 파일 업로드"
            className={`h-9 px-3 rounded-xl text-xs font-bold border transition-colors flex-shrink-0 ${
              uploading || sending || jobRunning
                ? "border-slate-100 text-slate-300 cursor-not-allowed"
                : "border-slate-200 text-slate-500 hover:border-blue-300 hover:text-blue-600"
            }`}
          >
            📎
          </button>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            placeholder="예: 'SARS-CoV-2 스파이크 단백질을 억제할 수 있는 약물 찾아줘' 또는 📎로 VCF/FASTA 파일 업로드"
            className="flex-1 text-xs border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || sending}
            className={`h-9 px-4 rounded-xl text-xs font-bold transition-colors ${
              !input.trim() || sending
                ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                : "bg-blue-600 text-white hover:bg-blue-700"
            }`}
          >
            전송
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 px-5 py-2.5 border-t border-slate-100">
        <p className="text-[10px] text-slate-300 text-center leading-relaxed">
          모든 구조/리간드/도킹 점수는 결정론적 엔진(RDKit, AlphaFold DB, ESMFold, AutoDock Vina) 전담 · 문헌/임상/화합물/경로 데이터는 실시간 조회(PubMed, ClinicalTrials.gov, PubChem, ChEMBL, UniProt, Reactome) — AI는 의도 파싱과 실제 데이터 기반 설명만 담당
        </p>
      </div>
    </div>
  );
}
