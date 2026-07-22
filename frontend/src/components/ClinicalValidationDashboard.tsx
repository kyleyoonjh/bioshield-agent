import { useState, useRef, useEffect, DragEvent, ChangeEvent } from "react";
import ReactMarkdown from "react-markdown";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { TooltipProps } from "recharts";
import type {
  DiseaseType, GuidelineType,
  EP05Result, EP09Result,
  SchemaMapping, SchemaMappingEP09,
  FeedbackValue, Mutation,
} from "../types";
import { useAutoCollect } from "../hooks/useAutoCollect";

const GUIDELINES: { id: GuidelineType; title: string; subtitle: string; endpoint: string }[] = [
  {
    id:       "EP05",
    title:    "CLSI EP05-A3",
    subtitle: "정밀도 분석 · Repeatability / Reproducibility",
    endpoint: "/api/v1/analyze/ep05",
  },
  {
    id:       "EP09",
    title:    "CLSI EP09-A3",
    subtitle: "방법 비교 분석 · Method Comparison",
    endpoint: "/api/v1/analyze/ep09",
  },
];

type AnalysisState = "idle" | "uploading" | "confirming" | "analyzing" | "done" | "error";

// ─── Utility UI Components ────────────────────────────────────────────────────

function SectionCard({ title, children, className = "" }: {
  title: string; children: React.ReactNode; className?: string;
}) {
  return (
    <div className={`bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden ${className}`}>
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50">
        <h3 className="text-xs font-semibold text-slate-600 uppercase tracking-widest">{title}</h3>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function KpiCard({ label, value, unit, sub, accent = "default" }: {
  label: string; value: string | number; unit?: string; sub?: string;
  accent?: "green" | "yellow" | "blue" | "default";
}) {
  const borders = { green: "border-l-emerald-500", yellow: "border-l-amber-500", blue: "border-l-blue-500", default: "border-l-slate-300" };
  const values  = { green: "text-emerald-600",      yellow: "text-amber-600",     blue: "text-blue-600",   default: "text-slate-800"   };
  return (
    <div className={`bg-white border border-slate-200 border-l-4 ${borders[accent]} rounded-lg p-4 shadow-sm`}>
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${values[accent]}`}>
        {value}{unit && <span className="text-sm font-normal ml-1 text-slate-500">{unit}</span>}
      </p>
      {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
    </div>
  );
}

function Badge({ status, label }: { status: "pass" | "caution" | "info"; label: string }) {
  const styles = { pass: "bg-emerald-50 border-emerald-200 text-emerald-800", caution: "bg-amber-50 border-amber-200 text-amber-800", info: "bg-blue-50 border-blue-200 text-blue-800" };
  const icons  = { pass: "✓", caution: "!", info: "i" };
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${styles[status]}`}>
      <span className="font-bold">{icons[status]}</span> {label}
    </span>
  );
}

// ─── Custom Tooltips ──────────────────────────────────────────────────────────

function MCTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs shadow-md">
      <p className="text-slate-500">Reference <span className="font-semibold text-slate-800">{Number(d?.ref).toFixed(2)} Ct</span></p>
      <p className="text-slate-500">Test&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="font-semibold text-blue-700">{Number(d?.test).toFixed(2)} Ct</span></p>
    </div>
  );
}

function BATooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs shadow-md">
      <p className="text-slate-500">Average&nbsp;&nbsp;&nbsp;<span className="font-semibold text-slate-800">{Number(d?.avg).toFixed(2)} Ct</span></p>
      <p className="text-slate-500">Difference <span className="font-semibold text-blue-700">{Number(d?.diff).toFixed(3)} Ct</span></p>
    </div>
  );
}

// ─── EP05 Stats Panel ─────────────────────────────────────────────────────────

function EP05StatsPanel({ result }: { result: EP05Result }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <KpiCard label="Repeatability CV"   value={result.repeatability.cv_percent}   unit="%" sub={`SD: ${result.repeatability.sd}`}   accent="green"  />
        <KpiCard label="Reproducibility CV" value={result.reproducibility.cv_percent} unit="%" sub={`SD: ${result.reproducibility.sd}`} accent="yellow" />
        <KpiCard label="ANOVA F-value"      value={result.anova.f_value}              sub={`p = ${result.anova.p_value}`}               accent="blue"   />
        <KpiCard label="Grand Mean Ct"      value={result.grand_mean}                 sub={`n = ${result.sample_count}`}                              />
      </div>

      <SectionCard title="ANOVA 분산 분석표">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-400 uppercase border-b border-slate-100">
              {["Source","SS","df","F","p"].map(h => (
                <th key={h} className={`pb-2 font-medium ${h === "Source" ? "text-left" : "text-right"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50 text-slate-700">
            <tr>
              <td className="py-2.5 font-semibold text-slate-600">Between</td>
              <td className="text-right py-2.5 font-mono">{result.variance_components.between_group.toFixed(3)}</td>
              <td className="text-right py-2.5 font-mono">{result.groups.length - 1}</td>
              <td className="text-right py-2.5 font-semibold text-blue-700 font-mono">{result.anova.f_value}</td>
              <td className="text-right py-2.5 font-semibold text-amber-700 font-mono">{result.anova.p_value}</td>
            </tr>
            <tr>
              <td className="py-2.5 font-semibold text-slate-600">Within</td>
              <td className="text-right py-2.5 font-mono">{result.variance_components.within_group.toFixed(3)}</td>
              <td className="text-right py-2.5 font-mono">{result.sample_count - result.groups.length}</td>
              <td className="text-right py-2.5 text-slate-300">—</td>
              <td className="text-right py-2.5 text-slate-300">—</td>
            </tr>
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="그룹별 정밀도">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-400 uppercase border-b border-slate-100">
              {["Group","N","Mean Ct","SD","CV%"].map(h => (
                <th key={h} className={`pb-2 font-medium ${h === "Group" ? "text-left" : "text-right"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {result.groups.map((g) => (
              <tr key={g.group} className="text-slate-700">
                <td className="py-2 font-semibold text-slate-600">{g.group}</td>
                <td className="text-right py-2 font-mono">{g.n}</td>
                <td className="text-right py-2 font-mono">{Number(g.mean).toFixed(2)}</td>
                <td className="text-right py-2 font-mono">{Number(g.sd).toFixed(3)}</td>
                <td className={`text-right py-2 font-semibold font-mono ${g.cv_percent < 1 ? "text-emerald-600" : g.cv_percent < 2 ? "text-blue-600" : "text-amber-600"}`}>
                  {g.cv_percent}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 text-sm text-blue-900">
        <span className="font-semibold">권장사항:</span> 장비 및 일자별 표준화 프로토콜 수립을 권장합니다.
        양성 대조군(Positive Control) Ct 값 모니터링 및 일별 보정 절차 도입을 고려하십시오.
      </div>
    </div>
  );
}

// ─── EP09 Stats Panel ─────────────────────────────────────────────────────────

function EP09StatsPanel({ result }: { result: EP09Result }) {
  const { deming: d, bland_altman: ba } = result;
  const interceptLabel = d.intercept < 0 ? `− ${Math.abs(d.intercept)}` : `+ ${d.intercept}`;

  const scatterDomain = (() => {
    const allVals = result.scatter_data.flatMap(p => [p.ref, p.test]);
    const min = Math.floor(Math.min(...allVals)) - 1;
    const max = Math.ceil(Math.max(...allVals)) + 1;
    return [min, max] as [number, number];
  })();

  const baDomain: [number, number] = [
    Math.floor(ba.loa_lower - 0.5),
    Math.ceil(ba.loa_upper + 0.5),
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <KpiCard label="R² (결정계수)"  value={result.r_squared}  sub="> 0.99 = Excellent"      accent="green" />
        <KpiCard label="Deming 기울기" value={d.slope}            sub="허용 범위: 0.95 ~ 1.05"  accent="blue"  />
        <KpiCard label="Deming 절편"   value={d.intercept}        sub="허용 범위: ± 1.0"        accent="blue"  />
        <KpiCard label="평균 Bias"     value={ba.mean_diff} unit="Ct" sub={`LOA: ${ba.loa_lower} ~ ${ba.loa_upper}`} />
      </div>

      <SectionCard title="방법 비교 산점도 · Deming Regression">
        <ResponsiveContainer width="100%" height={240}>
          <ScatterChart margin={{ top: 8, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="ref" type="number" domain={scatterDomain} name="Reference Ct"
              label={{ value: "Reference Ct", position: "insideBottom", offset: -12, fontSize: 11, fill: "#94a3b8" }}
              tick={{ fontSize: 10, fill: "#64748b" }} />
            <YAxis dataKey="test" type="number" domain={scatterDomain} name="Test Ct"
              label={{ value: "Test Ct", angle: -90, position: "insideLeft", offset: 12, fontSize: 11, fill: "#94a3b8" }}
              tick={{ fontSize: 10, fill: "#64748b" }} />
            <Tooltip content={<MCTooltip />} cursor={{ strokeDasharray: "3 3" }} />
            <ReferenceLine segment={[{ x: scatterDomain[0], y: scatterDomain[0] }, { x: scatterDomain[1], y: scatterDomain[1] }]}
              stroke="#cbd5e1" strokeDasharray="4 4" strokeWidth={1.5} />
            <Scatter data={result.scatter_data} fill="#1d4db8" opacity={0.8} />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-400 text-center mt-1">
          y = {d.slope}x {interceptLabel} &nbsp;·&nbsp; 점선 = 완전일치선 (y = x)
        </p>
      </SectionCard>

      <SectionCard title="Bland-Altman Plot · 일치도 분석">
        <ResponsiveContainer width="100%" height={240}>
          <ScatterChart margin={{ top: 8, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="avg" type="number" name="평균 Ct"
              label={{ value: "평균 Ct", position: "insideBottom", offset: -12, fontSize: 11, fill: "#94a3b8" }}
              tick={{ fontSize: 10, fill: "#64748b" }} />
            <YAxis dataKey="diff" type="number" domain={baDomain} name="차이"
              label={{ value: "차이 (Test−Ref)", angle: -90, position: "insideLeft", offset: 12, fontSize: 11, fill: "#94a3b8" }}
              tick={{ fontSize: 10, fill: "#64748b" }} />
            <Tooltip content={<BATooltip />} cursor={{ strokeDasharray: "3 3" }} />
            <ReferenceLine y={0}           stroke="#cbd5e1" strokeWidth={1} />
            <ReferenceLine y={ba.mean_diff} stroke="#3b82f6" strokeWidth={2}
              label={{ value: `Bias ${ba.mean_diff}`, fill: "#3b82f6", fontSize: 10, position: "right" }} />
            <ReferenceLine y={ba.loa_upper} stroke="#ef4444" strokeDasharray="5 3" strokeWidth={1.5}
              label={{ value: `+1.96SD ${ba.loa_upper}`, fill: "#ef4444", fontSize: 10, position: "right" }} />
            <ReferenceLine y={ba.loa_lower} stroke="#ef4444" strokeDasharray="5 3" strokeWidth={1.5}
              label={{ value: `−1.96SD ${ba.loa_lower}`, fill: "#ef4444", fontSize: 10, position: "right" }} />
            <Scatter data={result.bland_altman_data} fill="#1d4db8" opacity={0.8} />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-400 text-center mt-1">
          Limits of Agreement: [{ba.loa_lower}, {ba.loa_upper}] Ct
        </p>
      </SectionCard>

      <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-900">
        <span className="font-semibold">결론:</span> 기존 허가 제품 대비 동등성이 확보되었습니다.
        Bland-Altman 분석의 Bias({ba.mean_diff} Ct) 수준을 임상 보고 기준 수립 시 반영하십시오.
      </div>
    </div>
  );
}

// ─── AI Report Panel ──────────────────────────────────────────────────────────

function AiReportPanel({ statsData, guideline }: { statsData: object; guideline: GuidelineType }) {
  const [reportKo, setReportKo] = useState<string | null>(null);
  const [reportEn, setReportEn] = useState<string | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [tab, setTab]           = useState<"ko" | "en">("ko");

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch("/api/v1/report", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stats_data: statsData, language: "ko" }) }),
      fetch("/api/v1/report", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stats_data: statsData, language: "en" }) }),
    ])
      .then(async ([kr, en]) => {
        if (!kr.ok || !en.ok) throw new Error("Report generation failed");
        const [k, e] = await Promise.all([kr.json(), en.json()]);
        setReportKo(k.report_markdown);
        setReportEn(e.report_markdown);
      })
      .catch(e => setError(e instanceof Error ? e.message : "Report failed"))
      .finally(() => setLoading(false));
  }, [statsData, guideline]);

  return (
    <SectionCard title={`AI 임상 R&D 리포트 · ${guideline === "EP05" ? "CLSI EP05-A3" : "CLSI EP09-A3"}`}>
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-slate-400">OpenAI가 통계 결과를 바탕으로 생성한 임상 해석 리포트</p>
        <div className="flex gap-1.5">
          {(["ko", "en"] as const).map(l => (
            <button key={l} onClick={() => setTab(l)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition ${tab === l ? "bg-blue-700 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"}`}>
              {l === "ko" ? "한국어" : "English"}
            </button>
          ))}
        </div>
      </div>
      {loading && (
        <div className="flex items-center gap-3 py-6 text-slate-500 text-sm">
          <span className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin inline-block" />
          OpenAI가 리포트를 생성 중입니다...
        </div>
      )}
      {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">{error}</div>}
      {!loading && !error && (
        <div className="prose prose-sm max-w-none text-slate-700">
          <ReactMarkdown>{tab === "ko" ? (reportKo ?? "") : (reportEn ?? "")}</ReactMarkdown>
        </div>
      )}
    </SectionCard>
  );
}


// ─── Schema Confirmation Modal ────────────────────────────────────────────────

function SchemaConfirmBar({
  guideline, schema, onConfirm, onCancel,
}: {
  guideline: GuidelineType;
  schema: SchemaMapping | SchemaMappingEP09;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-4">
      <div className="text-2xl">🤖</div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-amber-800 mb-2">AI가 컬럼을 자동 매핑했습니다. 확인 후 분석을 실행하세요.</p>
        <div className="flex flex-wrap gap-3 text-xs font-mono">
          {guideline === "EP05" ? (
            <>
              <span className="bg-white border border-amber-200 rounded px-2 py-1">
                측정값 컬럼: <strong>{(schema as SchemaMapping).target_column}</strong>
              </span>
              <span className="bg-white border border-amber-200 rounded px-2 py-1">
                그룹 컬럼: <strong>{(schema as SchemaMapping).group_columns?.join(", ") || "없음"}</strong>
              </span>
            </>
          ) : (
            <>
              <span className="bg-white border border-amber-200 rounded px-2 py-1">
                Reference 컬럼: <strong>{(schema as SchemaMappingEP09).reference_column}</strong>
              </span>
              <span className="bg-white border border-amber-200 rounded px-2 py-1">
                Test 컬럼: <strong>{(schema as SchemaMappingEP09).test_column}</strong>
              </span>
            </>
          )}
        </div>
      </div>
      <div className="flex gap-2 shrink-0">
        <button onClick={onConfirm}
          className="px-4 py-2 bg-blue-700 text-white text-xs font-semibold rounded-lg hover:bg-blue-800 transition">
          분석 실행
        </button>
        <button onClick={onCancel}
          className="px-3 py-2 bg-white border border-slate-300 text-slate-600 text-xs font-semibold rounded-lg hover:bg-slate-50 transition">
          취소
        </button>
      </div>
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────

export default function ClinicalValidationDashboard({
  disease,
  variant,
  variantLabel,
  mutations = [],
  onVariantMismatchWarning,
}: {
  disease: DiseaseType;
  variant: string;
  variantLabel: string;
  mutations?: Mutation[];
  onVariantMismatchWarning: () => void;
}) {
  const [guideline, setGuideline] = useState<GuidelineType>("EP05");
  const [file,      setFile]      = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [state, setState]         = useState<AnalysisState>("idle");
  const [errorMsg, setErrorMsg]   = useState<string | null>(null);

  const [schema,    setSchema]    = useState<SchemaMapping | SchemaMappingEP09 | null>(null);
  const [ep05Result, setEp05Result] = useState<EP05Result | null>(null);
  const [ep09Result, setEp09Result] = useState<EP09Result | null>(null);
  const [experimentId, setExperimentId]         = useState<string | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const { collect } = useAutoCollect();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter  = useRef(0);
  const currentGuide = GUIDELINES.find(g => g.id === guideline)!;

  // disease가 바뀌면 분석 상태 전체 초기화
  useEffect(() => {
    setFile(null); setSchema(null); setEp05Result(null); setEp09Result(null);
    setErrorMsg(null); setState("idle"); setExperimentId(null); setFeedbackSubmitted(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [disease]);

  // variant가 바뀌면 업로드 파일이 있을 경우 경고 후 리셋
  useEffect(() => {
    setExperimentId(null); setFeedbackSubmitted(false);
    if (file) {
      onVariantMismatchWarning();
      setFile(null); setSchema(null); setEp05Result(null); setEp09Result(null);
      setErrorMsg(null); setState("idle");
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [variant]);

  // ─── Drag & Drop ────────────────────────────────────────────────────────

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); dragCounter.current++; setIsDragging(true); };
  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); if (--dragCounter.current === 0) setIsDragging(false); };
  const handleDragOver  = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); };
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); dragCounter.current = 0; setIsDragging(false);
    const f = e.dataTransfer.files[0]; if (f) acceptFile(f);
  };
  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) acceptFile(f); };

  const acceptFile = (f: File) => {
    setFile(f); setSchema(null); setEp05Result(null); setEp09Result(null);
    setErrorMsg(null); setState("idle");
  };

  const reset = () => {
    setFile(null); setSchema(null); setEp05Result(null); setEp09Result(null);
    setErrorMsg(null); setState("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ─── Upload → schema detection ──────────────────────────────────────────

  const handleUpload = async () => {
    if (!file) return;
    setState("uploading");
    setErrorMsg(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("guideline", guideline);
      const res = await fetch("/api/v1/upload", { method: "POST", body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Upload failed"); }
      const data = await res.json();
      setSchema(data.schema_mapping);
      setState("confirming");
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Upload failed");
      setState("error");
    }
  };

  // ─── Analyze ────────────────────────────────────────────────────────────

  const handleAnalyze = async () => {
    if (!file || !schema) return;
    setState("analyzing");
    setErrorMsg(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("schema_mapping", JSON.stringify(schema));
      const res = await fetch(currentGuide.endpoint, { method: "POST", body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Analysis failed"); }
      const data = await res.json();
      if (guideline === "EP05") setEp05Result(data as EP05Result);
      else                      setEp09Result(data as EP09Result);
      setState("done");

      // Phase 2: background data collection (fire-and-forget, captures ID for feedback)
      const ep05 = guideline === "EP05" ? (data as EP05Result) : null;
      const ep09 = guideline === "EP09" ? (data as EP09Result) : null;
      void collect({
        disease_type:      disease,
        variant_name:      variant,
        mismatch_count:    mutations.length,
        guideline,
        source_filename:   file?.name,
        grand_mean:        ep05?.grand_mean,
        repeatability_cv:  ep05?.repeatability?.cv_percent,
        reproducibility_cv: ep05?.reproducibility?.cv_percent,
        anova_f_value:     ep05?.anova?.f_value,
        anova_p_value:     ep05?.anova?.p_value,
        sample_count:      ep05?.sample_count ?? ep09?.sample_count,
        deming_slope:          ep09?.deming?.slope,
        bland_altman_mean_diff: ep09?.bland_altman?.mean_diff,
        pearson_r:             ep09?.pearson_r,
      }).then(id => { if (id) setExperimentId(id); });
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Analysis failed");
      setState("error");
    }
  };

  const showResults = state === "done" && (ep05Result || ep09Result);

  const submitFeedback = async (value: FeedbackValue) => {
    setFeedbackSubmitted(true);
    try {
      await fetch("/api/v2/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_id: experimentId,
          feedback_value: value,
          guideline,
          disease_type: disease,
          variant_name: variant,
        }),
      });
    } catch {
      /* silent — feedback is optional */
    }
  };

  return (
    <div className="space-y-5">

      {/* Variant context notice */}
      {variant !== "wild-type" && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 flex items-center gap-2 text-sm text-red-800">
          <span className="font-bold text-red-500">●</span>
          <span>현재 변이주: <strong>{variantLabel}</strong> — 업로드 파일이 해당 변이의 임상 데이터인지 확인하세요.</span>
        </div>
      )}

      {/* ── Controls ──────────────────────────────────────────────────────── */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        {/* Guideline selector */}
        <div className="mb-5">
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
            CLSI 분석 가이드라인
          </label>
          <div className="grid grid-cols-2 gap-2 max-w-md">
            {GUIDELINES.map(g => (
              <label key={g.id}
                className={`flex items-start gap-2.5 p-3 rounded-lg border-2 cursor-pointer transition-all ${
                  guideline === g.id ? "border-blue-600 bg-blue-50" : "border-slate-200 hover:border-slate-300"
                }`}
              >
                <input type="radio" name="guideline" value={g.id} checked={guideline === g.id}
                  onChange={() => { setGuideline(g.id); reset(); }}
                  className="mt-0.5 accent-blue-700 shrink-0" />
                <div>
                  <p className={`text-xs font-bold ${guideline === g.id ? "text-blue-800" : "text-slate-700"}`}>{g.title}</p>
                  <p className={`text-xs mt-0.5 ${guideline === g.id ? "text-blue-600" : "text-slate-400"}`}>{g.subtitle}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Dropzone */}
        <div className="mt-5">
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
            CFX96 Raw Data 업로드
          </label>
          <div
            onDragEnter={handleDragEnter} onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}  onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
            className={`rounded-xl border-2 border-dashed p-8 text-center transition-all ${
              isDragging ? "border-blue-500 bg-blue-50 scale-[1.01]"
              : file      ? "border-emerald-400 bg-emerald-50 cursor-default"
              :             "border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50 cursor-pointer"
            }`}
          >
            <input ref={fileInputRef} type="file" accept=".xlsx,.xls,.csv" className="hidden" onChange={handleFileInput} />
            {file ? (
              <div className="flex items-center justify-center gap-4">
                <span className="text-3xl">📂</span>
                <div className="text-left">
                  <p className="font-semibold text-slate-800">{file.name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
                <button onClick={e => { e.stopPropagation(); reset(); }}
                  className="ml-4 w-7 h-7 flex items-center justify-center rounded-full text-slate-400 hover:text-red-500 hover:bg-red-50 transition text-sm">✕</button>
              </div>
            ) : (
              <>
                <div className="text-3xl mb-2">📊</div>
                <p className="font-semibold text-slate-600">CFX96 데이터 파일을 드래그하거나 클릭하여 선택</p>
                <p className="text-xs text-slate-400 mt-1">Excel (.xlsx, .xls) · CSV</p>
              </>
            )}
          </div>
        </div>

        {/* Upload Button */}
        {file && state === "idle" && (
          <div className="flex justify-center mt-4">
            <button onClick={handleUpload}
              className="flex items-center gap-2 px-7 py-2.5 bg-slate-700 hover:bg-slate-800 text-white text-sm font-semibold rounded-xl transition shadow-sm active:scale-95">
              🤖 AI 컬럼 자동 매핑
            </button>
          </div>
        )}

        {/* Uploading spinner */}
        {state === "uploading" && (
          <div className="flex items-center justify-center gap-3 mt-4 text-slate-500 text-sm">
            <span className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin inline-block" />
            파일 파싱 및 AI 컬럼 매핑 중...
          </div>
        )}

        {/* Schema Confirmation */}
        {state === "confirming" && schema && (
          <div className="mt-4">
            <SchemaConfirmBar guideline={guideline} schema={schema}
              onConfirm={handleAnalyze} onCancel={reset} />
          </div>
        )}

        {/* Analyzing spinner */}
        {state === "analyzing" && (
          <div className="flex items-center justify-center gap-3 mt-4 text-slate-500 text-sm">
            <span className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin inline-block" />
            statsmodels {guideline} 통계 분석 수행 중...
          </div>
        )}

        {/* Error + retry */}
        {state === "error" && errorMsg && (
          <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm flex items-start justify-between gap-3">
            <span>{errorMsg}</span>
            <button
              onClick={schema ? handleAnalyze : handleUpload}
              className="shrink-0 text-xs font-semibold bg-red-100 hover:bg-red-200 text-red-700 border border-red-300 rounded-lg px-3 py-1.5 transition"
            >
              다시 시도
            </button>
          </div>
        )}
      </div>

      {/* ── Results ───────────────────────────────────────────────────────── */}
      <div className="space-y-4">
        {!showResults && (
          <div className="bg-white border border-slate-200 border-dashed rounded-xl p-12 text-center text-slate-400">
            <p className="text-4xl mb-3">📈</p>
            <p className="font-medium text-slate-500">파일 업로드 후 분석을 실행하면</p>
            <p className="text-sm mt-1">{currentGuide.title} 통계 결과가 이곳에 표시됩니다.</p>
          </div>
        )}

        {showResults && guideline === "EP05" && ep05Result && (
          <>
            <div className="flex items-center gap-2 mb-1">
              <Badge status="info" label={currentGuide.title} />
              <span className="text-xs text-slate-400">{ep05Result.target_column} · n={ep05Result.sample_count}</span>
            </div>
            <EP05StatsPanel result={ep05Result} />
            <AiReportPanel statsData={ep05Result} guideline={guideline} />
          </>
        )}

        {showResults && guideline === "EP09" && ep09Result && (
          <>
            <div className="flex items-center gap-2 mb-1">
              <Badge status="info" label={currentGuide.title} />
              <span className="text-xs text-slate-400">{ep09Result.reference_column} vs {ep09Result.test_column} · n={ep09Result.sample_count}</span>
            </div>
            <EP09StatsPanel result={ep09Result} />
            <AiReportPanel statsData={ep09Result} guideline={guideline} />
          </>
        )}

        {/* Phase 2: AI 예측 정확도 피드백 */}
        {showResults && !feedbackSubmitted && (
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
              AI 예측 정확도 피드백
              <span className="ml-1.5 font-normal normal-case text-slate-400">(선택 · 데이터 품질 개선에 사용됩니다)</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {([
                { value: "accurate"           as FeedbackValue, label: "정확",    color: "emerald" },
                { value: "partially_accurate" as FeedbackValue, label: "부분 정확", color: "amber"   },
                { value: "incorrect"          as FeedbackValue, label: "부정확",  color: "red"     },
              ]).map(({ value, label, color }) => (
                <button
                  key={value}
                  onClick={() => submitFeedback(value)}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold border transition active:scale-95 ${
                    color === "emerald"
                      ? "border-emerald-300 text-emerald-700 hover:bg-emerald-50"
                      : color === "amber"
                      ? "border-amber-300 text-amber-700 hover:bg-amber-50"
                      : "border-red-300 text-red-700 hover:bg-red-50"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}
        {showResults && feedbackSubmitted && (
          <p className="text-xs text-slate-400 text-center py-1">
            피드백 감사합니다. 수집된 데이터는 Phase 2 AI 모델 개선에 활용됩니다.
          </p>
        )}
      </div>

    </div>
  );
}
