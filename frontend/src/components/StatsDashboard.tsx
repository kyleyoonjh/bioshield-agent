import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import ReactMarkdown from "react-markdown";
import type { StatsResult } from "../types";

interface StatsDashboardProps {
  stats: StatsResult;
}

const PIE_COLORS = ["#22c55e", "#f59e0b"];

export default function StatsDashboard({ stats }: StatsDashboardProps) {
  const [reportKo, setReportKo] = useState<string | null>(null);
  const [reportEn, setReportEn] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"ko" | "en">("ko");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReports = async () => {
      setLoading(true);
      setError(null);
      try {
        const [koRes, enRes] = await Promise.all([
          fetch("/api/v1/report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ stats_data: stats, language: "ko" }),
          }),
          fetch("/api/v1/report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ stats_data: stats, language: "en" }),
          }),
        ]);
        if (!koRes.ok || !enRes.ok) throw new Error("Report generation failed");
        const koData = await koRes.json();
        const enData = await enRes.json();
        setReportKo(koData.report_markdown);
        setReportEn(enData.report_markdown);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Report failed");
      } finally {
        setLoading(false);
      }
    };
    fetchReports();
  }, [stats]);

  const varianceData = [
    { name: "군내 (Repeatability)", value: stats.variance_components.within_group_percent },
    { name: "군간 (Reproducibility)", value: stats.variance_components.between_group_percent },
  ];

  const groupChartData = stats.groups.map((g) => ({
    name: g.group.length > 20 ? g.group.slice(0, 20) + "…" : g.group,
    mean: g.mean,
    cv: g.cv_percent,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="F-value" value={stats.anova.f_value.toFixed(4)} />
        <StatCard label="p-value" value={stats.anova.p_value.toFixed(6)} />
        <StatCard
          label="Repeatability CV%"
          value={`${stats.repeatability.cv_percent}%`}
          sub={`SD: ${stats.repeatability.sd}`}
        />
        <StatCard
          label="Reproducibility CV%"
          value={`${stats.reproducibility.cv_percent}%`}
          sub={`SD: ${stats.reproducibility.sd}`}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-semibold mb-4">분산 성분 비율</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={varianceData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {varianceData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Legend />
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-semibold mb-4">그룹별 평균 Ct</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={groupChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="mean" fill="#22c55e" name="Mean Ct" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold">AI 임상 R&D 리포트</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab("ko")}
              className={`px-4 py-1 rounded-lg text-sm ${
                activeTab === "ko" ? "bg-bio-600 text-white" : "bg-gray-100 text-gray-600"
              }`}
            >
              한국어
            </button>
            <button
              onClick={() => setActiveTab("en")}
              className={`px-4 py-1 rounded-lg text-sm ${
                activeTab === "en" ? "bg-bio-600 text-white" : "bg-gray-100 text-gray-600"
              }`}
            >
              English
            </button>
          </div>
        </div>

        {loading && <p className="text-gray-500">OpenAI가 리포트를 생성 중...</p>}
        {error && <p className="text-red-600">{error}</p>}
        {!loading && !error && (
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{activeTab === "ko" ? reportKo ?? "" : reportEn ?? ""}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-gray-800 mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}
