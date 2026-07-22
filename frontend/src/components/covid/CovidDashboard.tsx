import { useState, useEffect, useMemo } from "react";
import {
  fetchGlobal, fetchCountries,
  KOREA_REGIONAL, fmtNum,
  type GlobalStats, type CountryData,
} from "../../services/covidService";
import CovidWorldMap from "./CovidWorldMap";
import CovidKoreaMap from "./CovidKoreaMap";

type ActiveMetric = "cases" | "deaths" | "active" | "todayCases" | "recovered" | "cfr" | "infectionRate";
type SortKey    = "cases" | "deaths" | "newConfirmed" | "deathRate" | "country";

const METRIC_MAP: Record<ActiveMetric, "cases" | "deaths" | "active" | "todayCases"> = {
  cases: "cases", deaths: "deaths", active: "active", todayCases: "todayCases",
  recovered: "cases", cfr: "deaths", infectionRate: "active",
};


export default function CovidDashboard() {
  const [global, setGlobal]       = useState<GlobalStats | null>(null);
  const [countries, setCountries] = useState<CountryData[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);

  const [selected, setSelected]   = useState<string>("Global");
  const [metric, setMetric]       = useState<ActiveMetric>("cases");
  const [sortKey, setSortKey]     = useState<SortKey>("cases");
  const [sortAsc, setSortAsc]     = useState(false);

  const isSouthKorea = useMemo(() => {
    const n = selected.toLowerCase();
    return n === "s. korea" || n === "south korea" || n === "kr";
  }, [selected]);

  const currentCountry = useMemo(
    () => countries.find(c => c.country === selected) ?? null,
    [countries, selected]
  );

  // Load global + countries on mount
  useEffect(() => {
    setLoading(true);
    Promise.all([fetchGlobal(), fetchCountries()])
      .then(([g, cs]) => {
        setGlobal(g);
        setCountries(cs);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // ── Displayed stats ─────────────────────────────────────────────────────────
  const stats = useMemo(() => {
    if (selected === "Global") return global;
    return currentCountry ?? global;
  }, [selected, global, currentCountry]);

  const cfr = useMemo(() => {
    if (!stats?.deaths || !stats?.cases) return null;
    return ((stats.deaths / stats.cases) * 100).toFixed(2);
  }, [stats]);

  const infectionRate = useMemo(() => {
    if (!stats?.active || !stats?.population) return null;
    return ((stats.active / stats.population) * 100).toFixed(2);
  }, [stats]);

  // ── Top countries table ─────────────────────────────────────────────────────
  const topCountries = useMemo(() => {
    const sorted = [...countries].sort((a, b) => {
      if (sortKey === "country") return a.country.localeCompare(b.country);
      if (sortKey === "deathRate") {
        const ar = a.cases > 0 ? a.deaths / a.cases : 0;
        const br = b.cases > 0 ? b.deaths / b.cases : 0;
        return sortAsc ? ar - br : br - ar;
      }
      if (sortKey === "newConfirmed") return sortAsc ? a.todayCases - b.todayCases : b.todayCases - a.todayCases;
      return sortAsc ? (a as never)[sortKey] - (b as never)[sortKey] : (b as never)[sortKey] - (a as never)[sortKey];
    });
    return sorted.slice(0, 10);
  }, [countries, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(p => !p);
    else { setSortKey(key); setSortAsc(false); }
  };

  // ── KPI cards config ─────────────────────────────────────────────────────────
  const kpiCards = [
    {
      id: "todayCases" as ActiveMetric,
      label: "신규 확진",
      sublabel: "오늘",
      value: fmtNum(stats?.todayCases),
      color: "from-orange-500 to-amber-400",
    },
    {
      id: "cases" as ActiveMetric,
      label: "누적 확진",
      sublabel: "",
      value: fmtNum(stats?.cases),
      color: "from-blue-600 to-blue-400",
    },
    {
      id: "active" as ActiveMetric,
      label: "활성 환자",
      sublabel: "",
      value: fmtNum(stats?.active),
      color: "from-yellow-500 to-yellow-300",
    },
    {
      id: "recovered" as ActiveMetric,
      label: "회복",
      sublabel: "",
      value: fmtNum(stats?.recovered),
      color: "from-emerald-500 to-green-400",
    },
    {
      id: "deaths" as ActiveMetric,
      label: "사망",
      sublabel: "",
      value: fmtNum(stats?.deaths),
      color: "from-red-600 to-rose-400",
    },
    {
      id: "cfr" as ActiveMetric,
      label: "치명률 (CFR)",
      sublabel: "",
      value: cfr != null ? `${cfr}%` : "N/A",
      color: "from-red-800 to-red-500",
    },
    {
      id: "infectionRate" as ActiveMetric,
      label: "현재 감염률",
      sublabel: "활성/인구",
      value: infectionRate != null ? `${infectionRate}%` : "N/A",
      color: "from-purple-600 to-violet-400",
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        <span className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mr-3" />
        COVID-19 데이터 로딩 중...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-32 text-red-400 text-sm">
        데이터 로드 실패: {error}
      </div>
    );
  }

  return (
    <div className="space-y-5">

      {/* ── Header bar ──────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3 justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-800">COVID-19 Pandemic Dashboard</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            출처: disease.sh &nbsp;·&nbsp; 업데이트: {global?.updated ? new Date(global.updated).toLocaleString("ko-KR") : "—"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Country selector */}
          <select
            value={selected}
            onChange={e => setSelected(e.target.value)}
            className="text-sm bg-white border border-slate-200 text-slate-700 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            <option value="Global">🌍 Global</option>
            {countries.map(c => (
              <option key={c.country} value={c.country}>{c.country}</option>
            ))}
          </select>
          <button
            onClick={() => setSelected("Global")}
            className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Global
          </button>
          <button
            onClick={() => {
              const kr = countries.find(c => c.countryInfo?.iso2 === "KR");
              setSelected(kr?.country ?? "S. Korea");
            }}
            className="px-3 py-1.5 text-sm bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
          >
            🇰🇷 한국
          </button>
        </div>
      </div>

      {/* ── KPI Cards ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {kpiCards.map(card => (
          <button
            key={card.id}
            onClick={() => setMetric(card.id)}
            className={`relative overflow-hidden rounded-xl p-3 text-left transition-all ${
              metric === card.id
                ? "ring-2 ring-white shadow-lg scale-105"
                : "opacity-85 hover:opacity-100 hover:scale-102"
            } bg-gradient-to-br ${card.color} text-white`}
          >
            <div className="text-xs font-semibold opacity-80 mb-1">{card.label}</div>
            <div className="text-lg font-bold leading-tight">{card.value}</div>
            {card.sublabel && <div className="text-xs opacity-60 mt-0.5">{card.sublabel}</div>}
          </button>
        ))}
      </div>

      {/* ── Map (full width) ─────────────────────────────────────────────────── */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-700">
            {isSouthKorea ? "🇰🇷 한국 지역별 현황" : "🌍 전 세계 팬데믹 확산"}
          </h3>
          <span className="text-xs text-slate-400">
            {METRIC_MAP[metric].charAt(0).toUpperCase() + METRIC_MAP[metric].slice(1)}
          </span>
        </div>
        <div className="p-2">
          {isSouthKorea ? (
            <CovidKoreaMap
              data={KOREA_REGIONAL}
              metric={metric === "deaths" ? "deaths" : metric === "todayCases" ? "newCases" : "cases"}
            />
          ) : (
            <CovidWorldMap
              countries={countries}
              metric={METRIC_MAP[metric]}
              focusCountry={selected !== "Global" ? selected : null}
              onCountryClick={c => setSelected(c.country)}
            />
          )}
        </div>
      </div>

      {/* ── South Korea Regional table ────────────────────────────────────────── */}
      {isSouthKorea && (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-100">
            <h3 className="text-sm font-semibold text-slate-700">🇰🇷 지역별 누적 현황</h3>
            <p className="text-xs text-slate-400 mt-0.5">기준: 2023-08-31 (data.go.kr ODMS_COVID_04)</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  {["지역", "누적 확진", "신규", "사망"].map(h => (
                    <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...KOREA_REGIONAL].sort((a, b) => b.confirmed - a.confirmed).map(row => (
                  <tr key={row.region} className="border-t border-slate-50 hover:bg-slate-50">
                    <td className="px-4 py-2 font-medium text-slate-700">{row.region}</td>
                    <td className="px-4 py-2 text-slate-600">{fmtNum(row.confirmed)}</td>
                    <td className="px-4 py-2 text-orange-600 font-medium">+{fmtNum(row.newConfirmed)}</td>
                    <td className="px-4 py-2 text-red-500">{fmtNum(row.deaths)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Top Countries table ───────────────────────────────────────────────── */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100">
          <h3 className="text-sm font-semibold text-slate-700">🏆 Top 10 국가</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                {(
                  [
                    { key: "country" as SortKey, label: "국가" },
                    { key: "cases" as SortKey, label: "누적 확진" },
                    { key: "newConfirmed" as SortKey, label: "신규 (오늘)" },
                    { key: "deaths" as SortKey, label: "사망" },
                    { key: "deathRate" as SortKey, label: "치명률" },
                  ]
                ).map(col => (
                  <th key={col.key}
                    className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 cursor-pointer hover:text-blue-600 select-none"
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    {sortKey === col.key ? (sortAsc ? " ↑" : " ↓") : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {topCountries.map((c, i) => (
                <tr
                  key={c.country}
                  className="border-t border-slate-50 hover:bg-blue-50 cursor-pointer"
                  onClick={() => setSelected(c.country)}
                >
                  <td className="px-4 py-2 flex items-center gap-2">
                    <span className="text-xs text-slate-400 w-5 text-right">{i + 1}</span>
                    {c.countryInfo?.flag && (
                      <img src={c.countryInfo.flag} alt="" className="w-5 h-3.5 object-cover rounded-sm" />
                    )}
                    <span className="font-medium text-slate-700">{c.country}</span>
                  </td>
                  <td className="px-4 py-2 text-blue-700 font-medium">{fmtNum(c.cases)}</td>
                  <td className="px-4 py-2 text-orange-600">+{fmtNum(c.todayCases)}</td>
                  <td className="px-4 py-2 text-red-500">{fmtNum(c.deaths)}</td>
                  <td className="px-4 py-2 text-slate-600">
                    {c.cases > 0 ? `${((c.deaths / c.cases) * 100).toFixed(2)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <div className="text-center text-xs text-slate-400 pb-4">
        데이터 출처: disease.sh (Johns Hopkins · WHO) &nbsp;·&nbsp; 한국 지역 데이터: data.go.kr (2023-08-31 기준)
      </div>
    </div>
  );
}
