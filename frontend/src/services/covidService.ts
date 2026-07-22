// COVID-19 data service — disease.sh REST API (demo mode: /api/demo/*)

import { logFetch } from '../utils/apiLogger';

const BASE = 'https://disease.sh/v3/covid-19';

// Cached demo mode flag — set by checkDemoMode() called from App on startup
let _demoMode: boolean | null = null;

export async function checkDemoMode(): Promise<boolean> {
  if (_demoMode !== null) return _demoMode;
  try {
    const res = await logFetch('/api/demo/mode');
    if (!res.ok) { _demoMode = false; return false; }
    const data = await res.json() as { demo_mode: boolean };
    _demoMode = data.demo_mode;
  } catch {
    _demoMode = false;
  }
  return _demoMode;
}

export function isDemoMode(): boolean {
  return _demoMode === true;
}

const get = async <T>(path: string): Promise<T> => {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${path}`);
  return res.json();
};

const getDemoProxy = async <T>(path: string): Promise<T> => {
  const res = await fetch(`/api/demo${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}: demo${path}`);
  return res.json();
};

export interface GlobalStats {
  cases: number;
  deaths: number;
  recovered: number;
  active: number;
  todayCases: number;
  todayDeaths: number;
  todayRecovered: number;
  critical: number;
  population: number;
  updated: number;
}

export interface CountryData {
  country: string;
  countryInfo: { iso2: string; iso3: string; flag: string; lat: number; long: number };
  cases: number;
  deaths: number;
  recovered: number;
  active: number;
  todayCases: number;
  todayDeaths: number;
  population: number;
  casesPerOneMillion: number;
  deathsPerOneMillion: number;
}

export interface HistoricalTimeline {
  cases: Record<string, number>;
  deaths: Record<string, number>;
  recovered: Record<string, number>;
}

export const fetchGlobal = (): Promise<GlobalStats> =>
  _demoMode ? getDemoProxy<GlobalStats>('/covid-global') : get<GlobalStats>('/all');

export const fetchCountries = (): Promise<CountryData[]> =>
  _demoMode ? getDemoProxy<CountryData[]>('/covid-countries') : get<CountryData[]>('/countries?sort=cases');

export const fetchGlobalHistory = (days = 180): Promise<HistoricalTimeline> =>
  _demoMode
    ? getDemoProxy<HistoricalTimeline>('/covid-historical')
    : get<HistoricalTimeline>(`/historical/all?lastdays=${days}`);

export const fetchCountryHistory = (country: string, days = 180): Promise<{ country: string; timeline: HistoricalTimeline }> =>
  get<{ country: string; timeline: HistoricalTimeline }>(
    `/historical/${encodeURIComponent(country)}?lastdays=${days}`
  );

// Korea regional mock (data.go.kr API stopped public access 2023-08)
export const KOREA_REGIONAL: { region: string; confirmed: number; newConfirmed: number; deaths: number }[] = [
  { region: '경기', confirmed: 8901209, newConfirmed: 10476, deaths: 8538 },
  { region: '서울', confirmed: 6698035, newConfirmed: 8825,  deaths: 6649 },
  { region: '경남', confirmed: 1912833, newConfirmed: 442,   deaths: 2003 },
  { region: '인천', confirmed: 1857275, newConfirmed: 884,   deaths: 1930 },
  { region: '부산', confirmed: 1963017, newConfirmed: 3934,  deaths: 2881 },
  { region: '경북', confirmed: 1457067, newConfirmed: 738,   deaths: 2123 },
  { region: '대구', confirmed: 1476145, newConfirmed: 2668,  deaths: 2051 },
  { region: '충남', confirmed: 1364723, newConfirmed: 1854,  deaths: 1634 },
  { region: '전북', confirmed: 1090547, newConfirmed: 356,   deaths: 1252 },
  { region: '충북', confirmed: 1073338, newConfirmed: 1705,  deaths: 1094 },
  { region: '전남', confirmed: 1048254, newConfirmed: 487,   deaths: 1025 },
  { region: '광주', confirmed: 997048,  newConfirmed: 557,   deaths: 854  },
  { region: '강원', confirmed: 984563,  newConfirmed: 647,   deaths: 1383 },
  { region: '대전', confirmed: 909928,  newConfirmed: 155,   deaths: 951  },
  { region: '울산', confirmed: 698215,  newConfirmed: 1242,  deaths: 540  },
  { region: '제주', confirmed: 414395,  newConfirmed: 382,   deaths: 298  },
  { region: '세종', confirmed: 269111,  newConfirmed: 392,   deaths: 58   },
];

export const fmtNum = (n: number | null | undefined): string => {
  if (n == null || !Number.isFinite(n)) return 'N/A';
  return n.toLocaleString('en-US');
};

export const fmtCompact = (n: number | null | undefined): string => {
  if (n == null || !Number.isFinite(n)) return '0';
  const v = Number(n);
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return Math.round(v).toString();
};
