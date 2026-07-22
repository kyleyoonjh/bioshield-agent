// Ported from COVID-19-Interactive-Data-Visualization-Dashboard/WorldMap.js
// Changes: removed Redux, removed airport markers for brevity, TypeScript types added
import { useEffect, useRef, useMemo, useState, useCallback } from "react";
import * as d3 from "d3";
import { fmtNum, fmtCompact, type CountryData } from "../../services/covidService";

// ── Types ──────────────────────────────────────────────────────────────────────
export type MapMetric = "cases" | "deaths" | "active" | "todayCases";

interface Props {
  countries: CountryData[];
  metric?: MapMetric;
  focusCountry?: string | null;
  onCountryClick?: (country: CountryData) => void;
  width?: number;
  height?: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────────
const getCountryName = (feature: GeoJSON.Feature): string =>
  (feature.properties?.name ?? feature.properties?.ADMIN ?? feature.properties?.NAME ?? "Unknown") as string;

const getShortName = (name: string): string => {
  const MAP: Record<string, string> = {
    "United States of America": "USA",
    "United Kingdom": "UK",
    "United Arab Emirates": "UAE",
    "Russian Federation": "Russia",
    "Democratic Republic of the Congo": "DR Congo",
    "Central African Republic": "CAR",
    "Bosnia and Herzegovina": "Bosnia",
    "Dominican Republic": "Dominican Rep.",
    "Papua New Guinea": "Papua N.G.",
    "Trinidad and Tobago": "Trinidad",
  };
  const mapped = MAP[name] ?? name;
  return mapped.length <= 14 ? mapped : `${mapped.slice(0, 13)}.`;
};

const getPer100k = (c: CountryData | undefined, metric: MapMetric): number => {
  if (!c) return 0;
  const val = Number(c[metric] ?? 0);
  const pop = Number(c.population ?? 0);
  if (!Number.isFinite(val) || val <= 0 || !Number.isFinite(pop) || pop <= 0) return 0;
  return (val / pop) * 100000;
};

const fmtCompactNum = (v: number): string => {
  if (!Number.isFinite(v) || v <= 0) return "";
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return `${Math.round(v)}`;
};

// ── Component ──────────────────────────────────────────────────────────────────
export default function CovidWorldMap({
  countries,
  metric = "cases",
  focusCountry = null,
  onCountryClick,
  width = 960,
  height = 500,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef       = useRef<SVGSVGElement>(null);
  const tooltipRef   = useRef<HTMLDivElement>(null);

  const [worldData, setWorldData]       = useState<GeoJSON.FeatureCollection | null>(null);
  const [isLoading, setIsLoading]       = useState(true);
  const [error, setError]               = useState<string | null>(null);
  const [containerWidth, setContainerWidth] = useState(width);

  const margin = { top: 10, right: 10, bottom: 10, left: 10 };
  const renderWidth  = Math.max(320, containerWidth);
  const renderHeight = Math.round((height / width) * renderWidth);

  // ── Load world GeoJSON ────────────────────────────────────────────────────
  useEffect(() => {
    setIsLoading(true);
    fetch("/assets/world.geojson")
      .then(r => { if (!r.ok) throw new Error("Failed to load geojson"); return r.json(); })
      .then((geo: GeoJSON.FeatureCollection) => { setWorldData(geo); setIsLoading(false); })
      .catch(e => { setError(e.message); setIsLoading(false); });
  }, []);

  // ── Responsive width ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => setContainerWidth(Math.max(320, Math.floor(el.getBoundingClientRect().width || width)));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [width]);

  // ── Build iso3 lookup map from disease.sh data ────────────────────────────
  const countryByIso3 = useMemo(() => {
    const m = new Map<string, CountryData>();
    countries.forEach(c => {
      const iso3 = c.countryInfo?.iso3;
      if (iso3) m.set(iso3.toUpperCase(), c);
    });
    return m;
  }, [countries]);

  const countryByName = useMemo(() => {
    const m = new Map<string, CountryData>();
    countries.forEach(c => { if (c.country) m.set(c.country.toLowerCase(), c); });
    return m;
  }, [countries]);

  const lookupCountry = useCallback((feature: GeoJSON.Feature): CountryData | undefined => {
    const iso3 = (feature.properties?.iso_a3 ?? feature.properties?.ISO_A3 ?? "") as string;
    return countryByIso3.get(iso3.toUpperCase())
      ?? countryByName.get(getCountryName(feature).toLowerCase());
  }, [countryByIso3, countryByName]);

  // ── Filter features (show only focus country when non-global) ────────────
  const filteredData = useMemo((): GeoJSON.FeatureCollection | null => {
    if (!worldData) return null;
    const focusName = String(focusCountry ?? "").toLowerCase();
    const isGlobal = !focusName || focusName === "global" || focusName === "all";
    if (isGlobal) {
      return { ...worldData, features: worldData.features.filter(f => {
        const iso3 = (f.properties?.ISO_A3 ?? f.properties?.iso_a3 ?? "") as string;
        const name = getCountryName(f).toLowerCase();
        return iso3 !== "ATA" && name !== "antarctica";
      })};
    }
    const focusCountryData = countries.find(c =>
      c.country.toLowerCase() === focusName ||
      c.countryInfo?.iso2?.toLowerCase() === focusName ||
      c.countryInfo?.iso3?.toLowerCase() === focusName
    );
    const focusIso3 = focusCountryData?.countryInfo?.iso3?.toUpperCase() ?? "";
    return {
      ...worldData,
      features: worldData.features.filter(f => {
        const iso3 = (f.properties?.iso_a3 ?? f.properties?.ISO_A3 ?? "") as string;
        if (focusIso3 && iso3.toUpperCase() === focusIso3) return true;
        return getCountryName(f).toLowerCase().includes(focusName);
      }),
    };
  }, [worldData, focusCountry, countries]);

  // ── D3 render ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!filteredData || !svgRef.current) return;

    const svg     = d3.select(svgRef.current);
    const tooltip = d3.select(tooltipRef.current!);
    const focusName = String(focusCountry ?? "").toLowerCase();
    const isGlobal  = !focusName || focusName === "global" || focusName === "all";

    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${renderWidth} ${renderHeight}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("width", "100%")
      .style("height", "auto");

    // Per-100k values for color scale
    const per100kVals = countries
      .map(c => getPer100k(c, metric))
      .filter(v => Number.isFinite(v) && v > 0);
    if (per100kVals.length === 0) return;

    const colorScale = d3.scaleSequential()
      .domain([0, d3.max(per100kVals) ?? 1])
      .interpolator(d3.interpolate("#2e7d32", "#c62828"));

    // Projection
    const projection = d3.geoMercator().fitExtent(
      [
        [margin.left + 12, margin.top + 52],
        [renderWidth - margin.right - 12, renderHeight - margin.bottom - 26],
      ],
      filteredData as d3.GeoPermissibleObjects
    );
    const pathGen = d3.geoPath().projection(projection);

    // Zoom
    const g = svg.append("g");
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 8])
      .on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);

    // Countries
    g.selectAll("path")
      .data(filteredData.features)
      .join("path")
      .attr("d", pathGen as never)
      .attr("fill", d => {
        const c = lookupCountry(d as GeoJSON.Feature);
        const v = getPer100k(c, metric);
        return v > 0 ? colorScale(v) : "#ddd";
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 0.5)
      .style("cursor", "pointer")
      .on("mouseover", (event, d) => {
        const c = lookupCountry(d as GeoJSON.Feature);
        const name = getCountryName(d as GeoJSON.Feature);
        const v100k = getPer100k(c, metric);
        d3.select(event.currentTarget as SVGPathElement)
          .attr("stroke", "#000").attr("stroke-width", 1.5);
        tooltip.style("opacity", "1")
          .style("left", `${event.pageX + 10}px`)
          .style("top", `${event.pageY - 28}px`)
          .html(c
            ? `<strong>${name}</strong><br/>${metric}: ${fmtNum(c[metric as keyof CountryData] as number)}<br/>per 100k: ${fmtNum(Math.round(v100k))}`
            : `<strong>${name}</strong><br/>No data`);
      })
      .on("mousemove", event => {
        tooltip.style("left", `${event.pageX + 10}px`).style("top", `${event.pageY - 28}px`);
      })
      .on("mouseout", event => {
        d3.select(event.currentTarget as SVGPathElement)
          .attr("stroke", "#fff").attr("stroke-width", 0.5);
        tooltip.style("opacity", "0");
      })
      .on("click", (_, d) => {
        const c = lookupCountry(d as GeoJSON.Feature);
        if (c && onCountryClick) onCountryClick(c);
      });

    // Country value labels (non-overlapping, top 80 by metric)
    const labelColorScale = d3.scaleLinear<string>()
      .domain(colorScale.domain())
      .range(["#2e7d32", "#c62828"])
      .clamp(true);

    type LabelEntry = { x: number; y: number; value: number; per100k: number; label: string };
    const candidates: LabelEntry[] = filteredData.features
      .map(f => {
        const c = lookupCountry(f as GeoJSON.Feature);
        const val = Number(c?.[metric as keyof CountryData] ?? 0);
        const p100k = getPer100k(c, metric);
        if (!c || !Number.isFinite(val) || val <= 0) return null;
        const centroid = pathGen.centroid(f as GeoJSON.Feature);
        if (!Number.isFinite(centroid[0])) return null;
        const short = getShortName(getCountryName(f as GeoJSON.Feature));
        return { x: centroid[0], y: centroid[1], value: val, per100k: p100k, label: `${short} ${fmtCompactNum(val)}` };
      })
      .filter(Boolean)
      .sort((a, b) => b!.value - a!.value)
      .slice(0, 80) as LabelEntry[];

    const selected: LabelEntry[] = [];
    const boxes: { l: number; r: number; t: number; b: number }[] = [];
    candidates.forEach(c => {
      const w = Math.max(38, c.label.length * 5.5);
      const box = { l: c.x - w / 2, r: c.x + w / 2, t: c.y - 12, b: c.y + 2 };
      const overlaps = boxes.some(b => box.l < b.r + 10 && box.r > b.l - 10 && box.t < b.b + 8 && box.b > b.t - 8);
      if (!overlaps) { selected.push(c); boxes.push(box); }
    });

    g.append("g")
      .selectAll("text")
      .data(selected)
      .join("text")
      .attr("x", d => d.x)
      .attr("y", d => d.y - 4)
      .attr("text-anchor", "middle")
      .attr("fill", d => labelColorScale(d.per100k))
      .style("font-size", "9px")
      .style("font-weight", "600")
      .style("pointer-events", "none")
      .style("paint-order", "stroke")
      .style("stroke", "rgba(255,255,255,0.7)")
      .style("stroke-width", "2px")
      .text(d => d.label);

    // Legend
    const lgW = Math.min(200, Math.max(120, renderWidth * 0.25));
    const lgH = 10;
    const lgX = renderWidth - lgW - 20;
    const lgY = renderHeight - 40;

    const defs = svg.append("defs");
    const grad = defs.append("linearGradient")
      .attr("id", `covid-legend-${metric}`)
      .attr("x1", "0%").attr("x2", "100%");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#2e7d32");
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#c62828");

    svg.append("g").attr("transform", `translate(${lgX},${lgY})`)
      .append("rect").attr("width", lgW).attr("height", lgH)
      .style("fill", `url(#covid-legend-${metric})`);

    const lgAxis = d3.axisBottom(d3.scaleLinear().domain(colorScale.domain()).range([0, lgW]))
      .ticks(5).tickFormat(v => fmtCompact(+v));
    svg.append("g").attr("transform", `translate(${lgX},${lgY + lgH})`).call(lgAxis)
      .selectAll("text").style("font-size", "9px").attr("fill", "#333");

    // Map title
    const titleMap: Record<string, string> = { todayCases: "New Cases (Today)" };
    const title = titleMap[metric] ?? `${metric.charAt(0).toUpperCase()}${metric.slice(1)}`;
    svg.append("text")
      .attr("x", renderWidth / 2).attr("y", 28)
      .attr("text-anchor", "middle")
      .style("font-size", "15px").style("font-weight", "bold").attr("fill", "#333")
      .text(`Pandemic ${title} by Country`);

    // Airport markers (global view only)
    if (isGlobal) {
      const AIRPORTS = [
        { iata:"ICN", name:"Incheon Int'l", lat:37.4602, lng:126.4407 },
        { iata:"JFK", name:"JFK Int'l",     lat:40.6413, lng:-73.7781  },
        { iata:"LHR", name:"Heathrow",      lat:51.47,   lng:-0.4543   },
        { iata:"HND", name:"Haneda",        lat:35.5494, lng:139.7798  },
        { iata:"CDG", name:"Charles de Gaulle", lat:49.0097, lng:2.5479 },
        { iata:"DXB", name:"Dubai Int'l",   lat:25.2532, lng:55.3657   },
        { iata:"SIN", name:"Singapore Changi", lat:1.3644, lng:103.9915 },
        { iata:"PEK", name:"Beijing Capital", lat:40.0799, lng:116.6031 },
        { iata:"LAX", name:"Los Angeles Int'l", lat:33.9416, lng:-118.4085 },
        { iata:"GRU", name:"São Paulo Guarulhos", lat:-23.4356, lng:-46.4731 },
        { iata:"JNB", name:"O.R. Tambo",    lat:-26.1337, lng:28.242   },
        { iata:"SYD", name:"Sydney",        lat:-33.9399, lng:151.1753  },
        { iata:"YYZ", name:"Toronto Pearson", lat:43.6777, lng:-79.6248 },
        { iata:"DEL", name:"Indira Gandhi", lat:28.5562, lng:77.1       },
        { iata:"DOH", name:"Hamad Int'l",   lat:25.2731, lng:51.6081    },
        { iata:"IST", name:"Istanbul",      lat:41.2753, lng:28.7519    },
        { iata:"SVO", name:"Sheremetyevo",  lat:55.9726, lng:37.4146    },
        { iata:"ATL", name:"Hartsfield-Jackson", lat:33.6407, lng:-84.4277 },
        { iata:"ORD", name:"O'Hare",        lat:41.9742, lng:-87.9073   },
        { iata:"DFW", name:"Dallas/Fort Worth", lat:32.8998, lng:-97.0403 },
      ];

      const airportGroup = g.append("g");
      const withPos = AIRPORTS.map(a => {
        const pt = projection([a.lng, a.lat]);
        return { ...a, x: pt?.[0], y: pt?.[1] };
      }).filter(a => Number.isFinite(a.x) && Number.isFinite(a.y));

      const markers = airportGroup.selectAll("g").data(withPos).join("g")
        .attr("transform", d => `translate(${d.x},${d.y})`);
      markers.append("circle").attr("r", 5.5).attr("fill", "rgba(0,0,0,0.15)");
      markers.append("circle").attr("r", 3.5).attr("fill", "#ff7f50")
        .attr("stroke", "#111").attr("stroke-width", 1.5)
        .style("cursor", "pointer")
        .on("mouseover", (event, d) => {
          tooltip.style("opacity", "1")
            .style("left", `${event.pageX + 10}px`)
            .style("top", `${event.pageY - 28}px`)
            .html(`<strong>${d.iata}</strong><br/>${d.name}`);
        })
        .on("mousemove", event => {
          tooltip.style("left", `${event.pageX + 10}px`).style("top", `${event.pageY - 28}px`);
        })
        .on("mouseout", () => tooltip.style("opacity", "0"));
    }
  }, [filteredData, countries, metric, renderWidth, renderHeight, margin, focusCountry, lookupCountry, onCountryClick]);

  if (isLoading) return (
    <div className="flex items-center justify-center h-64 text-slate-400 text-sm">
      <span className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mr-2" />
      세계 지도 로딩 중...
    </div>
  );
  if (error) return (
    <div className="flex items-center justify-center h-40 text-red-400 text-sm">{error}</div>
  );

  return (
    <div ref={containerRef} className="relative w-full">
      <svg ref={svgRef} className="w-full" />
      <div
        ref={tooltipRef}
        className="fixed z-50 pointer-events-none bg-white border border-slate-200 text-slate-800 text-xs px-3 py-2 rounded-lg shadow-lg"
        style={{ opacity: 0, transition: "opacity 0.1s" }}
      />
    </div>
  );
}
