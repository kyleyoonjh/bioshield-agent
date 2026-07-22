import { useEffect, useRef, useMemo, useState } from "react";
import * as d3 from "d3";
import { fmtCompact } from "../../services/covidService";

const GEOJSON_URL =
  "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo_simple.json";

const ALIASES: Record<string, string> = {
  서울특별시: "서울", Seoul: "서울",
  부산광역시: "부산", Busan: "부산",
  대구광역시: "대구", Daegu: "대구",
  인천광역시: "인천", Incheon: "인천",
  광주광역시: "광주", Gwangju: "광주",
  대전광역시: "대전", Daejeon: "대전",
  울산광역시: "울산", Ulsan: "울산",
  세종특별자치시: "세종", Sejong: "세종",
  경기도: "경기", Gyeonggi: "경기",
  강원도: "강원", Gangwon: "강원",
  충청북도: "충북", Chungbuk: "충북",
  충청남도: "충남", Chungnam: "충남",
  전라북도: "전북", Jeonbuk: "전북",
  전라남도: "전남", Jeonnam: "전남",
  경상북도: "경북", Gyeongbuk: "경북",
  경상남도: "경남", Gyeongnam: "경남",
  제주특별자치도: "제주", Jeju: "제주",
};

const LABEL_OFFSETS: Record<string, [number, number]> = {
  서울: [0, -8], 인천: [-10, 4], 세종: [10, -4], 대전: [8, 8],
  광주: [-6, 8], 대구: [8, 8], 울산: [12, 8], 부산: [12, 10], 제주: [0, -8],
};

const normalize = (name: string) => {
  const raw = String(name || "").trim();
  return ALIASES[raw] ?? raw.replace(/(특별시|광역시|특별자치시|특별자치도|도)$/g, "");
};

export type KoreaMetric = "cases" | "deaths" | "newCases";

interface RegionRow { region: string; confirmed: number; newConfirmed: number; deaths: number }
interface Props { data: RegionRow[]; metric?: KoreaMetric }

export default function CovidKoreaMap({ data, metric = "cases" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [geoData, setGeoData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [containerWidth, setContainerWidth] = useState(640);

  useEffect(() => {
    fetch(GEOJSON_URL).then(r => r.json()).then(setGeoData).catch(() => null);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => setContainerWidth(Math.max(320, el.getBoundingClientRect().width));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const regionalMap = useMemo(() => {
    const m = new Map<string, RegionRow>();
    data.forEach(row => m.set(normalize(row.region), row));
    return m;
  }, [data]);

  useEffect(() => {
    if (!geoData || !svgRef.current) return;
    const W = Math.max(320, Math.min(640, containerWidth));
    const H = Math.round((560 / 640) * W);

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("height", "auto");

    const projection = d3.geoMercator().fitExtent([[20, 24], [W - 20, H - 20]], geoData as d3.GeoPermissibleObjects);
    const path = d3.geoPath(projection);

    const getValue = (row: RegionRow) =>
      metric === "deaths" ? row.deaths : metric === "newCases" ? row.newConfirmed : row.confirmed;

    const vals = Array.from(regionalMap.values()).map(getValue).filter(v => v > 0);
    const color = d3.scaleSequential(d3.interpolateReds).domain([0, d3.max(vals) ?? 1]);

    const tooltip = d3.select(tooltipRef.current!);
    const metricLabel = metric === "deaths" ? "Deaths" : metric === "newCases" ? "New" : "Confirmed";

    svg.append("g")
      .selectAll("path")
      .data(geoData.features)
      .join("path")
      .attr("d", path as never)
      .attr("fill", d => {
        const name = normalize((d as GeoJSON.Feature).properties?.name ?? "");
        const row = regionalMap.get(name);
        return row ? color(getValue(row)) : "#e5e7eb";
      })
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 1.2)
      .on("mousemove", (event, d) => {
        const name = normalize((d as GeoJSON.Feature).properties?.name ?? "");
        const row = regionalMap.get(name);
        tooltip.style("opacity", "1")
          .style("left", `${event.pageX + 12}px`)
          .style("top", `${event.pageY - 32}px`)
          .html(row
            ? `<strong>${name}</strong><br/>${metricLabel}: ${fmtCompact(getValue(row))}<br/>Deaths: ${fmtCompact(row.deaths)}`
            : `<strong>${name}</strong><br/>No data`);
      })
      .on("mouseleave", () => tooltip.style("opacity", "0"));

    svg.append("g")
      .selectAll("text")
      .data(geoData.features)
      .join("text")
      .attr("transform", d => {
        const name = normalize((d as GeoJSON.Feature).properties?.name ?? "");
        const [x, y] = path.centroid(d as GeoJSON.Feature);
        const [dx, dy] = LABEL_OFFSETS[name] ?? [0, 0];
        return `translate(${x + dx},${y + dy})`;
      })
      .attr("text-anchor", "middle")
      .attr("pointer-events", "none")
      .style("paint-order", "stroke")
      .style("stroke", "#fff")
      .style("stroke-width", "2.6px")
      .style("font-size", W < 500 ? "8px" : "10px")
      .style("font-weight", "700")
      .style("fill", "#1f2937")
      .each(function (d) {
        const name = normalize((d as GeoJSON.Feature).properties?.name ?? "");
        const row = regionalMap.get(name);
        if (!row) return;
        const el = d3.select(this);
        el.append("tspan").attr("x", 0).attr("dy", 0).text(name);
        el.append("tspan").attr("x", 0).attr("dy", W < 500 ? 9 : 11)
          .style("font-size", W < 500 ? "7px" : "9px").style("fill", "#7f1d1d")
          .text(fmtCompact(Number(row.confirmed)));
      });
  }, [geoData, regionalMap, metric, containerWidth]);

  if (!geoData) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400 text-sm">
        한국 지도 로딩 중...
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full">
      <svg ref={svgRef} />
      <div
        ref={tooltipRef}
        className="fixed z-50 pointer-events-none bg-slate-800 text-white text-xs px-3 py-2 rounded-lg shadow-lg opacity-0 transition-opacity"
        style={{ opacity: 0 }}
      />
    </div>
  );
}
