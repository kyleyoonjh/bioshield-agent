export default function DemoBanner() {
  return (
    <div
      style={{
        position: "fixed",
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        background: "linear-gradient(90deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%)",
        borderTop: "2px solid #2563eb",
        padding: "6px 20px",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        fontSize: "12px",
        color: "#94a3b8",
      }}
    >
      <span
        style={{
          background: "#2563eb",
          color: "#fff",
          fontWeight: 700,
          fontSize: "11px",
          padding: "2px 8px",
          borderRadius: "3px",
          letterSpacing: "0.05em",
          flexShrink: 0,
        }}
      >
        DEMO MODE
      </span>
      <span>
        모든 데이터는 사전 로드된 목업 데이터입니다. NCBI GenBank · disease.sh · OpenAI · Supabase 외부 API 호출 없음.
      </span>
      <span style={{ marginLeft: "auto", color: "#475569", flexShrink: 0 }}>
        Phase 3 (Assay Design) 은 실제 API 사용
      </span>
    </div>
  );
}
