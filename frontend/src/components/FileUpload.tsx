import { useCallback, useState } from "react";
import type { SchemaMapping, UploadMetadata } from "../types";

interface FileUploadProps {
  onSchemaConfirmed: (file: File, schema: SchemaMapping, metadata: UploadMetadata) => void;
}

export default function FileUpload({ onSchemaConfirmed }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<UploadMetadata | null>(null);
  const [schema, setSchema] = useState<SchemaMapping | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(async (selected: File) => {
    setFile(selected);
    setError(null);
    setLoading(true);
    setMetadata(null);
    setSchema(null);

    const formData = new FormData();
    formData.append("file", selected);

    try {
      const res = await fetch("/api/v1/upload", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      setMetadata(data.metadata);
      setSchema(data.schema_mapping);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFile(dropped);
    },
    [handleFile]
  );

  const updateTargetColumn = (value: string) => {
    if (schema) setSchema({ ...schema, target_column: value });
  };

  const updateGroupColumn = (index: number, value: string) => {
    if (!schema) return;
    const cols = [...schema.group_columns];
    cols[index] = value;
    setSchema({ ...schema, group_columns: cols });
  };

  const addGroupColumn = () => {
    if (schema) setSchema({ ...schema, group_columns: [...schema.group_columns, ""] });
  };

  const removeGroupColumn = (index: number) => {
    if (!schema) return;
    setSchema({
      ...schema,
      group_columns: schema.group_columns.filter((_, i) => i !== index),
    });
  };

  const handleConfirm = () => {
    if (file && schema && metadata) {
      onSchemaConfirmed(file, schema, metadata);
    }
  };

  const columnOptions = metadata?.columns.map((c) => c.name) ?? [];

  return (
    <div className="space-y-6">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
          dragOver ? "border-bio-500 bg-bio-50" : "border-gray-300 bg-white"
        }`}
      >
        <div className="text-4xl mb-4">📊</div>
        <p className="text-lg font-medium text-gray-700">CLSI EP05-A3 엑셀 파일 업로드</p>
        <p className="text-sm text-gray-500 mt-2">.xlsx, .xls, .csv 지원 · 드래그앤드롭 또는 클릭</p>
        <input
          type="file"
          accept=".xlsx,.xls,.csv"
          className="hidden"
          id="file-input"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        <label
          htmlFor="file-input"
          className="mt-4 inline-block px-6 py-2 bg-bio-600 text-white rounded-lg cursor-pointer hover:bg-bio-700 transition"
        >
          파일 선택
        </label>
        {file && <p className="mt-3 text-sm text-gray-600">선택됨: {file.name}</p>}
      </div>

      {loading && (
        <div className="text-center py-8">
          <div className="inline-block w-8 h-8 border-4 border-bio-500 border-t-transparent rounded-full animate-spin" />
          <p className="mt-2 text-gray-600">OpenAI가 컬럼 스키마를 분석 중...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">{error}</div>
      )}

      {schema && metadata && (
        <div className="bg-white rounded-xl shadow-sm border p-6 space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">AI 스키마 매핑 결과</h3>
          <p className="text-sm text-gray-500">
            {metadata.column_count}개 컬럼 · 상위 {metadata.row_preview_count}행 분석됨
          </p>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">측정값 컬럼 (Target)</label>
            <select
              value={schema.target_column}
              onChange={(e) => updateTargetColumn(e.target.value)}
              className="w-full border rounded-lg px-3 py-2"
            >
              {columnOptions.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">그룹 컬럼</label>
            {schema.group_columns.map((col, i) => (
              <div key={i} className="flex gap-2 mb-2">
                <select
                  value={col}
                  onChange={(e) => updateGroupColumn(i, e.target.value)}
                  className="flex-1 border rounded-lg px-3 py-2"
                >
                  <option value="">-- 선택 --</option>
                  {columnOptions.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => removeGroupColumn(i)}
                  className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg"
                >
                  삭제
                </button>
              </div>
            ))}
            <button onClick={addGroupColumn} className="text-sm text-bio-600 hover:underline">
              + 그룹 컬럼 추가
            </button>
          </div>

          <button
            onClick={handleConfirm}
            className="w-full py-3 bg-bio-600 text-white rounded-lg font-medium hover:bg-bio-700 transition"
          >
            스키마 확인 · 분석 시작
          </button>
        </div>
      )}
    </div>
  );
}
