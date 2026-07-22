// ─── Domain types ─────────────────────────────────────────────────────────────

export type DiseaseType = "SARS-CoV-2" | "HPV" | "STI";
export type GuidelineType = "EP05" | "EP09";

export interface VariantInfo {
  id: string;
  label: string;
  lineage: string;
  who_label: string;
}

export interface Mutation {
  gene: string;
  position: number;
  ref: string;
  alt: string;
  effect: string;
}

// ─── Upload / Schema ──────────────────────────────────────────────────────────

export interface SchemaMapping {
  target_column: string;
  group_columns: string[];
}

export interface SchemaMappingEP09 {
  reference_column: string;
  test_column: string;
}

export interface UploadMetadata {
  filename: string;
  column_count: number;
  row_preview_count: number;
  columns: Array<{
    name: string;
    dtype: string;
    sample_values: string[];
  }>;
}

// ─── EP05 Result ──────────────────────────────────────────────────────────────

export interface EP05Result {
  anova: {
    f_value: number;
    p_value: number;
  };
  repeatability: {
    sd: number;
    cv_percent: number;
  };
  reproducibility: {
    sd: number;
    cv_percent: number;
  };
  variance_components: {
    within_group: number;
    between_group: number;
    within_group_percent: number;
    between_group_percent: number;
  };
  grand_mean: number;
  sample_count: number;
  groups: Array<{
    group: string;
    mean: number;
    sd: number;
    n: number;
    cv_percent: number;
  }>;
  target_column: string;
  group_columns: string[];
}

// Alias for legacy code
export type StatsResult = EP05Result;

// ─── EP09 Result ──────────────────────────────────────────────────────────────

export interface EP09Result {
  r_squared: number;
  pearson_r: number;
  deming: {
    slope: number;
    intercept: number;
  };
  bland_altman: {
    mean_diff: number;
    sd_diff: number;
    loa_upper: number;
    loa_lower: number;
  };
  scatter_data: Array<{ ref: number; test: number }>;
  bland_altman_data: Array<{ avg: number; diff: number }>;
  sample_count: number;
  reference_column: string;
  test_column: string;
}

// ─── Bio Context (NCBI) ───────────────────────────────────────────────────────

export interface BioAnnotation {
  name: string;
  start: number;
  end: number;
  color: string;
  strand: number;
}

export interface BioContextResponse {
  disease_type: DiseaseType;
  accession: string;
  rdrp_sequence: string;
  seq_start?: number;          // absolute genomic position of rdrp_sequence[0]
  annotations: BioAnnotation[];
  primer_structure: {
    sequence: string;
    length: number;
    abs_start?: number;        // absolute genomic start of forward primer
    tm_celsius: number | null;
    gc_percent: number;
    dot_bracket: string | null;
    mfe: number | null;
  };
  assay_info: {
    target_gene: string;
    organism: string;
    assay_type: string;
    standard: string;
    accession: string;
  };
  source: "ncbi" | "cache" | "fallback" | "demo";
}

// Legacy alias
export type BioContext = BioContextResponse;

// ─── Phase 2 ──────────────────────────────────────────────────────────────────

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW";
export type FeedbackValue = "accurate" | "partially_accurate" | "incorrect";

export interface CollectPayload {
  disease_type: DiseaseType;
  variant_name: string;
  mismatch_count?: number;
  three_prime_mismatch?: boolean;
  grand_mean?: number;
  repeatability_cv?: number;
  reproducibility_cv?: number;
  anova_f_value?: number;
  anova_p_value?: number;
  sample_count?: number;
  deming_slope?: number;
  bland_altman_mean_diff?: number;
  pearson_r?: number;
  instrument?: string;
  assay_type?: string;
  test_date?: string;
  guideline?: string;
  source_filename?: string;
}

export interface CollectResponse {
  id: string;
  risk_level: RiskLevel;
  message: string;
}

export interface DatasetStats {
  total_records: number;
  fine_tuning_threshold: number;
  readiness_percent: number;
  is_ready: boolean;
  by_disease: Record<string, number>;
  by_risk: Record<RiskLevel, number>;
  by_guideline: Record<string, number>;
  recent_30d: number;
}

