"""
MCP Tool Registry — AiRemedy-Agent.

Wraps the native Bio Computation Engine services as named, callable MCP tools.
Each tool is a thin adapter; all scientific computation stays in the original services.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_registry: dict[str, Callable] = {}
_metadata: dict[str, dict] = {}


def register(name: str, description: str = "", input_schema: dict | None = None):
    """Decorator: register a callable as a named MCP tool."""
    def decorator(fn: Callable) -> Callable:
        _registry[name] = fn
        _metadata[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema or {"type": "object", "properties": {}},
        }
        logger.debug("[tool_registry] registered: %s", name)
        return fn
    return decorator


def call(name: str, **kwargs: Any) -> Any:
    """Invoke a registered tool by name with keyword arguments."""
    fn = _registry.get(name)
    if not fn:
        raise KeyError(f"Tool not found: {name!r}. Available: {list(_registry)}")
    logger.info("[tool_registry] invoke tool=%s", name)
    return fn(**kwargs)


def list_tools() -> list[dict]:
    """Return MCP-compatible tool definitions for all registered tools."""
    return list(_metadata.values())


# ── Drug Discovery + mRNA-vaccine engine tools ─────────────────────
# Thin @register adapters dispatched through router_core.route(); all
# scientific computation stays in the underlying services.

@register(
    "fetch_protein_structure",
    description="Look up a pre-computed AlphaFold structure for a known UniProt ID from the EBI AlphaFold DB.",
    input_schema={
        "type": "object",
        "properties": {"uniprot_id": {"type": "string"}},
        "required": ["uniprot_id"],
    },
)
def fetch_protein_structure(uniprot_id: str) -> dict:
    from services.protein_structure_engine import fetch_known_structure
    return fetch_known_structure(uniprot_id)


@register(
    "predict_protein_structure",
    description="Fold an arbitrary protein sequence (<=400aa) via the public ESM Atlas (ESMFold) API.",
    input_schema={
        "type": "object",
        "properties": {"sequence": {"type": "string"}},
        "required": ["sequence"],
    },
)
def predict_protein_structure(sequence: str) -> dict:
    from services.protein_structure_engine import predict_structure_esmfold
    return predict_structure_esmfold(sequence)


@register(
    "analyze_ligand",
    description="Parse a SMILES string and compute RDKit drug-likeness descriptors (MW, LogP, TPSA, Lipinski).",
    input_schema={
        "type": "object",
        "properties": {"smiles": {"type": "string"}},
        "required": ["smiles"],
    },
)
def analyze_ligand(smiles: str) -> dict:
    from services.docking_engine import analyze_ligand as _analyze_ligand
    return _analyze_ligand(smiles)


@register(
    "dock_ligand",
    description="Dock a ligand against a prepared receptor via AutoDock Vina, or a heuristic proxy if Vina is unavailable.",
    input_schema={
        "type": "object",
        "properties": {
            "receptor_pdbqt_path": {"type": "string"},
            "ligand_smiles":       {"type": "string"},
            "center":              {"type": "array"},
            "box_size":            {"type": "array"},
            "exhaustiveness":      {"type": "integer"},
            "keep_pose":           {"type": "boolean"},
        },
        "required": ["receptor_pdbqt_path", "ligand_smiles", "center", "box_size"],
    },
)
def dock_ligand(receptor_pdbqt_path: str, ligand_smiles: str, center, box_size, exhaustiveness: int = 8,
                 keep_pose: bool = False) -> dict:
    from services.docking_engine import dock_ligand as _dock_ligand
    return _dock_ligand(receptor_pdbqt_path, ligand_smiles, tuple(center), tuple(box_size), exhaustiveness,
                         keep_pose=keep_pose)


@register(
    "prepare_receptor",
    description="Prepare a real PDBQT receptor + blind-docking search box from arbitrary resolved target PDB text.",
    input_schema={
        "type": "object",
        "properties": {
            "pdb_text": {"type": "string"},
            "padding":  {"type": "number"},
        },
        "required": ["pdb_text"],
    },
)
def prepare_receptor(pdb_text: str, padding: float = 8.0) -> dict:
    from services.receptor_prep_engine import prepare_receptor_from_pdb
    return prepare_receptor_from_pdb(pdb_text, padding)


@register(
    "screen_candidates",
    description="Dock every candidate in a drug library against one prepared receptor.",
    input_schema={
        "type": "object",
        "properties": {
            "receptor_pdbqt_path": {"type": "string"},
            "center":              {"type": "array"},
            "box_size":            {"type": "array"},
            "candidates":          {"type": "array"},
            "exhaustiveness":      {"type": "integer"},
        },
        "required": ["receptor_pdbqt_path", "center", "box_size", "candidates"],
    },
)
def screen_candidates(receptor_pdbqt_path: str, center, box_size, candidates: list, exhaustiveness: int = 8) -> list:
    from services.drug_screening_engine import screen_candidates as _screen_candidates
    return _screen_candidates(receptor_pdbqt_path, tuple(center), tuple(box_size), candidates, exhaustiveness)


@register(
    "filter_drug_candidates",
    description="Drop screened candidates that failed to dock or are not drug-like (deterministic quality gate).",
    input_schema={
        "type": "object",
        "properties": {"results": {"type": "array"}},
        "required": ["results"],
    },
)
def filter_drug_candidates(results: list) -> list:
    from services.drug_screening_engine import filter_candidates
    return filter_candidates(results)


@register(
    "rank_drug_candidates",
    description="Rank filtered drug candidates by weighted binding-affinity + drug-likeness score.",
    input_schema={
        "type": "object",
        "properties": {"screened": {"type": "array"}},
        "required": ["screened"],
    },
)
def rank_drug_candidates(screened: list) -> list:
    from services.drug_ranking_engine import rank_candidates
    return rank_candidates(screened)


@register(
    "parse_vcf",
    description="Parse VCF 4.x text into structured variant records (CHROM/POS/REF/ALT/INFO/samples).",
    input_schema={
        "type": "object",
        "properties": {"vcf_text": {"type": "string"}},
        "required": ["vcf_text"],
    },
)
def parse_vcf(vcf_text: str) -> list:
    from services.vcf_annotation_engine import parse_vcf as _parse_vcf
    return _parse_vcf(vcf_text)


@register(
    "annotate_variant_consequence",
    description="Real-time amino-acid consequence annotation for one variant via Ensembl VEP (GRCh38) — "
                "authoritative over any gene label the VCF itself carries.",
    input_schema={
        "type": "object",
        "properties": {
            "chrom": {"type": "string"},
            "pos":   {"type": "integer"},
            "ref":   {"type": "string"},
            "alt":   {"type": "string"},
        },
        "required": ["chrom", "pos", "ref", "alt"],
    },
)
def annotate_variant_consequence(chrom: str, pos: int, ref: str, alt: str) -> dict:
    from services.vcf_annotation_engine import annotate_variant_consequence as _annotate
    return _annotate(chrom, pos, ref, alt)


@register(
    "parse_fasta",
    description="Parse FASTA text into structured records (id/description/sequence).",
    input_schema={
        "type": "object",
        "properties": {"fasta_text": {"type": "string"}},
        "required": ["fasta_text"],
    },
)
def parse_fasta(fasta_text: str) -> list:
    from services.fasta_engine import parse_fasta as _parse_fasta
    return _parse_fasta(fasta_text)


@register(
    "search_literature",
    description="Real live PubMed search (esearch+efetch) for a target/drug/disease query — returns real "
                "papers (title/abstract/journal/year/PMID) only, never a fabricated summary.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
)
def search_literature(query: str, max_results: int = 5) -> dict:
    from services.literature_engine import search_pubmed
    return search_pubmed(query, max_results)


@register(
    "search_clinical_trials",
    description="Real live ClinicalTrials.gov v2 search for a target/drug/disease query — returns real trials "
                "(NCT ID/title/status/phase/conditions/interventions) only, never a fabricated landscape.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
)
def search_clinical_trials(query: str, max_results: int = 5) -> dict:
    from services.clinical_trials_engine import search_clinical_trials as _search
    return _search(query, max_results)


@register(
    "search_similar_compounds",
    description="Real live PubChem 2D similarity search from a reference SMILES — returns real compounds "
                "(CID/SMILES/IUPAC name/properties) only.",
    input_schema={
        "type": "object",
        "properties": {
            "smiles": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["smiles"],
    },
)
def search_similar_compounds(smiles: str, max_results: int = 10) -> dict:
    from services.compound_discovery_engine import search_similar_compounds_pubchem
    return search_similar_compounds_pubchem(smiles, max_results)


@register(
    "search_known_inhibitors",
    description="Real live ChEMBL search for measured IC50 inhibitor data against a UniProt target — "
                "returns real bioactivity records only, never an estimated potency.",
    input_schema={
        "type": "object",
        "properties": {
            "uniprot_id": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["uniprot_id"],
    },
)
def search_known_inhibitors(uniprot_id: str, max_results: int = 10) -> dict:
    from services.compound_discovery_engine import search_known_inhibitors_chembl
    return search_known_inhibitors_chembl(uniprot_id, max_results)


@register(
    "predict_admet_profile",
    description="Real-computable ADMET subset for a ligand: Veber's rule (oral absorption), a coarse "
                "hepatotoxicity structural-alert screen, RDKit's official PAINS filter catalog (assay "
                "interference), and synthetic accessibility (SA score). hERG/CYP/BBB/toxicity ML "
                "predictions are NOT included — no validated model/API for those exists here.",
    input_schema={
        "type": "object",
        "properties": {"smiles": {"type": "string"}},
        "required": ["smiles"],
    },
)
def predict_admet_profile(smiles: str) -> dict:
    from services.admet_engine import predict_admet_profile as _predict
    return _predict(smiles)


@register(
    "get_target_disease_associations",
    description="Real live UniProt DISEASE/FUNCTION comments for a target — returns real curated "
                "disease links (name/description/MIM ID) and function text only, never a guessed association.",
    input_schema={
        "type": "object",
        "properties": {"uniprot_id": {"type": "string"}},
        "required": ["uniprot_id"],
    },
)
def get_target_disease_associations(uniprot_id: str) -> dict:
    from services.target_intelligence_engine import get_target_disease_associations as _get
    return _get(uniprot_id)


@register(
    "get_target_pathways",
    description="Real live Reactome pathway mapping for a target UniProt ID — returns real pathway "
                "names/stable IDs only, empty list if the target has no indexed pathways.",
    input_schema={
        "type": "object",
        "properties": {
            "uniprot_id": {"type": "string"},
            "species_taxon": {"type": "integer"},
        },
        "required": ["uniprot_id"],
    },
)
def get_target_pathways(uniprot_id: str, species_taxon: int = 9606) -> dict:
    from services.target_intelligence_engine import get_target_pathways as _get
    return _get(uniprot_id, species_taxon)


@register(
    "get_opentargets_profile",
    description="Real live OpenTargets Platform lookup for a UniProt target — real disease association "
                "scores (0-1, genetics/literature/expression composite) and real small-molecule "
                "tractability flags. Empty result for non-human-gene targets (e.g. viral proteins).",
    input_schema={
        "type": "object",
        "properties": {"uniprot_id": {"type": "string"}},
        "required": ["uniprot_id"],
    },
)
def get_opentargets_profile(uniprot_id: str) -> dict:
    from services.opentargets_engine import get_opentargets_profile as _get
    return _get(uniprot_id)


@register(
    "run_sar_optimization",
    description="Real bioisosteric analog generation (RDKit reaction SMARTS) for a completed job's top "
                "candidate, each analog RE-DOCKED for a real recomputed affinity comparison — never a "
                "predicted 'expected effect'.",
    input_schema={
        "type": "object",
        "properties": {"job_result": {"type": "object"}},
        "required": ["job_result"],
    },
)
def run_sar_optimization(job_result: dict) -> dict:
    from services.sar_optimization_service import run_sar_optimization as _run
    return _run(job_result)


@register(
    "generate_decision_report",
    description="Aggregates a completed job's real docking/ADMET results plus real target literature/"
                "clinical-trial evidence into a transparent priority score (disclosed formula) and an "
                "LLM narrative strictly grounded in that real data.",
    input_schema={
        "type": "object",
        "properties": {
            "job_result": {"type": "object"},
            "target_name": {"type": "string"},
        },
        "required": ["job_result"],
    },
)
def generate_decision_report_tool(job_result: dict, target_name: str = "") -> dict:
    from services.decision_agent import get_top_candidate_scored, calculate_priority_score, generate_decision_report
    from services.literature_engine import search_pubmed
    from services.clinical_trials_engine import search_clinical_trials

    candidate = get_top_candidate_scored(job_result)
    if not candidate:
        return {"available": False, "reason": "No successfully docked candidate to evaluate"}
    papers, trials = [], []
    if target_name:
        papers = search_pubmed(target_name, max_results=3).get("papers") or []
        trials = search_clinical_trials(target_name, max_results=3).get("trials") or []
    scoring = calculate_priority_score(candidate, bool(papers), bool(trials))
    return {"available": True, **generate_decision_report(candidate, scoring, papers, trials)}
