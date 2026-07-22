FROM python:3.12-slim

WORKDIR /app

# ca-certificates for HTTPS trust (pip, the mhcflurry model download, and live
# external API calls). ncbi-blast+/curl removed: they only served the hg38
# off-target BLAST screening (off_target_filter_service.py) + genome fetch,
# which are no longer built into the image and aren't part of the deployed MCP
# tools. Re-add "ncbi-blast+ curl" here if you restore scripts/download_refs.sh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Real pretrained MHCflurry models (services/neoantigen_engine.py) — a
# one-time data fetch, not part of the pip package itself.
RUN mhcflurry-downloads fetch models_class1_presentation

COPY backend/ .

# Sample VCF/BAM demo files (used by the "mRNA 암 백신 데모" button's
# vcf_path/bam_path convenience — see api/drug_discovery_router.py's
# _resolve_sample_path()) live at backend/sample/, so `COPY backend/ .`
# above already includes them at /app/sample — no separate COPY needed.
# (Real reported bug this avoids: they used to be a repo-root sibling of
# backend/, which a backend/-only build context — e.g. backend/Dockerfile —
# can never see; moved inside backend/ so every build path just works.)

# NOTE: the hg38 BLAST-db build (scripts/download_refs.sh) is intentionally
# NOT run at image-build time. It downloaded ~1GB from UCSC
# (hgdownload.soe.ucsc.edu) and frequently timed out (curl exit 28), failing
# the whole image build; it also added ~3GB to the image. It's only used for
# off-target primer screening / somatic-oligo design, which are NOT part of
# the deployed MCP tools. To restore it, re-add:
#   RUN chmod +x scripts/download_refs.sh && ./scripts/download_refs.sh
ENV BLASTN_PATH=blastn
ENV BLAST_DB_PATH=refs/human_genome
# mhcgnomes (an mhcflurry dependency) opens its bundled YAML data files
# without an explicit encoding=, so it inherits the container locale's
# default codec — confirmed to break under a non-UTF8 locale on Windows;
# harmless to set explicitly here too rather than depend on the base
# image's locale.
ENV PYTHONUTF8=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
