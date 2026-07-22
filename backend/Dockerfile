# Standalone backend-only image — build context is this backend/ directory
# itself (e.g. Cloud Run deploying only this folder as its own service),
# unlike the repo-root Dockerfile which builds the whole monorepo. Same real
# runtime dependencies as that one, just with paths adjusted for this
# narrower build context; keep both in sync if either changes.
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Real pretrained MHCflurry models (services/neoantigen_engine.py) — a
# one-time data fetch, not part of the pip package itself.
RUN mhcflurry-downloads fetch models_class1_presentation

COPY . .

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

# Cloud Run injects $PORT at runtime; 8000 is only the local-run fallback.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
