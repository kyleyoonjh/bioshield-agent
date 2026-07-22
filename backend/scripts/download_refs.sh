#!/bin/sh
# Downloads the full hg38 human genome (chr1-22, X, Y, M) from UCSC and
# builds a BLAST nucleotide db at $REFS_DIR/human_genome, used by
# off_target_filter_service.py for real off-target specificity screening.
# The raw FASTA is kept (not deleted) because services/target_extractor.py
# also needs direct random-access reads via pyfaidx (auto-builds
# human_genome.fa.fai on first use) — this adds ~3GB to the image.
#
# Runs once at Docker build time (see Dockerfile). ~1GB compressed download,
# ~3.1GB decompressed FASTA, several minutes to index with makeblastdb.
set -eu

REFS_DIR="${REFS_DIR:-refs}"
BASE_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes"
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY chrM"

mkdir -p "$REFS_DIR"
cd "$REFS_DIR"

: > human_genome.fa
for c in $CHROMS; do
    echo "[download_refs] fetching $c ..."
    curl -fsSL "$BASE_URL/$c.fa.gz" -o "$c.fa.gz"
    gunzip -c "$c.fa.gz" >> human_genome.fa
    rm -f "$c.fa.gz"
done

echo "[download_refs] building BLAST db ..."
makeblastdb -in human_genome.fa -dbtype nucl -out human_genome -title "hg38 full genome" -parse_seqids

# human_genome.fa is intentionally kept — target_extractor.py reads it
# directly via pyfaidx for flanking-sequence extraction.

echo "[download_refs] done — $(ls -la human_genome.* | wc -l) files present (BLAST db + raw FASTA)"
