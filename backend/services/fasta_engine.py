"""
FASTA parsing — Drug Discovery Assistant, "Track A" file-upload gateway.
Standalone, no shared code with primer-design (mirrors vcf_annotation_engine.py's
isolation convention for this feature).

Pure text parsing, no network calls, no interpretation — extracting a
sequence from FASTA is deterministic. What that sequence resolves to
structurally (real AlphaFold DB lookup if it matches a known UniProt entry,
real ESMFold prediction otherwise, honestly failing if it exceeds ESMFold's
real length limit) is handled downstream by the existing
services/protein_structure_engine.py — this module never predicts or
fabricates structure/identity itself.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^>(\S+)(?:\s+(.*))?$")
_VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZJUO*]+$", re.IGNORECASE)


def parse_fasta(fasta_text: str) -> list[dict]:
    """
    Parses one or more FASTA records into
    [{"id": str, "description": str, "sequence": str}, ...].
    Whitespace within sequence lines is stripped; sequence is uppercased.
    Records with no header (text before the first '>') are ignored.
    """
    records: list[dict] = []
    current_id: str | None = None
    current_desc: str = ""
    current_seq: list[str] = []

    def _flush() -> None:
        if current_id is not None:
            records.append({
                "id": current_id,
                "description": current_desc,
                "sequence": "".join(current_seq).upper(),
            })

    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            match = _HEADER_RE.match(line)
            if match:
                current_id, current_desc = match.group(1), match.group(2) or ""
            else:
                current_id, current_desc = line[1:].strip(), ""
            current_seq = []
        elif current_id is not None:
            current_seq.append("".join(line.split()))

    _flush()
    return records


def first_sequence(fasta_text: str) -> str | None:
    """Convenience accessor for the common single-record-upload case."""
    records = parse_fasta(fasta_text)
    return records[0]["sequence"] if records else None


def is_valid_amino_acid_sequence(sequence: str) -> bool:
    """Real IUPAC amino-acid alphabet check (including ambiguity codes
    B/Z/J/X/U/O) — a coarse sanity check before handing the sequence to
    ESMFold, not a guarantee the sequence is biologically meaningful."""
    return bool(sequence) and bool(_VALID_AA_RE.match(sequence))
