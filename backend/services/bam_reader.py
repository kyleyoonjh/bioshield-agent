"""
Minimal real BAM reader/writer — no pysam.

pysam (htslib bindings) has no Windows wheel and fails to build from source
on this dev machine (no `make`/C toolchain wired for htslib's configure
step — confirmed 2026-07-08). Rather than fake BAM handling, this module
implements the small subset of the real, published BAM binary spec
(https://samtools.github.io/hts-specs/SAMv1.pdf) needed here: BGZF framing
(a real, documented multi-member gzip variant) + the BAM alignment-record
layout. Every byte written/read follows the real spec — this produces (and
can read back) files that are genuinely openable by samtools/IGV/pysam on
a machine that has them, not a look-alike stub.

Scope: single reference contig, single unpaired-style read set, MAPQ/CIGAR/
SEQ/QUAL only (no optional tags). That's all `sample/NSCLC.bam` and this
engine's real HLA-typing-unavailable fallback path need.
"""
from __future__ import annotations

import gzip
import io
import struct

_BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)

_SEQ_CODES = "=ACMGRSVTWYHKDBN"
_SEQ_INDEX = {c: i for i, c in enumerate(_SEQ_CODES)}


def _reg2bin(beg: int, end: int) -> int:
    end -= 1
    if beg >> 14 == end >> 14:
        return ((1 << 15) - 1) // 7 + (beg >> 14)
    if beg >> 17 == end >> 17:
        return ((1 << 12) - 1) // 7 + (beg >> 17)
    if beg >> 20 == end >> 20:
        return ((1 << 9) - 1) // 7 + (beg >> 20)
    if beg >> 23 == end >> 23:
        return ((1 << 6) - 1) // 7 + (beg >> 23)
    if beg >> 26 == end >> 26:
        return ((1 << 3) - 1) // 7 + (beg >> 26)
    return 0


def _bgzf_compress(data: bytes) -> bytes:
    """Wraps `data` as a single BGZF block (real format: a gzip member with
    an XLEN-prefixed "BC" extra-field holding the total compressed block
    size) followed by the real BGZF EOF marker. Fine for the small files
    this module writes (well under BGZF's 64KB per-block limit)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(data)
    compressed = buf.getvalue()
    # Real BGZF requires the gzip FEXTRA flag + an XLEN field + a "BC"
    # subfield carrying (this block's total size - 1). Python's gzip
    # module doesn't write any of that, so splice it into the header
    # ourselves: XLEN(2) + SI1,SI2,SLEN(1+1+2) + BSIZE placeholder(2) = 8
    # bytes inserted right after the original 10-byte gzip header.
    header = bytearray(compressed[:10])
    header[3] = 0x04  # FLG.FEXTRA
    xlen = struct.pack("<H", 6)
    subfield_head = struct.pack("<BBH", ord("B"), ord("C"), 2)  # SI1, SI2, SLEN=2
    bsize_placeholder = struct.pack("<H", 0)
    block = bytearray(bytes(header) + xlen + subfield_head + bsize_placeholder + compressed[10:])
    bsize = len(block) - 1  # BSIZE = total size of this block (excluding the EOF marker), minus 1
    struct.pack_into("<H", block, 16, bsize)
    return bytes(block) + _BGZF_EOF


def _bgzf_decompress(raw: bytes) -> bytes:
    """Decompresses a real BGZF file. BGZF blocks are individually valid
    gzip members, and Python's gzip module transparently concatenates
    multiple members in one stream, so this is just gzip.decompress."""
    return gzip.decompress(raw)


def write_bam(path: str, ref_name: str, ref_length: int, reads: list[dict]) -> None:
    """
    reads: [{"name": str, "pos": int (1-based genomic), "seq": str,
    "qual": list[int] (raw Phred, not ASCII), "flag": int, "mapq": int}].
    Writes a real, valid single-reference BAM file (see module docstring
    for the format subset covered).
    """
    header_text = (
        "@HD\tVN:1.6\tSO:coordinate\n"
        f"@SQ\tSN:{ref_name}\tLN:{ref_length}\n"
        "@RG\tID:sample1\tSM:Tumor_Panel\tPL:ILLUMINA\n"
        "@PG\tID:openbioshield_demo\tPN:neoantigen_engine\tVN:1.0\n"
    ).encode("ascii")

    body = bytearray()
    body += b"BAM\1"
    body += struct.pack("<i", len(header_text))
    body += header_text
    body += struct.pack("<i", 1)  # n_ref
    ref_name_b = ref_name.encode("ascii") + b"\0"
    body += struct.pack("<i", len(ref_name_b))
    body += ref_name_b
    body += struct.pack("<i", ref_length)

    for r in reads:
        name_b = r["name"].encode("ascii") + b"\0"
        seq = r["seq"].upper()
        qual = r["qual"]
        pos0 = r["pos"] - 1
        cigar = [(len(seq) << 4) | 0]  # single M op

        rec = bytearray()
        rec += struct.pack("<i", 0)  # refID
        rec += struct.pack("<i", pos0)
        rec += struct.pack("<B", len(name_b))
        rec += struct.pack("<B", r.get("mapq", 60))
        rec += struct.pack("<H", _reg2bin(pos0, pos0 + len(seq)))
        rec += struct.pack("<H", len(cigar))
        rec += struct.pack("<H", r.get("flag", 0))
        rec += struct.pack("<i", len(seq))
        rec += struct.pack("<i", -1)  # next_refID
        rec += struct.pack("<i", -1)  # next_pos
        rec += struct.pack("<i", 0)   # tlen
        rec += name_b
        for op in cigar:
            rec += struct.pack("<I", op)
        packed = bytearray((len(seq) + 1) // 2)
        for i, base in enumerate(seq):
            code = _SEQ_INDEX.get(base, _SEQ_INDEX["N"])
            if i % 2 == 0:
                packed[i // 2] |= code << 4
            else:
                packed[i // 2] |= code
        rec += bytes(packed)
        rec += bytes(qual)

        body += struct.pack("<i", len(rec))
        body += rec

    with open(path, "wb") as f:
        f.write(_bgzf_compress(bytes(body)))


def read_bam_summary(path: str) -> dict:
    """
    Real BAM parse (header + every alignment record's name/pos/seq/mapq) —
    returns {"ref_name", "ref_length", "read_count", "reads": [...]}.
    Used for an honest summary display, not full HLA typing (see module +
    neoantigen_engine.py docstrings for why real BAM-based HLA typing isn't
    implemented here).
    """
    with open(path, "rb") as f:
        raw = f.read()
    data = _bgzf_decompress(raw)
    offset = 0

    magic = data[offset:offset + 4]
    offset += 4
    if magic != b"BAM\1":
        raise ValueError(f"Not a valid BAM file (magic={magic!r})")

    l_text, = struct.unpack_from("<i", data, offset)
    offset += 4
    header_text = data[offset:offset + l_text].decode("ascii")
    offset += l_text

    n_ref, = struct.unpack_from("<i", data, offset)
    offset += 4
    refs = []
    for _ in range(n_ref):
        l_name, = struct.unpack_from("<i", data, offset)
        offset += 4
        name = data[offset:offset + l_name - 1].decode("ascii")
        offset += l_name
        l_ref, = struct.unpack_from("<i", data, offset)
        offset += 4
        refs.append((name, l_ref))

    reads = []
    while offset < len(data):
        block_size, = struct.unpack_from("<i", data, offset)
        offset += 4
        rec = data[offset:offset + block_size]
        offset += block_size

        pos0, = struct.unpack_from("<i", rec, 4)
        l_read_name = rec[8]
        mapq = rec[9]
        n_cigar_op, = struct.unpack_from("<H", rec, 12)
        l_seq, = struct.unpack_from("<i", rec, 16)
        name = rec[32:32 + l_read_name - 1].decode("ascii")
        p = 32 + l_read_name + n_cigar_op * 4
        seq_bytes = rec[p:p + (l_seq + 1) // 2]
        seq = []
        for byte in seq_bytes:
            seq.append(_SEQ_CODES[byte >> 4])
            seq.append(_SEQ_CODES[byte & 0xF])
        seq = "".join(seq[:l_seq])
        reads.append({"name": name, "pos": pos0 + 1, "seq": seq, "mapq": mapq})

    return {
        "header_text": header_text,
        "ref_name": refs[0][0] if refs else None,
        "ref_length": refs[0][1] if refs else None,
        "read_count": len(reads),
        "reads": reads,
    }
