"""
Memory-leak test (the one item the test plan left unverified).

Runs the server under sustained, realistic load — docking jobs, vaccine jobs, and
read-only tool calls — and samples the uvicorn process's RSS between cycles. A
leak here is not academic: this process is long-lived on Kakao's side, and the
docking result alone carries the target's full PDB text (1.6 MB for the spike
protein), so anything that retains finished jobs retains those megabytes with them.

What a pass looks like: RSS rises during the first cycles (interpreter warm-up,
mhcflurry model load, connection pools) and then FLATTENS. Steady growth that
tracks the number of jobs is the leak.

Usage:
    python scripts/mcp_leak.py            # 12 cycles
    python scripts/mcp_leak.py -n 30
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import re
import subprocess
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

PORT = 8001  # overridden by --port; a second server can be tested without disturbing the first
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


def _url() -> str:
    return f"http://127.0.0.1:{PORT}/mcp"


# Kept as a module-level name because other harnesses import it.
URL = _url()


def _server_pid() -> int | None:
    """The PID listening on PORT (netstat is the portable-enough option here)."""
    # Windows console tools emit the OEM codepage, not UTF-8 — decode leniently or
    # this dies on the first localised word.
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout or ""
    for line in out.splitlines():
        if f":{PORT} " in line and "LISTENING" in line:
            return int(line.split()[-1])
    return None


def _rss_mb(pid: int) -> float:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout or ""
    # tasklist prints the working set with thousands separators INSIDE the quoted
    # field: "python.exe","1234","Console","1","804,108 K". Splitting the line on ","
    # tears that number in half and yields a plausible-looking 0.4 MB — which is how
    # this harness first reported a server holding 785 MB as perfectly flat. A
    # measuring tool that quietly reads the wrong number is worse than no tool.
    row = next(csv.reader(io.StringIO(out.strip())), None)
    if not row:
        return 0.0
    kb = re.sub(r"[^0-9]", "", row[-1])
    return int(kb) / 1024 if kb else 0.0


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


async def _job(session: ClientSession, tool: str, args: dict) -> None:
    body = _text(await session.call_tool(tool, args))
    job = next((l.split("**:")[-1].strip() for l in body.splitlines() if "job_id" in l), "")
    if not job:
        return
    for _ in range(15):
        body = _text(await session.call_tool("get_drug_discovery_job_status", {"job_id": job}))
        if "RUNNING" not in body[:200]:
            return


async def _cycle(session: ClientSession) -> None:
    """One realistic user's worth of work."""
    await _job(session, "predict_drug_binding", {"uniprot_id": "P0DTC2", "screen_library": True})
    await _job(session, "predict_neoantigen_candidates", {})
    for tool, args in [
        ("predict_admet_profile", {"smiles": ASPIRIN}),
        ("get_target_disease_associations", {"uniprot_id": "P00533"}),
        ("search_similar_compounds", {"smiles": ASPIRIN, "max_results": 5}),
    ]:
        await session.call_tool(tool, args)


async def main() -> None:
    global PORT
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=12, help="사이클 수")
    p.add_argument("--port", type=int, default=PORT, help="테스트할 서버 포트")
    args = p.parse_args()
    PORT = args.port

    pid = _server_pid()
    if not pid:
        sys.exit(f"{PORT}에서 서버를 찾지 못했습니다.")

    samples: list[tuple[int, float, int]] = []
    print(f"서버 PID {pid} | 사이클마다 도킹 잡 + 백신 잡 + 조회 3건\n")
    print(f"{'사이클':>5} {'RSS(MB)':>9} {'증가':>8}  {'잡 수':>6}")

    async with streamablehttp_client(_url()) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            base = _rss_mb(pid)
            print(f"{'시작':>5} {base:>9.1f} {'—':>8}  {0:>6}")

            for i in range(1, args.n + 1):
                await _cycle(session)
                # Let the server settle before sampling: a job that just finished may
                # still be referenced by the task that ran it.
                await asyncio.sleep(1.0)
                rss = _rss_mb(pid)
                jobs = await _job_count(session)
                samples.append((i, rss, jobs))
                print(f"{i:>5} {rss:>9.1f} {rss - base:>+8.1f}  {jobs:>6}")

    # A leak shows up as growth that does not flatten. Compare the second half's
    # slope with the first half's: warm-up is front-loaded, a leak is not.
    if len(samples) >= 6:
        mid = len(samples) // 2
        early = (samples[mid - 1][1] - samples[0][1]) / max(1, mid - 1)
        late = (samples[-1][1] - samples[mid][1]) / max(1, len(samples) - 1 - mid)
        total = samples[-1][1] - base
        print(f"\n총 증가: {total:+.1f} MB / {args.n}사이클")
        print(f"전반 기울기: {early:+.2f} MB/사이클   후반 기울기: {late:+.2f} MB/사이클")
        leaking = late > 1.0
        print(f"\n판정: {'✗ 누수 — 후반에도 사이클마다 증가' if leaking else '✓ 안정 — 워밍업 후 평탄화'}")
        sys.exit(1 if leaking else 0)


async def _job_count(session: ClientSession) -> int:
    """How many jobs the server is still holding. Growth here IS the leak — every
    retained docking job pins the target's full PDB text."""
    import urllib.request, json
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/api/drug-discovery/debug/store-size", timeout=5) as r:
            return json.loads(r.read()).get("jobs", -1)
    except Exception:
        return -1


if __name__ == "__main__":
    asyncio.run(main())
