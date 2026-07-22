"""
Latency benchmark for every MCP tool, measured over the real Streamable HTTP
transport at /mcp — the same path Kakao PlayMCP calls.

Kakao enforces a ~10s hard timeout per tool call, so the bar here is 9s with
room to spare. Timing the Python functions directly would understate it: this
measures the whole round trip, including transport and the markdown rendering.

Every tool is called once COLD (a freshly started server, no warm caches, no
open keep-alive connections to PubChem/ChEMBL/UniProt) because that is the call
a real user makes first and the only one at risk of the timeout. The background
jobs (docking / SAR / neoantigen) return a job_id immediately, so what is timed
for those is the START call plus every status poll that follows — a poll now
holds the call open server-side, which is exactly what has to stay under 9s.

Usage (server must already be running):
    python scripts/mcp_bench.py
"""
from __future__ import annotations

import asyncio
import re
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://127.0.0.1:8001/mcp"
KAKAO_TIMEOUT_S = 10.0
BUDGET_S = 9.0

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
EGFR, KRAS, SPIKE = "P00533", "P01116", "P0DTC2"

# One representative call per tool, with the arguments a real conversation
# produces. Job-starting tools are handled separately (they need polling).
SIMPLE_CALLS: list[tuple[str, dict]] = [
    ("search_literature", {"query": "KRAS G12D neoantigen mRNA vaccine", "max_results": 3}),
    ("search_clinical_trials", {"query": "pancreatic cancer KRAS vaccine", "max_results": 3}),
    ("search_similar_compounds", {"smiles": ASPIRIN, "max_results": 5}),
    ("search_known_inhibitors", {"uniprot_id": EGFR, "max_results": 5}),
    ("predict_admet_profile", {"smiles": ASPIRIN}),
    ("get_target_disease_associations", {"uniprot_id": EGFR}),
    ("get_target_pathways", {"uniprot_id": EGFR}),
    ("get_opentargets_profile", {"uniprot_id": KRAS}),
]

JOB_CALLS: list[tuple[str, dict]] = [
    ("predict_neoantigen_candidates", {"vcf_content": ""}),
    ("predict_drug_binding", {"uniprot_id": SPIKE, "screen_library": True}),
]

# Report tools run only after their job completes, so they're timed inside the
# job flow below rather than standalone.
REPORT_FOR_MODE = {
    "neoantigen": "generate_vaccine_report",
    "screen": "generate_decision_report",
    "sar": "generate_decision_report",
}


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def _verdict(seconds: float) -> str:
    if seconds >= KAKAO_TIMEOUT_S:
        return "TIMEOUT"
    if seconds > BUDGET_S:
        return "OVER 9s"
    return "OK"


async def _timed(session: ClientSession, name: str, args: dict):
    t0 = time.monotonic()
    result = await session.call_tool(name, args)
    return _text(result), bool(result.isError), time.monotonic() - t0


def _row(label: str, seconds: float, chars: int, is_error: bool) -> None:
    print(f"{label:42} {seconds:>6.2f}s  {chars:>6}자  "
          f"{'isError' if is_error else 'ok':<8} {_verdict(seconds)}")


async def main() -> None:
    worst: list[tuple[float, str]] = []

    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"{'tool':42} {'time':>7}  {'size':>6}  status\n" + "-" * 78)

            for name, args in SIMPLE_CALLS:
                body, err, secs = await _timed(session, name, args)
                _row(name, secs, len(body), err)
                worst.append((secs, name))

            for name, args in JOB_CALLS:
                body, err, secs = await _timed(session, name, args)
                _row(f"{name} (start)", secs, len(body), err)
                worst.append((secs, f"{name} (start)"))

                match = re.search(r"job_id\*\*:\s*(\S+)", body)
                if not match:
                    print(f"  !! {name}: job_id 없음 — 폴링 생략")
                    continue
                job_id = match.group(1)

                mode, slowest_poll, polls, t0 = "", 0.0, 0, time.monotonic()
                for polls in range(1, 21):
                    body, err, secs = await _timed(
                        session, "get_drug_discovery_job_status", {"job_id": job_id})
                    slowest_poll = max(slowest_poll, secs)
                    if "RUNNING" not in body[:200]:
                        mode = (re.search(r"mode\*\*:\s*(\w+)", body) or [None, ""])[1]
                        break
                total = time.monotonic() - t0
                _row(f"  └ status poll (최악, {polls}회/{total:.0f}s)", slowest_poll, len(body), err)
                worst.append((slowest_poll, f"get_drug_discovery_job_status ({name})"))

                report_tool = REPORT_FOR_MODE.get(mode)
                if report_tool:
                    body, err, secs = await _timed(session, report_tool, {"job_id": job_id})
                    _row(f"  └ {report_tool}", secs, len(body), err)
                    worst.append((secs, f"{report_tool} ({mode})"))

                # SAR optimizes an existing docking result, so it can only be
                # timed once a docking job has completed — it takes that job's
                # id, not a target.
                if mode == "screen":
                    body, err, secs = await _timed(session, "run_sar_optimization", {"job_id": job_id})
                    _row("  └ run_sar_optimization (start)", secs, len(body), err)
                    worst.append((secs, "run_sar_optimization (start)"))
                    match = re.search(r"job_id\*\*:\s*(\S+)", body)
                    if match:
                        sar_id, slowest_poll, t0 = match.group(1), 0.0, time.monotonic()
                        for polls in range(1, 21):
                            body, err, secs = await _timed(
                                session, "get_drug_discovery_job_status", {"job_id": sar_id})
                            slowest_poll = max(slowest_poll, secs)
                            if "RUNNING" not in body[:200]:
                                break
                        _row(f"    └ status poll (최악, {polls}회/{time.monotonic() - t0:.0f}s)",
                             slowest_poll, len(body), err)
                        worst.append((slowest_poll, "get_drug_discovery_job_status (sar)"))

    print("-" * 78)
    worst.sort(reverse=True)
    slowest, name = worst[0]
    print(f"가장 느린 호출: {name} — {slowest:.2f}s  ({_verdict(slowest)})")
    over = [(s, n) for s, n in worst if s > BUDGET_S]
    print(f"9초 초과: {len(over)}건" + ("".join(f"\n  - {n}: {s:.2f}s" for s, n in over) if over else " — 전부 통과"))


if __name__ == "__main__":
    asyncio.run(main())
