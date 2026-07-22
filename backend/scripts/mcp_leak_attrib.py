"""
Leak attribution — WHICH workload is leaking.

mcp_leak.py proved there IS a leak (+188 MB over 60 cycles, and still +1.67
MB/cycle in the second half, long after the job store hit its 50-job cap). Capping
the store was necessary but not sufficient: something else grows.

A mixed-workload harness cannot tell you what. This one runs each workload ALONE
against the already-warm server and compares the RSS slopes, so the growth lands on
a specific tool instead of on "the server".

Read it as a comparison, not as absolute numbers: the server is warm and the job
store is already at its cap, so warm-up and job retention are common-mode and
cancel out. The workload with the outlier slope is the one holding the memory.

Usage:
    python scripts/mcp_leak_attrib.py            # 10 cycles per workload
    python scripts/mcp_leak_attrib.py -n 15
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from mcp_leak import URL, ASPIRIN, _rss_mb, _server_pid, _text, _job


async def _docking(session: ClientSession) -> None:
    await _job(session, "predict_drug_binding", {"uniprot_id": "P0DTC2", "screen_library": True})


async def _vaccine(session: ClientSession) -> None:
    await _job(session, "predict_neoantigen_candidates", {})


async def _readonly(session: ClientSession) -> None:
    for tool, args in [
        ("predict_admet_profile", {"smiles": ASPIRIN}),
        ("get_target_disease_associations", {"uniprot_id": "P00533"}),
        ("search_similar_compounds", {"smiles": ASPIRIN, "max_results": 5}),
    ]:
        await session.call_tool(tool, args)


async def _idle(session: ClientSession) -> None:
    """Control: the transport and the sampling loop themselves, doing no real work.
    If this slopes up too, the leak is in the MCP session layer, not in a tool."""
    await session.call_tool("list_available_targets", {})


WORKLOADS = [
    ("도킹 잡만",     _docking),
    ("백신 잡만",     _vaccine),
    ("조회 툴만",     _readonly),
    ("대조군(유휴)",  _idle),
]


async def _measure(session: ClientSession, pid: int, label: str, fn, cycles: int) -> float:
    base = _rss_mb(pid)
    for _ in range(cycles):
        await fn(session)
        await asyncio.sleep(1.0)  # let the finishing task drop its references
    rss = _rss_mb(pid)
    slope = (rss - base) / cycles
    print(f"  {label:<14} {base:>8.1f} → {rss:>8.1f} MB   "
          f"{rss - base:>+7.1f} MB   {slope:>+6.2f} MB/사이클")
    return slope


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=10, help="워크로드당 사이클 수")
    args = p.parse_args()

    pid = _server_pid()
    if not pid:
        sys.exit("8001에서 서버를 찾지 못했습니다.")

    print(f"서버 PID {pid} | 워크로드당 {args.n}사이클, 각각 단독 실행\n")
    print(f"  {'워크로드':<14} {'시작':>8}   {'종료':>8}   {'증가':>8}   {'기울기':>8}")

    slopes: list[tuple[str, float]] = []
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for label, fn in WORKLOADS:
                slopes.append((label, await _measure(session, pid, label, fn, args.n)))

    worst, worst_slope = max(slopes, key=lambda s: s[1])
    print(f"\n최대 증가: {worst} ({worst_slope:+.2f} MB/사이클)")
    if worst_slope <= 0.5:
        print("판정: ✓ 단독 실행에서는 어떤 워크로드도 뚜렷하게 증가하지 않습니다")
    else:
        print(f"판정: ✗ '{worst}'가 사이클마다 메모리를 붙잡고 있습니다 — 여기부터 파세요")


if __name__ == "__main__":
    asyncio.run(main())
