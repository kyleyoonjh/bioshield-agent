"""
Talk to the MCP server exactly the way Kakao PlayMCP does — over the real
Streamable HTTP transport at /mcp, with real tools/call requests — instead of
through the REST router the web UI uses.

The distinction is not cosmetic. Everything that has actually broken in PlayMCP
was invisible from the REST side: the tool DESCRIPTIONS the remote model reads,
the markdown _md() renders (not the JSON the router returns), the isError flag,
and above all the SIZE of a result, which is what a remote client truncates. The
REST endpoint returned a perfectly good 18,000-character job and PlayMCP still
reported it as a failure. So test what Kakao calls.

Usage (server must already be running):
    python scripts/mcp_probe.py                # list tools + sizes
    python scripts/mcp_probe.py vaccine        # the full 3-command vaccine flow
    python scripts/mcp_probe.py call <tool> '{"json": "args"}'
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://127.0.0.1:8001/mcp"

# PlayMCP's published response-time budget. A poll that holds the call open to
# wait for the job must still land under the p99 cap.
P99_BUDGET_MS = 3000


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


async def _call(session: ClientSession, name: str, args: dict) -> tuple[str, bool, float]:
    t0 = time.monotonic()
    result = await session.call_tool(name, args)
    ms = (time.monotonic() - t0) * 1000
    return _text(result), bool(result.isError), ms


def _report(name: str, body: str, is_error: bool, ms: float) -> None:
    flag = "실패(isError)" if is_error else "성공"
    warn = "  ⚠ p99 초과" if ms > P99_BUDGET_MS else ""
    print(f"\n=== {name} → {flag} | {ms:.0f}ms | {len(body)}자{warn}")


async def list_tools(session: ClientSession) -> None:
    tools = await session.list_tools()
    print(f"{len(tools.tools)} tools\n")
    print(f"{'tool':34}{'desc':>6}")
    for t in tools.tools:
        desc = t.description or ""
        over = "  ⚠ >1000" if len(desc) > 1000 else ""
        print(f"{t.name:34}{len(desc):>6}{over}")


async def vaccine_flow(session: ClientSession) -> None:
    """The three commands a Kakao user actually types, in order, with the model's
    polling behaviour reproduced faithfully: no sleeping between calls."""
    body, err, ms = await _call(session, "predict_neoantigen_candidates", {"vcf_content": ""})
    _report("① predict_neoantigen_candidates(vcf_content='')", body, err, ms)
    print(body[:600])

    job_id = ""
    for line in body.splitlines():
        if "job_id" in line:
            job_id = line.split("**:")[-1].strip()
            break
    if not job_id:
        print("!! job_id를 응답에서 못 찾음")
        return

    # A model polls back-to-back — it has no way to sleep. If the server doesn't
    # wait, this loop burns through every attempt in under a second.
    t0 = time.monotonic()
    for i in range(1, 13):
        body, err, ms = await _call(session, "get_drug_discovery_job_status", {"job_id": job_id})
        done = "RUNNING" not in body[:200]
        print(f"  poll {i:>2}: {ms:>6.0f}ms  {len(body):>6}자  "
              f"{'COMPLETED/FAILED' if done else 'RUNNING'}"
              f"{'  ⚠ p99 초과' if ms > P99_BUDGET_MS else ''}")
        if done:
            print(f"  → {i}번 폴링 / 총 {time.monotonic() - t0:.1f}s 만에 종료 상태 도달")
            _report("② get_drug_discovery_job_status", body, err, ms)
            print(body[:1200])
            break
    else:
        print("  !! 12번 폴링에도 끝나지 않음")
        return

    body, err, ms = await _call(session, "generate_vaccine_report", {"job_id": job_id})
    _report("③ generate_vaccine_report", body, err, ms)
    print(body[:1500])


async def main() -> None:
    argv = sys.argv[1:]
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if not argv:
                await list_tools(session)
            elif argv[0] == "vaccine":
                await vaccine_flow(session)
            elif argv[0] == "call":
                body, err, ms = await _call(session, argv[1], json.loads(argv[2]) if len(argv) > 2 else {})
                _report(argv[1], body, err, ms)
                print(body)
            else:
                print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
