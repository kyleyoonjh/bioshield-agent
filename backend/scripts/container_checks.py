"""
The checks that can ONLY be run inside the deployed Linux container.

Everything else in scripts/ runs against a server on the dev machine — which is
Windows, with a .venv, a layout that cannot exist in the image. Two real bugs lived
in exactly that gap and were invisible to every other harness here:

  1. receptor_prep_engine hardcoded .venv/Scripts/mk_prepare_receptor.exe. In the
     container that file never existed, prep raised into a broad `except`, returned
     {"prepared": False}, and every docking job silently degraded — no receptor, no
     pocket analysis. It never crashed. It never logged an error anyone looked at.
  2. The report worker forked uvicorn's live event loop + thread pool. Linux defaults
     to fork; Windows only HAS spawn. A lock held by another thread at fork time is
     inherited locked and never released, so the child deadlocks on its first
     logging/import call — intermittently, as "a report that never comes back".

Run inside the running container (see .github/workflows/docker-verify.yml):
    docker exec airemedy python scripts/container_checks.py

Deliberately a FILE, not a `python - <<EOF` heredoc: multiprocessing's spawn start
method re-imports the main module in the child, and a script fed through stdin is not
importable. Piping this in would fail for a reason that has nothing to do with the
product — a test that fails for its own reasons teaches you to ignore it.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
import shutil
import sys

# /app (== backend/) must be importable. Running `python scripts/container_checks.py`
# puts sys.path[0] at /app/scripts, and the CWD is NOT added — so `import services`
# fails even though the server imports it fine (uvicorn is launched from /app, which
# does put /app on the path). Three checks died on ModuleNotFoundError before this,
# which reads like a container bug and is not one.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
_DRUG_RESULT = {
    "target": {"uniprot_id": "P0DTC2", "name": "Spike"},
    "docking": {"affinity_kcal_per_mol": -6.2},
    "screened": [{"name": "aspirin", "smiles": ASPIRIN, "affinity_kcal_per_mol": -5.1}],
    "mode": "screen",
}

_failures: list[str] = []

# GitHub's Actions LOG api requires authentication even on a public repo, but the
# check-run ANNOTATIONS api does not. So emit every failure as a ::error:: workflow
# command: the reason a build failed is then readable straight off the public run
# page — and by anyone diagnosing it — instead of being locked behind a token.
_IN_CI = bool(os.getenv("GITHUB_ACTIONS"))


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}", flush=True)
    if _IN_CI:
        # Newlines would truncate the annotation; %0A is the documented escape.
        print(f"::error::{msg}".replace("\n", "%0A"), flush=True)
    _failures.append(msg)


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def check_binaries() -> None:
    """Bug #1. Assert the resolver produces a binary that ACTUALLY RUNS.

    Existence is not enough, and neither is a plausible-looking string. meeko declares
    its console script as "mk_prepare_receptor.py" — pip keeps that literal name on
    Linux and turns it into mk_prepare_receptor.exe on Windows, so a resolver that
    searched only the bare name passed on the dev box and returned None in the
    container. Actually invoking it is the only assertion that would have caught that.
    """
    print("\n[1] 외부 바이너리가 리눅스 PATH에서 해석되고 실행되는가")
    import subprocess

    from services import receptor_prep_engine

    path = receptor_prep_engine._MK_PREPARE_RECEPTOR_PATH
    if not (os.path.isfile(path) or shutil.which(path)):
        _fail(f"mk_prepare_receptor를 찾을 수 없습니다 ({path}) — 도킹이 조용히 강등됩니다")
        return
    try:
        proc = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        _fail(f"mk_prepare_receptor를 실행할 수 없습니다 ({path}): {exc}")
        return
    if proc.returncode != 0:
        _fail(f"mk_prepare_receptor --help 가 exit {proc.returncode} "
              f"({(proc.stderr or '')[-200:]}) — 파일은 있으나 실행되지 않습니다")
        return
    _ok(f"mk_prepare_receptor 실행 확인 → {path}")


def check_sample_data() -> None:
    """The vaccine demo runs on this with no user upload. If COPY missed it, the
    flagship flow fails on the user's very first message."""
    print("\n[2] 내장 샘플 데이터가 이미지에 있는가")
    p = "sample/NSCLC_variants.vcf"
    if os.path.isfile(p):
        _ok(f"{p} ({os.path.getsize(p)} bytes)")
    else:
        _fail(f"{p} 이 이미지에 없습니다 — 내장 백신 데모가 실패합니다")


def check_mhcflurry_models() -> None:
    """A 135 MB build-time download. Without it the whole vaccine track is dead."""
    print("\n[3] MHCflurry 사전학습 모델이 빌드에 포함됐는가")
    try:
        from services.neoantigen_engine import _get_predictor
        _get_predictor()
        _ok("Class1PresentationPredictor 로드 성공")
    except Exception as exc:  # noqa: BLE001
        _fail(f"MHCflurry 모델 로드 실패 — 백신 트랙 전체가 죽습니다: {exc}")


async def check_report_worker() -> None:
    """Bug #2. A deadlocked child hangs forever, so the caller's timeout IS the
    assertion. Render more times than max_tasks_per_child so a child is actually
    RETIRED mid-run — one lucky render would otherwise pass."""
    print("\n[4] 리포트 워커가 리눅스에서 데드락 없이 렌더링하는가")
    os.environ["REPORT_WORKER_MAX_TASKS"] = "3"
    from services import report_worker

    try:
        for i in range(7):  # > 3, so the child is recycled at least twice
            out = await asyncio.wait_for(
                report_worker.render_drug_report(f"ci-{i}", _DRUG_RESULT, "요약"),
                timeout=120,
            )
            if not out.get("html_path"):
                _fail(f"{i}번째 리포트가 렌더링되지 않았습니다")
                return
    except asyncio.TimeoutError:
        _fail("리포트 렌더링이 응답하지 않습니다 — fork 데드락이 의심됩니다")
        return

    if report_worker._pool_broken:
        _fail("서브프로세스 풀이 죽어 스레드로 폴백했습니다 — 메모리 누수가 돌아옵니다")
        return
    _ok(f"리포트 7건 렌더링, 자식 재활용 통과 (start_method="
        f"{multiprocessing.get_start_method(allow_none=True)})")


async def check_mcp_tools() -> None:
    """Kakao registers whatever tools/list returns. Zero tools is the '등록된 Tool이
    없습니다' failure — and from the server's side it is completely silent."""
    print("\n[5] MCP 툴이 등록되는가 (카카오가 실제로 보는 것)")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools

    if not tools:
        _fail("툴이 0개 — 카카오가 아무것도 등록하지 못합니다")
        return

    bad = False
    if len(tools) > 20:
        _fail(f"툴 {len(tools)}개 — PlayMCP 상한은 20개")
        bad = True
    over = [t.name for t in tools if len(t.description or "") > 1000]
    if over:
        _fail(f"설명이 1000자를 넘습니다 (PlayMCP가 거부): {over}")
        bad = True
    # Judged on THIS check's own findings. Gating on the global _failures list made
    # a passing check go silent whenever an earlier, unrelated check had failed.
    if not bad:
        _ok(f"툴 {len(tools)}개 등록, 설명 전부 1000자 이내")


async def check_vaccine_end_to_end() -> None:
    """The flagship flow, on the real image, with the real pretrained models.

    No OPENAI_API_KEY is set in CI on purpose: the pipeline must still complete via
    its rule-based summary path. If the science only works when an LLM key is
    present, the science is not the science.
    """
    print("\n[6] 암 백신 잡 엔드투엔드 (진짜 MHCflurry 모델, LLM 키 없이)")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()

            body = _text(await s.call_tool("predict_neoantigen_candidates", {}))
            job = next((l.split("**:")[-1].strip()
                        for l in body.splitlines() if "job_id" in l), "")
            if not job:
                _fail(f"job_id를 받지 못했습니다: {body[:300]}")
                return

            for _ in range(60):
                body = _text(await s.call_tool(
                    "get_drug_discovery_job_status", {"job_id": job}))
                if "RUNNING" not in body[:200]:
                    break
            else:
                _fail("잡이 끝나지 않았습니다")
                return

            if "FAILED" in body[:200]:
                _fail(f"잡 실패: {body[:400]}")
                return
            # A COMPLETED job that produced no peptide is a failure wearing a
            # success label — exactly the mode the container bugs failed in.
            if "HLA" not in body:
                _fail(f"COMPLETED이지만 HLA/펩타이드 결과가 없습니다: {body[:300]}")
                return
            _ok("백신 잡 COMPLETED — 실제 MHCflurry 결과 포함")

            rep = _text(await s.call_tool("generate_vaccine_report", {"job_id": job}))
            if "결론" not in rep:
                _fail(f"리포트에 결론이 없습니다: {rep[:300]}")
                return
            _ok("종합 리포트 생성 (서브프로세스 워커 경유)")


async def _run(check) -> None:
    """Run one check, converting an unexpected exception into a reported failure.

    Without this, a check that raises kills the script with a bare traceback and no
    ::error:: annotation — leaving the CI failure just as unreadable as the log it
    was supposed to replace. It also means one broken check no longer hides the
    verdict of every check after it.
    """
    try:
        result = check()
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:  # noqa: BLE001
        _fail(f"{check.__name__} 이 예외로 죽었습니다: {type(exc).__name__}: {exc}")


async def main() -> None:
    print("=" * 70)
    print("컨테이너 내부 검증 — 배포 환경에서만 드러나는 것들")
    print("=" * 70)

    for check in (check_binaries, check_sample_data, check_mhcflurry_models,
                  check_report_worker, check_mcp_tools, check_vaccine_end_to_end):
        await _run(check)

    print("\n" + "=" * 70)
    if _failures:
        print(f"실패 {len(_failures)}건:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("전부 통과 — 이 이미지는 배포 가능합니다")


if __name__ == "__main__":
    asyncio.run(main())
