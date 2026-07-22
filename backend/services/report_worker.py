"""
Report rendering, isolated in a RECYCLED subprocess.

Why this exists: a 60-cycle sustained-load run grew the server by 188 MB and never
plateaued. Attribution (scripts/mcp_leak_attrib.py) put the growth entirely on
docking jobs — vaccine jobs and read-only tools were flat — and tracemalloc showed
the PYTHON heap growing only 0.02 MB/cycle. So nothing was retaining Python objects:
the growth was native, in the C/allocator layer under reportlab + jinja2. Stubbing
out report generation dropped the pipeline from +0.52 to +0.05 MB/cycle, i.e. the
report writer was ~90% of it.

Native allocator growth cannot be fixed from Python. gc.collect() does not touch it,
and freed C++ blocks are not returned to the OS — they just fragment the heap. The
only reliable way to give that memory back is for the process holding it to exit. So
rendering happens in a child process that is retired every _MAX_TASKS_PER_CHILD
reports: the memory is bounded by construction rather than by hoping an allocator
behaves.

max_tasks_per_child is the whole point — a plain ProcessPoolExecutor would just move
the identical leak into a child that never dies. The count trades spawn cost against
the ceiling: 20 reports x ~0.5 MB ≈ 10 MB of headroom before the child is recycled,
while paying process-spawn cost only once per 20 reports rather than once per report.

If the pool cannot be used at all (a container that forbids subprocesses, a spawn
failure), rendering falls back to a thread. A degraded-but-working report beats a
failed job; the fallback logs loudly so the leak's return is not silent.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_MAX_TASKS_PER_CHILD = int(os.getenv("REPORT_WORKER_MAX_TASKS", "20"))

_pool: ProcessPoolExecutor | None = None
_pool_broken = False


def _get_pool() -> ProcessPoolExecutor | None:
    """Created lazily: building it at import time would fork/spawn a child during
    module import, which under uvicorn's reloader means a child per reload."""
    global _pool, _pool_broken
    if _pool_broken:
        return None
    if _pool is None:
        try:
            _pool = ProcessPoolExecutor(
                max_workers=1, max_tasks_per_child=_MAX_TASKS_PER_CHILD,
                # "spawn", explicitly — NOT the platform default. On Linux (the
                # deployed container) that default is fork, and forking a process
                # that holds a live asyncio loop plus uvicorn's thread pool copies
                # only the calling thread while every lock stays in whatever state
                # it was in: a lock held by another thread at fork time is inherited
                # locked and never released, and the child deadlocks on the first
                # logging/import call. It is timing-dependent, so it would surface
                # as an occasional report that simply never returns.
                #
                # This dev box is Windows, where spawn is already the only option —
                # so no amount of local testing could have caught the fork variant.
                # Pinning spawn makes the container behave the way the tests ran.
                mp_context=multiprocessing.get_context("spawn"),
            )
        except Exception as exc:  # noqa: BLE001 - any spawn failure must not kill the job
            logger.warning(
                "[report_worker] 서브프로세스 풀 생성 실패 — 스레드로 대체합니다 "
                "(리포트 생성 메모리가 다시 누적됩니다) | error=%s", exc)
            _pool_broken = True
            return None
    return _pool


# Top-level so they are picklable under spawn (Windows/macOS). The heavy imports are
# INSIDE the functions so the parent never pays for reportlab/jinja2 it will not use.
def _render_drug(job_id: str, result: dict, ai_summary: str) -> dict:
    from services.drug_report_service import generate_drug_report
    return generate_drug_report(job_id, result, ai_summary)


def _render_neoantigen(job_id: str, result: dict) -> dict:
    from services.drug_report_service import generate_neoantigen_report
    return generate_neoantigen_report(job_id, result)


async def _run(fn, *args) -> dict:
    global _pool, _pool_broken
    pool = _get_pool()
    if pool is not None:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(pool, fn, *args)
        except Exception as exc:  # noqa: BLE001
            # A BrokenProcessPool leaves the executor permanently unusable; drop it so
            # the next report retries with a fresh one instead of failing forever.
            logger.warning("[report_worker] 서브프로세스 렌더링 실패 — 스레드로 재시도 | error=%s", exc)
            _pool = None
    return await asyncio.to_thread(fn, *args)


async def render_drug_report(job_id: str, result: dict, ai_summary: str = "") -> dict:
    return await _run(_render_drug, job_id, result, ai_summary)


async def render_neoantigen_report(job_id: str, result: dict) -> dict:
    return await _run(_render_neoantigen, job_id, result)
