"""
A shared wall-clock budget for the external biomedical APIs this project calls.

Every one of them — UniProt, Reactome, OpenTargets, PubMed, ClinicalTrials.gov,
ChEMBL, PubChem — is a public service with no latency guarantee, and each engine
independently chose a 15s timeout. Kakao PlayMCP kills a tool call at ~10s, so
those timeouts could never fire: the client gave up first and the user got
nothing at all. Measured, not theorised — ChEMBL answered a target in 0.7s and
then stalled for a full 20s on the identical query moments later, and Reactome
took 15.1s on a lookup that normally costs 1.4s.

A longer timeout does not buy resilience against that; it just converts a
hiccup into a dead tool call. What actually helps is the opposite: give each
attempt a short leash, retry once (a stalled connection almost always answers
immediately on a fresh one), and cap the whole thing well inside the client's
patience. A slightly thinner answer beats no answer.

The pooled client matters too. Each engine used to open a brand-new
httpx.Client per call — a fresh TCP + TLS handshake to the same handful of hosts
every single time, measured at roughly 0.8s of pure setup. One client per
process keeps those connections warm. httpx.Client is thread-safe, which is what
lets these run under asyncio.to_thread.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time

import httpx

logger = logging.getLogger(__name__)

# Same convention as the *_engine.py modules — this environment can sit behind a
# proxy that breaks strict TLS verification.
_VERIFY_SSL = os.getenv("NCBI_VERIFY_SSL", "false").lower() == "true"

# Kakao PlayMCP's hard kill is ~10s. The budget leaves headroom for the tool's
# own work (rendering, an LLM narrative on top) on the far side of the request.
# This is the ceiling for a SYNCHRONOUS tool the client is timing.
DEFAULT_BUDGET_S = 8.0
ATTEMPT_TIMEOUT_S = 3.0

# A background pipeline (docking / vaccine job) is NOT under Kakao's 10s kill — the
# model polls a job store while it runs — so its external calls can afford to wait
# out a slow-but-alive upstream instead of failing it. Ensembl VEP legitimately
# answers in 12-15s on a bad day; cutting that at 8s would MANUFACTURE failures for
# answers the server was willing to give. Job-context callers pass a total up to
# this ceiling; the clamp below still stops anything unbounded.
MAX_BUDGET_S = 20.0

# Transient upstream 5xx (502/503/504, and 500 from overloaded EBI/NCBI) is a blip,
# not a verdict — it clears on an immediate retry the same way a stalled socket does.
# A 4xx (404 "no such entry", 400 bad query) is determinate and must NOT be retried.
_RETRYABLE_STATUS = {500, 502, 503, 504}

# Chaos hooks — OFF unless the env vars are set, so production behaviour is
# byte-for-byte unchanged. The point of the budget is that this project keeps
# working when a public API stalls or dies, and the only honest way to check that
# is to make one stall or die on demand instead of waiting for Reactome to have a
# bad day (it did, mid-benchmark, which is how the 15s timeout was found).
#   HTTP_CHAOS_DELAY_MS=4000  — add this much latency to every upstream request
#   HTTP_CHAOS_FAIL_RATE=0.5  — fail this fraction of them outright
_CHAOS_DELAY_MS = float(os.getenv("HTTP_CHAOS_DELAY_MS", "0"))
_CHAOS_FAIL_RATE = float(os.getenv("HTTP_CHAOS_FAIL_RATE", "0"))


def _chaos(url: str, timeout: float) -> None:
    """Behave like a slow/failing SERVER, which means the injected latency has to
    be subject to the caller's timeout exactly as real latency would be. Sleeping
    the full delay regardless would be slower than any real upstream can be — the
    request would blow the budget from outside it, and the harness would be
    measuring the harness."""
    if _CHAOS_DELAY_MS:
        delay = _CHAOS_DELAY_MS / 1000.0
        if delay > timeout:
            time.sleep(timeout)
            raise httpx.ReadTimeout(f"[chaos] {url} exceeded the {timeout:.1f}s attempt timeout")
        time.sleep(delay)
    if _CHAOS_FAIL_RATE and random.random() < _CHAOS_FAIL_RATE:
        raise httpx.ConnectError(f"[chaos] injected failure for {url}")


_client: httpx.Client | None = None
_client_lock = threading.Lock()


def http() -> httpx.Client:
    """The process-wide pooled client. Connections stay warm across calls."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    verify=_VERIFY_SSL,
                    timeout=httpx.Timeout(ATTEMPT_TIMEOUT_S, connect=5.0),
                    limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=300.0),
                )
    return _client


class Budget:
    """A wall-clock deadline shared by every request made in one tool call."""

    def __init__(self, total: float = DEFAULT_BUDGET_S) -> None:
        # Clamp to MAX_BUDGET_S, not DEFAULT_BUDGET_S: a synchronous Kakao-timed tool
        # passes <=8s and is unaffected, while a background job may ask for more (up
        # to the ceiling) so a slow-but-alive upstream is waited out, not failed.
        self.deadline = time.monotonic() + min(total, MAX_BUDGET_S)

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def timeout(self, attempt: int = 0) -> float:
        """
        Timeout for one attempt, never longer than what's left of the budget.

        The two attempts are not the same. The first is a stall detector: a
        connection that has hung answers instantly once reopened, so waiting on it
        is pure loss — cut it at ATTEMPT_TIMEOUT_S and reconnect. The second is
        where the answer actually has to come from, so it gets everything that
        remains.

        Giving both attempts the same short leash looked tidy and was wrong: an
        upstream that reliably takes 4s (slow, but perfectly alive) would be cut
        off twice and reported as a failure, and we'd have thrown away an answer
        the server was willing to give. Found by injecting 4s of latency into every
        request and watching working tools report isError.
        """
        limit = self.remaining() if attempt else min(ATTEMPT_TIMEOUT_S, self.remaining())
        return max(0.1, limit)


def get(url: str, params: dict | None = None, budget: Budget | None = None,
        client: httpx.Client | None = None, **kwargs) -> httpx.Response:
    """
    GET with one retry on a stall OR a transient 5xx, both attempts inside the budget.

    A 4xx is raised untouched so the caller can distinguish "404, no such entry" (an
    honest empty result) from a real outage. A 5xx is retried once (overloaded
    EBI/NCBI throw these and clear immediately) and only then raised untouched, so a
    persistent 5xx still surfaces as the outage it is.
    """
    budget = budget or Budget()
    client = client or http()
    last: httpx.HTTPError | None = None

    for attempt in range(2):
        if budget.remaining() <= 0:
            break
        try:
            _chaos(url, budget.timeout(attempt))
            resp = client.get(url, params=params, timeout=budget.timeout(attempt), **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            last = exc
            if exc.response.status_code in _RETRYABLE_STATUS and attempt == 0 and budget.remaining() > 0:
                logger.info("[http_budget] %s, retrying once | url=%s", exc.response.status_code, url)
                continue
            raise  # 4xx, or a 5xx that survived the retry — the caller must see it
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt == 0:
                logger.info("[http_budget] stalled, retrying once | url=%s", url)

    raise httpx.ReadTimeout(f"{url} did not respond within the budget") from last


def post(url: str, budget: Budget | None = None, client: httpx.Client | None = None,
         **kwargs) -> httpx.Response:
    """POST counterpart — OpenTargets' GraphQL endpoint needs it."""
    budget = budget or Budget()
    client = client or http()
    last: httpx.HTTPError | None = None

    for attempt in range(2):
        if budget.remaining() <= 0:
            break
        try:
            _chaos(url, budget.timeout(attempt))
            resp = client.post(url, timeout=budget.timeout(attempt), **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            last = exc
            if exc.response.status_code in _RETRYABLE_STATUS and attempt == 0 and budget.remaining() > 0:
                logger.info("[http_budget] %s, retrying once | url=%s", exc.response.status_code, url)
                continue
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt == 0:
                logger.info("[http_budget] stalled, retrying once | url=%s", url)

    raise httpx.ReadTimeout(f"{url} did not respond within the budget") from last
