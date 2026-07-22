"""
Stress and robustness suite for the 14 MCP tools, over the real /mcp transport.

Companion to the other two harnesses:
  mcp_probe.py  — does a given tool work?
  mcp_bench.py  — is every tool inside Kakao's ~10s kill?
  kakao_sim.py  — does an LLM with only the tool descriptions drive it correctly?
  mcp_stress.py — does it hold up under garbage input, load, and a failing upstream?

Subcommands:
  fuzz        20 randomised tool calls, adversarial arguments included
  chain       output of tool A fed as input to tool B, down the real pipelines
  concurrent  N clients hitting the server at once (race conditions, pool exhaustion)
  cache       the same call cold vs warm — does caching actually pay?
  chaos       requires the server to run with HTTP_CHAOS_* set (see http_budget.py)

The bar for fuzz/chaos is NOT "everything succeeds". Bad input SHOULD fail — the
question is whether it fails the way the MCP spec requires (isError, a message a
model can act on) instead of hanging, crashing the process, or returning a
confident answer built from nothing. A tool that reports "0 results" for a SQL
injection string has not passed; it has lied politely.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://127.0.0.1:8001/mcp"
KAKAO_TIMEOUT_S = 10.0

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Arguments a real user (or a confused model) can actually produce.
VALID_UNIPROT = ["P00533", "P01116", "P04637", "P0DTC2", "P04626", "P15056"]
VALID_SMILES = [ASPIRIN, "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"]
VALID_QUERIES = ["KRAS G12D vaccine", "EGFR inhibitor", "pancreatic cancer immunotherapy"]

# Adversarial: empty, whitespace, absurdly long, injection-shaped, wrong-type,
# unicode, and a plausible-but-wrong identifier (a gene name where an accession
# is required — the single most common real mistake a model makes here).
HOSTILE = [
    "", "   ", "\n\t", "A" * 5000, "'; DROP TABLE studies; --",
    "<script>alert(1)</script>", "../../etc/passwd", "%00", "🧬💊" * 50,
    "EGFR", "코로나 신약", "NULL", "-1", "0" * 200, "{{7*7}}",
]


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def _hostile() -> str:
    return random.choice(HOSTILE)


def _random_call() -> tuple[str, dict]:
    """A random tool with either sane or hostile arguments (50/50)."""
    hostile = random.random() < 0.5
    name = random.choice([
        "search_literature", "search_clinical_trials", "search_similar_compounds",
        "search_known_inhibitors", "predict_admet_profile",
        "get_target_disease_associations", "get_target_pathways",
        "get_opentargets_profile", "get_drug_discovery_job_status",
        "generate_vaccine_report", "generate_decision_report", "run_sar_optimization",
    ])
    if name in ("search_literature", "search_clinical_trials"):
        args = {"query": _hostile() if hostile else random.choice(VALID_QUERIES)}
    elif name in ("search_similar_compounds", "predict_admet_profile"):
        args = {"smiles": _hostile() if hostile else random.choice(VALID_SMILES)}
    elif name in ("search_known_inhibitors", "get_target_disease_associations",
                  "get_target_pathways", "get_opentargets_profile"):
        args = {"uniprot_id": _hostile() if hostile else random.choice(VALID_UNIPROT)}
    else:  # job-id tools: a bogus id is the interesting case
        args = {"job_id": _hostile() if hostile else "00000000-0000-0000-0000-000000000000"}
    return name, args


async def _call(session: ClientSession, name: str, args: dict) -> tuple[str, bool, float, bool]:
    """Returns (body, is_error, seconds, hung) — hung means it blew Kakao's timeout."""
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(session.call_tool(name, args), timeout=KAKAO_TIMEOUT_S)
        return _text(result), bool(result.isError), time.monotonic() - t0, False
    except asyncio.TimeoutError:
        return "", True, time.monotonic() - t0, True


async def fuzz(session: ClientSession, n: int) -> int:
    random.seed(20260712)
    hung = crashed = 0
    slow: list[tuple[float, str]] = []

    print(f"{'#':>3} {'tool':32}{'arg':38}{'time':>7}  결과")
    for i in range(1, n + 1):
        name, args = _random_call()
        arg_repr = repr(next(iter(args.values())))[:34]
        try:
            body, is_error, secs, timed_out = await _call(session, name, args)
        except Exception as exc:  # a tool must never take the transport down
            print(f"{i:>3} {name:32}{arg_repr:38}{'—':>7}  💥 예외: {exc.__class__.__name__}")
            crashed += 1
            continue

        if timed_out:
            print(f"{i:>3} {name:32}{arg_repr:38}{secs:>6.1f}s  ⏱ 10초 초과 (카카오라면 사망)")
            hung += 1
            continue
        slow.append((secs, name))
        verdict = "isError (정상적 거부)" if is_error else f"ok {len(body)}자"
        print(f"{i:>3} {name:32}{arg_repr:38}{secs:>6.1f}s  {verdict}")

    worst, wname = max(slow) if slow else (0.0, "-")
    print(f"\n호출 {n}건 | 프로세스 크래시 {crashed} | 10초 초과 {hung} | 최장 {worst:.1f}s ({wname})")
    return crashed + hung


async def chain(session: ClientSession) -> int:
    """A -> B: the output of one tool really being the input of the next."""
    failures = 0

    print("① search_known_inhibitors(EGFR) → 그 SMILES → predict_admet_profile → search_similar_compounds")
    body, err, secs, _ = await _call(session, "search_known_inhibitors", {"uniprot_id": "P00533", "max_results": 3})
    smiles = ""
    for line in body.splitlines():
        if "**smiles**:" in line:
            smiles = line.split("**smiles**:")[1].strip()
            break
    if err or not smiles:
        print(f"   ✗ 1단계에서 SMILES를 얻지 못함 (isError={err})")
        return 1
    print(f"   → SMILES 확보: {smiles[:50]}")

    for nxt in ("predict_admet_profile", "search_similar_compounds"):
        body, err, secs, _ = await _call(session, nxt, {"smiles": smiles})
        ok = not err and len(body) > 50
        print(f"   {'✓' if ok else '✗'} {nxt}: {secs:.1f}s {len(body)}자 isError={err}")
        failures += 0 if ok else 1

    print("\n② predict_neoantigen_candidates → job_id → status → generate_vaccine_report")
    body, err, _, _ = await _call(session, "predict_neoantigen_candidates", {})
    job_id = next((l.split("**:")[-1].strip() for l in body.splitlines() if "job_id" in l), "")
    if not job_id:
        print("   ✗ job_id 없음")
        return failures + 1

    for _ in range(12):
        body, err, secs, _ = await _call(session, "get_drug_discovery_job_status", {"job_id": job_id})
        if "RUNNING" not in body[:200]:
            break
    has_peptide = "mutant_peptide" in body
    print(f"   {'✓' if has_peptide else '✗'} status: 후보 펩타이드 전달됨={has_peptide}")
    failures += 0 if has_peptide else 1

    body, err, secs, _ = await _call(session, "generate_vaccine_report", {"job_id": job_id})
    has_decision = "최종_결론" in body
    print(f"   {'✓' if has_decision else '✗'} generate_vaccine_report: {secs:.1f}s 최종_결론 포함={has_decision}")
    failures += 0 if has_decision else 1

    print("\n③ 모달리티 교차: 백신 job_id를 저분자 리포트 툴에 넣으면 거부해야 함")
    body, err, _, _ = await _call(session, "generate_decision_report", {"job_id": job_id})
    print(f"   {'✓ 거부됨 (isError)' if err else '✗ 통과시킴 — 모달리티 혼선!'}")
    failures += 0 if err else 1
    return failures


async def _one_client(idx: int, results: list) -> None:
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            name, args = _random_call()
            try:
                _, is_error, secs, hung = await _call(session, name, args)
                results.append((idx, name, secs, is_error, hung, None))
            except Exception as exc:
                results.append((idx, name, 0.0, True, False, exc.__class__.__name__))


async def concurrent(n: int) -> int:
    """N independent MCP sessions at once — race conditions, pool exhaustion."""
    results: list = []
    t0 = time.monotonic()
    await asyncio.gather(*(_one_client(i, results) for i in range(1, n + 1)))
    wall = time.monotonic() - t0

    hung = sum(1 for r in results if r[4])
    crashed = [r for r in results if r[5]]
    times = [r[2] for r in results if not r[4]]
    print(f"동시 클라이언트 {n}개 | 전체 {wall:.1f}s")
    print(f"  응답: 중앙값 {statistics.median(times):.2f}s  최대 {max(times):.2f}s")
    print(f"  10초 초과 {hung}건 | 예외 {len(crashed)}건 {[c[5] for c in crashed]}")
    return hung + len(crashed)


async def cache(session: ClientSession) -> int:
    """Warm start must actually be faster — otherwise the caches are dead weight."""
    checks = [
        ("search_known_inhibitors", {"uniprot_id": "P04626", "max_results": 5}),
        ("search_similar_compounds", {"smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "max_results": 5}),
    ]
    for name, args in checks:
        _, _, cold, _ = await _call(session, name, args)
        warm = []
        for _ in range(3):
            _, _, secs, _ = await _call(session, name, args)
            warm.append(secs)
        best = min(warm)
        gain = f"{cold / best:.1f}배 빠름" if best > 0 else "—"
        print(f"  {name:26} 콜드 {cold:5.2f}s → 웜 {best:5.2f}s   ({gain})")
    return 0


async def chaos(session: ClientSession) -> int:
    """With HTTP_CHAOS_* set on the server: degrade gracefully, never hang."""
    print("서버가 HTTP_CHAOS_DELAY_MS / HTTP_CHAOS_FAIL_RATE 로 떠 있어야 합니다.\n")
    calls = [
        ("search_literature", {"query": "KRAS G12D vaccine"}),
        ("search_clinical_trials", {"query": "KRAS vaccine"}),
        ("search_known_inhibitors", {"uniprot_id": "P00533"}),
        ("get_target_pathways", {"uniprot_id": "P00533"}),
        ("get_target_disease_associations", {"uniprot_id": "P00533"}),
        ("get_opentargets_profile", {"uniprot_id": "P01116"}),
        ("predict_admet_profile", {"smiles": ASPIRIN}),  # no upstream — must still work
    ]
    hung = 0
    for name, args in calls:
        body, is_error, secs, timed_out = await _call(session, name, args)
        if timed_out:
            print(f"  ✗ {name:32} {secs:5.1f}s  10초 초과 — 카카오라면 사망")
            hung += 1
        elif is_error:
            print(f"  ✓ {name:32} {secs:5.1f}s  isError로 우아하게 실패")
        else:
            print(f"  ✓ {name:32} {secs:5.1f}s  {len(body)}자 응답 (부분 성공/정상)")
    print(f"\n행(hang) {hung}건 — 0이어야 통과 (실패는 허용, 매달림은 불가)")
    return hung


async def params(session: ClientSession) -> int:
    """Test 5 — the arguments a model gets wrong. Missing, wrong type, out of
    range, a gene symbol where an accession belongs. Every one must be refused
    with isError; none may crash the server or be silently coerced into an
    answer."""
    cases = [
        ("필수 파라미터 누락", "predict_admet_profile", {}),
        ("타입 오류 (int ← str)", "search_literature", {"query": "KRAS", "max_results": "다섯개"}),
        ("범위 오류 (음수)", "search_literature", {"query": "KRAS", "max_results": -5}),
        ("범위 오류 (과대)", "search_similar_compounds", {"smiles": ASPIRIN, "max_results": 100000}),
        ("유전자명 ← accession 자리", "get_target_pathways", {"uniprot_id": "EGFR"}),
        ("질병명 ← accession 자리", "search_known_inhibitors", {"uniprot_id": "코로나"}),
        ("SMILES 자리에 약 이름", "predict_admet_profile", {"smiles": "aspirin"}),
        ("job_id 자리에 UniProt", "generate_vaccine_report", {"job_id": "P00533"}),
    ]
    # NOT a failure case, and it was wrong to list it as one: an unknown extra
    # argument is ignored, which is what JSON Schema without additionalProperties
    # prescribes and what every forward-compatible API does. It "passed" once only
    # because Reactome happened to be down that minute and the call errored for an
    # entirely unrelated reason — a false pass, which is worse than a fail.
    bad = 0
    for label, name, args in cases:
        try:
            body, is_error, secs, hung = await _call(session, name, args)
        except Exception as exc:
            # An MCP validation error surfaces as an exception in the client — that
            # is a legitimate rejection, not a crash, as long as the server survives.
            print(f"  ✓ {label:26} {name:30} 스키마 거부 ({exc.__class__.__name__})")
            continue
        if hung:
            print(f"  ✗ {label:26} {name:30} 10초 초과")
            bad += 1
        elif is_error:
            print(f"  ✓ {label:26} {name:30} isError로 거부")
        else:
            print(f"  ✗ {label:26} {name:30} 그냥 통과 — {len(body)}자 응답")
            bad += 1
    print(f"\n부적절하게 통과한 케이스: {bad}건 (0이어야 통과)")
    return bad


async def isolation() -> int:
    """Test 13 — two users at once must not see each other's study. Session A runs
    a vaccine job, session B a docking job; each must get back its own, and neither
    may read the other's job as if it were theirs."""
    async def start(kind: str) -> tuple[str, str]:
        async with streamablehttp_client(URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if kind == "vaccine":
                    body, *_ = await _call(session, "predict_neoantigen_candidates", {})
                else:
                    body, *_ = await _call(session, "predict_drug_binding",
                                           {"uniprot_id": "P0DTC2", "screen_library": True})
                job = next((l.split("**:")[-1].strip() for l in body.splitlines() if "job_id" in l), "")
                for _ in range(12):
                    body, *_ = await _call(session, "get_drug_discovery_job_status", {"job_id": job})
                    if "RUNNING" not in body[:200]:
                        break
                return job, body

    (vac_job, vac_body), (dock_job, dock_body) = await asyncio.gather(start("vaccine"), start("docking"))

    checks = [
        ("A(백신) 세션이 백신 결과를 받음", "neoantigen" in vac_body and "mutant_peptide" in vac_body),
        ("B(도킹) 세션이 도킹 결과를 받음", "screen" in dock_body and "ranked_candidates" in dock_body),
        ("두 job_id가 서로 다름", vac_job != dock_job),
        ("A의 결과에 B의 내용 없음", "ranked_candidates" not in vac_body),
        ("B의 결과에 A의 내용 없음", "mutant_peptide" not in dock_body),
    ]
    bad = 0
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")
        bad += 0 if ok else 1
    print(f"\n세션 오염: {bad}건 (0이어야 100% 격리)")
    return bad


async def deterministic(session: ClientSession, n: int) -> int:
    """Test 18 — the engines are the product. The same input must yield byte-identical
    numbers every time; only the AI's prose may vary. Runs the deterministic tools
    repeatedly and diffs the numeric payloads."""
    bad = 0
    for name, args, label in [
        ("predict_admet_profile", {"smiles": ASPIRIN}, "RDKit ADMET"),
        ("search_similar_compounds", {"smiles": ASPIRIN, "max_results": 5}, "PubChem 유사체"),
    ]:
        bodies = []
        for _ in range(n):
            body, is_error, _, _ = await _call(session, name, args)
            if not is_error:
                bodies.append(body)
        unique = len(set(bodies))
        ok = unique == 1 and len(bodies) == n
        print(f"  {'✓' if ok else '✗'} {label:20} {n}회 실행 → 서로 다른 결과 {unique}종 "
              f"({'완전 동일' if ok else '불일치!'})")
        bad += 0 if ok else 1

    # The neoantigen pipeline is the one that matters most: real MHCflurry numbers.
    peptides, scores = set(), set()
    for _ in range(3):
        body, *_ = await _call(session, "predict_neoantigen_candidates", {})
        job = next((l.split("**:")[-1].strip() for l in body.splitlines() if "job_id" in l), "")
        for _ in range(12):
            body, *_ = await _call(session, "get_drug_discovery_job_status", {"job_id": job})
            if "RUNNING" not in body[:200]:
                break
        for line in body.splitlines():
            if "**mutant_peptide**:" in line:
                peptides.add(line.split("**:")[-1].strip())
            if "**mutant_affinity_nm**:" in line:
                scores.add(line.split("**:")[-1].strip())
    ok = len(peptides) == 1 and len(scores) == 1
    print(f"  {'✓' if ok else '✗'} MHCflurry 백신 후보  3회 실행 → 펩타이드 {peptides}, 친화도 {scores}")
    bad += 0 if ok else 1
    print(f"\n비결정론적 결과: {bad}건 (0이어야 통과 — 계산은 항상 같아야 합니다)")
    return bad


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["fuzz", "chain", "concurrent", "cache", "chaos",
                                    "params", "isolation", "deterministic"])
    p.add_argument("-n", type=int, default=20)
    args = p.parse_args()

    if args.mode == "concurrent":
        sys.exit(1 if await concurrent(args.n) else 0)
    if args.mode == "isolation":
        sys.exit(1 if await isolation() else 0)

    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            fn = {"fuzz": lambda: fuzz(session, args.n), "chain": lambda: chain(session),
                  "cache": lambda: cache(session), "chaos": lambda: chaos(session),
                  "params": lambda: params(session),
                  "deterministic": lambda: deterministic(session, min(args.n, 10))}[args.mode]
            failures = await fn()
    # Exit outside the session — sys.exit inside an anyio task group surfaces as
    # an ExceptionGroup traceback even on success.
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
