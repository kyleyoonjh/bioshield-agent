"""
Kakao PlayMCP simulator — drive this MCP server the way Kakao actually drives it.

scripts/mcp_probe.py already speaks the real transport, but it calls tools with
arguments *I* chose. That is not what Kakao does. Kakao hands a generic LLM the
tool list and nothing else, and the model decides — from the descriptions alone —
which tool to call, with what arguments, whether to keep polling, and when to
stop. Every bug that reached production lived in exactly that gap:

  - the model docked KRAS in the middle of a vaccine study (a tool DESCRIPTION
    told it to)
  - it gave up after three instant polls and declared a running job failed
  - it read the key "all_scored" out of a truncated result and reported it as
    the failure reason

None of those reproduce when a script supplies the arguments. They reproduce
here.

What this simulates, and what it deliberately does NOT:

  - NO system prompt of ours. The local /playground has an elaborate one with
    modality guards and polling rules; Kakao has none of it. If a behaviour only
    holds because of the playground prompt, it does not hold in production, and
    the point of this harness is to catch that. The tool descriptions are the
    only steering that exists here — which is why they carry the guards.
  - A HARD 10s timeout per tool call, cancelled client-side like Kakao's, and
    reported back to the model as a failure.
  - Result truncation. Kakao's exact cap is not published; --max-chars is an
    assumption (default 4000) and every truncation is printed loudly, because a
    result that gets cut in the wrong place is how a successful job came to be
    reported as a failure.

Usage (server running, OPENAI_API_KEY set):
    python scripts/kakao_sim.py --scenario vaccine
    python scripts/kakao_sim.py "아스피린이 신약 후보로 적합한지 분석해줘."
    python scripts/kakao_sim.py --scenario vaccine --max-chars 2000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PORT = 8001  # --port lets a second (e.g. freshly-fixed) server be tested without
             # disturbing whatever is already serving 8001
URL = f"http://127.0.0.1:{PORT}/mcp"
KAKAO_TOOL_TIMEOUT_S = 10.0
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# The three commands the app's own onboarding tells a user to type.
SCENARIOS = {
    "vaccine": [
        "환자의 암 정보를 분석해서 맞춤형 mRNA 암 백신을 설계해줘.",
        "백신 후보 논문과 임상 근거를 분석해줘.",
        "종합 연구 리포트로 만들어줘.",
    ],
    "smallmolecule": [
        "코로나 스파이크 단백질을 막는 약 후보를 찾아줘.",
        "1위 후보를 더 좋게 개선할 수 있어?",
        "종합 연구 리포트로 만들어줘.",
    ],
}

C_DIM, C_TOOL, C_WARN, C_OK, C_END = "\033[90m", "\033[36m", "\033[33m", "\033[32m", "\033[0m"


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


async def _completion(client: OpenAI, **kwargs):
    """The harness's own OpenAI key has a 30k TPM ceiling, and a few scenarios
    back-to-back will trip it. That is a limit of this test rig, not of the MCP
    server under test — Kakao brings its own model capacity — so back off and
    retry rather than let a 429 abort a run and look like a product failure."""
    from openai import RateLimitError

    delay = 3.0
    for attempt in range(5):
        try:
            return await asyncio.to_thread(client.chat.completions.create, **kwargs)
        except RateLimitError:
            if attempt == 4:
                raise
            print(f"{C_DIM}    (하네스 OpenAI 429 — {delay:.0f}초 후 재시도){C_END}")
            await asyncio.sleep(delay)
            delay *= 2


async def _call_like_kakao(session: ClientSession, name: str, args: dict,
                           max_chars: int) -> tuple[str, bool, float]:
    """One tool call under Kakao's rules: hard timeout, then truncation."""
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(session.call_tool(name, args),
                                        timeout=KAKAO_TOOL_TIMEOUT_S)
        body, is_error = _text(result), bool(result.isError)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        print(f"{C_WARN}    ⚠ {KAKAO_TOOL_TIMEOUT_S:.0f}초 타임아웃 — 카카오라면 여기서 호출이 죽습니다{C_END}")
        return f"Tool call timed out after {KAKAO_TOOL_TIMEOUT_S:.0f}s.", True, elapsed

    elapsed = time.monotonic() - t0
    if len(body) > max_chars:
        print(f"{C_WARN}    ⚠ 응답 {len(body)}자 → {max_chars}자로 잘림 "
              f"(뒤 {len(body) - max_chars}자 유실){C_END}")
        body = body[:max_chars]
    return body, is_error, elapsed


async def run(messages: list[str], max_chars: int) -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            } for t in listed.tools]
            print(f"{C_DIM}툴 {len(tools)}개 로드 — 시스템 프롬프트 없음 (카카오와 동일: 툴 설명만이 유일한 안내){C_END}\n")

            history: list[dict] = []
            for turn, user_msg in enumerate(messages, 1):
                print(f"\n{'=' * 74}\n👤 {user_msg}\n{'=' * 74}")
                history.append({"role": "user", "content": user_msg})

                for _ in range(25):  # generous: a real poll loop needs many hops
                    resp = await _completion(
                        client, model=MODEL, messages=history, tools=tools, temperature=0.2)
                    msg = resp.choices[0].message
                    history.append(msg.model_dump(exclude_none=True))

                    if not msg.tool_calls:
                        print(f"\n🤖 {msg.content}\n")
                        break

                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments or "{}")
                        body, is_error, secs = await _call_like_kakao(
                            session, tc.function.name, args, max_chars)
                        flag = f"{C_WARN}실패{C_END}" if is_error else f"{C_OK}성공{C_END}"
                        print(f"  {C_TOOL}🔧 {tc.function.name}{C_END}({json.dumps(args, ensure_ascii=False)[:70]}) "
                              f"→ {flag} {secs:.1f}s {len(body)}자")
                        history.append({"role": "tool", "tool_call_id": tc.id, "content": body})
                else:
                    print(f"{C_WARN}  ⚠ 25홉 안에 끝나지 않음{C_END}")


def main() -> None:
    global URL
    p = argparse.ArgumentParser()
    p.add_argument("message", nargs="*", help="한 번만 보낼 메시지")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), help="정해진 다단계 시나리오")
    p.add_argument("--max-chars", type=int, default=4000,
                   help="툴 결과 잘림 한도 (카카오 실제 값은 비공개 — 가정치)")
    p.add_argument("--port", type=int, default=PORT, help="테스트할 로컬 서버 포트")
    p.add_argument("--url", default="", help="원격 MCP 엔드포인트 (예: 배포된 PlayMCP URL)")
    args = p.parse_args()
    URL = args.url or f"http://127.0.0.1:{args.port}/mcp"

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 필요합니다 (backend/.env).")

    messages = SCENARIOS[args.scenario] if args.scenario else [" ".join(args.message)]
    if not messages or not messages[0]:
        sys.exit("메시지나 --scenario 중 하나가 필요합니다.")

    asyncio.run(run(messages, args.max_chars))


if __name__ == "__main__":
    main()
