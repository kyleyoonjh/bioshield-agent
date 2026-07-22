"""
Deploy preflight — the checks that only fail on the DEPLOYED container.

Everything else in scripts/ tests the running server on this machine. That is not
where the product runs. Kakao runs the Docker image: Linux, no virtualenv, a
different filesystem layout, and whatever the Dockerfile actually copied. Bugs
that live in that gap are invisible to every other harness here, and they are the
ones that reach users.

The bug that motivated this file: receptor_prep_engine hardcoded
`.venv/Scripts/mk_prepare_receptor.exe` as its default. That path exists on this
Windows dev box and cannot exist in the container (pip installs console scripts to
/usr/local/bin there; no .venv is ever created). So on the deployed server the
receptor prep raised FileNotFoundError into a broad `except Exception`, returned
{"prepared": False}, and every docking job quietly degraded to the heuristic
scorer with no receptor and no binding-pocket analysis. It never crashed. It never
logged an error the tests looked at. It just silently stopped doing the science.

Run this before every deploy. It does not need Docker.
"""
from __future__ import annotations

import ast
import io
import os
import re
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BACKEND, "..")


def _read(path: str) -> str:
    with io.open(path, encoding="utf-8") as f:
        return f.read()


_WINDOWS_ISH = re.compile(r"\.venv|Scripts|\.exe")


def check_no_windows_only_paths() -> list[str]:
    """
    A Windows-only path is a bug only when NOTHING resolves it at runtime. Inside a
    resolver — one that asks os.getenv() and shutil.which() first — the same string
    is the correct last-resort fallback for the dev machine. So judge the enclosing
    function, not the line: flag a Windows-ish path literal only if its function
    never consults the environment or PATH. (A line-level check flagged the fix
    itself, plus every "vina.exe exited…" error message, which is how a preflight
    trains people to ignore it.)
    """
    problems = []
    for dirpath, _, files in os.walk(os.path.join(BACKEND, "services")):
        for name in sorted(f for f in files if f.endswith(".py")):
            path = os.path.join(dirpath, name)
            tree = ast.parse(_read(path))

            for func in [n for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                body = ast.unparse(func)
                if "shutil.which" in body or "getenv" in body or "os.environ" in body:
                    continue  # it resolves at runtime — this is the fixed shape
                docstring = ast.get_docstring(func)
                for node in ast.walk(func):
                    if isinstance(node, ast.Raise):
                        continue  # error text, not a path
                    if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                            and node.value != docstring  # prose about a path is not a path
                            and "\n" not in node.value
                            and _WINDOWS_ISH.search(node.value)
                            and ("/" in node.value or "\\" in node.value
                                 or node.value.endswith(".exe"))):
                        problems.append(f"{name}:{node.lineno}  {func.name}() → {node.value!r}")

            # Module-level literals (constants assigned outside any function) have no
            # resolver at all by definition.
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                src = ast.unparse(node)
                if _WINDOWS_ISH.search(src) and "getenv" not in src and "shutil.which" not in src:
                    problems.append(f"{name}:{node.lineno}  모듈 상수 → {src[:80]}")
    return problems


def check_binaries_resolve_on_linux() -> list[str]:
    """The two external binaries must be found via PATH in the container.

    Checked STATICALLY, by parsing the source — deliberately not by importing the
    modules. Importing docking_engine drags in RDKit, so an import-based check only
    runs where the full scientific stack is already installed. That is exactly the
    machine this preflight is least needed on, and it made the check unrunnable on a
    bare CI runner (it failed on ImportError, which looks like a finding but is not).
    A preflight that cannot run before the build is not a preflight.
    """
    problems = []
    for label, filename, fn in [
        ("mk_prepare_receptor", "receptor_prep_engine.py", "_find_mk_prepare_receptor"),
        ("vina", "docking_engine.py", "_find_vina"),
    ]:
        path = os.path.join(BACKEND, "services", filename)
        if not os.path.isfile(path):
            problems.append(f"{label}: {filename} 이 없습니다")
            continue
        src = _read(path)
        tree = ast.parse(src)
        finder = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == fn), None)
        if finder is None:
            problems.append(f"{label}: PATH를 탐색하는 {fn}()가 없습니다 — 컨테이너에서 못 찾습니다")
            continue
        if "shutil.which" not in ast.unparse(finder):
            problems.append(f"{label}: {fn}()에 shutil.which() 탐색이 없습니다")
    return problems


def check_dockerfile_copies_what_the_code_reads() -> list[str]:
    """The sample VCF/BAM the demo auto-runs on must actually be in the image."""
    problems = []
    for rel in ["backend/sample/NSCLC_variants.vcf", "backend/sample/NSCLC.bam"]:
        if not os.path.isfile(os.path.join(ROOT, rel)):
            problems.append(f"{rel} 이 저장소에 없습니다 — 내장 데모가 배포본에서 실패합니다")

    dockerignore = os.path.join(ROOT, ".dockerignore")
    if os.path.isfile(dockerignore):
        for line in _read(dockerignore).splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "sample" in line:
                problems.append(f".dockerignore가 sample을 제외합니다: {line!r}")
    return problems


def check_tool_descriptions() -> list[str]:
    """PlayMCP rejects a description over 1000 chars. The tools are the API."""
    problems = []
    tree = ast.parse(_read(os.path.join(BACKEND, "mcp_server.py")))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        if not any("mcp" in ast.unparse(d) and "tool" in ast.unparse(d)
                   for d in node.decorator_list):
            continue
        count += 1
        doc = ast.get_docstring(node) or ""
        if not doc:
            problems.append(f"{node.name}: 설명이 없습니다")
        elif len(doc) > 1000:
            problems.append(f"{node.name}: 설명 {len(doc)}자 (1000자 초과)")
    if count > 20:
        problems.append(f"툴이 {count}개 — PlayMCP 상한은 20개")
    if count == 0:
        problems.append("툴이 하나도 등록되지 않았습니다")
    return problems


def check_chaos_hooks_are_off() -> list[str]:
    """A chaos env var left set in a deploy would sabotage production."""
    problems = []
    for var in ("HTTP_CHAOS_DELAY_MS", "HTTP_CHAOS_FAIL_RATE"):
        if os.getenv(var):
            problems.append(f"{var}={os.getenv(var)} 이 설정되어 있습니다 — 배포 전 제거하세요")
    return problems


CHECKS = [
    ("윈도우 전용 경로 하드코딩", check_no_windows_only_paths),
    ("외부 바이너리가 리눅스 PATH에서 해석되는가", check_binaries_resolve_on_linux),
    ("Docker 이미지가 코드가 읽는 파일을 담는가", check_dockerfile_copies_what_the_code_reads),
    ("MCP 툴 설명 (개수/길이)", check_tool_descriptions),
    ("카오스 훅이 꺼져 있는가", check_chaos_hooks_are_off),
]


def main() -> None:
    failed = 0
    for label, check in CHECKS:
        problems = check()
        if problems:
            failed += 1
            print(f"✗ {label}")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"✓ {label}")
    print()
    if failed:
        print(f"배포 차단: {failed}개 항목 실패")
    else:
        print("배포 가능: 전부 통과")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
