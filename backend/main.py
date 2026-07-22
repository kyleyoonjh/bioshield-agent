"""FastAPI entrypoint for AiRemedy-Agent."""

from __future__ import annotations

import logging
import logging.handlers
import os
import time

from dotenv import load_dotenv

# override=False (the default) so a REAL environment variable beats the .env file.
# It used to be True, which inverts the precedence every deployment platform
# assumes: whatever Cloud Run / Kakao sets in the environment would be silently
# overwritten by whatever a stale .env happened to say. The container excludes .env
# (see .dockerignore) so production was not actually affected — but locally it made
# the setting untestable: MCP_AUTH_REQUIRED=true on the command line was discarded
# in favour of .env's `false`, and the auth guard appeared to be broken when it was
# simply never switched on.
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.drug_discovery_router import router as drug_discovery_router
from api.playground_router import router as playground_router
from auth.router import router as oauth_router
from mcp_server import mcp, mcp_app


# File-based logging, in addition to console — every logger.info/warning/
# error/exception call anywhere in the app (already used throughout
# services/api) previously only went to the console, which meant the only
# way to review what happened after the fact was to have kept a live
# terminal/task buffer open across the whole session. RotatingFileHandler
# caps growth (10MB x 5 backups) instead of one unbounded file.
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
    force=True,  # uvicorn's own dictConfig (or another import) may have
                 # already installed root handlers before this line runs;
                 # without force=True, basicConfig() silently no-ops and the
                 # FileHandler never gets attached (confirmed empirically —
                 # the log file stayed empty despite requests being served).
)
logger = logging.getLogger("openbioshield")
logger.info("Logging initialized | file=%s", _LOG_FILE)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="AiRemedy-Agent API",
    description="Scientific AI Agent — MCP-native drug-discovery & mRNA-vaccine design platform",
    version="3.0.0",
    lifespan=_lifespan,
)

# ── OAuth 2.1 Bearer auth guard on /mcp paths ─────────────────────────────────

# Whether to enforce OAuth Bearer auth on /mcp. Defaults to OFF: this server is
# registered with Kakao PlayMCP without OAuth, and with the guard on an
# unauthenticated tools/list gets 401 so the client registers ZERO tools — the
# exact "등록된 Tool이 없습니다 / Failed" symptom. Set MCP_AUTH_REQUIRED=true to
# re-enable the Bearer guard (e.g. when fronting /mcp with real OAuth).
_MCP_AUTH_REQUIRED = os.getenv("MCP_AUTH_REQUIRED", "false").lower() not in ("false", "0", "no")
logger.info("MCP auth guard on /mcp: %s", "enforced" if _MCP_AUTH_REQUIRED else "DISABLED (MCP_AUTH_REQUIRED=false)")


class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _MCP_AUTH_REQUIRED and request.url.path.startswith("/mcp"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="AiRemedy MCP"'},
                )
            token = auth[7:].strip()
            from auth import validate_token
            if validate_token(token) is None:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )
        return await call_next(request)


# Middleware order: last-added = outermost (runs first on request).
# Add MCPAuthMiddleware first so CORS (added last) is outermost and handles preflight.
app.add_middleware(MCPAuthMiddleware)

# Real reported gap: the Vercel-deployed frontend (https://bioshield-agent
# .vercel.app) couldn't reach this Cloud Run backend — its origin was never
# in the allow-list, so the browser blocked every request before it even
# reached FastAPI. Included in the default fallback (not just requiring the
# env var) so it works even if BACKEND_CORS_ORIGINS isn't set on the Cloud
# Run service itself; still override via the env var if the Vercel URL
# ever changes (e.g. a custom domain).
cors_origins = os.getenv(
    "BACKEND_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,https://bioshield-agent.vercel.app",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth 2.1 endpoints at root level (required by MCP Authorization spec)
app.include_router(oauth_router)
app.include_router(drug_discovery_router)
app.include_router(playground_router)  # local KakaoTalk-style MCP chat tester at /playground
# Graft the MCP streamable route (an exact /mcp Route, see mcp_server.py) onto
# this app instead of app.mount("/mcp", ...). A mount makes /mcp 307-redirect
# to /mcp/ — and behind Kakao PlayMCP's TLS gateway that redirect Location
# downgrades to http and breaks the client, so PlayMCP registers zero tools.
# The session manager lifespan is already run in _lifespan above.
app.router.routes.extend(mcp_app.routes)


@app.on_event("startup")
async def _log_routes():
    routes = [f"  {m:6s} {r.path}" for r in app.routes if hasattr(r, "methods") for m in r.methods]
    logger.info("[startup] 등록된 라우트 %d개:\n%s", len(routes), "\n".join(sorted(routes)))


@app.exception_handler(404)
async def _not_found_handler(request: Request, exc: Exception):
    from fastapi.exceptions import HTTPException as FastAPIHTTPException
    if isinstance(exc, FastAPIHTTPException) and exc.detail and exc.detail != "Not Found":
        detail = exc.detail
    else:
        logger.warning("[404] %s %s", request.method, request.url.path)
        detail = f"Not Found: {request.method} {request.url.path}"
    return JSONResponse(status_code=404, content={"detail": detail})


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":        "AiRemedy-Agent API",
        "version":        "3.0.0",
        "docs":           "/docs",
        "health":         "/api/v1/health",
        "drug_discovery": "/api/drug-discovery",
        "mcp":            "/mcp",
        "oauth_metadata": "/.well-known/oauth-authorization-server",
        "oauth_register": "/register",
        "oauth_authorize":"/authorize",
        "oauth_token":    "/token",
    }


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "service": "AiRemedy-Agent v3"}
