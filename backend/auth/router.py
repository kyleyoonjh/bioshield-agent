"""
OAuth 2.1 endpoints — RFC 8414 (Metadata) + RFC 7591 (DCR) + Authorization Code + PKCE.
All endpoints are mounted at the root level so MCP clients can discover them.
"""
from __future__ import annotations

import os
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .oauth_server import (
    create_auth_code,
    exchange_code_for_token,
    get_client,
    refresh_access_token,
    register_client,
)

router = APIRouter(tags=["OAuth 2.1"])

_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


# ── RFC 8414 — Server Metadata ────────────────────────────────────────────────

@router.get("/.well-known/oauth-authorization-server")
async def server_metadata():
    return {
        "issuer":                                   _BASE_URL,
        "authorization_endpoint":                   f"{_BASE_URL}/authorize",
        "token_endpoint":                           f"{_BASE_URL}/token",
        "registration_endpoint":                    f"{_BASE_URL}/register",
        "response_types_supported":                 ["code"],
        "grant_types_supported":                    ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported":    ["client_secret_post", "client_secret_basic", "none"],
        "code_challenge_methods_supported":         ["S256"],
        "scopes_supported":                         ["mcp"],
        "service_documentation":                    f"{_BASE_URL}/docs",
    }


# ── RFC 7591 — Dynamic Client Registration ────────────────────────────────────

@router.post("/register")
async def dynamic_client_registration(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    redirect_uris = body.get("redirect_uris", [])
    if not redirect_uris or not isinstance(redirect_uris, list):
        raise HTTPException(status_code=400, detail="redirect_uris is required")

    result = register_client(
        redirect_uris=redirect_uris,
        client_name=body.get("client_name", "Unknown Client"),
        grant_types=body.get("grant_types"),
        response_types=body.get("response_types"),
        token_endpoint_auth_method=body.get("token_endpoint_auth_method", "client_secret_basic"),
    )
    return JSONResponse(content=result, status_code=201)


# ── Authorization endpoint — consent page ────────────────────────────────────

@router.get("/authorize", response_class=HTMLResponse)
async def authorize(
    response_type:          str = Query(...),
    client_id:              str = Query(...),
    redirect_uri:           str = Query(...),
    code_challenge:         str = Query(...),
    code_challenge_method:  str = Query(default="S256"),
    state:                  str = Query(default=""),
    scope:                  str = Query(default="mcp"),
):
    client = get_client(client_id)
    if client is None:
        raise HTTPException(status_code=400, detail="Unknown client_id")
    if response_type != "code":
        raise HTTPException(status_code=400, detail="Only response_type=code is supported")
    if redirect_uri not in client.redirect_uris:
        raise HTTPException(status_code=400, detail="Invalid redirect_uri")
    if code_challenge_method != "S256":
        raise HTTPException(status_code=400, detail="Only code_challenge_method=S256 is supported")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AiRemedy — Authorization</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh; background: #f8fafc;
    }}
    .card {{
      background: white; border-radius: 16px; padding: 40px 48px;
      max-width: 440px; width: 100%;
      box-shadow: 0 4px 32px rgba(0,0,0,.08);
    }}
    .logo {{ font-size: 2rem; margin-bottom: 12px; }}
    h1 {{ font-size: 1.5rem; color: #1e293b; font-weight: 700; margin-bottom: 6px; }}
    .subtitle {{ color: #64748b; font-size: .9rem; margin-bottom: 28px; }}
    .app-name {{ font-weight: 600; color: #1e293b; }}
    .scope-box {{
      background: #f1f5f9; border-radius: 10px; padding: 14px 18px;
      margin-bottom: 28px;
    }}
    .scope-label {{ font-size: .75rem; font-weight: 600; color: #94a3b8;
                   text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }}
    .scope-item {{ font-size: .9rem; color: #334155; display: flex; align-items: center; gap: 8px; }}
    .scope-item::before {{ content: "✓"; color: #22c55e; font-weight: 700; }}
    .buttons {{ display: flex; gap: 12px; }}
    button {{
      flex: 1; padding: 12px; border: none; border-radius: 10px;
      font-size: .95rem; cursor: pointer; font-weight: 600; transition: .15s;
    }}
    .allow {{ background: #2563eb; color: white; }}
    .allow:hover {{ background: #1d4ed8; }}
    .deny {{ background: #f1f5f9; color: #475569; }}
    .deny:hover {{ background: #e2e8f0; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🧬</div>
    <h1>AiRemedy</h1>
    <p class="subtitle">
      <span class="app-name">{client.client_name}</span> is requesting access to your account.
    </p>
    <div class="scope-box">
      <div class="scope-label">Requested permissions</div>
      <div class="scope-item">Access MCP tools (primer design, chat, memory)</div>
    </div>
    <form method="POST" action="/authorize/confirm">
      <input type="hidden" name="client_id"             value="{client_id}">
      <input type="hidden" name="redirect_uri"           value="{redirect_uri}">
      <input type="hidden" name="code_challenge"         value="{code_challenge}">
      <input type="hidden" name="code_challenge_method"  value="{code_challenge_method}">
      <input type="hidden" name="state"                  value="{state}">
      <input type="hidden" name="scope"                  value="{scope}">
      <div class="buttons">
        <button type="submit" name="decision" value="allow" class="allow">Allow</button>
        <button type="submit" name="decision" value="deny"  class="deny">Deny</button>
      </div>
    </form>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.post("/authorize/confirm")
async def authorize_confirm(
    client_id:              str = Form(...),
    redirect_uri:           str = Form(...),
    code_challenge:         str = Form(...),
    code_challenge_method:  str = Form(default="S256"),
    state:                  str = Form(default=""),
    scope:                  str = Form(default="mcp"),
    decision:               str = Form(...),
):
    if decision != "allow":
        params: dict = {"error": "access_denied", "error_description": "User denied authorization"}
        if state:
            params["state"] = state
        return RedirectResponse(url=f"{redirect_uri}?{urlencode(params)}", status_code=302)

    client = get_client(client_id)
    if client is None or redirect_uri not in client.redirect_uris:
        raise HTTPException(status_code=400, detail="Invalid client or redirect_uri")

    code = create_auth_code(
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        scope=scope,
    )
    params = {"code": code}
    if state:
        params["state"] = state
    return RedirectResponse(url=f"{redirect_uri}?{urlencode(params)}", status_code=302)


# ── Token endpoint ────────────────────────────────────────────────────────────

@router.post("/token")
async def token_endpoint(
    grant_type:     str = Form(...),
    client_id:      str = Form(...),
    client_secret:  str = Form(default=""),
    code:           str = Form(default=""),
    redirect_uri:   str = Form(default=""),
    code_verifier:  str = Form(default=""),
    refresh_token:  str = Form(default=""),
):
    client = get_client(client_id)
    if client is None:
        raise HTTPException(status_code=401, detail="Unknown client_id")

    if grant_type == "authorization_code":
        if not code or not redirect_uri or not code_verifier:
            raise HTTPException(
                status_code=400,
                detail="code, redirect_uri, and code_verifier are required",
            )
        result = exchange_code_for_token(code, client_id, redirect_uri, code_verifier)
        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid, expired, or already-used authorization code",
            )
        return JSONResponse(content=result)

    if grant_type == "refresh_token":
        if not refresh_token:
            raise HTTPException(status_code=400, detail="refresh_token is required")
        result = refresh_access_token(refresh_token, client_id)
        if result is None:
            raise HTTPException(status_code=400, detail="Invalid or expired refresh_token")
        return JSONResponse(content=result)

    raise HTTPException(status_code=400, detail=f"Unsupported grant_type: {grant_type}")
