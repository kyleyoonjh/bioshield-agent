"""
OAuth 2.1 Authorization Server — in-memory implementation.
RFC 7591 (Dynamic Client Registration) + RFC 8414 (Server Metadata) + PKCE (RFC 7636).
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field

# ── TTLs ─────────────────────────────────────────────────────────────────────

ACCESS_TOKEN_TTL  = 3600          # 1 hour
AUTH_CODE_TTL     = 600           # 10 minutes
REFRESH_TOKEN_TTL = 86400 * 30   # 30 days

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class _Client:
    client_id:                  str
    client_secret:              str
    redirect_uris:              list[str]
    client_name:                str
    grant_types:                list[str]
    response_types:             list[str]
    token_endpoint_auth_method: str
    registered_at:              float = field(default_factory=time.time)


@dataclass
class _AuthCode:
    code:                   str
    client_id:              str
    redirect_uri:           str
    code_challenge:         str
    code_challenge_method:  str
    scope:                  str
    expires_at:             float


@dataclass
class _Token:
    access_token:   str
    token_type:     str
    expires_at:     float
    scope:          str
    client_id:      str
    refresh_token:  str | None = None


# ── In-memory stores ──────────────────────────────────────────────────────────

_CLIENTS:        dict[str, _Client]  = {}
_AUTH_CODES:     dict[str, _AuthCode] = {}
_TOKENS:         dict[str, _Token]   = {}
_REFRESH_TOKENS: dict[str, str]      = {}  # refresh_token → access_token


# ── PKCE (RFC 7636) ───────────────────────────────────────────────────────────

def verify_pkce(code_verifier: str, code_challenge: str, method: str = "S256") -> bool:
    if method == "S256":
        digest   = hashlib.sha256(code_verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return secrets.compare_digest(expected, code_challenge)
    # plain — not recommended but spec-compliant
    return secrets.compare_digest(code_verifier, code_challenge)


# ── Client registration (RFC 7591) ────────────────────────────────────────────

def register_client(
    redirect_uris:              list[str],
    client_name:                str = "Unknown Client",
    grant_types:                list[str] | None = None,
    response_types:             list[str] | None = None,
    token_endpoint_auth_method: str = "client_secret_basic",
) -> dict:
    client_id     = secrets.token_urlsafe(16)
    client_secret = secrets.token_urlsafe(32)
    client = _Client(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uris=redirect_uris,
        client_name=client_name,
        grant_types=grant_types or ["authorization_code", "refresh_token"],
        response_types=response_types or ["code"],
        token_endpoint_auth_method=token_endpoint_auth_method,
    )
    _CLIENTS[client_id] = client
    return {
        "client_id":                    client_id,
        "client_secret":                client_secret,
        "redirect_uris":                redirect_uris,
        "client_name":                  client_name,
        "grant_types":                  client.grant_types,
        "response_types":               client.response_types,
        "token_endpoint_auth_method":   token_endpoint_auth_method,
        "client_id_issued_at":          int(client.registered_at),
        "client_secret_expires_at":     0,  # never expires
    }


def get_client(client_id: str) -> _Client | None:
    return _CLIENTS.get(client_id)


# ── Authorization code ────────────────────────────────────────────────────────

def create_auth_code(
    client_id:              str,
    redirect_uri:           str,
    code_challenge:         str,
    code_challenge_method:  str = "S256",
    scope:                  str = "mcp",
) -> str:
    code = secrets.token_urlsafe(32)
    _AUTH_CODES[code] = _AuthCode(
        code=code,
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        scope=scope,
        expires_at=time.time() + AUTH_CODE_TTL,
    )
    return code


# ── Token exchange ────────────────────────────────────────────────────────────

def exchange_code_for_token(
    code:           str,
    client_id:      str,
    redirect_uri:   str,
    code_verifier:  str,
) -> dict | None:
    auth_code = _AUTH_CODES.pop(code, None)
    if auth_code is None or auth_code.expires_at < time.time():
        return None
    if auth_code.client_id != client_id or auth_code.redirect_uri != redirect_uri:
        return None
    if not verify_pkce(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
        return None

    access_token  = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(32)
    _TOKENS[access_token] = _Token(
        access_token=access_token,
        token_type="Bearer",
        expires_at=time.time() + ACCESS_TOKEN_TTL,
        scope=auth_code.scope,
        client_id=client_id,
        refresh_token=refresh_token,
    )
    _REFRESH_TOKENS[refresh_token] = access_token
    return {
        "access_token":  access_token,
        "token_type":    "Bearer",
        "expires_in":    ACCESS_TOKEN_TTL,
        "refresh_token": refresh_token,
        "scope":         auth_code.scope,
    }


def refresh_access_token(refresh_token: str, client_id: str) -> dict | None:
    old_access = _REFRESH_TOKENS.pop(refresh_token, None)
    if old_access is None:
        return None
    old_token = _TOKENS.pop(old_access, None)
    if old_token is None or old_token.client_id != client_id:
        return None

    access_token      = secrets.token_urlsafe(32)
    new_refresh_token = secrets.token_urlsafe(32)
    _TOKENS[access_token] = _Token(
        access_token=access_token,
        token_type="Bearer",
        expires_at=time.time() + ACCESS_TOKEN_TTL,
        scope=old_token.scope,
        client_id=client_id,
        refresh_token=new_refresh_token,
    )
    _REFRESH_TOKENS[new_refresh_token] = access_token
    return {
        "access_token":  access_token,
        "token_type":    "Bearer",
        "expires_in":    ACCESS_TOKEN_TTL,
        "refresh_token": new_refresh_token,
        "scope":         old_token.scope,
    }


# ── Token validation ──────────────────────────────────────────────────────────

def validate_token(token: str) -> dict | None:
    t = _TOKENS.get(token)
    if t is None:
        return None
    if t.expires_at < time.time():
        _TOKENS.pop(token, None)
        return None
    return {"client_id": t.client_id, "scope": t.scope, "expires_at": t.expires_at}
