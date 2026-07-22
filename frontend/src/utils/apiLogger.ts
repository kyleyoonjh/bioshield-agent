// Centralized fetch logging — added because Vercel's own dashboard doesn't
// expose client-side console output, so API failures in production were
// invisible unless the user happened to have DevTools open at the right
// moment with nothing already scrolled past. Every call site below wraps
// the same native `fetch` and logs request/response/error with a shared,
// greppable "[API #n]" prefix — transparent pass-through otherwise (same
// Response object, same thrown errors), so no caller's existing
// try/catch/.ok handling needs to change.
let _seq = 0;

// Real reported gap: every `/api/...` call in this app is a relative path,
// which works in dev (Vite's own proxy in vite.config.ts forwards /api to
// localhost:8001) but resolves to the VERCEL frontend's own origin in
// production — Vercel only serves the static build, it isn't the FastAPI
// backend (deployed separately on Cloud Run), so every request 404'd
// silently. VITE_API_BASE_URL (a Vercel project env var, baked in at build
// time) is preferred when set; the Cloud Run URL below is a working
// fallback so this functions even without configuring that env var,
// overridable if the backend URL ever changes.
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ||
  (import.meta.env.PROD ? "https://bioshield2-agent-git-665023959219.asia-northeast3.run.app" : "");

function resolveUrl(url: string): string {
  return url.startsWith("/api/") ? `${API_BASE_URL}${url}` : url;
}

function summarizeBody(body: BodyInit | null | undefined): unknown {
  if (body == null) return undefined;
  if (typeof body === "string") {
    try {
      return JSON.parse(body);
    } catch {
      return body.length > 500 ? `${body.slice(0, 500)}…` : body;
    }
  }
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    const entries: Record<string, string> = {};
    body.forEach((value, key) => {
      entries[key] =
        typeof File !== "undefined" && value instanceof File
          ? `File(${value.name}, ${value.size}b)`
          : String(value).slice(0, 200);
    });
    return entries;
  }
  return "[non-string/non-FormData body]";
}

export async function logFetch(url: string, init?: RequestInit): Promise<Response> {
  const id = ++_seq;
  const method = init?.method ?? "GET";
  const resolvedUrl = resolveUrl(url);
  const startedAt = performance.now();
  console.log(`[API #${id}] -> ${method} ${resolvedUrl}`, summarizeBody(init?.body));

  let res: Response;
  try {
    res = await fetch(resolvedUrl, init);
  } catch (err) {
    const elapsedMs = Math.round(performance.now() - startedAt);
    console.error(`[API #${id}] <- ${method} ${resolvedUrl} NETWORK ERROR after ${elapsedMs}ms`, err);
    throw err;
  }

  const elapsedMs = Math.round(performance.now() - startedAt);
  if (!res.ok) {
    console.error(`[API #${id}] <- ${method} ${resolvedUrl} FAILED status=${res.status} (${elapsedMs}ms)`);
    // Clone before reading so the actual caller can still consume the body
    // (res.json()/.text()) exactly as before — a Response body can only be
    // read once.
    res
      .clone()
      .text()
      .then(text => console.error(`[API #${id}] error body:`, text.slice(0, 2000)))
      .catch(() => {});
  } else {
    console.log(`[API #${id}] <- ${method} ${resolvedUrl} ${res.status} (${elapsedMs}ms)`);
  }
  return res;
}
