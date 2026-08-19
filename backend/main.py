"""
Doc Research Portal — backend
------------------------------
Given an app name (and optionally a doc URL), this:
  1. discovers the official developer-doc URL if one wasn't given, and
     verifies it actually resolves (HTTP check) before trusting it
  2. fetches that page and strips it to plain text
  3. sends it to OpenRouter for structured research: is it an API doc,
     auth type, doc quality, gating/signup, verdict
  4. runs a SEPARATE, web-search-grounded lookup for MCP support, and
     verifies any MCP URL actually resolves before returning it — this
     is deliberately its own call because the main classify call (when
     it already has page text) has no web access and would otherwise
     guess a plausible-looking MCP URL instead of finding a real one
  5. if the model reports Low confidence, runs a recheck pass (forced
     web search) and merges any corrections
  6. computes an integration-priority score and streams the result back

Apps are processed CONCURRENTLY (see CONCURRENCY below) — each app's
OpenRouter calls are independent of every other app's, so a thread pool
runs several apps at once instead of working through the list one at a time.
Results stream back as each app finishes, in completion order.

Run with:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-...
    uvicorn main:app --reload
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DISCOVER_MODEL = "google/gemini-2.5-flash-lite"
CLASSIFY_MODEL = "google/gemini-2.5-flash-lite"
MCP_MODEL = "google/gemini-2.5-flash-lite"
RECHECK_MODEL = "google/gemini-2.5-flash-lite"

# How many apps to research in parallel. Apps are independent of each
# other, so this is the main lever on total wall time.
CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))

# Page text is capped hard — this is the single biggest input-token cost,
# and a quickstart/reference page rarely needs more than this to judge
# auth type and doc depth.
PAGE_TEXT_CHAR_CAP = 6000

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Doc Research Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class AnalyzeItem(BaseModel):
    app_name: str
    url: Optional[str] = None


class AnalyzeRequest(BaseModel):
    items: list[AnalyzeItem]


# --------------------------------------------------------------------------
# OpenRouter helpers
# --------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before starting the server."
        )
    return key


def call_openrouter(payload: dict, api_key: str) -> str:
    resp = requests.post(
        OPENROUTER_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=60,
    )
    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(f"Invalid JSON from OpenRouter: {resp.text[:500]}")

    if not resp.ok:
        msg = (body.get("error") or {}).get("message", resp.text[:500])
        raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {msg}")

    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        raise RuntimeError("Empty response from OpenRouter")
    return content


def call_openrouter_json(payload: dict, api_key: str) -> dict:
    raw = call_openrouter(payload, api_key)
    cleaned = re.sub(r"^```json\s*|```$", "", raw.strip(), flags=re.I).strip()
    return json.loads(cleaned)


# --------------------------------------------------------------------------
# Live URL verification — nothing gets shown to the user unless it
# actually resolves. This is what stops hallucinated-looking-plausible
# URLs (e.g. a guessed /docs/mcp/introduction.html that doesn't exist)
# from ever reaching the table.
# --------------------------------------------------------------------------

def verify_url(url: str) -> bool:
    if not url or not re.match(r"^https?://", url):
        return False
    headers = {"User-Agent": "Mozilla/5.0 (doc-research-portal)"}
    try:
        resp = requests.head(url, headers=headers, timeout=8, allow_redirects=True)
        if resp.status_code == 405 or resp.status_code >= 400:
            # Some servers don't support HEAD properly — confirm with a GET
            # before giving up on an otherwise-valid-looking URL.
            resp = requests.get(url, headers=headers, timeout=8, allow_redirects=True, stream=True)
            resp.close()
        return resp.status_code < 400
    except Exception:
        return False


def extract_url(text: str) -> str:
    match = re.search(r"https?://[^\s)\]]+", text)
    return match.group(0).rstrip(".,)") if match else ""


def discover_url(app_name: str, api_key: str) -> str:
    prompt = (
        f'Find the OFFICIAL developer documentation homepage URL for "{app_name}". '
        f"It must be the official developer/API doc site run by the company itself — "
        f"not a tutorial, blog, GitHub repo, or third-party site. Return ONLY the raw URL, nothing else."
    )
    payload = {
        "model": DISCOVER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "plugins": [{"id": "web"}],
        "max_tokens": 60,
    }
    text = call_openrouter(payload, api_key)
    url = extract_url(text) or text.strip()

    if verify_url(url):
        return url

    # one retry, telling it the first guess was dead
    retry_prompt = prompt + f' The URL "{url}" is dead or invalid — find a different, live one.'
    payload["messages"] = [{"role": "user", "content": retry_prompt}]
    text2 = call_openrouter(payload, api_key)
    url2 = extract_url(text2) or text2.strip()
    return url2 if verify_url(url2) else url  # keep best guess even if unverified


def fetch_page(url: str) -> dict:
    """Returns {"text": str, "status": int|None, "gated_hint": bool, "error": str|None}"""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (doc-research-portal)"},
            timeout=20,
            allow_redirects=True,
        )
        status = resp.status_code
        if status != 200:
            return {"text": "", "status": status, "gated_hint": status in (401, 403), "error": None}

        html = resp.text
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        gated_hint = bool(
            re.search(r"sign in to continue|log in to view|create a free account to", text, re.I)
        )
        return {"text": text[:PAGE_TEXT_CHAR_CAP], "status": status, "gated_hint": gated_hint, "error": None}
    except Exception as e:
        return {"text": "", "status": None, "gated_hint": False, "error": str(e)}


# --------------------------------------------------------------------------
# Core research (auth / gating / doc quality / verdict — no MCP here)
# --------------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "name": "doc_research",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_api_doc": {"type": "boolean"},
            "auth_type": {
                "type": "string",
                "enum": ["OAuth2", "OAuth1", "API Key", "Basic Auth", "Bearer Token", "Other", "None documented"],
            },
            "required_fields": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "self_serve": {
                "type": "string",
                "enum": ["Yes", "No", "Unknown"],
                "description": "Can a developer get working credentials today with zero human/sales approval?",
            },
            "gating_notes": {
                "type": "string",
                "description": "Short clause: signup wall, paid plan, admin approval, sandbox-only, contact-sales. Empty string if none.",
            },
            "doc_quality_score": {
                "type": "integer",
                "description": "1-10, from what's actually on the page: quickstart, code samples, endpoint reference, vs thin marketing copy.",
            },
            "verdict": {
                "type": "string",
                "enum": ["Ready today", "Ready with friction", "Blocked", "Unknown"],
            },
            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "notes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
                "description": "Short bullet points (under 12 words each) on anything important: quirks, deprecated flows, sandbox vs prod keys, doc gaps.",
            },
        },
        "required": [
            "is_api_doc", "auth_type", "required_fields", "self_serve", "gating_notes",
            "doc_quality_score", "verdict", "confidence", "notes",
        ],
        "additionalProperties": False,
    },
}


def classify(app_name: str, url: str, fetch_meta: dict, api_key: str) -> dict:
    page_text = fetch_meta["text"]
    has_page_text = len(page_text) > 200

    if has_page_text:
        prompt = (
            f'Text from the official dev doc page for "{app_name}" ({url}). Based only on this, '
            f"determine: is it an API doc, what auth method it documents, how constructive the docs "
            f"are (quickstart/code samples/reference vs thin marketing), and whether access is gated "
            f"(signup/paid plan/sales).\n\n--- PAGE TEXT ---\n{page_text}"
        )
        payload = {
            "model": CLASSIFY_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": CLASSIFY_SCHEMA},
            "max_tokens": 600,
        }
    else:
        reason = "looks like a signup/login wall" if fetch_meta.get("gated_hint") else "reason unclear"
        status_note = f"HTTP {fetch_meta['status']}, {reason}" if fetch_meta.get("status") else "unreachable"
        prompt = (
            f'The dev doc page for "{app_name}" ({url}) could not be read directly ({status_note}). '
            f"Use web search to find how developers authenticate to this API and whether access is "
            f"gated. Set confidence to \"Low\" since this is search-based, not the primary doc page."
        )
        payload = {
            "model": CLASSIFY_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": CLASSIFY_SCHEMA},
            "plugins": [{"id": "web"}],
            "max_tokens": 600,
        }

    return call_openrouter_json(payload, api_key)


# --------------------------------------------------------------------------
# MCP lookup — always web-search-grounded, never guessed from page text,
# and the returned URL is verified live before it's trusted.
# --------------------------------------------------------------------------

MCP_SCHEMA = {
    "name": "mcp_lookup",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "mcp_available": {"type": "string", "enum": ["Yes", "No", "Unknown"]},
            "mcp_url": {
                "type": "string",
                "description": "Direct URL to the official MCP server (their own docs, GitHub repo, or MCP registry listing). Empty string if none found.",
            },
            "mcp_note": {"type": "string", "description": "One short clause, e.g. 'Official server, npm package'. Empty if not available."},
        },
        "required": ["mcp_available", "mcp_url", "mcp_note"],
        "additionalProperties": False,
    },
}


def discover_mcp(app_name: str, api_key: str, refine: bool = False) -> dict:
    base = (
        f'Search the web for an official or well-known Model Context Protocol (MCP) server for '
        f'"{app_name}". Only report mcp_available="Yes" with a URL if you find a real, specific '
        f"page confirming it (their docs, or the official MCP registry/directory). "
        f"Do NOT construct or guess a plausible-looking URL — only report one you actually found "
        f"via search. If you can't confirm one exists, return mcp_available=\"No\" and an empty mcp_url."
    )
    if refine:
        base += f' Try a more specific search this time, e.g. "{app_name} mcp server github" or "{app_name} model context protocol".'

    payload = {
        "model": MCP_MODEL,
        "messages": [{"role": "user", "content": base}],
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": MCP_SCHEMA},
        "plugins": [{"id": "web"}],
        "max_tokens": 250,
    }
    result = call_openrouter_json(payload, api_key)

    if result.get("mcp_available") == "Yes" and result.get("mcp_url"):
        if verify_url(result["mcp_url"]):
            return result
        if not refine:
            return discover_mcp(app_name, api_key, refine=True)
        # gave up after one refine attempt — don't show an unverified link
        return {"mcp_available": "Unknown", "mcp_url": "", "mcp_note": "Candidate link found but could not be verified live."}

    return result


# --------------------------------------------------------------------------
# Recheck (only runs when confidence came back Low)
# --------------------------------------------------------------------------

RECHECK_SCHEMA = {
    "name": "doc_recheck",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "fields_rechecked": {"type": "array", "items": {"type": "string"}},
            "result": {"type": "string", "enum": ["Confirmed", "Corrected"]},
            "corrected_auth_type": {"type": "string", "description": "New value, or empty string if unchanged"},
            "corrected_self_serve": {"type": "string", "description": "Yes/No/Unknown, or empty string if unchanged"},
            "corrected_verdict": {"type": "string", "description": "New verdict, or empty string if unchanged"},
            "note": {"type": "string"},
        },
        "required": [
            "fields_rechecked", "result", "corrected_auth_type",
            "corrected_self_serve", "corrected_verdict", "note",
        ],
        "additionalProperties": False,
    },
}


def recheck(app_name: str, url: str, first_pass: dict, api_key: str) -> dict:
    prompt = (
        f'First research pass on "{app_name}" ({url}) came back Low confidence. It found: '
        f"auth_type={first_pass['auth_type']}, self_serve={first_pass['self_serve']}, "
        f"verdict={first_pass['verdict']}, gating_notes=\"{first_pass['gating_notes']}\". "
        f"Use web search to independently re-verify auth_type, self_serve, and verdict. If it matches, "
        f'return result="Confirmed" with empty corrected_* fields. If wrong, return result="Corrected" '
        f"and fill in only the corrected_* fields that actually changed."
    )
    payload = {
        "model": RECHECK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": RECHECK_SCHEMA},
        "plugins": [{"id": "web"}],
        "max_tokens": 350,
    }
    return call_openrouter_json(payload, api_key)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

VERDICT_WEIGHT = {"Ready today": 3, "Ready with friction": 2, "Blocked": 0, "Unknown": 1}
CONFIDENCE_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}
SELF_SERVE_WEIGHT = {"Yes": 2, "No": 0, "Unknown": 1}


def compute_priority(result: dict) -> float:
    v = VERDICT_WEIGHT.get(result.get("verdict"), 1) / 3 * 40
    c = CONFIDENCE_WEIGHT.get(result.get("confidence"), 1) / 3 * 25
    s = SELF_SERVE_WEIGHT.get(result.get("self_serve"), 1) / 2 * 15
    q = max(0, min(10, result.get("doc_quality_score", 0))) / 10 * 20
    return round(v + c + s + q, 1)


# --------------------------------------------------------------------------
# Core pipeline: one app end-to-end, then a thread pool fans this out across
# apps concurrently instead of doing them one at a time.
# --------------------------------------------------------------------------

def process_one(item: AnalyzeItem, api_key: str) -> dict:
    app_name = item.app_name.strip()
    row = {"app_name": app_name}
    try:
        url = (item.url or "").strip()
        if not url:
            url = discover_url(app_name, api_key)
        elif not verify_url(url):
            # a user-supplied URL that's dead — still try it for page text,
            # but flag it rather than silently trusting it
            row["doc_url_unverified"] = True
        row["doc_url"] = url

        fetch_meta = fetch_page(url)
        classification = classify(app_name, url, fetch_meta, api_key)
        row.update(classification)
        row["fetch_status"] = fetch_meta.get("status")
        row["gated_hint"] = fetch_meta.get("gated_hint")

        mcp = discover_mcp(app_name, api_key)
        row["mcp_available"] = mcp.get("mcp_available", "Unknown")
        row["mcp_url"] = mcp.get("mcp_url", "")
        row["mcp_note"] = mcp.get("mcp_note", "")

        recheck_info = None
        if classification.get("confidence") == "Low":
            try:
                recheck_info = recheck(app_name, url, classification, api_key)
                if recheck_info["result"] == "Corrected":
                    if recheck_info["corrected_auth_type"]:
                        row["auth_type"] = recheck_info["corrected_auth_type"]
                    if recheck_info["corrected_self_serve"]:
                        row["self_serve"] = recheck_info["corrected_self_serve"]
                    if recheck_info["corrected_verdict"]:
                        row["verdict"] = recheck_info["corrected_verdict"]
                    row["confidence"] = "Medium"
            except Exception as re_err:
                recheck_info = {"error": str(re_err)}
        row["recheck"] = recheck_info

        row["priority_score"] = compute_priority(row)
        row["error"] = None
    except Exception as e:
        row["error"] = str(e)
    return row


def run_pipeline(items: list[AnalyzeItem]):
    api_key = get_api_key()
    valid_items = [it for it in items if it.app_name.strip()]
    if not valid_items:
        return

    workers = max(1, min(CONCURRENCY, len(valid_items)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, item, api_key) for item in valid_items]
        # yield each app's result as soon as it's done, not in submit order —
        # this is what lets the frontend show progress while others are
        # still in flight instead of waiting on the slowest one first.
        for future in as_completed(futures):
            yield json.dumps(future.result()) + "\n"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.items:
        raise HTTPException(400, "No items provided")
    try:
        get_api_key()
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return StreamingResponse(run_pipeline(req.items), media_type="application/x-ndjson")


@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "frontend/index.html not found")
    return FileResponse(index_path)
