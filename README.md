# PipeRube

A small local tool that replaces the two Google Apps Script files: point it at
an app name (or a name + doc URL), and it researches the real developer docs —
auth type, MCP support, how constructive the docs are, whether access is
gated, and a buildability verdict — then ranks everything by an integration
priority score, the same way the CompR case study does it.

## What it does per app

1. **Discover** — if no URL is given, asks OpenRouter (with web search) for
   the official developer-doc homepage.
2. **Fetch** — pulls the real page and strips it to plain text. If the page
   can't be read (login wall, blocked, error), that's recorded and the
   research falls back to web search instead of guessing.
3. **Research** — one structured OpenRouter call returns: `is_api_doc`,
   `auth_type`, `required_fields`, `self_serve`, `gating_notes`,
   `mcp_available` + note, `doc_quality_score` (1–10) + reasoning, `verdict`
   (Ready today / Ready with friction / Blocked), `confidence`, and `notes`.
4. **Recheck** — if confidence came back **Low**, a second forced-web-search
   pass independently re-verifies auth/MCP/self-serve/verdict and reports
   `fields_rechecked`, `result` (Confirmed/Corrected), and a note — the same
   `App | Fields re-checked | Result | Note` shape from the case study.
5. **Score** — a 0–100 priority score is computed from verdict (40%),
   confidence (25%), self-serve (15%), and doc quality (20%), and results are
   sorted so the easiest wins float to the top.

## Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then edit .env with your real key
export OPENROUTER_API_KEY=sk-or-...

uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** — FastAPI serves the frontend directly, so
there's nothing separate to run or configure for CORS.

## Using it

- **Manual entry** — add rows of app name (+ optional doc URL). Leave the URL
  blank and it'll be discovered for you.
- **Import CSV** — a CSV with `app_name,url` columns (header optional, `url`
  can be blank per row).
- Hit **Run research**. Results stream in live, then auto-sort by priority
  score once the batch finishes.
- Click any row to expand gating notes, MCP notes, required fields, doc
  quality reasoning, and — for low-confidence apps — the recheck detail.
- **Export CSV** downloads the full results table.

## Speed

Apps are researched **concurrently**, not one at a time — a thread pool runs
several apps' research in flight at once (default 5, set via `CONCURRENCY` in
`.env` or the shell). Each app still makes its own 1–3 sequential OpenRouter
calls (discover → classify → recheck-if-low-confidence), but different apps
no longer wait on each other, and the artificial pacing delay between calls
was removed. A batch of 20 apps that used to take a few minutes should now
land in well under a minute at the default concurrency of 5 — raise
`CONCURRENCY` if you're not hitting OpenRouter rate limits, lower it if you
are (a `429` from OpenRouter is the signal to dial it back).

```bash
export CONCURRENCY=8   # optional, defaults to 5
```

Results stream back in **completion order**, not input order — a fast app
that needed no recheck can finish before a slow one still discovering its
URL. The UI doesn't care: it places each row by app name and re-sorts by
priority once the whole batch is done.

## Notes

- Model used for all three OpenRouter calls is `google/gemini-2.5-flash-lite`
  by default (matching your original scripts) — change `DISCOVER_MODEL` /
  `CLASSIFY_MODEL` / `RECHECK_MODEL` in `backend/main.py` if you want
  something else. Structured-output (`json_schema`) support varies by model
  and provider on OpenRouter, so check a model's page there before swapping.
- Each app costs 2–3 OpenRouter calls (discover if needed, classify, recheck
  if low-confidence). See **Speed** above for how apps are parallelized.
- This is a research aid, not a final answer — same caveat as the case study:
  verify anything you're about to build against before committing engineering
  time to it.
