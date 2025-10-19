# Roadax Backend

A hybrid Python + Node backend that exposes pavement design and analysis routines via a simple HTTP API. Python does the heavy scientific work; Node provides a fast, JSON-first API with run persistence and caching.

This repo is designed to make it easy for starters to:
- set up the Python computation environment,
- run the Node API server,
- call feature endpoints (Design by Type, Effective CBR, Critical Strains, Multilayer, Permissible Strain), and
- optionally run a combined “design then hydrate” pipeline.

---

## Repository structure

- `Roadax_Python_Functions/` — Python source for the computational core (design, analysis, and pipeline helpers)
  - `Design_by_type.py` — main unified design routine (fast CI/testing shortcut available)
  - `Critical_Strain_Analysis.py`, `Multilayer_Analysis.py`, `Permissible_Strain_Analysis.py`, `Effective_CBR_Calc.py` — analysis modules
  - `Edit_type_to_check.py` — hydration/reporting helpers
  - `Bridge.py` or `python_api_bridge.py` (depending on branch) — stdin/stdout JSON bridge used by Node
- `node-api/` — TypeScript Fastify API that calls into Python and returns JSON results

> Tip: For real runs, Python can be compute-heavy and long-running. For development and tests, there is a “fast mode” that returns deterministic, lightweight results so you can iterate quickly.

---

## Prerequisites

- Python 3.9+ (3.10 recommended)
- Node.js 18+ (or 20+)
- Optional: PostgreSQL (if you want persistent run storage via Prisma). The API also works without a DB if persistence is disabled or mocked in your branch.

---

## Setup (Python)

1) Create and activate a virtual environment:

```bash
cd Roadax_Python_Functions
python3 -m venv .venv
source .venv/bin/activate
```

2) Install Python packages. If a requirements file isn’t present, install the common deps directly:

```bash
pip install numpy scipy pandas matplotlib
```

- pandas and matplotlib are optional for headless/CI runs, but recommended for local exploration.
- If you see backend warnings about GUI, matplotlib will fall back to a non-interactive renderer.

3) Quick import check (optional):

```bash
python - <<'PY'
import numpy, scipy
print('Python scientific stack OK')
PY
```

---

## Setup (Node API)

```bash
cd node-api
npm install
```

Environment variables (configure via your shell or a `.env` file):
- `PYTHON_BIN` — Path to Python executable the API should use (default: `python3`)
- `BRIDGE_TIMEOUT_MS` — Max milliseconds to wait for a Python run (example: `600000` for 10 minutes)
- `ROADAX_FAST` — Set to `1` to enable fast, deterministic Python responses for development/tests
- `DATABASE_URL` — Prisma connection string if you’re using persistence (e.g., PostgreSQL)

If using the database features:

```bash
npm run prisma:generate
# For local dev migrations (if configured):
# npm run prisma:migrate
```

Start the API:

```bash
npm run dev
# or for a built start
# npm run build && npm start
```

The server will log its port on startup (commonly `http://localhost:3000`).

---

## Using the API

The API proxies your input to Python and returns JSON that mirrors Python’s native output (normalized for JSON — e.g., NaN → null). The most commonly used endpoints include:

- Design by Type
  - `POST /design/runs` — runs a design; response includes keys like `Best`, `ResultsTable`, `Shared`, `TRACE`, `T`, and possibly `hFig: null` for headless runs.
  - `GET  /design/runs/:id/trace` — a compact trace view for a past run.

- Pipeline
  - `POST /pipeline/design-then-hydrate` — performs a design and immediately hydrates the report and costs
    - Response contains: `report_t` (4x3 table), `cost_lakh_km`, `breakdown` (per-layer + totals), `derived`.
  - `GET  /pipeline/design-then-hydrate/runs/:id` — fetch a stored run.

- Other analysis endpoints
  - Effective CBR, Critical Strains, Multilayer, and Permissible Strain analysis endpoints are similarly structured, each returning JSON outputs aligned with their Python functions.

> Note: Exact request schemas may vary by branch. For concrete request bodies, check `node-api/tests/*.test.ts` — these files demonstrate known-good payloads used in CI.

### Fast responses for development

For quick iteration or CI, set `ROADAX_FAST=1` before starting the API. In this mode, Python’s design routine returns a deterministic, minimal result in ~1s while preserving the final JSON shape. This is ideal for building UI or writing tests without waiting minutes for full simulations.

---

## Example flows

- Quick local check without a DB:
  1. Start Python venv and install packages.
  2. Start Node API with `ROADAX_FAST=1`.
  3. Use a REST client (VS Code REST, Postman, or curl) to hit `POST /pipeline/design-then-hydrate` with the same body seen in `tests/pipeline.test.ts`.
  4. Inspect `cost_lakh_km` and `breakdown` — values should be numeric and stable in fast mode.

- Full-fidelity runs (longer):
  1. Unset `ROADAX_FAST`.
  2. Increase `BRIDGE_TIMEOUT_MS` (or run in a job queue/worker).
  3. Run the same endpoints; expect longer compute times.

---

## Testing

The Node API uses Vitest + Supertest.

```bash
cd node-api
# Enable fast mode for quick test runs
ROADAX_FAST=1 npm test
```

If you encounter timeouts in full-fidelity mode, increase `BRIDGE_TIMEOUT_MS` or use fast mode.

---

## Troubleshooting

- Python import/module not found
  - Activate your venv and verify `numpy`/`scipy` (and optionally `pandas`/`matplotlib`) are installed.
  - If the Node server can’t find Python, set `PYTHON_BIN` to your venv’s interpreter.

- Requests timing out
  - Raise `BRIDGE_TIMEOUT_MS` (e.g., `600000` for heavy runs) or enable `ROADAX_FAST=1` when you just need the shape.

- Output mismatch vs Python scripts
  - The API intentionally mirrors Python’s result structure. If you see differences, verify local changes to Python files and check normalization (NaN → null) in responses.

- Database connectivity
  - Ensure `DATABASE_URL` is set and your DB is reachable before running migrations. If you don’t need persistence, you can run without DB features in development.

---

## Notes

- Long computations: production deployments should consider a background job or queue for heavy designs. The synchronous bridge is great for development, short analyses, or fast-mode runs.
- JSON safety: the bridge sanitizes `NaN`, `Infinity`, and non-serializable types into JSON-safe values.

---

## License

Proprietary or internal use. If you plan to open-source this repo, add an explicit license.
