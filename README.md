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
  - `Design_by_type.py` — main unified design routine
  - `Critical_Strain_Analysis.py`, `Multilayer_Analysis.py`, `Permissible_Strain_Analysis.py`, `Effective_CBR_Calc.py` — analysis modules
  - `Edit_type_to_check.py` — hydration/reporting helpers
  - `Bridge.py` or `python_api_bridge.py` (depending on branch) — stdin/stdout JSON bridge used by Node
- `node-api/` — TypeScript Fastify API that calls into Python and returns JSON results

> Note: Full-fidelity Python runs can take time (tens of seconds to minutes) depending on inputs. Ensure API timeouts are configured generously in development and production.

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
- `PYTHON_BIN` — Path to Python executable the API should use (default: `python3`). Prefer the venv Python.
- `PYTHON_BRIDGE_PATH` — Absolute path to the Python bridge file (e.g., `/abs/path/to/Roadax_Python_Functions/python_api_bridge.py`).
- `BRIDGE_TIMEOUT_MS` — Max milliseconds to wait for a Python run (example: `600000` for 10 minutes).
- `DATABASE_URL` — Prisma connection string if you’re using persistence (e.g., PostgreSQL).

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

### Available endpoints and Python mapping

- Health
  - `GET /health` → Node-only health probe.
  - `GET /health/python` → Bridge no-op check (invokes the Python bridge; verifies readiness).

- Effective CBR Calculation
  - `POST /effective-cbr/runs` → `Roadax_Python_Functions/Effective_CBR_Calc.py`
  - `GET  /effective-cbr/runs/:id` → fetch run by id (if persistence enabled)

- Critical Strain Analysis
  - `POST /criticals/runs` → `Roadax_Python_Functions/Critical_Strain_Analysis.py`
  - `GET  /criticals/runs/:id` → fetch run by id (if persistence enabled)

- Multilayer Analysis
  - `POST /multilayer/runs` → `Roadax_Python_Functions/Multilayer_Analysis.py`
  - `GET  /multilayer/runs/:id` → fetch run by id (if persistence enabled)

- Permissible Strain Analysis
  - `POST /permissible-strain/runs` → `Roadax_Python_Functions/Permissible_Strain_Analysis.py`
  - `GET  /permissible-strain/runs/:id` → fetch run by id (if persistence enabled)

- Design by Type
  - `POST /design/runs` → `Roadax_Python_Functions/Design_by_type.py` (unified design)
  - `GET  /design/runs/:id` → fetch run by id (if persistence enabled)
  - `GET  /design/runs/:id/trace` → returns `TRACE` and `T` from the stored result

- Pipeline
  - `POST /pipeline/design-then-hydrate` → composite pipeline using `Design_by_type.py` + hydration from `Edit_type_to_check.py`
  - `GET  /pipeline/design-then-hydrate/runs/:id` → fetch run by id (if persistence enabled)

> Note: GET endpoints that fetch by id require the database to be configured (Prisma/Postgres). If persistence is not set up, rely on the immediate POST response body (which already includes the computed result and runId).

## Example flows

- Quick local check without a DB:
  1. Start Python venv and install packages.
  2. Start the Node API.
  3. Use a REST client (VS Code REST, Postman, or curl) to hit `POST /pipeline/design-then-hydrate` with the same body seen in `tests/pipeline.test.ts`.
  4. Inspect `cost_lakh_km` and `breakdown` — values should be numeric and consistent with inputs.

- Full-fidelity runs (longer):
  1. Increase `BRIDGE_TIMEOUT_MS` (or run in a job queue/worker).
  3. Run the same endpoints; expect longer compute times.

---

## Testing

The Node API uses Vitest + Supertest.

```bash
cd node-api
npm test
```

If you encounter timeouts during tests, increase `BRIDGE_TIMEOUT_MS` or temporarily narrow the test set.

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

- Long computations: production deployments should consider a background job or queue for heavy designs. The synchronous bridge is great for development and short analyses.
- JSON safety: the bridge sanitizes `NaN`, `Infinity`, and non-serializable types into JSON-safe values. Python prints/logs are routed to stderr to keep stdout strict JSON.
- Pandas optionality: where applicable (e.g., design), results can be returned as plain dict-shaped tables if pandas isn’t available; installing pandas restores full DataFrame-shaped JSON.

---

## License

Proprietary or internal use. If you plan to open-source this repo, add an explicit license.


## How to call Features

### Health

curl -sS http://localhost:3000/health | jq

curl -sS http://localhost:3000/health/python | jq

### Effective CBR Calculation

curl -sS -X POST http://localhost:3000/effective-cbr/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "number_of_layer": 5,
    "thk": [200, 300, 100, 400],
    "CBR": [10, 5, 10, 5, 8],
    "Poisson_r": [0.35, 0.35, 0.35, 0.35, 0.35]
  }' | jq

# Fetch by run id (replace RUN_ID)
curl -sS http://localhost:3000/effective-cbr/runs/RUN_ID | jq

### Critical Strain Analysis (CFD off)

curl -sS -X POST http://localhost:3000/criticals/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "Number_of_layers": 5,
    "Thickness_layers": [100.0, 200.0, 200.0, 250.0],
    "Modulus_layers": [3000.0, 400.0, 5000.0, 800.0, 100.0],
    "Poissons": [0.35, 0.35, 0.25, 0.35, 0.35],
    "Eva_depth_bituminous": 100.0,
    "Eva_depth_base": 500.0,
    "Eva_depth_Subgrade": 750.0,
    "CFD_Check": 0
  }' | jq

# Fetch by run id (replace RUN_ID)
curl -sS http://localhost:3000/criticals/runs/RUN_ID | jq


### Permissible Strain Analysis

curl -sS -X POST http://localhost:3000/permissible-strain/runs \
  -H "Content-Type: application/json" \
  -d '{
    "Design_Traffic": 50,
    "Reliability": 90,
    "Va": 4,
    "Vbe": 11,
    "BT_Mod": 3000,
    "Base_ctb": 0,
    "Base_Mod": null,
    "RF_CTB": null
  }'

# Fetch by run id (replace RUN_ID)
curl -sS http://localhost:3000/permissible-strain/runs/RUN_ID | jq

### Multilayer Analysis

curl -sS -X POST http://localhost:3000/multilayer/runs \
  -H "Content-Type: application/json" \
  -d '{
    "Number_of_layers": 3,
    "Thickness_layers": [100, 200],
    "Modulus_layers": [3000, 600, 150],
    "Poissons": [0.35, 0.35, 0.45],
    "Tyre_pressure": 0.8,
    "wheel_load": 40,
    "wheel_set": 1,
    "analysis_points": 3,
    "depths": [50, 100, 300],
    "radii": [0, 150, 300],
    "isbonded": true,
    "center_spacing": 0,
    "alpha_deg": 0
  }'

# Fetch by run id (replace RUN_ID)
curl -sS http://localhost:3000/multilayer/runs/RUN_ID | jq

### Design By Type

curl -sS -X POST http://localhost:3000/design/runs \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{
  "Type": 2,
  "Design_Traffic": 30,
  "Effective_Subgrade_CBR": 8,
  "Reliability": 90,
  "Va": 4,
  "Vbe": 11,
  "BT_Mod": 3000,
  "BC_cost": 6000,
  "DBM_cost": 5000,
  "BC_DBM_width": 3.5,
  "Base_cost": 2000,
  "Subbase_cost": 1500,
  "Base_Sub_width": 3.75
}
JSON

# Fetch by run id (replace RUN_ID)
curl -sS http://localhost:3000/design/runs/RUN_ID | jq

# Fetch trace (TRACE + T)
curl -sS http://localhost:3000/design/runs/RUN_ID/trace | jq


### Design Then Hydrate

curl -sS -X POST http://localhost:3000/pipeline/design-then-hydrate \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{
  "Type": 2,
  "Design_Traffic": 30,
  "Effective_Subgrade_CBR": 8,
  "Reliability": 90,
  "Va": 4,
  "Vbe": 11,
  "BT_Mod": 3000,
  "BC_cost": 6000,
  "DBM_cost": 5000,
  "BC_DBM_width": 3.5,
  "Base_cost": 2000,
  "Subbase_cost": 1500,
  "Base_Sub_width": 3.75,
  "cfdchk_UI": null, "FS_CTB_UI": null, "RF_UI": null,
  "CRL_cost_UI": null, "SAMI_cost_UI": null,
  "Rtype_UI": null, "is_wmm_r_UI": null, "R_Base_UI": null,
  "is_gsb_r_UI": null, "R_Subbase_UI": null,
  "wmm_r_cost_UI": null, "gsb_r_cost_UI": null,
  "SA_M_UI": null, "TaA_M_UI": null, "TrA_M_UI": null,
  "AIL_Mod_UI": null, "WMM_Mod_UI": null, "ETB_Mod_UI": null,
  "CTB_Mod_UI": null, "CTSB_Mod_UI": null
}
JSON

# Fetch by run id (replace RUN_ID)
curl -sS http://localhost:3000/pipeline/design-then-hydrate/runs/RUN_ID | jq

## Instruction to Run

1) Create Python Virtual Environment inside the Roadax_Python_Functions folder and activate it

2) Install: scipy, pandas, numpy, matplotlib

4) Set Environment Variables in the Root Folder

> export PYTHON_BIN="$(pwd)/Roadax_Python_Functions/.venv/bin/python"
> export PYTHON_BRIDGE_PATH="$(pwd)/Roadax_Python_Functions/python_api_bridge.py"
> export BRIDGE_TIMEOUT_MS=600000

5) Install NPM Dependencies and Start the Server

> npm install
> npx prisma migrate
> npm run dev

6) Make requests using Curl by syntax above

---

## Deployment checklist

- Use absolute path for `PYTHON_BRIDGE_PATH` so the process manager/service can find the bridge reliably.
- Point `PYTHON_BIN` to the correct virtual environment interpreter.
- Increase `BRIDGE_TIMEOUT_MS` to cover longest expected design runs (e.g., 10–20 minutes).
- Align upstream proxy timeouts (nginx, load balancer) with the backend timeout.
- Ensure Python dependencies are installed in the target environment (numpy/scipy; pandas optional).
- If using Prisma/Postgres, ensure `DATABASE_URL` is set and DB is reachable.
- After changing `.env`, restart the Node process to apply.