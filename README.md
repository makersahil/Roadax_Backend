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


## How to call Features

### Effective CBR Calculation

curl -sS -X POST http://localhost:3000/effective-cbr/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "number_of_layer": 5,
    "thk": [200, 300, 100, 400],
    "CBR": [10, 5, 10, 5, 8],
    "Poisson_r": [0.35, 0.35, 0.35, 0.35, 0.35]
  }' | jq

### Critical Strain Analysis

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
    "CFD_Check": 1,
    "FS_CTB_T": 1.4,
    "SA_M_T": [[185, 195, 70000],[175, 185, 90000]],
    "TaA_M_T": [[390, 410, 200000],[370, 390, 230000]],
  }' | jqM_T": [[585, 615, 35000],[555, 585, 40000]]


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

## Instruction to Run

1) Create Python Virtual Environment inside the Roadax_Python_Functions folder and activate it

2) Install: scipy, pandas, numpy, matplotlib

4) Set Environment Variables in the Root Folder

> export ROADAX_FAST=0
> export BRIDGE_TIMEOUT_MS=600000

5) Install NPM Dependencies and Start the Server

> npm install
> npx prisma migrate
> npm run dev

6) Make requests using Curl by syntax above