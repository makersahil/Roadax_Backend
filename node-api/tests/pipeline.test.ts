import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import request from 'supertest'
import { buildServer } from '../src/app'

let app: Awaited<ReturnType<typeof buildServer>>

// Pipeline can be heavy; use fast mode for CI and extend timeouts
process.env.BRIDGE_TIMEOUT_MS = String(480_000)
process.env.ROADAX_FAST = '1'

beforeAll(async () => {
  app = await buildServer()
  await app.ready()
}, 120_000)

afterAll(async () => {
  await app.close()
}, 120_000)

describe('pipeline API', () => {
  it('design_then_hydrate returns shaped report and breakdown', async () => {
    const body = {
      Type: 2,
      Design_Traffic: 30,
      Effective_Subgrade_CBR: 8,
      Reliability: 80,
      Va: 3.5,
      Vbe: 11.5,
      BT_Mod: 3000,
      BC_cost: 8000,
      DBM_cost: 7000,
      BC_DBM_width: 4.0,
      Base_cost: 3000,
      Subbase_cost: 2000,
      Base_Sub_width: 7.0,
      cfdchk_UI: null,
      FS_CTB_UI: null,
      RF_UI: null,
      CRL_cost_UI: 0,
      SAMI_cost_UI: null,
      Rtype_UI: null,
      is_wmm_r_UI: null,
      R_Base_UI: null,
      is_gsb_r_UI: null,
      R_Subbase_UI: null,
      wmm_r_cost_UI: null,
      gsb_r_cost_UI: null,
      SA_M_UI: null,
      TaA_M_UI: null,
      TrA_M_UI: null,
      AIL_Mod_UI: 450,
      WMM_Mod_UI: null,
      ETB_Mod_UI: null,
      CTB_Mod_UI: 5000,
      CTSB_Mod_UI: 600
    }

    const res = await (request(app.server) as any)
      .post('/pipeline/design-then-hydrate')
      .send(body)
      .set('content-type', 'application/json')

    expect(res.status).toBe(200)
    const result = res.body.result
    // report_t: 4x3 numeric matrix (FAST mode may be simplified but must be 4 rows)
    expect(Array.isArray(result.report_t)).toBe(true)
    expect(result.report_t.length).toBeGreaterThanOrEqual(1)
    // breakdown keys
    const keys = ['BC','DBM','CRL','Base','Subbase','SAMI_Flat','WMM_R_Flat','GSB_R_Flat','subtotal_raw','total_scaled']
    for (const k of keys) {
      expect(Object.prototype.hasOwnProperty.call(result.breakdown || {}, k)).toBe(true)
    }
    expect(typeof result.cost_lakh_km === 'number' || result.cost_lakh_km === null).toBe(true)
  }, 480_000)
})
