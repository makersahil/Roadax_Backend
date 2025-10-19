import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import request from 'supertest'
import { buildServer } from '../src/app'

let app: Awaited<ReturnType<typeof buildServer>>

// Design by Type can take much longer; bump Python bridge and hook timeouts
process.env.BRIDGE_TIMEOUT_MS = String(480_000)
// Enable fast mode on Python side during tests to keep CI practical
process.env.ROADAX_FAST = '1'

beforeAll(async () => {
  app = await buildServer()
  await app.ready()
}, 120_000)

afterAll(async () => {
  await app.close()
}, 120_000)

describe('design API', () => {
  it('runs Type 2 sample and returns python-shaped output', async () => {
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
      // leave optional fields as nulls
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
      .post('/design/runs')
      .send(body)
      .set('content-type', 'application/json')

    expect(res.status).toBe(200)
    const result = res.body.result
    // Python-shaped keys
    expect(result).toHaveProperty('Best')
    expect(result).toHaveProperty('ResultsTable')
    expect(result).toHaveProperty('Shared')
    expect(result).toHaveProperty('TRACE')
    expect(result).toHaveProperty('T')
    expect(result).toHaveProperty('hFig')
    // Tables look like DataFrame-shaped dicts
    expect(result.Best?._type).toBe('DataFrame')
    expect(Array.isArray(result.Best?.columns)).toBe(true)
    expect(Array.isArray(result.Best?.data)).toBe(true)
    expect(result.T?._type).toBe('DataFrame')
    expect(Array.isArray(result.T?.columns)).toBe(true)
    expect(Array.isArray(result.T?.data)).toBe(true)
    // TRACE is a 2D array (may be empty)
    expect(Array.isArray(result.TRACE)).toBe(true)
    // Shared has key fields
    expect(result.Shared?.BT_thk).toSatisfy((x: any) => typeof x === 'number')
    expect(result.Shared?.Base_thk).toSatisfy((x: any) => typeof x === 'number')
    expect(result.Shared?.Subbase_thk).toSatisfy((x: any) => typeof x === 'number')
  }, 480_000)
})
