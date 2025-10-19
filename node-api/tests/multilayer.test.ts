import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import request from 'supertest'
import { buildServer } from '../src/app'

let app: Awaited<ReturnType<typeof buildServer>>

beforeAll(async () => {
  app = await buildServer()
  await app.ready()
})

afterAll(async () => {
  await app.close()
})

describe('multilayer API', () => {
  it('returns a JSON-shaped table with correct columns and rows', async () => {
    const body = {
      Number_of_layers: 4,
      Thickness_layers: [100.00, 240.00, 200.00],
      Modulus_layers: [3000.00, 617.94, 300.11, 76.83],
      Poissons: [0.35, 0.35, 0.35, 0.35],
      Tyre_pressure: 0.56,
      wheel_load: 20000,
      wheel_set: 2,
      analysis_points: 4,
      depths: [100, 100, 540, 540],
      radii: [0, 155, 0, 155],
      isbonded: true,
      center_spacing: 310,
      alpha_deg: 0
    }
    const res = await request(app.server).post('/multilayer/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(200)
    const table = res.body.result.table
    expect(table.columns).toEqual(["Point","Depth","Radius_from_left","Sigma_z","Sigma_t","Sigma_r","Tau_xz","w","ez","et","er"])
    expect(Array.isArray(table.data)).toBe(true)
    // Dual wheel + 4 analysis points -> may return 4 rows (not every interface split). Check row count >= analysis_points
    expect(table.data.length).toBeGreaterThanOrEqual(4)
    // no NaN/Infinity
    const hasNonFinite = table.data.flat().some((v: any) => typeof v === 'number' && !Number.isFinite(v))
    expect(hasNonFinite).toBe(false)
  }, 15000)
})
