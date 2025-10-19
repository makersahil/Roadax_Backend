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

describe('permissible strain API', () => {
  it('rejects when Base_ctb=1 but Base_Mod/RF_CTB are missing', async () => {
    const body = {
      Design_Traffic: 300,
      Reliability: 80,
      Va: 3.5,
      Vbe: 11.5,
      BT_Mod: 3000,
      Base_ctb: 1,
      Base_Mod: null,
      RF_CTB: null,
    }
  const res = await (request(app.server) as any).post('/permissible-strain/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(400)
  })

  it('rejects when Base_ctb=0 but Base_Mod/RF_CTB provided', async () => {
    const body = {
      Design_Traffic: 300,
      Reliability: 80,
      Va: 3.5,
      Vbe: 11.5,
      BT_Mod: 3000,
      Base_ctb: 0,
      Base_Mod: 600,
      RF_CTB: 1,
    }
  const res = await (request(app.server) as any).post('/permissible-strain/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(400)
  })

  it('creates a run (no CTB) and returns expected shape with Base_micro=null', async () => {
    const body = {
      Design_Traffic: 300,
      Reliability: 80,
      Va: 3.5,
      Vbe: 11.5,
      BT_Mod: 3000,
      Base_ctb: 0,
      Base_Mod: null,
      RF_CTB: null,
    }
  const res = await (request(app.server) as any).post('/permissible-strain/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(200)
    const result = res.body.result
    expect(Array.isArray(result.vec4)).toBe(true)
    expect(result.vec4.length).toBe(4)
    // vec4[2] should be exactly 1 (per specification)
    expect(result.vec4[2]).toBe(1)
    // Base_micro should be null when no CTB
    expect(result.named.Base_micro).toBe(null)
    // Permissible mapping keys exist and si_cfat_micro is null when no CTB
    expect(result.permissible.si_bfat_micro).toSatisfy((x: any) => typeof x === 'number')
    expect(result.permissible.si_rut_micro).toSatisfy((x: any) => typeof x === 'number')
    expect(result.permissible.si_cfat_micro).toBe(null)
  })
})
