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

describe('criticals API', () => {
  it('rejects invalid interface depths', async () => {
    const body = {
      Number_of_layers: 3,
      Thickness_layers: [100, 200],
      Modulus_layers: [3000, 400, 100],
      Poissons: [0.35, 0.35, 0.35],
      Eva_depth_bituminous: 150, // not an interface (should be 100 or 300)
      Eva_depth_base: 200,
      Eva_depth_Subgrade: 300,
      CFD_Check: 0
    }
    const res = await request(app.server).post('/criticals/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(400)
  })

  it('requires traffic arrays when CFD_Check=1', async () => {
    const body = {
      Number_of_layers: 3,
      Thickness_layers: [100, 200],
      Modulus_layers: [3000, 400, 100],
      Poissons: [0.35, 0.35, 0.35],
      Eva_depth_bituminous: 100,
      Eva_depth_base: 300,
      Eva_depth_Subgrade: 300,
      CFD_Check: 1,
      FS_CTB_T: 1.4
    }
    const res = await request(app.server).post('/criticals/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(400)
  })

  it('creates a run and returns vector4 length 4', async () => {
    const body = {
      Number_of_layers: 5,
      Thickness_layers: [100.0, 200.0, 200.0, 250.0],
      Modulus_layers: [3000.0, 400.0, 5000.0, 800.0, 100.0],
      Poissons: [0.35, 0.35, 0.25, 0.35, 0.35],
      Eva_depth_bituminous: 100.0,
      Eva_depth_base: 500.0,
      Eva_depth_Subgrade: 750.0,
      CFD_Check: 1,
      FS_CTB_T: 1.4,
      SA_M_T: [
        [185, 195, 70000],
        [175, 185, 90000]
      ],
      TaA_M_T: [
        [390, 410, 200000],
        [370, 390, 230000]
      ],
      TrA_M_T: [
        [585, 615, 35000],
        [555, 585, 40000]
      ]
    }
    const res = await request(app.server).post('/criticals/runs').send(body).set('content-type', 'application/json')
    expect(res.status).toBe(200)
    expect(res.body.status).toBe('success')
    expect(res.body.result.vector4.length).toBe(4)
    const id = res.body.runId
    const getRes = await request(app.server).get(`/criticals/runs/${id}`)
    expect(getRes.status).toBe(200)
    expect(getRes.body.id).toBe(id)
  }, 20000)
})
