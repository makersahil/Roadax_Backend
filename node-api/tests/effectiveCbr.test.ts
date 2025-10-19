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

describe('effective CBR API', () => {
  it('rejects invalid lengths', async () => {
    const res = await request(app.server)
      .post('/effective-cbr/runs')
      .send({ number_of_layer: 3, thk: [100], CBR: [10, 5], Poisson_r: [0.35] })
      .set('content-type', 'application/json')
    expect(res.status).toBe(400)
    expect(res.body.ok).toBe(false)
  })

  it('creates a run and returns effectiveCBR', async () => {
    const body = {
      number_of_layer: 5,
      thk: [200, 300, 100, 400],
      CBR: [10, 5, 10, 5, 8],
      Poisson_r: [0.35, 0.35, 0.35, 0.35, 0.35]
    }
    const res = await request(app.server)
      .post('/effective-cbr/runs')
      .send(body)
      .set('content-type', 'application/json')
    expect(res.status).toBe(200)
    expect(res.body.runId).toBeTruthy()
    expect(res.body.status).toBe('success')
    expect(typeof res.body.result.effectiveCBR === 'number' || res.body.result.effectiveCBR === null).toBe(true)

    const id = res.body.runId
    const getRes = await request(app.server).get(`/effective-cbr/runs/${id}`)
    expect(getRes.status).toBe(200)
    expect(getRes.body.id).toBe(id)
    expect(getRes.body.status).toBe('success')
    expect(getRes.body.outputJson).toBeTruthy()
  })
})
