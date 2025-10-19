import { describe, it, expect } from 'vitest'
import request from 'supertest'
import { buildServer } from '../src/app'

describe('health routes', () => {
  it('GET /health should return ok', async () => {
    const app = await buildServer()
    await app.ready()
  const res = await request(app.server).get('/health')
    expect(res.status).toBe(200)
    expect(res.body.ok).toBe(true)
    expect(typeof res.body.ts).toBe('string')
    await app.close()
  })

  it('GET /health/python should return bridge ready (if python available)', async () => {
    const app = await buildServer()
    await app.ready()
  const res = await request(app.server).get('/health/python')
    // 200 if ready, 500 if not configured; both are acceptable signals
    expect([200, 500]).toContain(res.status)
    if (res.status === 200) {
      expect(res.body.ok).toBe(true)
      expect(res.body.bridge).toBe('ready')
    } else {
      expect(res.body.ok).toBe(false)
    }
    await app.close()
  })
})
