import { FastifyInstance, FastifyPluginCallback } from 'fastify'
import { runNoop } from '../services/pythonBridge'

export const healthRoutes: FastifyPluginCallback = (app: FastifyInstance, _opts, done) => {
  app.get('/', async (_req, _reply) => {
    return {
      ok: true,
      ts: new Date().toISOString(),
      version: '0.1.0'
    }
  })

  app.get('/python', async (_req, reply) => {
    const result = await runNoop()
    if (result.ok) {
      return reply.send({ ok: true, bridge: 'ready', data: result.data })
    }
    return reply.code(500).send({ ok: false, error: result.error })
  })

  done()
}
