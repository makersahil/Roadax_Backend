import fastify from 'fastify'
import pino from 'pino'
import { healthRoutes } from './routes/health'
import { effectiveCbrRoutes } from './routes/effectiveCbr'
import { criticalsRoutes } from './routes/criticals'
import { multilayerRoutes } from './routes/multilayer'
import { permissibleRoutes } from './routes/permissible'
import { designRoutes } from './routes/design'
import { pipelineRoutes } from './routes/pipeline'

const logger = pino({ level: process.env.LOG_LEVEL || 'info' })

export async function buildServer() {
  const app = fastify({ logger })

  // request id + timing
  app.addHook('onRequest', async (req, reply) => {
    const start = Date.now()
    ;(req as any).start = start
    const rid = (req as any).id ?? Math.random().toString(36).slice(2)
    reply.header('x-request-id', String(rid))
  })
  app.addHook('onResponse', async (req, reply) => {
    const start = (req as any).start as number | undefined
    if (start) {
      const dur = Date.now() - start
      reply.header('x-duration-ms', String(dur))
    }
  })

  app.register(healthRoutes, { prefix: '/health' })
  app.register(effectiveCbrRoutes, { prefix: '/effective-cbr' })
  app.register(criticalsRoutes, { prefix: '/criticals' })
  app.register(multilayerRoutes, { prefix: '/multilayer' })
  app.register(permissibleRoutes, { prefix: '/permissible-strain' })
  app.register(designRoutes, { prefix: '/design' })
  app.register(pipelineRoutes, { prefix: '/pipeline' })

  return app
}
