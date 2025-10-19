import { FastifyInstance, FastifyPluginCallback } from 'fastify'
import { z } from 'zod'
import { createRun, completeRun, failRun, findCached } from '../db/runRepo'
import { stableHash } from '../utils/hash'
import { runFeature } from '../services/pythonBridge'

const EffectiveCbrRequest = z.object({
  number_of_layer: z.number().int().gte(2),
  thk: z.array(z.number()),
  CBR: z.array(z.number()),
  Poisson_r: z.array(z.number())
}).superRefine((val, ctx) => {
  const n = val.number_of_layer
  if (val.thk.length !== n - 1) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: `thk length must be n-1 (${n - 1}), got ${val.thk.length}` })
  }
  if (val.CBR.length !== n) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: `CBR length must be n (${n}), got ${val.CBR.length}` })
  }
  if (val.Poisson_r.length !== n) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Poisson_r length must be n (${n}), got ${val.Poisson_r.length}` })
  }
})

export const effectiveCbrRoutes: FastifyPluginCallback = (app: FastifyInstance, _opts, done) => {
  // Create run
  app.post('/runs', async (req, reply) => {
    const parse = EffectiveCbrRequest.safeParse(req.body)
    if (!parse.success) {
      return reply.code(400).send({ ok: false, error: { type: 'ValidationError', message: 'Invalid request', details: parse.error.flatten() } })
    }
    const input = parse.data
    const payload = { feature: 'effective_cbr_calc', args: input }
    const inputHash = stableHash(payload)

    // Return cached success if exists
  const cached = await findCached('effective_cbr_calc' as any, inputHash)
    if (cached) {
      return reply.send({ runId: cached.id, status: cached.status, result: { effectiveCBR: cached.outputJson && (cached.outputJson as any).effectiveCBR } })
    }

    const startedAt = new Date()
  const run = await createRun('effective_cbr_calc' as any, input, inputHash)
    const bridge = await runFeature<number>('effective_cbr_calc', input)

    if (!bridge.ok) {
      const failed = await failRun(run.id, bridge.error, startedAt)
      return reply.code(502).send({ runId: failed.id, status: failed.status, error: bridge.error })
    }

    const num = Number(bridge.data)
    const effectiveCBR = Number.isFinite(num) ? num : null
    const completed = await completeRun(run.id, { effectiveCBR }, startedAt)
    return reply.send({ runId: completed.id, status: completed.status, result: { effectiveCBR } })
  })

  // Get run by id
  app.get('/runs/:id', async (req, reply) => {
    const id = (req.params as any).id as string
    try {
      const { prisma } = await import('../db/client')
      const fresh = await prisma.run.findUnique({ where: { id } })
      if (!fresh) return reply.code(404).send({ ok: false, error: { type: 'NotFound', message: 'Run not found' } })
      return reply.send(fresh)
    } catch (e: any) {
      return reply.code(500).send({ ok: false, error: { type: 'ServerError', message: String(e?.message || e) } })
    }
  })

  done()
}
