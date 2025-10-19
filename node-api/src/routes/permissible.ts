import { FastifyInstance, FastifyPluginCallback } from 'fastify'
import { z } from 'zod'
import { stableHash } from '../utils/hash'
import { createRun, completeRun, failRun, findCached } from '../db/runRepo'
import { runFeature, jsonNormalize } from '../services/pythonBridge'

const PermissibleRequest = z.object({
  Design_Traffic: z.number().positive(),
  Reliability: z.union([z.literal(80), z.literal(90)]),
  Va: z.number(),
  Vbe: z.number(),
  BT_Mod: z.number().positive(),
  Base_ctb: z.union([z.literal(0), z.literal(1)]),
  Base_Mod: z.number().positive().nullable(),
  RF_CTB: z.number().positive().nullable(),
}).superRefine((val, ctx) => {
  if (val.Base_ctb === 1) {
    if (val.Base_Mod == null) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Base_Mod is required when Base_ctb=1' })
    if (val.RF_CTB == null) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'RF_CTB is required when Base_ctb=1' })
  } else {
    if (val.Base_Mod != null) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Base_Mod must be null when Base_ctb=0' })
    if (val.RF_CTB != null) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'RF_CTB must be null when Base_ctb=0' })
  }
})

export const permissibleRoutes: FastifyPluginCallback = (app: FastifyInstance, _opts, done) => {
  app.post('/runs', async (req, reply) => {
    const parse = PermissibleRequest.safeParse(req.body)
    if (!parse.success) return reply.code(400).send({ ok: false, error: { type: 'ValidationError', message: 'Invalid request', details: parse.error.flatten() } })

    const input = parse.data
    const payload = { feature: 'permissible_strain_analysis', args: input }
    const inputHash = stableHash(payload)

    const cached = await findCached('permissible_strain_analysis' as any, inputHash)
    if (cached?.outputJson) {
      return reply.send({ runId: cached.id, status: cached.status, result: cached.outputJson })
    }

    const startedAt = new Date()
    const run = await createRun('permissible_strain_analysis' as any, input, inputHash)
    const bridge = await runFeature<any>('permissible_strain_analysis', input)
    if (!bridge.ok) {
      const failed = await failRun(run.id, bridge.error, startedAt)
      return reply.code(502).send({ runId: failed.id, status: failed.status, error: bridge.error })
    }

    // Python fast-path returns { vec4, Perm_Si_R, out }
    const data = bridge.data as any

    const vec4_raw = Array.isArray(data?.vec4) ? data.vec4 : []
    // normalize NaN->null
    const vec4 = (jsonNormalize(vec4_raw) as any[])
    // permissible mapping
    const perm_raw = Array.isArray(data?.Perm_Si_R) ? data.Perm_Si_R : []
    const perm = jsonNormalize(perm_raw) as [number | null, number | null, number | null]
    const named = data?.out || {}
    const result = {
      vec4: vec4,
      permissible: {
        si_bfat_micro: perm[0] ?? null,
        si_rut_micro: perm[1] ?? null,
        si_cfat_micro: perm[2] ?? null,
      },
      named: {
        Bituminous_micro: (named.Bituminous_micro ?? null),
        Subgrade_micro: (named.Subgrade_micro ?? null),
        Base_micro: (Number.isFinite(named.Base_micro) ? named.Base_micro : null),
      },
    }

    const completed = await completeRun(run.id, result, startedAt)
    return reply.send({ runId: completed.id, status: completed.status, result })
  })

  app.get('/runs/:id', async (req, reply) => {
    const id = (req.params as any).id as string
    const { prisma } = await import('../db/client')
    const fresh = await prisma.run.findUnique({ where: { id } })
    if (!fresh) return reply.code(404).send({ ok: false, error: { type: 'NotFound', message: 'Run not found' } })
    return reply.send(fresh)
  })

  done()
}
