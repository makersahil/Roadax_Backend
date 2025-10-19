import { FastifyInstance, FastifyPluginCallback } from 'fastify'
import { z } from 'zod'
import { createRun, completeRun, failRun, findCached } from '../db/runRepo'
import { stableHash } from '../utils/hash'
import { runFeature } from '../services/pythonBridge'

const Triple = z.tuple([z.number(), z.number(), z.number()])

const CriticalsRequest = z.object({
  Number_of_layers: z.number().int().gte(2),
  Thickness_layers: z.array(z.number()),
  Modulus_layers: z.array(z.number()),
  Poissons: z.array(z.number()),
  Eva_depth_bituminous: z.number(),
  Eva_depth_base: z.number(),
  Eva_depth_Subgrade: z.number(),
  CFD_Check: z.union([z.literal(0), z.literal(1)]),
  FS_CTB_T: z.number().positive().optional(),
  SA_M_T: z.array(Triple).optional(),
  TaA_M_T: z.array(Triple).optional(),
  TrA_M_T: z.array(Triple).optional(),
}).superRefine((val, ctx) => {
  const n = val.Number_of_layers
  if (val.Thickness_layers.length !== n - 1) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Thickness_layers length must be n-1 (${n - 1})` })
  if (val.Modulus_layers.length !== n) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Modulus_layers length must be n (${n})` })
  if (val.Poissons.length !== n) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Poissons length must be n (${n})` })

  // Interface depths must be in cumulative sums of Thickness_layers
  if (val.Thickness_layers.length > 0) {
    const cum: number[] = []
    let s = 0
    for (const t of val.Thickness_layers) { s += t; cum.push(Number(s)) }
    const inCum = (x: number) => cum.some(v => Math.abs(v - x) < 1e-6)
    if (!inCum(val.Eva_depth_bituminous)) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Eva_depth_bituminous must equal a layer interface depth' })
    if (!inCum(val.Eva_depth_base)) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Eva_depth_base must equal a layer interface depth' })
    if (!inCum(val.Eva_depth_Subgrade)) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Eva_depth_Subgrade must equal a layer interface depth' })
  }

  // CFD branch requirements
  if (val.CFD_Check === 1) {
    if (typeof val.FS_CTB_T !== 'number' || !(val.FS_CTB_T > 0)) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'FS_CTB_T > 0 is required when CFD_Check=1' })
    const arrays = [val.SA_M_T, val.TaA_M_T, val.TrA_M_T]
    for (const arr of arrays) {
      if (!arr || arr.length === 0) { ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'SA_M_T, TaA_M_T, TrA_M_T are required when CFD_Check=1' }); break }
      for (const row of arr) {
        if (row.length !== 3) { ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Each traffic row must be a triple [kN_low, kN_high, reps]' }); break }
      }
    }
  }
})

export const criticalsRoutes: FastifyPluginCallback = (app: FastifyInstance, _opts, done) => {
  // Create run
  app.post('/runs', async (req, reply) => {
    const parse = CriticalsRequest.safeParse(req.body)
    if (!parse.success) return reply.code(400).send({ ok: false, error: { type: 'ValidationError', message: 'Invalid request', details: parse.error.flatten() } })

    // Normalize optional arrays for CFD=0
    const input = { ...parse.data }
    if (input.CFD_Check === 0) {
      if (!input.FS_CTB_T) input.FS_CTB_T = 1.0
      input.SA_M_T = input.SA_M_T || []
      input.TaA_M_T = input.TaA_M_T || []
      input.TrA_M_T = input.TrA_M_T || []
    }

    const payload = { feature: 'critical_strain_analysis', args: input }
    const inputHash = stableHash(payload)

    // cache
    const cached = await findCached('critical_strain_analysis' as any, inputHash)
    if (cached) {
      const out = (cached.outputJson as any) || {}
      return reply.send({ runId: cached.id, status: cached.status, result: out })
    }

    const startedAt = new Date()
    const run = await createRun('critical_strain_analysis' as any, input, inputHash)
    const bridge = await runFeature<number[] | number>('critical_strain_analysis', input)
    if (!bridge.ok) {
      const failed = await failRun(run.id, bridge.error, startedAt)
      return reply.code(502).send({ runId: failed.id, status: failed.status, error: bridge.error })
    }
    // Expect array length 4; coerce to numbers
    const arr = Array.isArray(bridge.data) ? bridge.data : [bridge.data]
    const vector4 = arr.map((x) => Number(x)).slice(0, 4)
    while (vector4.length < 4) vector4.push(NaN)
    const labels = ["BSi_micro", "CTB_micro", "CFD", "Subgrade_micro"]
    const output = { vector4, labels }

    const completed = await completeRun(run.id, output, startedAt)
    return reply.send({ runId: completed.id, status: completed.status, result: output })
  })

  // Get run by id
  app.get('/runs/:id', async (req, reply) => {
    const id = (req.params as any).id as string
    const { prisma } = await import('../db/client')
    const fresh = await prisma.run.findUnique({ where: { id } })
    if (!fresh) return reply.code(404).send({ ok: false, error: { type: 'NotFound', message: 'Run not found' } })
    return reply.send(fresh)
  })

  done()
}
