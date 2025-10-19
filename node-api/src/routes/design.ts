import { FastifyInstance, FastifyPluginCallback } from 'fastify'
import { z } from 'zod'
import { stableHash } from '../utils/hash'
import { createRun, completeRun, failRun, findCached } from '../db/runRepo'
import { runFeature, jsonNormalize } from '../services/pythonBridge'

const num = () => z.number()
const pos = () => z.number().positive()
const int1to8 = z.number().int().min(1).max(8)
const numArr2D = z.array(z.array(z.number()))

const DesignRequest = z.object({
  Type: int1to8,
  Design_Traffic: pos(),
  Effective_Subgrade_CBR: pos(),
  Reliability: z.union([z.literal(80), z.literal(90)]),
  Va: num(),
  Vbe: num(),
  BT_Mod: pos(),
  BC_cost: num(),
  DBM_cost: num(),
  BC_DBM_width: num(),
  Base_cost: num(),
  Subbase_cost: num(),
  Base_Sub_width: num(),
  // Optional / type-dependent
  cfdchk_UI: z.number().int().nullable().optional(),
  FS_CTB_UI: z.number().nullable().optional(),
  RF_UI: z.number().nullable().optional(),
  CRL_cost_UI: z.number().nullable().optional(),
  SAMI_cost_UI: z.number().nullable().optional(),
  Rtype_UI: z.number().int().nullable().optional(),
  is_wmm_r_UI: z.number().int().nullable().optional(),
  R_Base_UI: z.number().nullable().optional(),
  is_gsb_r_UI: z.number().int().nullable().optional(),
  R_Subbase_UI: z.number().nullable().optional(),
  wmm_r_cost_UI: z.number().nullable().optional(),
  gsb_r_cost_UI: z.number().nullable().optional(),
  SA_M_UI: numArr2D.nullable().optional(),
  TaA_M_UI: numArr2D.nullable().optional(),
  TrA_M_UI: numArr2D.nullable().optional(),
  AIL_Mod_UI: z.number().nullable().optional(),
  WMM_Mod_UI: z.number().nullable().optional(),
  ETB_Mod_UI: z.number().nullable().optional(),
  CTB_Mod_UI: z.number().nullable().optional(),
  CTSB_Mod_UI: z.number().nullable().optional(),
}).superRefine((v, ctx) => {
  // Basic shape checks for axle arrays if provided
  for (const k of ['SA_M_UI','TaA_M_UI','TrA_M_UI'] as const) {
    const arr = v[k]
    if (arr) {
      for (const row of arr) {
        if (row.length < 3) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `${k} rows must have >=3 numbers [min_kN, max_kN, reps]` })
      }
    }
  }
})

// No table reshaping here; we mirror Python output shape (Best, ResultsTable, Shared, TRACE, T, hFig)

export const designRoutes: FastifyPluginCallback = (app: FastifyInstance, _opts, done) => {
  app.post('/runs', async (req, reply) => {
    const parse = DesignRequest.safeParse(req.body)
    if (!parse.success) return reply.code(400).send({ ok: false, error: { type: 'ValidationError', message: 'Invalid request', details: parse.error.flatten() } })

    const input = parse.data
    const payload = { feature: 'design_by_type', args: input }
    const inputHash = stableHash(payload)

    const cached = await findCached('design_by_type' as any, inputHash)
    if (cached?.outputJson) {
      return reply.send({ runId: cached.id, status: cached.status, result: cached.outputJson })
    }

    const startedAt = new Date()
    const run = await createRun('design_by_type' as any, input, inputHash)
    const bridge = await runFeature<any>('design_by_type', input)
    if (!bridge.ok) {
      const failed = await failRun(run.id, bridge.error, startedAt)
      return reply.code(502).send({ runId: failed.id, status: failed.status, error: bridge.error })
    }

    const data = bridge.data as any
    // Pass through Python-shaped output for design_by_type
    const result = jsonNormalize({
      Best: data?.Best ?? null,
      ResultsTable: data?.ResultsTable ?? null,
      Shared: data?.Shared ?? null,
      TRACE: data?.TRACE ?? null,
      T: data?.T ?? null,
      hFig: null,
    })

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

  // Optional trace endpoint: return TRACE and T tables
  app.get('/runs/:id/trace', async (req, reply) => {
    const id = (req.params as any).id as string
    const { prisma } = await import('../db/client')
    const fresh = await prisma.run.findUnique({ where: { id } })
    if (!fresh || !fresh.outputJson) return reply.code(404).send({ ok: false, error: { type: 'NotFound', message: 'Run not found or no output' } })
    const out = fresh.outputJson as any
    if (out?.TRACE || out?.T) return reply.send({ TRACE: out.TRACE ?? [], T: out.T ?? null })
    return reply.send({ TRACE: [], T: null })
  })

  done()
}
