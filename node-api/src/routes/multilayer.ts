import { FastifyInstance, FastifyPluginCallback } from 'fastify'
import { z } from 'zod'
import { createRun, completeRun, failRun, findCached } from '../db/runRepo'
import { stableHash } from '../utils/hash'
import { runFeature, jsonNormalize } from '../services/pythonBridge'

const MultilayerRequest = z.object({
  Number_of_layers: z.number().int().gte(2),
  Thickness_layers: z.array(z.number()),
  Modulus_layers: z.array(z.number()),
  Poissons: z.array(z.number()),
  Tyre_pressure: z.number().positive(),
  wheel_load: z.number().positive(),
  wheel_set: z.union([z.literal(1), z.literal(2)]),
  analysis_points: z.number().int().gte(1),
  depths: z.array(z.number()),
  radii: z.array(z.number()),
  isbonded: z.boolean(),
  center_spacing: z.number().gte(0),
  alpha_deg: z.number(),
}).superRefine((val, ctx) => {
  const n = val.Number_of_layers
  if (val.Thickness_layers.length !== n - 1) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Thickness_layers length must be n-1 (${n - 1})` })
  if (val.Modulus_layers.length !== n) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Modulus_layers length must be n (${n})` })
  if (val.Poissons.length !== n) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Poissons length must be n (${n})` })
  if (val.depths.length !== val.analysis_points) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `depths length must equal analysis_points (${val.analysis_points})` })
  if (val.radii.length !== val.analysis_points) ctx.addIssue({ code: z.ZodIssueCode.custom, message: `radii length must equal analysis_points (${val.analysis_points})` })
})

function toTable(jsonResultTable: any): { columns: string[]; data: any[][] } {
  // We expect unified_api JSON-shaping to return { _type:'DataFrame', columns: string[], data: object[] }
  // Fallback if Python returned a dict-equivalent.
  if (jsonResultTable && typeof jsonResultTable === 'object') {
    if (Array.isArray(jsonResultTable.columns) && Array.isArray(jsonResultTable.data)) {
      const cols: string[] = jsonResultTable.columns
      const rows = jsonResultTable.data.map((row: any) => cols.map(c => {
        const v = row?.[c]
        if (typeof v === 'number' && !Number.isFinite(v)) return null
        return typeof v === 'number' || v === null ? v : (v == null ? null : Number(v))
      }))
      return { columns: cols, data: rows }
    }
  }
  // Unknown shape; return empty
  return { columns: [], data: [] }
}

export const multilayerRoutes: FastifyPluginCallback = (app: FastifyInstance, _opts, done) => {
  app.post('/runs', async (req, reply) => {
    const parse = MultilayerRequest.safeParse(req.body)
    if (!parse.success) return reply.code(400).send({ ok: false, error: { type: 'ValidationError', message: 'Invalid request', details: parse.error.flatten() } })

    const input = parse.data
    const payload = { feature: 'multilayer_analysis', args: input }
    const inputHash = stableHash(payload)

    const cached = await findCached('multilayer_analysis' as any, inputHash)
    if (cached?.outputJson) {
      return reply.send({ runId: cached.id, status: cached.status, result: cached.outputJson })
    }

    const startedAt = new Date()
    const run = await createRun('multilayer_analysis' as any, input, inputHash)
    const bridge = await runFeature<any>('multilayer_analysis', input)
    if (!bridge.ok) {
      const failed = await failRun(run.id, bridge.error, startedAt)
      return reply.code(502).send({ runId: failed.id, status: failed.status, error: bridge.error })
    }

    // Expect { Report_arr, ResultTable } where ResultTable is JSON-shaped DataFrame
    const res = bridge.data as any
    const shaped = toTable(res?.ResultTable)
    // sanitize non-finite to null, ensure numbers
    const sanitized = jsonNormalize({ columns: shaped.columns, data: shaped.data }) as any
    const result = {
      table: { columns: sanitized.columns, data: sanitized.data },
      shape: [Array.isArray(sanitized.data) ? sanitized.data.length : 0, 11] as [number, number]
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
