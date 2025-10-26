import { createHash } from 'node:crypto'

function stableStringify(value: unknown): string {
  // Recursively sort object keys to produce a deterministic JSON string
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'number' || typeof value === 'boolean') return JSON.stringify(value)
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) return '[' + value.map((v) => stableStringify(v)).join(',') + ']'
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    const body = entries.map(([k, v]) => JSON.stringify(k) + ':' + stableStringify(v)).join(',')
    return '{' + body + '}'
  }
  // functions/symbols/etc.
  return 'null'
}

export function stableHash(obj: unknown): string {
  // Deep stable stringify then sha256
  const json = stableStringify(obj)
  return createHash('sha256').update(json).digest('hex')
}
