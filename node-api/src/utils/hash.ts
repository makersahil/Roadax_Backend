import { createHash } from 'node:crypto'

export function stableHash(obj: unknown): string {
  // stable stringify then sha256
  const json = JSON.stringify(obj, Object.keys(obj as any).sort())
  return createHash('sha256').update(json).digest('hex')
}
