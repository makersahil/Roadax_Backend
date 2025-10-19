import { spawn } from 'node:child_process'

export type BridgeOk<T = unknown> = { ok: true; data: T }
export type BridgeErr = { ok: false; error: { type: string; message: string; details?: unknown } }
export type BridgeResult<T = unknown> = BridgeOk<T> | BridgeErr

function getEnv(name: string, fallback?: string): string {
  const v = process.env[name]
  if (!v || v.trim() === '') {
    if (fallback !== undefined) return fallback
    throw new Error(`Missing env ${name}`)
  }
  return v
}

export async function runBridge<T = unknown>(payload: unknown): Promise<BridgeResult<T>> {
  const PYTHON_BIN = getEnv('PYTHON_BIN', 'python3')
  const BRIDGE = getEnv('PYTHON_BRIDGE_PATH', '../Roadax_Python_Functions/python_api_bridge.py')
  const TIMEOUT_MS = Number(process.env.BRIDGE_TIMEOUT_MS || 60000)

  return new Promise((resolve) => {
    const child = spawn(PYTHON_BIN, [BRIDGE], { stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    const to = setTimeout(() => {
      child.kill('SIGKILL')
    }, TIMEOUT_MS)

    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (d) => { stdout += String(d) })
    child.stderr.on('data', (d) => { stderr += String(d) })

    child.on('error', (err) => {
      clearTimeout(to)
      resolve({ ok: false, error: { type: 'SpawnError', message: String(err), details: { stderr } } })
    })

    child.on('close', (_code) => {
      clearTimeout(to)
      try {
        const parsed = JSON.parse(stdout)
        if (parsed && typeof parsed === 'object' && 'ok' in parsed) {
          resolve(parsed as BridgeResult<T>)
        } else {
          resolve({ ok: false, error: { type: 'InvalidBridgeResponse', message: 'No ok field', details: parsed } })
        }
      } catch (e) {
        resolve({ ok: false, error: { type: 'ParseError', message: String(e), details: { stdout, stderr } } })
      }
    })

    child.stdin.write(JSON.stringify(payload))
    child.stdin.end()
  })
}

export function jsonNormalize(value: unknown): unknown {
  // Replace NaN/Infinity with null recursively
  if (value === null || value === undefined) return null
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }
  if (Array.isArray(value)) return value.map(jsonNormalize)
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = jsonNormalize(v)
    }
    return out
  }
  return value
}

export async function runNoop(): Promise<BridgeResult<{ uptime: boolean; python: string }>> {
  // lightweight health call that doesn't import heavy modules
  return runBridge<{ uptime: boolean; python: string }>({ action: 'noop' })
}

export async function runFeature<T = unknown>(feature: string, args: unknown): Promise<BridgeResult<T>> {
  return runBridge<T>({ feature, args })
}
