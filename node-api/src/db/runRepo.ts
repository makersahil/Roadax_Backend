import { Feature, Run, RunStatus, Prisma } from '@prisma/client'
import { prisma } from './client'

export async function createRun(feature: Feature, inputJson: unknown, inputHash: string): Promise<Run> {
  return prisma.run.create({ data: { feature, inputJson: (inputJson as any), inputHash } })
}

export async function completeRun(id: string, outputJson: unknown, startedAt?: Date): Promise<Run> {
  const finishedAt = new Date()
  const durationMs = startedAt ? finishedAt.getTime() - startedAt.getTime() : null
  return prisma.run.update({ where: { id }, data: { status: RunStatus.success, outputJson: (outputJson as any), finishedAt, durationMs: durationMs ?? undefined } })
}

export async function failRun(id: string, errorJson: unknown, startedAt?: Date): Promise<Run> {
  const finishedAt = new Date()
  const durationMs = startedAt ? finishedAt.getTime() - startedAt.getTime() : null
  return prisma.run.update({ where: { id }, data: { status: RunStatus.error, errorJson: (errorJson as any), finishedAt, durationMs: durationMs ?? undefined } })
}

export async function findCached(feature: Feature, inputHash: string): Promise<Run | null> {
  return prisma.run.findFirst({ where: { feature, inputHash, status: RunStatus.success } })
}
