-- CreateEnum
CREATE TYPE "Feature" AS ENUM ('design_by_type', 'critical_strain_analysis', 'multilayer_analysis', 'permissible_strain_analysis', 'effective_cbr_calc', 'design_then_hydrate');

-- CreateEnum
CREATE TYPE "RunStatus" AS ENUM ('pending', 'success', 'error');

-- CreateTable
CREATE TABLE "Run" (
    "id" TEXT NOT NULL,
    "feature" "Feature" NOT NULL,
    "status" "RunStatus" NOT NULL DEFAULT 'pending',
    "inputJson" JSONB NOT NULL,
    "outputJson" JSONB,
    "errorJson" JSONB,
    "inputHash" VARCHAR(128) NOT NULL,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finishedAt" TIMESTAMP(3),
    "durationMs" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Run_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Attachment" (
    "id" TEXT NOT NULL,
    "runId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "mimeType" TEXT NOT NULL,
    "data" BYTEA NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Attachment_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Run_feature_inputHash_idx" ON "Run"("feature", "inputHash");

-- AddForeignKey
ALTER TABLE "Attachment" ADD CONSTRAINT "Attachment_runId_fkey" FOREIGN KEY ("runId") REFERENCES "Run"("id") ON DELETE CASCADE ON UPDATE CASCADE;
