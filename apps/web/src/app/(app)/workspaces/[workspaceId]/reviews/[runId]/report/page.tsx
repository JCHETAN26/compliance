import type { Metadata } from "next";

import { ReportView } from "@/components/reviews/report-view";

export const metadata: Metadata = { title: "Review report" };

export default async function ReviewReportPage({
  params,
}: {
  params: Promise<{ workspaceId: string; runId: string }>;
}) {
  const { runId } = await params;
  return <ReportView runId={runId} />;
}
