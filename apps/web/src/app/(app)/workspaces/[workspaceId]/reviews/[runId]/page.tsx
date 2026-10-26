import type { Metadata } from "next";

import { RunView } from "@/components/reviews/run-view";

export const metadata: Metadata = { title: "Review run" };

export default async function ReviewRunPage({ params }: { params: Promise<{ workspaceId: string; runId: string }> }) {
  const { runId } = await params;
  return <RunView runId={runId} />;
}
