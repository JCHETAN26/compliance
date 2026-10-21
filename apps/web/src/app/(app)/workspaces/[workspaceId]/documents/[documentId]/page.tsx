import type { Metadata } from "next";

import { DocumentViewerRoute } from "@/components/viewer/document-viewer";

export const metadata: Metadata = { title: "Document" };

export default async function DocumentPage({
  params,
}: {
  params: Promise<{ workspaceId: string; documentId: string }>;
}) {
  const { documentId } = await params;
  return <DocumentViewerRoute documentId={documentId} />;
}
