import type { Metadata } from "next";

import { WorkspaceDocuments } from "@/components/documents/workspace-documents";

export const metadata: Metadata = { title: "Documents" };

export default function WorkspaceDocumentsPage() {
  return <WorkspaceDocuments />;
}
