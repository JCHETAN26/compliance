"use client";

import { useWorkspace } from "@/components/workspaces/workspace-detail";

import { DocumentsPage } from "./documents-page";

export function WorkspaceDocuments() {
  return <DocumentsPage workspace={useWorkspace()} />;
}
