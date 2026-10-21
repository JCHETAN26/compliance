import type { ReactNode } from "react";

import { WorkspaceShell } from "@/components/workspaces/workspace-detail";

export default async function WorkspaceLayout({
  children,
  params,
}: {
  children: ReactNode;
  params: Promise<{ workspaceId: string }>;
}) {
  const { workspaceId } = await params;
  return <WorkspaceShell workspaceId={workspaceId}>{children}</WorkspaceShell>;
}
