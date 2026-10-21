import type { Metadata } from "next";

import { WorkspaceSettings } from "@/components/workspaces/workspace-detail";

export const metadata: Metadata = { title: "Workspace settings" };

export default function WorkspaceSettingsPage() {
  return <WorkspaceSettings />;
}
