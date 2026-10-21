import { Badge } from "@/components/ui/badge";
import type { WorkspaceRole } from "@/lib/api/client";

const LABELS: Record<WorkspaceRole, string> = {
  owner: "Owner",
  reviewer: "Reviewer",
  viewer: "Viewer",
};

export function RoleBadge({ role }: { role: WorkspaceRole }) {
  return (
    <Badge variant={role === "owner" ? "default" : role === "reviewer" ? "secondary" : "outline"}>
      {LABELS[role]}
    </Badge>
  );
}
