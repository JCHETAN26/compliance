import { api, unwrap } from "./client";

export const queryKeys = {
  workspaces: ["workspaces"] as const,
  workspace: (id: string) => ["workspaces", id] as const,
  members: (id: string) => ["workspaces", id, "members"] as const,
};

export async function fetchWorkspaces() {
  return unwrap(await api.GET("/api/v1/workspaces"), "/api/v1/workspaces");
}

export async function fetchWorkspace(workspaceId: string) {
  return unwrap(
    await api.GET("/api/v1/workspaces/{workspace_id}", {
      params: { path: { workspace_id: workspaceId } },
    }),
    "/api/v1/workspaces/{workspace_id}",
  );
}

export async function fetchMembers(workspaceId: string) {
  return unwrap(
    await api.GET("/api/v1/workspaces/{workspace_id}/members", {
      params: { path: { workspace_id: workspaceId } },
    }),
    "/api/v1/workspaces/{workspace_id}/members",
  );
}
