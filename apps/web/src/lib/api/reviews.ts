import {
  api,
  unwrap,
  type Report,
  type ReviewSettingsUpdate,
  type RulePackSpec,
  type StartReview,
} from "./client";

export const rulePackKeys = {
  list: (workspaceId: string) => ["workspaces", workspaceId, "rule-packs"] as const,
  detail: (packId: string) => ["rule-packs", packId] as const,
};

export const reviewKeys = {
  list: (workspaceId: string) => ["workspaces", workspaceId, "reviews"] as const,
  run: (runId: string) => ["reviews", runId] as const,
  tasks: (runId: string) => ["reviews", runId, "tasks"] as const,
  findings: (runId: string) => ["reviews", runId, "findings"] as const,
  report: (runId: string) => ["reviews", runId, "report"] as const,
  citation: (citationId: string) => ["citations", citationId] as const,
};

export const settingsKeys = {
  llm: ["settings", "llm"] as const,
  review: (workspaceId: string) => ["workspaces", workspaceId, "review-settings"] as const,
};

export async function fetchRulePacks(workspaceId: string) {
  return unwrap(
    await api.GET("/api/v1/rule-packs", { params: { query: { workspace_id: workspaceId } } }),
    "/api/v1/rule-packs",
  );
}

export async function fetchRulePack(packId: string) {
  return unwrap(
    await api.GET("/api/v1/rule-packs/{rule_pack_id}", { params: { path: { rule_pack_id: packId } } }),
    "/api/v1/rule-packs/{rule_pack_id}",
  );
}

export async function importRulePack(workspaceId: string, yaml: string) {
  return unwrap(
    await api.POST("/api/v1/rule-packs/import", { body: { workspace_id: workspaceId, yaml } }),
    "/api/v1/rule-packs/import",
  );
}

export async function updateRulePack(packId: string, spec: RulePackSpec) {
  return unwrap(
    await api.PUT("/api/v1/rule-packs/{rule_pack_id}", {
      params: { path: { rule_pack_id: packId } },
      body: spec,
    }),
    "/api/v1/rule-packs/{rule_pack_id}",
  );
}

export function rulePackExportUrl(packId: string): string {
  return `/api/v1/rule-packs/${encodeURIComponent(packId)}/export`;
}

export async function exportRulePack(packId: string): Promise<string> {
  const response = await fetch(rulePackExportUrl(packId), { credentials: "same-origin" });
  if (!response.ok) throw new Error(`Export failed with status ${response.status}`);
  return response.text();
}

export async function fetchReviews(workspaceId: string) {
  return unwrap(
    await api.GET("/api/v1/workspaces/{workspace_id}/reviews", {
      params: { path: { workspace_id: workspaceId } },
    }),
    "/api/v1/workspaces/{workspace_id}/reviews",
  );
}

export async function startReview(workspaceId: string, body: StartReview) {
  return unwrap(
    await api.POST("/api/v1/workspaces/{workspace_id}/reviews", {
      params: { path: { workspace_id: workspaceId } },
      body,
    }),
    "/api/v1/workspaces/{workspace_id}/reviews",
  );
}

export async function fetchRun(runId: string) {
  return unwrap(
    await api.GET("/api/v1/reviews/{run_id}", { params: { path: { run_id: runId } } }),
    "/api/v1/reviews/{run_id}",
  );
}

export async function controlRun(runId: string, action: "pause" | "resume" | "cancel") {
  return unwrap(
    await api.POST("/api/v1/reviews/{run_id}/control", {
      params: { path: { run_id: runId } },
      body: { action },
    }),
    "/api/v1/reviews/{run_id}/control",
  );
}

export async function fetchTasks(runId: string) {
  return unwrap(
    await api.GET("/api/v1/reviews/{run_id}/tasks", { params: { path: { run_id: runId } } }),
    "/api/v1/reviews/{run_id}/tasks",
  );
}

export async function fetchFindings(runId: string) {
  return unwrap(
    await api.GET("/api/v1/reviews/{run_id}/findings", {
      params: { path: { run_id: runId }, query: { limit: 1000 } },
    }),
    "/api/v1/reviews/{run_id}/findings",
  );
}

export async function overrideFinding(
  findingId: string,
  body: { status: "satisfied" | "violated" | "not_mentioned" | "needs_review"; comment: string },
) {
  return unwrap(
    await api.POST("/api/v1/findings/{finding_id}/override", {
      params: { path: { finding_id: findingId } },
      body,
    }),
    "/api/v1/findings/{finding_id}/override",
  );
}

export async function fetchReport(runId: string): Promise<Report> {
  // The endpoint also serves Markdown (?format=md), so the generated type is a union.
  const result = await api.GET("/api/v1/reviews/{run_id}/report", {
    params: { path: { run_id: runId }, query: { format: "json" } },
  });
  return unwrap<unknown>(result, "/api/v1/reviews/{run_id}/report") as Report;
}

export function reportExportUrl(runId: string, format: "md" | "json"): string {
  return `/api/v1/reviews/${encodeURIComponent(runId)}/report?format=${format}`;
}

export async function fetchCitation(citationId: string) {
  return unwrap(
    await api.GET("/api/v1/citations/{citation_id}", {
      params: { path: { citation_id: citationId } },
    }),
    "/api/v1/citations/{citation_id}",
  );
}

export async function fetchEventHistory(runId: string, after = 0, limit = 1000) {
  return unwrap(
    await api.GET("/api/v1/reviews/{run_id}/events/history", {
      params: { path: { run_id: runId }, query: { after, limit } },
    }),
    "/api/v1/reviews/{run_id}/events/history",
  );
}

export function eventStreamUrl(runId: string, after: number): string {
  return `/api/v1/reviews/${encodeURIComponent(runId)}/events?after=${after}`;
}

export async function fetchLlmSettings() {
  return unwrap(await api.GET("/api/v1/settings/llm"), "/api/v1/settings/llm");
}

export async function fetchReviewSettings(workspaceId: string) {
  return unwrap(
    await api.GET("/api/v1/workspaces/{workspace_id}/review-settings", {
      params: { path: { workspace_id: workspaceId } },
    }),
    "/api/v1/workspaces/{workspace_id}/review-settings",
  );
}

export async function saveReviewSettings(workspaceId: string, body: ReviewSettingsUpdate) {
  return unwrap(
    await api.PUT("/api/v1/workspaces/{workspace_id}/review-settings", {
      params: { path: { workspace_id: workspaceId } },
      body,
    }),
    "/api/v1/workspaces/{workspace_id}/review-settings",
  );
}
