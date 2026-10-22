import { api, unwrap, type SearchRequest } from "./client";

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

export const documentKeys = {
  list: (workspaceId: string) => ["workspaces", workspaceId, "documents"] as const,
  detail: (documentId: string) => ["documents", documentId] as const,
  chunks: (documentId: string) => ["documents", documentId, "chunks"] as const,
  pages: (documentId: string) => ["documents", documentId, "pages"] as const,
  blocks: (documentId: string) => ["documents", documentId, "blocks"] as const,
  sheets: (documentId: string) => ["documents", documentId, "sheets"] as const,
  rows: (documentId: string, sheet: number) => ["documents", documentId, "rows", sheet] as const,
  entities: (documentId: string) => ["documents", documentId, "entities"] as const,
};

export async function fetchDocuments(workspaceId: string) {
  return unwrap(
    await api.GET("/api/v1/workspaces/{workspace_id}/documents", {
      params: { path: { workspace_id: workspaceId } },
    }),
    "/api/v1/workspaces/{workspace_id}/documents",
  );
}

export async function fetchDocument(documentId: string) {
  return unwrap(
    await api.GET("/api/v1/documents/{document_id}", {
      params: { path: { document_id: documentId } },
    }),
    "/api/v1/documents/{document_id}",
  );
}

/** All chunks of a document (pages through the API's 2000-item limit). */
export async function fetchAllChunks(documentId: string) {
  const items = [];
  for (let offset = 0; ; offset += 2000) {
    const page = unwrap(
      await api.GET("/api/v1/documents/{document_id}/chunks", {
        params: { path: { document_id: documentId }, query: { offset, limit: 2000 } },
      }),
      "/api/v1/documents/{document_id}/chunks",
    );
    items.push(...page.items);
    if (items.length >= page.total || page.items.length === 0) return items;
  }
}

export async function fetchPages(documentId: string) {
  return unwrap(
    await api.GET("/api/v1/documents/{document_id}/pages", {
      params: { path: { document_id: documentId }, query: { include_text: true } },
    }),
    "/api/v1/documents/{document_id}/pages",
  );
}

export async function fetchBlocks(documentId: string) {
  return unwrap(
    await api.GET("/api/v1/documents/{document_id}/blocks", {
      params: { path: { document_id: documentId } },
    }),
    "/api/v1/documents/{document_id}/blocks",
  );
}

export async function fetchSheets(documentId: string) {
  return unwrap(
    await api.GET("/api/v1/documents/{document_id}/sheets", {
      params: { path: { document_id: documentId } },
    }),
    "/api/v1/documents/{document_id}/sheets",
  );
}

export async function fetchRows(documentId: string, sheetIndex: number, offset = 0, limit = 1000) {
  return unwrap(
    await api.GET("/api/v1/documents/{document_id}/rows", {
      params: {
        path: { document_id: documentId },
        query: { sheet_index: sheetIndex, offset, limit },
      },
    }),
    "/api/v1/documents/{document_id}/rows",
  );
}

export async function fetchEntities(documentId: string) {
  return unwrap(
    await api.GET("/api/v1/documents/{document_id}/entities", {
      params: { path: { document_id: documentId } },
    }),
    "/api/v1/documents/{document_id}/entities",
  );
}

export function documentFileUrl(documentId: string, download = false): string {
  return `/api/v1/documents/${encodeURIComponent(documentId)}/file${download ? "?download=true" : ""}`;
}

export const searchKeys = {
  capabilities: ["search", "capabilities"] as const,
  results: (workspaceId: string, request: SearchRequest) =>
    ["workspaces", workspaceId, "search", request] as const,
};

export async function fetchSearchCapabilities() {
  return unwrap(await api.GET("/api/v1/search/capabilities"), "/api/v1/search/capabilities");
}

export async function searchWorkspace(workspaceId: string, body: SearchRequest) {
  return unwrap(
    await api.POST("/api/v1/workspaces/{workspace_id}/search", {
      params: { path: { workspace_id: workspaceId } },
      body,
    }),
    "/api/v1/workspaces/{workspace_id}/search",
  );
}

export const modelKeys = {
  status: ["models", "status"] as const,
};

export async function fetchModelStatus() {
  return unwrap(await api.GET("/api/v1/models/status"), "/api/v1/models/status");
}
