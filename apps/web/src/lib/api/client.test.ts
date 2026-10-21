import { ApiError, createApiClient, errorMessage, unwrap } from "./client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("ApiError", () => {
  it("reads the API error envelope", () => {
    const error = new ApiError(
      409,
      { error: { code: "conflict", message: "Already a member." } },
      "/api/v1/workspaces/{workspace_id}/members",
    );
    expect(error.message).toBe("Already a member.");
    expect(error.code).toBe("conflict");
    expect(error.sessionExpired).toBe(false);
  });

  it("flags 401s on authenticated endpoints as an expired session", () => {
    expect(new ApiError(401, {}, "/api/v1/workspaces").sessionExpired).toBe(true);
    expect(new ApiError(401, {}, "/api/v1/auth/login").sessionExpired).toBe(false);
    expect(new ApiError(401, {}, "/api/v1/auth/register").sessionExpired).toBe(false);
  });

  it("falls back to a generic message for non-envelope bodies", () => {
    expect(new ApiError(502, "<html>", "/x").message).toBe("Request failed with status 502");
  });
});

describe("errorMessage", () => {
  it("flattens validation details without the body prefix", () => {
    const error = new ApiError(
      422,
      {
        error: {
          code: "validation_error",
          message: "Request validation failed.",
          details: [{ loc: ["body", "name"], msg: "String should have at least 1 character", type: "x" }],
        },
      },
      "/api/v1/workspaces",
    );
    expect(errorMessage(error)).toBe("name: String should have at least 1 character");
  });

  it("explains network failures", () => {
    expect(errorMessage(new TypeError("fetch failed"))).toMatch(/Could not reach the server/);
  });
});

describe("API client", () => {
  it("sends the CSRF cookie value as a header on unsafe methods only", async () => {
    document.cookie = "cw_csrf=token-123; path=/";
    const seen: Request[] = [];
    const client = createApiClient(async (request: Request) => {
      seen.push(request);
      return request.method === "GET"
        ? jsonResponse(200, [])
        : jsonResponse(201, {
            id: "1",
            name: "W",
            owner_id: "u",
            created_at: "2026-01-01T00:00:00Z",
            role: "owner",
            member_count: 1,
          });
    });
    await client.GET("/api/v1/workspaces");
    await client.POST("/api/v1/workspaces", { body: { name: "W" } });
    expect(seen[0]?.headers.get("X-CSRF-Token")).toBeNull();
    expect(seen[1]?.headers.get("X-CSRF-Token")).toBe("token-123");
    expect(new URL(seen[1]!.url).pathname).toBe("/api/v1/workspaces");
  });

  it("unwrap throws ApiError for error responses", async () => {
    const client = createApiClient(async () =>
      jsonResponse(404, { error: { code: "not_found", message: "Workspace not found." } }),
    );
    const result = await client.GET("/api/v1/workspaces/{workspace_id}", {
      params: { path: { workspace_id: "x" } },
    });
    expect(() => unwrap(result, "/api/v1/workspaces/{workspace_id}")).toThrowError(
      "Workspace not found.",
    );
  });
});
