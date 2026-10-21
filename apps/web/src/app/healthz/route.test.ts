import { vi } from "vitest";

import { GET } from "./route";

beforeEach(() => {
  process.env.API_INTERNAL_URL = "http://api.internal:8000";
});

describe("/healthz", () => {
  it("is healthy only when the API answers", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(Response.json({ status: "ok", service: "api" }));
    const response = await GET();
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({ status: "ok", checks: { api: { status: "ok" } } });
    expect(String(fetchSpy.mock.calls[0]![0])).toBe("http://api.internal:8000/api/v1/health");
  });

  it("is unhealthy when the API returns an error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("down", { status: 503 }));
    const response = await GET();
    expect(response.status).toBe(503);
    expect(await response.json()).toMatchObject({
      status: "error",
      checks: { api: { status: "error", detail: "API returned HTTP 503" } },
    });
  });

  it("is unhealthy when the API is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    const response = await GET();
    expect(response.status).toBe(503);
  });
});
