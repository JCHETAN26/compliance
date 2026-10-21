import { NextRequest } from "next/server";
import { vi } from "vitest";

import { DELETE, GET, POST } from "./route";

function ctx(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

beforeEach(() => {
  process.env.API_INTERNAL_URL = "http://api.internal:8000";
});

describe("/api/v1 proxy route", () => {
  it("forwards method, path, query, cookies, CSRF header and body", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ id: "w1" }), {
        status: 201,
        headers: { "content-type": "application/json" },
      }),
    );
    const request = new NextRequest("http://localhost:3000/api/v1/workspaces?x=1", {
      method: "POST",
      headers: {
        cookie: "cw_access=tok; cw_csrf=c",
        "x-csrf-token": "c",
        "content-type": "application/json",
        host: "localhost:3000",
      },
      body: JSON.stringify({ name: "W" }),
    });
    const response = await POST(request, ctx(["workspaces"]));
    expect(response.status).toBe(201);
    expect(await response.json()).toEqual({ id: "w1" });

    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(String(url)).toBe("http://api.internal:8000/api/v1/workspaces?x=1");
    const headers = new Headers((init as RequestInit).headers);
    expect(headers.get("cookie")).toBe("cw_access=tok; cw_csrf=c");
    expect(headers.get("x-csrf-token")).toBe("c");
    expect(headers.get("host")).toBeNull();
    expect((init as RequestInit).method).toBe("POST");
    const body = await new Response((init as RequestInit).body).text();
    expect(JSON.parse(body)).toEqual({ name: "W" });
  });

  it("passes through multiple Set-Cookie headers from the API", async () => {
    const upstreamHeaders = new Headers({ "content-type": "application/json" });
    upstreamHeaders.append("set-cookie", "cw_access=a; HttpOnly; Path=/; SameSite=lax");
    upstreamHeaders.append("set-cookie", "cw_csrf=b; Path=/; SameSite=lax");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("{}", { status: 200, headers: upstreamHeaders }),
    );
    const request = new NextRequest("http://localhost:3000/api/v1/auth/login", {
      method: "POST",
      body: "{}",
    });
    const response = await POST(request, ctx(["auth", "login"]));
    expect(response.headers.getSetCookie()).toEqual([
      "cw_access=a; HttpOnly; Path=/; SameSite=lax",
      "cw_csrf=b; Path=/; SameSite=lax",
    ]);
  });

  it("returns 204 responses without a body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 204 }));
    const request = new NextRequest("http://localhost:3000/api/v1/workspaces/1", {
      method: "DELETE",
    });
    const response = await DELETE(request, ctx(["workspaces", "1"]));
    expect(response.status).toBe(204);
    expect(await response.text()).toBe("");
  });

  it("rejects path traversal without contacting the API", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const request = new NextRequest("http://localhost:3000/api/v1/x");
    const response = await GET(request, ctx(["..", "internal"]));
    expect(response.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("reports 502 in the error envelope when the API is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    const response = await GET(new NextRequest("http://localhost:3000/api/v1/auth/me"), ctx(["auth", "me"]));
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({
      error: { code: "bad_gateway", message: "The API service is unavailable." },
    });
  });
});
