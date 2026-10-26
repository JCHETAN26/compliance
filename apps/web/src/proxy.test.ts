import { NextRequest } from "next/server";

import { PATHNAME_HEADER } from "@/lib/session-cookies";

import { config, proxy } from "./proxy";

describe("auth proxy", () => {
  it("redirects to /login with the requested path when there is no session cookie", () => {
    const response = proxy(new NextRequest("http://localhost:3000/workspaces/abc?tab=1"));
    expect(response.status).toBe(307);
    const location = new URL(response.headers.get("location")!);
    expect(location.pathname).toBe("/login");
    expect(location.searchParams.get("next")).toBe("/workspaces/abc?tab=1");
  });

  it("omits next for the root path", () => {
    const response = proxy(new NextRequest("http://localhost:3000/"));
    expect(response.headers.get("location")).toBe("http://localhost:3000/login");
  });

  it("lets requests with a session cookie through and records the path", () => {
    const request = new NextRequest("http://localhost:3000/workspaces", {
      headers: { cookie: "cw_access=token" },
    });
    const response = proxy(request);
    expect(response.headers.get("location")).toBeNull();
    expect(response.headers.get(`x-middleware-request-${PATHNAME_HEADER}`)).toBe("/workspaces");
  });

  it("only guards application pages", () => {
    expect(config.matcher).toEqual(["/", "/workspaces/:path*", "/settings/:path*"]);
  });
});
