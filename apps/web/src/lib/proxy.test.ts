import { buildUpstreamUrl, forwardRequestHeaders, forwardResponseHeaders } from "./proxy";

describe("buildUpstreamUrl", () => {
  it("maps segments onto the API prefix and keeps the query string", () => {
    const url = buildUpstreamUrl("http://api:8000/", ["workspaces", "abc", "members"], "?x=1");
    expect(url?.toString()).toBe("http://api:8000/api/v1/workspaces/abc/members?x=1");
  });

  it("percent-encodes segment content", () => {
    const url = buildUpstreamUrl("http://api:8000", ["search", "a b?c#d"], "");
    expect(url?.pathname).toBe("/api/v1/search/a%20b%3Fc%23d");
  });

  it.each([[[".."]], [["workspaces", ".."]], [["."]], [["a/b"]], [["a\\b"]], [[""]]])(
    "rejects traversal or encoded separators %j",
    (segments) => {
      expect(buildUpstreamUrl("http://api:8000", segments, "")).toBeNull();
    },
  );
});

describe("header forwarding", () => {
  it("forwards only the allow-listed request headers", () => {
    const incoming = new Headers({
      cookie: "cw_access=t",
      "x-csrf-token": "c",
      "content-type": "application/json",
      host: "evil.example",
      authorization: "Bearer stolen",
      "x-forwarded-for": "1.2.3.4",
    });
    const forwarded = forwardRequestHeaders(incoming);
    expect(Object.fromEntries(forwarded.entries())).toEqual({
      cookie: "cw_access=t",
      "x-csrf-token": "c",
      "content-type": "application/json",
    });
  });

  it("keeps every Set-Cookie and drops hop-by-hop/encoding headers", () => {
    const upstream = new Headers();
    upstream.append("set-cookie", "cw_access=a; HttpOnly; Path=/");
    upstream.append("set-cookie", "cw_csrf=b; Path=/");
    upstream.set("content-type", "application/json");
    upstream.set("content-encoding", "gzip");
    upstream.set("content-length", "10");
    upstream.set("connection", "keep-alive");
    upstream.set("x-request-id", "rid");
    const out = forwardResponseHeaders(upstream);
    expect(out.getSetCookie()).toEqual(["cw_access=a; HttpOnly; Path=/", "cw_csrf=b; Path=/"]);
    expect(out.get("content-type")).toBe("application/json");
    expect(out.get("x-request-id")).toBe("rid");
    expect(out.has("content-encoding")).toBe(false);
    expect(out.has("content-length")).toBe(false);
    expect(out.has("connection")).toBe(false);
  });
});
