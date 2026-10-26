/**
 * Pure helpers for the same-origin `/api/v1/*` backend-for-frontend proxy.
 * Kept free of Next.js imports so they are unit-testable.
 */

/** Request headers forwarded to the API. Everything else (e.g. Host) is dropped. */
export const FORWARDED_REQUEST_HEADERS = [
  "accept",
  "accept-language",
  "content-type",
  "cookie",
  "last-event-id",
  "user-agent",
  "x-csrf-token",
  "x-request-id",
] as const;

/** Hop-by-hop or body-encoding headers that must not be copied to the client. */
const DROPPED_RESPONSE_HEADERS = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "set-cookie",
]);

/**
 * Build the upstream URL for a proxied request, or return null if the path is unsafe
 * (dot segments or encoded slashes could otherwise escape the /api/v1 prefix).
 */
export function buildUpstreamUrl(base: string, segments: string[], search: string): URL | null {
  for (const segment of segments) {
    if (segment === "" || segment === "." || segment === ".." || /[/\\]/.test(segment)) {
      return null;
    }
  }
  const path = segments.map(encodeURIComponent).join("/");
  const url = new URL(`${base.replace(/\/+$/, "")}/api/v1/${path}`);
  if (!url.pathname.startsWith("/api/v1/")) return null;
  url.search = search;
  return url;
}

export function forwardRequestHeaders(incoming: Headers): Headers {
  const headers = new Headers();
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = incoming.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

export function forwardResponseHeaders(upstream: Headers): Headers {
  const headers = new Headers();
  upstream.forEach((value, name) => {
    if (!DROPPED_RESPONSE_HEADERS.has(name.toLowerCase())) headers.set(name, value);
  });
  // Multiple Set-Cookie headers must stay separate, so copy them individually.
  for (const cookie of upstream.getSetCookie()) headers.append("set-cookie", cookie);
  return headers;
}
