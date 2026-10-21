import type { NextRequest } from "next/server";

import { buildUpstreamUrl, forwardRequestHeaders, forwardResponseHeaders } from "@/lib/proxy";
import { apiBaseUrl } from "@/lib/server/upstream";

/**
 * Same-origin proxy to the FastAPI service. The browser only ever talks to the web
 * origin, so the httpOnly session cookie is first-party and SameSite=Lax applies.
 * Bodies are streamed in both directions (uploads and SSE in later phases).
 */
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

async function proxy(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  const upstreamUrl = buildUpstreamUrl(apiBaseUrl(), path, request.nextUrl.search);
  if (upstreamUrl === null) {
    return Response.json(
      { error: { code: "bad_request", message: "Invalid API path." } },
      { status: 400 },
    );
  }

  const hasBody = !["GET", "HEAD"].includes(request.method);
  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: request.method,
      headers: forwardRequestHeaders(request.headers),
      body: hasBody ? request.body : undefined,
      // Required by Node's fetch when streaming a request body.
      ...(hasBody ? { duplex: "half" } : {}),
      redirect: "manual",
      cache: "no-store",
      signal: request.signal,
    } as RequestInit);
  } catch {
    return Response.json(
      { error: { code: "bad_gateway", message: "The API service is unavailable." } },
      { status: 502 },
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: forwardResponseHeaders(upstream.headers),
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
