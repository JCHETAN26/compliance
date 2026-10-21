import { apiBaseUrl } from "@/lib/server/upstream";

/**
 * Container healthcheck. Healthy only if this Next.js server is serving requests AND
 * can reach the API through the same upstream URL the /api/v1 proxy uses.
 */
export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  let api: "ok" | "error" = "error";
  let detail: string | undefined;
  try {
    const response = await fetch(`${apiBaseUrl()}/api/v1/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3000),
    });
    api = response.ok ? "ok" : "error";
    if (!response.ok) detail = `API returned HTTP ${response.status}`;
  } catch (error) {
    detail = error instanceof Error ? error.message : "unreachable";
  }
  const status = api === "ok" ? "ok" : "error";
  return Response.json(
    { status, service: "web", checks: { api: { status: api, detail } } },
    { status: status === "ok" ? 200 : 503, headers: { "Cache-Control": "no-store" } },
  );
}
