import "server-only";

/** Base URL of the FastAPI service as reachable from the Next.js server (not the browser). */
export function apiBaseUrl(): string {
  return (process.env.API_INTERNAL_URL ?? "http://localhost:8000").replace(/\/+$/, "");
}
