import createClient, { type Middleware } from "openapi-fetch";

import { CSRF_COOKIE, CSRF_HEADER, readCookie } from "@/lib/session-cookies";

import type { components, paths } from "./schema";

export type User = components["schemas"]["UserRead"];
export type Workspace = components["schemas"]["WorkspaceRead"];
export type Member = components["schemas"]["MemberRead"];
export type WorkspaceRole = Workspace["role"];
type ErrorEnvelope = components["schemas"]["ErrorResponse"];

const AUTH_ENTRY_PATHS = ["/api/v1/auth/login", "/api/v1/auth/register"];

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: ErrorEnvelope["error"]["details"];
  /** True when an authenticated request was rejected because the session ended. */
  readonly sessionExpired: boolean;

  constructor(status: number, body: unknown, path: string) {
    const envelope = isErrorEnvelope(body) ? body.error : undefined;
    super(envelope?.message ?? `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.code = envelope?.code ?? "error";
    this.details = envelope?.details;
    this.sessionExpired = status === 401 && !AUTH_ENTRY_PATHS.includes(path);
  }
}

function isErrorEnvelope(body: unknown): body is ErrorEnvelope {
  return (
    typeof body === "object" &&
    body !== null &&
    "error" in body &&
    typeof (body as { error: unknown }).error === "object" &&
    (body as { error: { message?: unknown } }).error !== null &&
    typeof (body as { error: { message?: unknown } }).error.message === "string"
  );
}

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/** Echo the session-bound CSRF cookie on state-changing requests. */
export const csrfMiddleware: Middleware = {
  onRequest({ request }) {
    if (SAFE_METHODS.has(request.method) || typeof document === "undefined") return request;
    const token = readCookie(document.cookie, CSRF_COOKIE);
    if (token) request.headers.set(CSRF_HEADER, token);
    return request;
  },
};

export function createApiClient(fetchImpl?: (request: Request) => Promise<Response>) {
  const client = createClient<paths>({
    baseUrl: typeof window === "undefined" ? "" : window.location.origin,
    credentials: "same-origin",
    // Resolve at call time so tests can stub the global fetch.
    fetch: fetchImpl ?? ((input: Request) => globalThis.fetch(input)),
  });
  client.use(csrfMiddleware);
  return client;
}

export const api = createApiClient();

/** Convert an openapi-fetch result into data or a thrown ApiError. */
export function unwrap<T>(
  result: { data?: T; error?: unknown; response: Response },
  path: string,
): T {
  if (result.error !== undefined || !result.response.ok) {
    throw new ApiError(result.response.status, result.error, path);
  }
  return result.data as T;
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "validation_error" && error.details?.length) {
      return error.details
        .map((d) => {
          const field = d.loc.filter((part) => part !== "body").join(".");
          return field ? `${field}: ${d.msg}` : d.msg;
        })
        .join("; ");
    }
    return error.message;
  }
  if (error instanceof TypeError) return "Could not reach the server. Check your connection.";
  return "Something went wrong. Please try again.";
}
