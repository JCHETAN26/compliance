/** Cookie names set by the API (see apps/api/src/compliance_api/deps.py). */
export const ACCESS_COOKIE = "cw_access";
export const CSRF_COOKIE = "cw_csrf";
export const CSRF_HEADER = "X-CSRF-Token";
/** Request header set by src/proxy.ts with the path being rendered. */
export const PATHNAME_HEADER = "x-cw-pathname";

export function readCookie(cookieHeader: string, name: string): string | null {
  for (const part of cookieHeader.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return decodeURIComponent(rest.join("="));
  }
  return null;
}
