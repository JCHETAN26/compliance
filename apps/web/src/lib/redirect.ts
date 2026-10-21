/**
 * Returns a same-origin path to continue to after sign-in, or the fallback.
 * Rejects absolute URLs, protocol-relative URLs and backslash tricks (open-redirect safe).
 */
export function safeNextPath(raw: string | null | undefined, fallback = "/workspaces"): string {
  if (!raw || raw.length > 512) return fallback;
  if (!raw.startsWith("/") || raw.startsWith("//") || raw.includes("\\")) return fallback;
  if (/[\u0000-\u001f]/.test(raw)) return fallback;
  if (raw.startsWith("/login") || raw.startsWith("/signup")) return fallback;
  return raw;
}
