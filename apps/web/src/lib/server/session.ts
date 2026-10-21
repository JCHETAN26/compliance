import "server-only";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import type { User } from "@/lib/api/client";
import { ACCESS_COOKIE } from "@/lib/session-cookies";

import { apiBaseUrl } from "./upstream";

/** Verify the session with the API; redirect to /login if it is missing or invalid. */
export async function requireUser(nextPath: string): Promise<User> {
  const cookieStore = await cookies();
  const token = cookieStore.get(ACCESS_COOKIE)?.value;
  const login = `/login?next=${encodeURIComponent(nextPath)}` as const;
  if (!token) redirect(login);

  const response = await fetch(`${apiBaseUrl()}/api/v1/auth/me`, {
    headers: { cookie: `${ACCESS_COOKIE}=${token}`, accept: "application/json" },
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (response.status === 401) redirect(`${login}&expired=1`);
  if (!response.ok) throw new Error(`Session check failed with HTTP ${response.status}`);
  return (await response.json()) as User;
}
