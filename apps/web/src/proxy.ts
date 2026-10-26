import { NextResponse, type NextRequest } from "next/server";

import { ACCESS_COOKIE, PATHNAME_HEADER } from "@/lib/session-cookies";

/**
 * Optimistic auth redirect: pages behind sign-in bounce to /login when there is no
 * session cookie at all. Real verification happens in the (app) layout and in the API,
 * which rejects every unauthenticated or revoked request.
 */
export function proxy(request: NextRequest) {
  const current = `${request.nextUrl.pathname}${request.nextUrl.search}`;
  if (request.cookies.has(ACCESS_COOKIE)) {
    // Let the server layout know where to send the user back after re-authentication.
    const headers = new Headers(request.headers);
    headers.set(PATHNAME_HEADER, current);
    return NextResponse.next({ request: { headers } });
  }
  const login = new URL("/login", request.url);
  if (current !== "/") login.searchParams.set("next", current);
  return NextResponse.redirect(login);
}

export const config = {
  matcher: ["/", "/workspaces/:path*", "/settings/:path*"],
};
