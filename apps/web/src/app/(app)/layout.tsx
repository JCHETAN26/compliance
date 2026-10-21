import { headers } from "next/headers";
import type { ReactNode } from "react";

import { AppShell } from "@/components/app-shell";
import { safeNextPath } from "@/lib/redirect";
import { requireUser } from "@/lib/server/session";
import { PATHNAME_HEADER } from "@/lib/session-cookies";

export default async function AuthenticatedLayout({ children }: { children: ReactNode }) {
  const requestHeaders = await headers();
  const user = await requireUser(safeNextPath(requestHeaders.get(PATHNAME_HEADER)));
  return <AppShell user={user}>{children}</AppShell>;
}
