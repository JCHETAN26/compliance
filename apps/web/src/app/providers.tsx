"use client";

import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { Toaster } from "sonner";

import { ApiError } from "@/lib/api/client";

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 15_000,
        // Client errors (4xx) are final; only retry transient failures.
        retry: (failureCount, error) =>
          !(error instanceof ApiError && error.status < 500) && failureCount < 2,
      },
    },
  });
}

/** Sends the user to /login when any authenticated query or mutation returns 401. */
export function SessionExpiryRedirect() {
  const client = useQueryClient();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const check = (error: unknown) => {
      if (error instanceof ApiError && error.sessionExpired) {
        client.clear();
        router.replace(`/login?expired=1&next=${encodeURIComponent(pathname)}`);
      }
    };
    const unsubscribeQueries = client.getQueryCache().subscribe((event) => {
      if (event.type === "updated" && event.action.type === "error") check(event.action.error);
    });
    const unsubscribeMutations = client.getMutationCache().subscribe((event) => {
      if (event.type === "updated" && event.action.type === "error") check(event.action.error);
    });
    return () => {
      unsubscribeQueries();
      unsubscribeMutations();
    };
  }, [client, router, pathname]);

  return null;
}

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(makeQueryClient);
  return (
    <QueryClientProvider client={client}>
      <SessionExpiryRedirect />
      {children}
      <Toaster richColors position="bottom-right" closeButton />
    </QueryClientProvider>
  );
}
