"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, LogOut, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createContext, useContext, type ReactNode } from "react";
import { toast } from "sonner";

import { ModelStatusBanner } from "@/components/model-status-banner";
import { Button } from "@/components/ui/button";
import { api, errorMessage, unwrap, type User } from "@/lib/api/client";

const UserContext = createContext<User | null>(null);

export function useCurrentUser(): User {
  const user = useContext(UserContext);
  if (!user) throw new Error("useCurrentUser must be used inside AppShell");
  return user;
}

export function AppShell({ user, children }: { user: User; children: ReactNode }) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const logout = useMutation({
    mutationFn: async () => unwrap(await api.POST("/api/v1/auth/logout"), "/api/v1/auth/logout"),
    onSuccess: () => {
      queryClient.clear();
      router.replace("/login");
      router.refresh();
    },
    onError: (error) => toast.error(errorMessage(error)),
  });

  return (
    <UserContext.Provider value={user}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="border-b bg-card">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
          <Link
            href="/workspaces"
            className="flex items-center gap-2 font-semibold focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
          >
            <ShieldCheck className="size-5 text-primary" aria-hidden />
            Compliance Workspace
          </Link>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-muted-foreground sm:inline" title="Signed in as">
              {user.email}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => logout.mutate()}
              disabled={logout.isPending}
            >
              {logout.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <LogOut aria-hidden />}
              Sign out
            </Button>
          </div>
        </div>
      </header>
      <ModelStatusBanner />
      <main id="main" className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {children}
      </main>
    </UserContext.Provider>
  );
}
