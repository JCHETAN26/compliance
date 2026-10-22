"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowLeft, Loader2 } from "lucide-react";
import type { Route } from "next";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useId, useState, type FormEvent, type ReactNode } from "react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api, errorMessage, unwrap, type Workspace } from "@/lib/api/client";
import { fetchWorkspace, queryKeys } from "@/lib/api/queries";
import { formatDate } from "@/lib/utils";

import { MembersPanel } from "./members-panel";
import { RoleBadge } from "./role-badge";
import { WORKSPACE_NAME_MAX } from "./workspace-list";

const WorkspaceContext = createContext<Workspace | null>(null);

export function useWorkspace(): Workspace {
  const workspace = useContext(WorkspaceContext);
  if (!workspace) throw new Error("useWorkspace must be used inside WorkspaceShell");
  return workspace;
}

/** Loads the workspace once and renders its header, tabs and the active tab. */
export function WorkspaceShell({ workspaceId, children }: { workspaceId: string; children: ReactNode }) {
  const pathname = usePathname();
  const workspace = useQuery({
    queryKey: queryKeys.workspace(workspaceId),
    queryFn: () => fetchWorkspace(workspaceId),
  });
  const base = `/workspaces/${workspaceId}`;
  const tabs = [
    { href: base, label: "Documents", active: pathname === base || pathname.startsWith(`${base}/documents`) },
    { href: `${base}/search`, label: "Search", active: pathname.startsWith(`${base}/search`) },
    { href: `${base}/settings`, label: "Settings & members", active: pathname.startsWith(`${base}/settings`) },
  ] as const;

  return (
    <div className="grid gap-6">
      <Link
        href="/workspaces"
        className="inline-flex w-fit items-center gap-1 text-sm text-muted-foreground hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
      >
        <ArrowLeft className="size-4" aria-hidden />
        All workspaces
      </Link>
      {workspace.isPending ? (
        <div className="grid gap-4" aria-busy="true" aria-label="Loading workspace">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-40 rounded-xl" />
        </div>
      ) : workspace.isError ? (
        <WorkspaceLoadError error={workspace.error} onRetry={() => workspace.refetch()} />
      ) : (
        <WorkspaceContext.Provider value={workspace.data}>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight break-words">{workspace.data.name}</h1>
            <RoleBadge role={workspace.data.role} />
          </div>
          <nav aria-label="Workspace sections" className="-mt-2 flex gap-1 border-b">
            {tabs.map((tab) => (
              <Link
                key={tab.href}
                href={tab.href as Route}
                aria-current={tab.active ? "page" : undefined}
                className={
                  "-mb-px border-b-2 px-3 py-2 text-sm font-medium focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none " +
                  (tab.active
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground")
                }
              >
                {tab.label}
              </Link>
            ))}
          </nav>
          {children}
        </WorkspaceContext.Provider>
      )}
    </div>
  );
}

export function WorkspaceSettings() {
  return <LoadedWorkspace workspace={useWorkspace()} />;
}

function WorkspaceLoadError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const notFound = error instanceof ApiError && (error.status === 404 || error.status === 422);
  return (
    <Alert variant="destructive">
      <AlertTriangle aria-hidden />
      <AlertTitle>{notFound ? "Workspace not found" : "Could not load this workspace"}</AlertTitle>
      <AlertDescription>
        {notFound
          ? "It may have been deleted, or you are not a member of it."
          : errorMessage(error)}
        {!notFound ? (
          <div className="mt-3">
            <Button variant="outline" size="sm" onClick={onRetry}>
              Retry
            </Button>
          </div>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}

function LoadedWorkspace({ workspace }: { workspace: Workspace }) {
  const isOwner = workspace.role === "owner";
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Details</CardTitle>
          <CardDescription>
            Created {formatDate(workspace.created_at)} · {workspace.member_count}{" "}
            {workspace.member_count === 1 ? "member" : "members"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isOwner ? (
            <RenameForm workspace={workspace} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Only the workspace owner can rename or delete this workspace.
            </p>
          )}
        </CardContent>
      </Card>
      <MembersPanel workspace={workspace} />
      {isOwner ? <DangerZone workspace={workspace} /> : null}
    </>
  );
}

function useWorkspaceCacheUpdate() {
  const queryClient = useQueryClient();
  return (updated: Workspace) => {
    queryClient.setQueryData(queryKeys.workspace(updated.id), updated);
    void queryClient.invalidateQueries({ queryKey: queryKeys.workspaces, exact: true });
  };
}

function RenameForm({ workspace }: { workspace: Workspace }) {
  const id = useId();
  const [name, setName] = useState(workspace.name);
  const [validationError, setValidationError] = useState<string | null>(null);
  const updateCache = useWorkspaceCacheUpdate();
  const rename = useMutation({
    mutationFn: async (newName: string) =>
      unwrap(
        await api.PATCH("/api/v1/workspaces/{workspace_id}", {
          params: { path: { workspace_id: workspace.id } },
          body: { name: newName },
        }),
        "/api/v1/workspaces/{workspace_id}",
      ),
    onSuccess: (updated) => {
      updateCache(updated);
      setName(updated.name);
      toast.success("Workspace renamed");
    },
  });

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      setValidationError("Enter a workspace name.");
      return;
    }
    setValidationError(null);
    rename.mutate(trimmed);
  }

  const error = validationError ?? (rename.isError ? errorMessage(rename.error) : null);
  const unchanged = name.trim() === workspace.name;

  return (
    <form onSubmit={onSubmit} noValidate className="flex max-w-xl flex-col gap-2">
      <Label htmlFor={`${id}-name`}>Workspace name</Label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <Input
          id={`${id}-name`}
          value={name}
          maxLength={WORKSPACE_NAME_MAX}
          onChange={(e) => setName(e.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? `${id}-error` : undefined}
        />
        <Button type="submit" variant="outline" disabled={rename.isPending || unchanged}>
          {rename.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
          Save name
        </Button>
      </div>
      <p id={`${id}-error`} className="min-h-5 text-sm text-destructive" aria-live="polite">
        {error}
      </p>
    </form>
  );
}

function DangerZone({ workspace }: { workspace: Workspace }) {
  const id = useId();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const remove = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.DELETE("/api/v1/workspaces/{workspace_id}", {
          params: { path: { workspace_id: workspace.id } },
        }),
        "/api/v1/workspaces/{workspace_id}",
      ),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: queryKeys.workspace(workspace.id) });
      void queryClient.invalidateQueries({ queryKey: queryKeys.workspaces, exact: true });
      toast.success(`Workspace "${workspace.name}" deleted`);
      router.replace("/workspaces");
    },
  });

  return (
    <Card className="border-destructive/40">
      <CardHeader>
        <CardTitle>Delete workspace</CardTitle>
        <CardDescription>
          Permanently deletes this workspace and removes every member&apos;s access. This cannot be
          undone.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Dialog
          open={open}
          onOpenChange={(next) => {
            setOpen(next);
            if (!next) {
              setConfirmation("");
              remove.reset();
            }
          }}
        >
          <DialogTrigger asChild>
            <Button variant="destructive">Delete workspace</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete &ldquo;{workspace.name}&rdquo;?</DialogTitle>
              <DialogDescription>
                Type the workspace name to confirm. All members lose access immediately.
              </DialogDescription>
            </DialogHeader>
            <form
              className="grid gap-4"
              onSubmit={(event) => {
                event.preventDefault();
                if (confirmation === workspace.name) remove.mutate();
              }}
            >
              <div className="grid gap-2">
                <Label htmlFor={`${id}-confirm`}>Workspace name</Label>
                <Input
                  id={`${id}-confirm`}
                  value={confirmation}
                  onChange={(e) => setConfirmation(e.target.value)}
                  autoComplete="off"
                />
              </div>
              {remove.isError ? (
                <p className="text-sm text-destructive" role="alert">
                  {errorMessage(remove.error)}
                </p>
              ) : null}
              <DialogFooter>
                <DialogClose asChild>
                  <Button type="button" variant="outline">
                    Cancel
                  </Button>
                </DialogClose>
                <Button
                  type="submit"
                  variant="destructive"
                  disabled={confirmation !== workspace.name || remove.isPending}
                >
                  {remove.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
                  Delete permanently
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}
