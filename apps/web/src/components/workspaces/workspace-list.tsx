"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, FolderOpen, Loader2, Plus, Users } from "lucide-react";
import Link from "next/link";
import { useId, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage, unwrap, type Workspace } from "@/lib/api/client";
import { fetchWorkspaces, queryKeys } from "@/lib/api/queries";
import { formatDate } from "@/lib/utils";

import { RoleBadge } from "./role-badge";

export const WORKSPACE_NAME_MAX = 120;

export function WorkspaceList() {
  const workspaces = useQuery({ queryKey: queryKeys.workspaces, queryFn: fetchWorkspaces });

  return (
    <div className="grid gap-8">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight">Workspaces</h1>
        <p className="text-muted-foreground">
          Each workspace holds the documents, rule packs and reviews for one compliance project.
        </p>
      </div>

      <CreateWorkspaceForm />

      <section aria-labelledby="workspace-list-heading" className="grid gap-4">
        <h2 id="workspace-list-heading" className="sr-only">
          Your workspaces
        </h2>
        {workspaces.isPending ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true" aria-label="Loading workspaces">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-32 rounded-xl" />
            ))}
          </div>
        ) : workspaces.isError ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not load workspaces</AlertTitle>
            <AlertDescription>
              {errorMessage(workspaces.error)}
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={() => workspaces.refetch()}>
                  Retry
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        ) : workspaces.data.length === 0 ? (
          <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-12 text-center">
            <FolderOpen className="size-8 text-muted-foreground" aria-hidden />
            <p className="font-medium">No workspaces yet</p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Create your first workspace above. You can invite reviewers and viewers once it
              exists.
            </p>
          </div>
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {workspaces.data.map((workspace) => (
              <li key={workspace.id}>
                <WorkspaceCard workspace={workspace} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function WorkspaceCard({ workspace }: { workspace: Workspace }) {
  return (
    <Link
      href={`/workspaces/${workspace.id}`}
      className="block rounded-xl focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
    >
      <Card className="h-full transition-colors hover:border-ring/60 hover:bg-accent/40">
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="text-base break-words">{workspace.name}</CardTitle>
            <RoleBadge role={workspace.role} />
          </div>
          <CardDescription>Created {formatDate(workspace.created_at)}</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Users className="size-4" aria-hidden />
          {workspace.member_count} {workspace.member_count === 1 ? "member" : "members"}
        </CardContent>
      </Card>
    </Link>
  );
}

function CreateWorkspaceForm() {
  const id = useId();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: async (workspaceName: string) =>
      unwrap(
        await api.POST("/api/v1/workspaces", { body: { name: workspaceName } }),
        "/api/v1/workspaces",
      ),
    onSuccess: (workspace) => {
      setName("");
      queryClient.setQueryData<Workspace[]>(queryKeys.workspaces, (current) => [
        workspace,
        ...(current ?? []).filter((w) => w.id !== workspace.id),
      ]);
      void queryClient.invalidateQueries({ queryKey: queryKeys.workspaces });
      toast.success(`Workspace "${workspace.name}" created`);
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
    create.mutate(trimmed);
  }

  const error = validationError ?? (create.isError ? errorMessage(create.error) : null);

  return (
    <Card>
      <CardContent className="pt-6">
        <form onSubmit={onSubmit} noValidate className="flex flex-col gap-2">
          <Label htmlFor={`${id}-name`}>New workspace</Label>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              id={`${id}-name`}
              placeholder="e.g. Vendor confidentiality review"
              value={name}
              maxLength={WORKSPACE_NAME_MAX}
              onChange={(e) => setName(e.target.value)}
              aria-invalid={error ? true : undefined}
              aria-describedby={error ? `${id}-error` : undefined}
            />
            <Button type="submit" disabled={create.isPending}>
              {create.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Plus aria-hidden />}
              Create workspace
            </Button>
          </div>
          <p id={`${id}-error`} className="min-h-5 text-sm text-destructive" aria-live="polite">
            {error}
          </p>
        </form>
      </CardContent>
    </Card>
  );
}
