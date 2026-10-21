"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Loader2, UserPlus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useId, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { useCurrentUser } from "@/components/app-shell";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage, unwrap, type Member, type Workspace } from "@/lib/api/client";
import { fetchMembers, queryKeys } from "@/lib/api/queries";
import { formatDate } from "@/lib/utils";

import { RoleBadge } from "./role-badge";

type AssignableRole = "reviewer" | "viewer";

const ROLE_HELP: Record<AssignableRole, string> = {
  reviewer: "Reviewer",
  viewer: "Viewer",
};

function useInvalidateMembership(workspaceId: string) {
  const queryClient = useQueryClient();
  return () => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.members(workspaceId) });
    void queryClient.invalidateQueries({ queryKey: queryKeys.workspace(workspaceId), exact: true });
    void queryClient.invalidateQueries({ queryKey: queryKeys.workspaces, exact: true });
  };
}

export function MembersPanel({ workspace }: { workspace: Workspace }) {
  const members = useQuery({
    queryKey: queryKeys.members(workspace.id),
    queryFn: () => fetchMembers(workspace.id),
  });
  const isOwner = workspace.role === "owner";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Members</CardTitle>
        <CardDescription>
          Owners manage the workspace. Reviewers will run reviews and override findings; viewers
          have read-only access.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6">
        {isOwner ? <AddMemberForm workspaceId={workspace.id} /> : null}
        {members.isPending ? (
          <div className="grid gap-2" aria-busy="true" aria-label="Loading members">
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
          </div>
        ) : members.isError ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not load members</AlertTitle>
            <AlertDescription>
              {errorMessage(members.error)}
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={() => members.refetch()}>
                  Retry
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">Workspace members</caption>
              <thead className="border-b text-left text-muted-foreground">
                <tr>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Email
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Role
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Joined
                  </th>
                  <th scope="col" className="py-2 font-medium">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {members.data.map((member) => (
                  <MemberRow key={member.user_id} member={member} workspace={workspace} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MemberRow({ member, workspace }: { member: Member; workspace: Workspace }) {
  const currentUser = useCurrentUser();
  const router = useRouter();
  const queryClient = useQueryClient();
  const invalidate = useInvalidateMembership(workspace.id);
  const isOwner = workspace.role === "owner";
  const isSelf = member.user_id === currentUser.id;
  const isWorkspaceOwner = member.role === "owner";
  const path = { workspace_id: workspace.id, user_id: member.user_id };

  const changeRole = useMutation({
    mutationFn: async (role: AssignableRole) =>
      unwrap(
        await api.PATCH("/api/v1/workspaces/{workspace_id}/members/{user_id}", {
          params: { path },
          body: { role },
        }),
        "/api/v1/workspaces/{workspace_id}/members/{user_id}",
      ),
    onSuccess: (updated) => {
      invalidate();
      toast.success(`${updated.email} is now a ${updated.role}`);
    },
    onError: (error) => toast.error(errorMessage(error)),
  });

  const remove = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.DELETE("/api/v1/workspaces/{workspace_id}/members/{user_id}", {
          params: { path },
        }),
        "/api/v1/workspaces/{workspace_id}/members/{user_id}",
      ),
    onSuccess: () => {
      if (isSelf) {
        queryClient.removeQueries({ queryKey: queryKeys.workspace(workspace.id) });
        void queryClient.invalidateQueries({ queryKey: queryKeys.workspaces, exact: true });
        toast.success(`You left "${workspace.name}"`);
        router.replace("/workspaces");
        return;
      }
      invalidate();
      toast.success(`Removed ${member.email}`);
    },
    onError: (error) => toast.error(errorMessage(error)),
  });

  const canEditRole = isOwner && !isWorkspaceOwner;
  const canRemove = !isWorkspaceOwner && (isOwner || isSelf);

  return (
    <tr className="border-b last:border-0">
      <td className="py-3 pr-4 break-all">
        {member.email}
        {isSelf ? <span className="ml-2 text-xs text-muted-foreground">(you)</span> : null}
      </td>
      <td className="py-3 pr-4">
        {canEditRole ? (
          <NativeSelect
            aria-label={`Role for ${member.email}`}
            value={member.role}
            disabled={changeRole.isPending}
            onChange={(e) => changeRole.mutate(e.target.value as AssignableRole)}
          >
            <option value="reviewer">{ROLE_HELP.reviewer}</option>
            <option value="viewer">{ROLE_HELP.viewer}</option>
          </NativeSelect>
        ) : (
          <RoleBadge role={member.role} />
        )}
      </td>
      <td className="py-3 pr-4 whitespace-nowrap text-muted-foreground">
        {formatDate(member.joined_at)}
      </td>
      <td className="py-3 text-right">
        {canRemove ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => remove.mutate()}
            disabled={remove.isPending}
            aria-label={isSelf ? "Leave workspace" : `Remove ${member.email}`}
          >
            {remove.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            {isSelf ? "Leave" : "Remove"}
          </Button>
        ) : null}
      </td>
    </tr>
  );
}

function AddMemberForm({ workspaceId }: { workspaceId: string }) {
  const id = useId();
  const invalidate = useInvalidateMembership(workspaceId);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<AssignableRole>("reviewer");
  const [validationError, setValidationError] = useState<string | null>(null);

  const add = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/v1/workspaces/{workspace_id}/members", {
          params: { path: { workspace_id: workspaceId } },
          body: { email: email.trim(), role },
        }),
        "/api/v1/workspaces/{workspace_id}/members",
      ),
    onSuccess: (member) => {
      setEmail("");
      invalidate();
      toast.success(`Added ${member.email} as ${member.role}`);
    },
  });

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setValidationError("Enter the email address of a registered user.");
      return;
    }
    setValidationError(null);
    add.mutate();
  }

  const error = validationError ?? (add.isError ? errorMessage(add.error) : null);

  return (
    <form onSubmit={onSubmit} noValidate className="grid gap-2">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
        <div className="grid flex-1 gap-2">
          <Label htmlFor={`${id}-email`}>Add a member by email</Label>
          <Input
            id={`${id}-email`}
            type="email"
            placeholder="colleague@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? `${id}-error` : `${id}-hint`}
          />
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-role`}>Role</Label>
          <NativeSelect
            id={`${id}-role`}
            value={role}
            onChange={(e) => setRole(e.target.value as AssignableRole)}
          >
            <option value="reviewer">{ROLE_HELP.reviewer}</option>
            <option value="viewer">{ROLE_HELP.viewer}</option>
          </NativeSelect>
        </div>
        <Button type="submit" disabled={add.isPending}>
          {add.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <UserPlus aria-hidden />}
          Add member
        </Button>
      </div>
      {error ? (
        <p id={`${id}-error`} className="text-sm text-destructive" aria-live="polite">
          {error}
        </p>
      ) : (
        <p id={`${id}-hint`} className="text-sm text-muted-foreground">
          The person must already have an account.
        </p>
      )}
    </form>
  );
}
