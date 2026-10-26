"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, Copy, KeyRound, Loader2 } from "lucide-react";
import { useId, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
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
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  errorMessage,
  type ApiToken,
  type ApiTokenCreated,
  type ApiTokenScope,
  type Workspace,
} from "@/lib/api/client";
import { fetchWorkspaces, queryKeys } from "@/lib/api/queries";
import { createApiToken, fetchApiTokens, revokeApiToken, tokenKeys } from "@/lib/api/tokens";
import { formatDate } from "@/lib/utils";

export const EXPIRY_OPTIONS = [7, 30, 90, 180, 365] as const;

const SCOPE_HELP: Record<ApiTokenScope, string> = {
  read: "List workspaces and documents, search, read runs, reports and citations (always included).",
  review: "Start reviews and rule checks. These spend the workspace's LLM budget.",
  upload: "Upload documents (local stdio connections only).",
};

/** Ready-to-paste client configurations for a freshly created token. */
export function mcpConfigSnippets(token: string) {
  const server = (command: string, args: string[], env?: Record<string, string>) =>
    JSON.stringify({ mcpServers: { "compliance-workspace": { command, args, ...(env ? { env } : {}) } } }, null, 2);
  const stdioArgs = ["python", "-m", "compliance_mcp_server", "--transport", "stdio"];
  return {
    docker: server("docker", ["exec", "-i", "-e", "COMPLIANCE_MCP_TOKEN", "compliance-mcp-server-1", ...stdioArgs], {
      COMPLIANCE_MCP_TOKEN: token,
    }),
    ssh: server("ssh", ["-T", "-o", "BatchMode=yes", "you@your-linux-host", "~/bin/compliance-mcp-stdio"]),
    http: `claude mcp add --transport http compliance-workspace http://127.0.0.1:8020/mcp --header "Authorization: Bearer ${token}"`,
  };
}

function statusVariant(status: ApiToken["status"]) {
  return status === "active" ? "secondary" : "outline";
}

export function ApiTokensPanel() {
  const tokens = useQuery({ queryKey: tokenKeys.list, queryFn: fetchApiTokens });
  const workspaces = useQuery({ queryKey: queryKeys.workspaces, queryFn: fetchWorkspaces });
  const [created, setCreated] = useState<ApiTokenCreated | null>(null);
  return (
    <Card>
      <CardHeader>
        <CardTitle>API tokens for MCP clients</CardTitle>
        <CardDescription>
          Tokens let Claude Desktop, Claude Code and other MCP clients use the MCP server as you. A token never has more
          access than your role in a workspace, always expires, and can be revoked at any time. These are not OAuth
          credentials.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6">
        <CreateTokenForm workspaces={workspaces.data ?? []} onCreated={setCreated} />
        {created ? <RevealedToken token={created} onDismiss={() => setCreated(null)} /> : null}
        {tokens.isPending ? (
          <div className="grid gap-2" aria-busy="true" aria-label="Loading API tokens">
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
          </div>
        ) : tokens.isError ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not load API tokens</AlertTitle>
            <AlertDescription>
              {errorMessage(tokens.error)}
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={() => tokens.refetch()}>
                  Retry
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        ) : tokens.data.length === 0 ? (
          <p className="text-sm text-muted-foreground">You have no API tokens yet.</p>
        ) : (
          <TokenTable tokens={tokens.data} workspaces={workspaces.data ?? []} />
        )}
      </CardContent>
    </Card>
  );
}

function CreateTokenForm({
  workspaces,
  onCreated,
}: {
  workspaces: Workspace[];
  onCreated: (token: ApiTokenCreated) => void;
}) {
  const id = useId();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<ApiTokenScope[]>(["read"]);
  const [days, setDays] = useState("90");
  const [restrict, setRestrict] = useState(false);
  const [allowed, setAllowed] = useState<string[]>([]);
  const create = useMutation({
    mutationFn: () =>
      createApiToken({
        name: name.trim(),
        scopes,
        expires_in_days: Number(days),
        workspace_ids: restrict ? allowed : null,
      }),
    onSuccess: (token) => {
      onCreated(token);
      setName("");
      setScopes(["read"]);
      setRestrict(false);
      setAllowed([]);
      void queryClient.invalidateQueries({ queryKey: tokenKeys.list });
    },
  });
  function toggle<T>(list: T[], value: T, on: boolean): T[] {
    return on ? [...new Set([...list, value])] : list.filter((v) => v !== value);
  }
  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim() || (restrict && allowed.length === 0)) return;
    create.mutate();
  }
  return (
    <form onSubmit={onSubmit} className="grid gap-4 rounded-lg border p-4" noValidate aria-label="Create API token">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="grid gap-2">
          <Label htmlFor={`${id}-name`}>Token name</Label>
          <Input
            id={`${id}-name`}
            value={name}
            maxLength={100}
            placeholder="Claude Desktop on my laptop"
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-expiry`}>Expires after</Label>
          <NativeSelect id={`${id}-expiry`} value={days} onChange={(e) => setDays(e.target.value)}>
            {EXPIRY_OPTIONS.map((d) => (
              <option key={d} value={String(d)}>
                {d} days
              </option>
            ))}
          </NativeSelect>
        </div>
      </div>
      <fieldset className="grid gap-2">
        <legend className="mb-1 text-sm font-medium">Permissions</legend>
        {(Object.keys(SCOPE_HELP) as ApiTokenScope[]).map((scope) => (
          <label key={scope} className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-1"
              checked={scopes.includes(scope)}
              disabled={scope === "read"}
              onChange={(e) => setScopes((s) => toggle(s, scope, e.target.checked))}
            />
            <span>
              <span className="font-medium">{scope}</span>{" "}
              <span className="text-muted-foreground">{SCOPE_HELP[scope]}</span>
            </span>
          </label>
        ))}
      </fieldset>
      <fieldset className="grid gap-2">
        <legend className="mb-1 text-sm font-medium">Workspaces</legend>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={restrict} onChange={(e) => setRestrict(e.target.checked)} />
          Only these workspaces (otherwise every workspace you belong to)
        </label>
        {restrict ? (
          workspaces.length === 0 ? (
            <p className="text-sm text-muted-foreground">You are not a member of any workspace.</p>
          ) : (
            <div className="grid gap-1 pl-6">
              {workspaces.map((w) => (
                <label key={w.id} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={allowed.includes(w.id)}
                    onChange={(e) => setAllowed((a) => toggle(a, w.id, e.target.checked))}
                  />
                  {w.name} <span className="text-muted-foreground">({w.role})</span>
                </label>
              ))}
            </div>
          )
        ) : null}
      </fieldset>
      {create.isError ? (
        <p className="text-sm text-destructive" role="alert">
          {errorMessage(create.error)}
        </p>
      ) : null}
      <div>
        <Button type="submit" disabled={create.isPending || !name.trim() || (restrict && allowed.length === 0)}>
          {create.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <KeyRound aria-hidden />}
          Create token
        </Button>
      </div>
    </form>
  );
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Copy failed: select the text and copy it manually.");
    }
  }
  return (
    <Button type="button" variant="outline" size="sm" onClick={() => void copy()} aria-label={label}>
      {copied ? <Check aria-hidden /> : <Copy aria-hidden />}
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function RevealedToken({ token, onDismiss }: { token: ApiTokenCreated; onDismiss: () => void }) {
  const snippets = mcpConfigSnippets(token.token);
  return (
    <Alert>
      <KeyRound aria-hidden />
      <AlertTitle>Copy your new token now: it will not be shown again</AlertTitle>
      <AlertDescription className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <code className="rounded bg-muted px-2 py-1 font-mono text-xs break-all" data-testid="new-token">
            {token.token}
          </code>
          <CopyButton text={token.token} label="Copy token" />
        </div>
        <p className="text-xs">
          Treat it like a password. Never put it in a URL. Paste one of these into your MCP client; see the MCP guide
          (apps/mcp-server/README.md) for the Mac-to-Linux SSH setup.
        </p>
        <Snippet title="Claude Desktop, MCP server in Docker on the same machine" text={snippets.docker} />
        <Snippet title="Claude Desktop on a Mac, server on a Linux host over SSH (token stays on the host)" text={snippets.ssh} />
        <Snippet title="Claude Code or another client over streamable HTTP (no uploads)" text={snippets.http} />
        <div>
          <Button type="button" variant="ghost" size="sm" onClick={onDismiss}>
            I have copied it
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}

function Snippet({ title, text }: { title: string; text: string }) {
  return (
    <details className="rounded-md border p-2">
      <summary className="cursor-pointer text-xs font-medium">{title}</summary>
      <div className="mt-2 grid gap-2">
        <pre className="max-h-60 overflow-auto rounded bg-muted p-2 text-xs">{text}</pre>
        <div>
          <CopyButton text={text} label={`Copy: ${title}`} />
        </div>
      </div>
    </details>
  );
}

function TokenTable({ tokens, workspaces }: { tokens: ApiToken[]; workspaces: Workspace[] }) {
  const names = new Map(workspaces.map((w) => [w.id, w.name]));
  const [revoking, setRevoking] = useState<ApiToken | null>(null);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Your API tokens</caption>
        <thead className="border-b text-left text-muted-foreground">
          <tr>
            <th className="py-2 pr-4 font-medium">Name</th>
            <th className="py-2 pr-4 font-medium">Token</th>
            <th className="py-2 pr-4 font-medium">Permissions</th>
            <th className="py-2 pr-4 font-medium">Workspaces</th>
            <th className="py-2 pr-4 font-medium">Last used</th>
            <th className="py-2 pr-4 font-medium">Expires</th>
            <th className="py-2 pr-4 font-medium">Status</th>
            <th className="py-2 font-medium">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {tokens.map((t) => (
            <tr key={t.id} className="border-b last:border-0">
              <td className="py-2 pr-4">{t.name}</td>
              <td className="py-2 pr-4 font-mono text-xs">{t.prefix}…</td>
              <td className="py-2 pr-4">{t.scopes.join(", ")}</td>
              <td className="py-2 pr-4">
                {t.workspace_ids ? t.workspace_ids.map((w) => names.get(w) ?? "a workspace you left").join(", ") : "All"}
              </td>
              <td className="py-2 pr-4">{t.last_used_at ? formatDate(t.last_used_at) : "Never"}</td>
              <td className="py-2 pr-4">{formatDate(t.expires_at)}</td>
              <td className="py-2 pr-4">
                <Badge variant={statusVariant(t.status)}>{t.status}</Badge>
              </td>
              <td className="py-2">
                {t.status === "active" ? (
                  <Button variant="outline" size="sm" onClick={() => setRevoking(t)}>
                    Revoke
                  </Button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <RevokeDialog token={revoking} onClose={() => setRevoking(null)} />
    </div>
  );
}

function RevokeDialog({ token, onClose }: { token: ApiToken | null; onClose: () => void }) {
  const queryClient = useQueryClient();
  const revoke = useMutation({
    mutationFn: (id: string) => revokeApiToken(id),
    onSuccess: () => {
      toast.success("Token revoked");
      void queryClient.invalidateQueries({ queryKey: tokenKeys.list });
      onClose();
    },
  });
  return (
    <Dialog open={token !== null} onOpenChange={(open) => (!open ? onClose() : undefined)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Revoke &ldquo;{token?.name}&rdquo;?</DialogTitle>
          <DialogDescription>
            MCP clients using this token stop working on their next request. This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        {revoke.isError ? (
          <p className="text-sm text-destructive" role="alert">
            {errorMessage(revoke.error)}
          </p>
        ) : null}
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Cancel</Button>
          </DialogClose>
          <Button
            variant="destructive"
            disabled={revoke.isPending || token === null}
            onClick={() => token && revoke.mutate(token.id)}
          >
            {revoke.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            Revoke token
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
