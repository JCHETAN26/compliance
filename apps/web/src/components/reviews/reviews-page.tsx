"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, FlaskConical, ListChecks, Loader2, Play } from "lucide-react";
import type { Route } from "next";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useId, useMemo, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { useWorkspace } from "@/components/workspaces/workspace-detail";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage, type DocumentInfo, type ReviewRun, type Workspace } from "@/lib/api/client";
import { documentKeys, fetchDocuments } from "@/lib/api/queries";
import {
  fetchLlmSettings,
  fetchReviewSettings,
  fetchReviews,
  fetchRulePacks,
  reviewKeys,
  rulePackKeys,
  settingsKeys,
  startReview,
} from "@/lib/api/reviews";
import { formatDate } from "@/lib/utils";

import { ACTIVE_RUN_STATUSES, formatUsd, RunStatusChip, totalCount } from "./labels";

export function MockBanner() {
  const llm = useQuery({ queryKey: settingsKeys.llm, queryFn: fetchLlmSettings, staleTime: 60_000 });
  if (!llm.data?.mock_only && llm.data?.rule_check_backend !== "mock") return null;
  return (
    <Alert>
      <FlaskConical aria-hidden />
      <AlertTitle>Offline mock mode</AlertTitle>
      <AlertDescription>
        {llm.data?.mock_only
          ? "No real LLM provider is configured on this server: LLM steps use the deterministic mock (lexical heuristics, no real model, cost $0). "
          : ""}
        {llm.data?.rule_check_backend === "mock"
          ? "Model-supported rules use the mock rule-check backend, not the trained model. "
          : ""}
        Results are labelled as mock everywhere and are not legal analysis.
      </AlertDescription>
    </Alert>
  );
}

export function ReviewsPage() {
  const workspace = useWorkspace();
  const canWrite = workspace.role !== "viewer";
  const runs = useQuery({
    queryKey: reviewKeys.list(workspace.id),
    queryFn: () => fetchReviews(workspace.id),
    refetchInterval: (q) =>
      q.state.data?.some((r) => ACTIVE_RUN_STATUSES.includes(r.status)) ? 3000 : false,
  });
  return (
    <div className="grid gap-6">
      <MockBanner />
      {canWrite ? (
        <NewReviewForm workspace={workspace} />
      ) : (
        <p className="rounded-lg border bg-card px-4 py-3 text-sm text-muted-foreground">
          You have viewer access: you can follow runs and read reports, but not start reviews.
        </p>
      )}
      <section aria-labelledby="runs-heading" className="grid gap-3">
        <h2 id="runs-heading" className="text-lg font-semibold">
          Review runs
        </h2>
        {runs.isPending ? (
          <div className="grid gap-2" aria-busy="true" aria-label="Loading review runs">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        ) : runs.isError ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not load review runs</AlertTitle>
            <AlertDescription>{errorMessage(runs.error)}</AlertDescription>
          </Alert>
        ) : runs.data.length === 0 ? (
          <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-10 text-center">
            <ListChecks className="size-8 text-muted-foreground" aria-hidden />
            <p className="font-medium">No reviews yet</p>
            <p className="max-w-sm text-sm text-muted-foreground">
              Start a review to check this workspace&apos;s documents against a rule pack.
            </p>
          </div>
        ) : (
          <RunTable workspaceId={workspace.id} runs={runs.data} />
        )}
      </section>
    </div>
  );
}

function RunTable({ workspaceId, runs }: { workspaceId: string; runs: ReviewRun[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border bg-card">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-left text-xs text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-medium">Started</th>
            <th className="px-3 py-2 font-medium">Rule pack</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 font-medium">Findings</th>
            <th className="px-3 py-2 font-medium">LLM</th>
            <th className="px-3 py-2 font-medium">Cost</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const total = totalCount(run.totals, "findings_by_status");
            const review = totalCount(run.totals, "findings_by_status", "needs_review");
            return (
              <tr key={run.id} className="border-b last:border-0 hover:bg-accent/40">
                <td className="px-3 py-2">
                  <Link
                    href={`/workspaces/${workspaceId}/reviews/${run.id}` as Route}
                    className="font-medium underline-offset-2 hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
                  >
                    {formatDate(run.created_at)}
                  </Link>
                </td>
                <td className="px-3 py-2">
                  {run.rule_pack.name} <span className="text-muted-foreground">{run.rule_pack.version}</span>
                </td>
                <td className="px-3 py-2">
                  <RunStatusChip status={run.status} />
                </td>
                <td className="px-3 py-2 text-muted-foreground">
                  {total ? `${total} (${review} need review)` : "—"}
                </td>
                <td className="px-3 py-2">
                  {run.llm_provider}
                  {run.llm_is_mock ? " (mock)" : ""}
                </td>
                <td className="px-3 py-2">{formatUsd(run.budget.usd_used)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function NewReviewForm({ workspace }: { workspace: Workspace }) {
  const id = useId();
  const router = useRouter();
  const queryClient = useQueryClient();
  const packs = useQuery({ queryKey: rulePackKeys.list(workspace.id), queryFn: () => fetchRulePacks(workspace.id) });
  const docs = useQuery({ queryKey: documentKeys.list(workspace.id), queryFn: () => fetchDocuments(workspace.id) });
  const defaults = useQuery({
    queryKey: settingsKeys.review(workspace.id),
    queryFn: () => fetchReviewSettings(workspace.id),
  });
  const ready = useMemo(() => (docs.data ?? []).filter((d) => d.status === "ready"), [docs.data]);
  const [packChoice, setPackId] = useState("");
  const [selected, setSelected] = useState<Set<string> | null>(null);
  const [budgetUsd, setBudgetUsd] = useState("");
  const [budgetTokens, setBudgetTokens] = useState("");
  const [pause, setPause] = useState<boolean | null>(null);

  const defaultPack =
    packs.data?.find((p) => p.key === "vendor_confidentiality_review") ?? packs.data?.[0] ?? null;
  const packId = packChoice || (defaultPack?.id ?? "");

  const chosen = selected ?? new Set(ready.map((d) => d.id));
  const start = useMutation({
    mutationFn: () =>
      startReview(workspace.id, {
        rule_pack_id: packId,
        document_ids: chosen.size === ready.length ? null : [...chosen],
        budget_usd: budgetUsd ? Number(budgetUsd) : null,
        budget_tokens: budgetTokens ? Number(budgetTokens) : null,
        pause_on_needs_review: pause,
      }),
    onSuccess: (run) => {
      void queryClient.invalidateQueries({ queryKey: reviewKeys.list(workspace.id) });
      toast.success("Review queued");
      router.push(`/workspaces/${workspace.id}/reviews/${run.id}` as Route);
    },
  });

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!packId || chosen.size === 0) return;
    start.mutate();
  }

  function toggle(doc: DocumentInfo) {
    const next = new Set(chosen);
    if (next.has(doc.id)) next.delete(doc.id);
    else next.add(doc.id);
    setSelected(next);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>New review</CardTitle>
        <CardDescription>
          The agents plan checks, extract facts, retrieve evidence, apply the trained model and the
          LLM, reason across documents, verify every citation and write the report. Runs execute in
          the background; you can leave this page.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="grid gap-4" noValidate>
          <div className="grid gap-2 sm:max-w-md">
            <Label htmlFor={`${id}-pack`}>Rule pack</Label>
            <NativeSelect
              id={`${id}-pack`}
              value={packId}
              onChange={(e) => setPackId(e.target.value)}
              disabled={packs.isPending}
            >
              {(packs.data ?? []).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} {p.version} ({p.rule_count} rules{p.builtin ? ", built-in" : ""})
                </option>
              ))}
            </NativeSelect>
          </div>
          <fieldset className="grid gap-2">
            <legend className="mb-1 text-sm font-medium">
              Documents ({chosen.size} of {ready.length} ready)
            </legend>
            {docs.isPending ? (
              <Skeleton className="h-16" />
            ) : ready.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No processed documents yet. Upload documents and wait for processing to finish.
              </p>
            ) : (
              <div className="grid max-h-56 gap-1 overflow-auto rounded-md border p-2 sm:grid-cols-2">
                {ready.map((doc) => (
                  <label key={doc.id} className="flex items-center gap-2 rounded px-2 py-1 text-sm hover:bg-accent">
                    <input type="checkbox" checked={chosen.has(doc.id)} onChange={() => toggle(doc)} />
                    <span className="truncate">{doc.filename}</span>
                    <span className="ml-auto text-xs text-muted-foreground">{doc.doc_type ?? doc.file_format}</span>
                  </label>
                ))}
              </div>
            )}
          </fieldset>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="grid gap-2">
              <Label htmlFor={`${id}-usd`}>Budget cap (USD)</Label>
              <Input
                id={`${id}-usd`}
                type="number"
                min={0}
                step="0.01"
                placeholder={defaults.data ? String(defaults.data.budget_usd) : ""}
                value={budgetUsd}
                onChange={(e) => setBudgetUsd(e.target.value)}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor={`${id}-tokens`}>Token cap</Label>
              <Input
                id={`${id}-tokens`}
                type="number"
                min={1000}
                step={1000}
                placeholder={defaults.data ? String(defaults.data.budget_tokens) : ""}
                value={budgetTokens}
                onChange={(e) => setBudgetTokens(e.target.value)}
              />
            </div>
            <label className="flex items-center gap-2 self-end pb-2 text-sm">
              <input
                type="checkbox"
                checked={pause ?? defaults.data?.pause_on_needs_review ?? false}
                onChange={(e) => setPause(e.target.checked)}
              />
              Pause for human review when findings need it
            </label>
          </div>
          {defaults.data ? (
            <p className="text-xs text-muted-foreground">
              LLM: {defaults.data.llm_provider} / {defaults.data.llm_model}. Cascade threshold tau:{" "}
              {defaults.data.tau ?? "the model's DEV-selected value"}. Change defaults in Settings.
            </p>
          ) : null}
          {start.isError ? (
            <p className="text-sm text-destructive" role="alert">
              {errorMessage(start.error)}
            </p>
          ) : null}
          <div>
            <Button type="submit" disabled={start.isPending || !packId || chosen.size === 0}>
              {start.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
              Start review
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
