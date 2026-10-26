"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, FileBarChart, Loader2, Pause, Play, Radio, Square } from "lucide-react";
import type { Route } from "next";
import Link from "next/link";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { useWorkspace } from "@/components/workspaces/workspace-detail";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { NativeSelect } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ApiError,
  errorMessage,
  type AgentEvent,
  type ReviewRun,
  type ReviewTasks,
} from "@/lib/api/client";
import { controlRun, fetchRun, fetchTasks, reviewKeys } from "@/lib/api/reviews";
import { cn, formatDate } from "@/lib/utils";

import { ACTIVE_RUN_STATUSES, formatUsd, RunStatusChip, totalCount } from "./labels";
import { MockBanner } from "./reviews-page";
import { useRunEvents } from "./use-run-events";

export const AGENT_LABEL: Record<string, string> = {
  planner: "Planner",
  extraction: "Extraction",
  retrieval: "Retrieval",
  rule_check: "Rule check",
  cross_document: "Cross-document",
  citation_verifier: "Citation verifier",
  report_writer: "Report writer",
  human_review: "Human review",
  human: "Reviewer",
  system: "System",
};

export function RunView({ runId }: { runId: string }) {
  const workspace = useWorkspace();
  const queryClient = useQueryClient();
  const run = useQuery({
    queryKey: reviewKeys.run(runId),
    queryFn: () => fetchRun(runId),
    refetchInterval: (q) => (q.state.data && ACTIVE_RUN_STATUSES.includes(q.state.data.status) ? 2000 : false),
  });
  const active = !!run.data && ACTIVE_RUN_STATUSES.includes(run.data.status);
  const tasks = useQuery({
    queryKey: reviewKeys.tasks(runId),
    queryFn: () => fetchTasks(runId),
    refetchInterval: active ? 3000 : false,
  });
  const labels = useMemo(() => taskLabels(tasks.data), [tasks.data]);
  const { events, status: streamStatus } = useRunEvents(runId, active, () => {
    void queryClient.invalidateQueries({ queryKey: reviewKeys.run(runId) });
    void queryClient.invalidateQueries({ queryKey: reviewKeys.tasks(runId) });
  });

  if (run.isPending) return <Skeleton className="h-64 rounded-xl" />;
  if (run.isError) {
    const notFound = run.error instanceof ApiError && [404, 422].includes(run.error.status);
    return (
      <Alert variant="destructive">
        <AlertTriangle aria-hidden />
        <AlertTitle>{notFound ? "Review not found" : "Could not load this review"}</AlertTitle>
        <AlertDescription>{notFound ? "It may have been deleted." : errorMessage(run.error)}</AlertDescription>
      </Alert>
    );
  }
  const data = run.data;
  return (
    <div className="grid gap-5">
      <MockBanner />
      <RunHeader run={data} canWrite={workspace.role !== "viewer"} workspaceId={workspace.id} />
      {data.status === "awaiting_review" ? (
        <Alert>
          <AlertTriangle aria-hidden />
          <AlertTitle>Waiting for human review</AlertTitle>
          <AlertDescription>
            {totalCount(data.totals, "findings_by_status", "needs_review")} finding(s) need review. Open the report,
            record decisions with a comment, then resume the run to write the final report.
          </AlertDescription>
        </Alert>
      ) : null}
      {data.status === "failed" ? (
        <Alert variant="destructive">
          <AlertTriangle aria-hidden />
          <AlertTitle>The run stopped with an error</AlertTitle>
          <AlertDescription>
            {data.error_message ?? data.error_code}. Completed steps are saved; resuming continues
            from the last checkpoint without repeating (or re-billing) finished work.
          </AlertDescription>
        </Alert>
      ) : null}
      <Counters run={data} />
      <section aria-labelledby="grid-heading" className="grid gap-2">
        <h2 id="grid-heading" className="text-lg font-semibold">
          Progress
        </h2>
        {tasks.isPending ? <Skeleton className="h-40" /> : tasks.data ? <ProgressGrid tasks={tasks.data} /> : null}
      </section>
      <Timeline events={events} streamStatus={active ? streamStatus : "idle"} labels={labels} />
    </div>
  );
}

function RunHeader({ run, canWrite, workspaceId }: { run: ReviewRun; canWrite: boolean; workspaceId: string }) {
  const queryClient = useQueryClient();
  const control = useMutation({
    mutationFn: (action: "pause" | "resume" | "cancel") => controlRun(run.id, action),
    onSuccess: (updated) => {
      queryClient.setQueryData(reviewKeys.run(run.id), updated);
      toast.success(updated.control ? `Requested: ${updated.control}` : `Run ${updated.status}`);
    },
    onError: (error) => toast.error(errorMessage(error)),
  });
  const canPause = ["queued", "running"].includes(run.status) && run.control !== "pause";
  const canResume = ["paused", "failed", "awaiting_review"].includes(run.status);
  const canCancel = !["completed", "cancelled"].includes(run.status) && run.control !== "cancel";
  return (
    <div className="flex flex-wrap items-start justify-between gap-4 rounded-xl border bg-card p-4">
      <div className="grid gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-xl font-semibold">
            {run.rule_pack.name} <span className="text-muted-foreground">{run.rule_pack.version}</span>
          </h2>
          <RunStatusChip status={run.status} />
          {run.control ? <span className="text-xs text-muted-foreground">({run.control} requested)</span> : null}
        </div>
        <p className="text-sm text-muted-foreground">
          Started {formatDate(run.created_at)} · {run.document_count} document(s) · LLM {run.llm_provider}/
          {run.llm_model}
          {run.llm_is_mock ? " (mock)" : ""} · rule-check {run.rule_check_backend === "mock" ? "mock backend" : run.model_version ?? "model"}
          {run.tau !== null && run.tau !== undefined ? ` · tau ${run.tau}` : ""}
        </p>
      </div>
      <div className="flex flex-wrap gap-2">
        {canWrite && canPause ? (
          <Button variant="outline" size="sm" onClick={() => control.mutate("pause")} disabled={control.isPending}>
            <Pause aria-hidden />
            Pause
          </Button>
        ) : null}
        {canWrite && canResume ? (
          <Button size="sm" onClick={() => control.mutate("resume")} disabled={control.isPending}>
            {control.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
            Resume
          </Button>
        ) : null}
        {canWrite && canCancel ? (
          <Button variant="outline" size="sm" onClick={() => control.mutate("cancel")} disabled={control.isPending}>
            <Square aria-hidden />
            Cancel
          </Button>
        ) : null}
        <Button asChild size="sm" variant={run.status === "completed" ? "default" : "outline"}>
          <Link href={`/workspaces/${workspaceId}/reviews/${run.id}/report` as Route}>
            <FileBarChart aria-hidden />
            {run.status === "completed" ? "Open report" : "Findings so far"}
          </Link>
        </Button>
      </div>
    </div>
  );
}

function Counters({ run }: { run: ReviewRun }) {
  const b = run.budget;
  const items: [string, string, string?][] = [
    ["Tokens", `${b.tokens_used.toLocaleString()} / ${b.tokens_cap.toLocaleString()}`, b.usage_source === "mock_estimate" ? "mock estimate" : undefined],
    ["Cost", `${formatUsd(b.usd_used)} / ${formatUsd(b.usd_cap)}`, run.llm_is_mock ? "mock: $0" : "includes held reservations"],
    ["LLM calls", String(b.calls), b.calls_unknown ? `${b.calls_unknown} with unknown outcome (charged)` : undefined],
    [
      "Findings",
      String(totalCount(run.totals, "findings_by_status")),
      `${totalCount(run.totals, "findings_by_status", "needs_review")} need review`,
    ],
  ];
  return (
    <dl className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Run counters">
      {items.map(([label, value, note]) => (
        <div key={label} className="rounded-lg border bg-card px-3 py-2">
          <dt className="text-xs text-muted-foreground">{label}</dt>
          <dd className="text-lg font-semibold tabular-nums">{value}</dd>
          {note ? <dd className="text-xs text-muted-foreground">{note}</dd> : null}
        </div>
      ))}
    </dl>
  );
}

const CELL: Record<string, string> = {
  pending: "bg-slate-100 text-slate-500",
  running: "bg-sky-100 text-sky-800 animate-pulse",
  failed: "bg-red-200 text-red-900",
  skipped: "bg-slate-100 text-slate-400",
  satisfied: "bg-emerald-100 text-emerald-800",
  violated: "bg-red-100 text-red-800",
  not_mentioned: "bg-slate-200 text-slate-700",
  needs_review: "bg-amber-100 text-amber-900",
};

const CELL_TEXT: Record<string, string> = {
  pending: "…",
  running: "⟳",
  failed: "!",
  skipped: "–",
  satisfied: "✓",
  violated: "✗",
  not_mentioned: "∅",
  needs_review: "?",
};

export function ProgressGrid({ tasks }: { tasks: ReviewTasks }) {
  const subjects = useMemo(() => {
    const seen = new Map<string, string>();
    for (const t of tasks.tasks) if (!seen.has(t.subject_key)) seen.set(t.subject_key, t.subject_label);
    return [...seen.entries()];
  }, [tasks.tasks]);
  const byCell = useMemo(() => new Map(tasks.tasks.map((t) => [`${t.rule_key}|${t.subject_key}`, t])), [tasks.tasks]);
  if (!tasks.tasks.length) return <p className="text-sm text-muted-foreground">The planner has not created tasks yet.</p>;
  return (
    <div className="overflow-auto rounded-xl border bg-card">
      <table className="text-xs" aria-label="Rules by subject progress">
        <thead>
          <tr>
            <th className="sticky left-0 bg-card px-2 py-2 text-left font-medium">Subject</th>
            {tasks.rules.map((r) => (
              <th key={r.key} className="max-w-28 px-1 py-2 text-left font-medium" title={r.title}>
                <span className="line-clamp-2">{r.title}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {subjects.map(([key, label]) => (
            <tr key={key} className="border-t">
              <th scope="row" className="sticky left-0 max-w-56 truncate bg-card px-2 py-1 text-left font-normal" title={label}>
                {label}
              </th>
              {tasks.rules.map((r) => {
                const task = byCell.get(`${r.key}|${key}`);
                if (!task) return <td key={r.key} className="px-1 py-1" aria-label="not applicable" />;
                const state = task.status === "done" ? (task.finding_status ?? "needs_review") : task.status;
                return (
                  <td key={r.key} className="px-1 py-1">
                    <span
                      className={cn("flex size-7 items-center justify-center rounded font-semibold", CELL[state])}
                      title={`${r.title} — ${label}: ${state.replace("_", " ")}`}
                      aria-label={`${r.title}, ${label}: ${state.replace("_", " ")}`}
                    >
                      {CELL_TEXT[state]}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t px-3 py-2 text-xs text-muted-foreground">
        ✓ satisfied · ✗ violated · ∅ not mentioned · ? needs review · … pending · ⟳ running
      </p>
    </div>
  );
}

export type TaskLabels = Map<string, string>;

/** "Rule title — subject" for every task key of the run. */
export function taskLabels(tasks: ReviewTasks | undefined): TaskLabels {
  const titles = new Map((tasks?.rules ?? []).map((r) => [r.key, r.title]));
  return new Map(
    (tasks?.tasks ?? []).map((t) => [t.task_key, `${titles.get(t.rule_key) ?? t.rule_key} — ${t.subject_label}`]),
  );
}

const TOOL_LABEL: Record<string, string> = {
  "llm.rule_verdict": "LLM verdict",
  "llm.support_check": "LLM support check",
  "llm.extract_entities": "LLM extraction",
  "model.check": "trained-model check",
  "register.link": "register linking",
  "retrieval.hybrid": "hybrid retrieval",
  "retrieval.keyword": "keyword retrieval",
  "retrieval.document_scope": "all document sentences",
};

/**
 * A readable one-line summary of an event (rule, subject, status, usage) plus the raw
 * identifiers, which the timeline shows only on demand (tooltip).
 */
export function describeEvent(event: AgentEvent, labels: TaskLabels = new Map()): { text: string; ids: string } {
  const p = event.payload as Record<string, unknown>;
  const parts: string[] = [];
  const add = (value: unknown, prefix = "") => {
    if (value !== undefined && value !== null && value !== "") parts.push(`${prefix}${String(value)}`);
  };
  const taskKey = typeof p.task_key === "string" ? p.task_key : undefined;
  if (taskKey) add(labels.get(taskKey) ?? "a check");
  if (typeof p.tool === "string") add(TOOL_LABEL[p.tool] ?? p.tool);
  const status = p.status ?? p.machine_status;
  if (typeof status === "string") add(status.replaceAll("_", " "));
  if (typeof p.verified === "boolean") add(p.verified ? "verified" : "not verified");
  if (typeof p.input_tokens === "number") add(`${p.input_tokens}+${String(p.output_tokens)}`, "tokens ");
  if (typeof p.cost_usd === "number") add(formatUsd(p.cost_usd), "cost ");
  if (p.usage_source === "mock_estimate") parts.push("(mock estimate)");
  if (typeof p.tasks === "number") add(p.tasks, "tasks ");
  if (typeof p.documents === "number") add(p.documents, "documents ");
  if (typeof p.findings === "number") add(p.findings, "findings ");
  if (typeof p.accepted === "number") add(p.accepted, "entities accepted ");
  add(p.code, "code ");
  if (typeof p.message === "string") add(p.message.replace(/\b[0-9a-f]{8}-[0-9a-f-]{27}\b/g, "…"));
  const ids = ["task_key", "finding_id", "document_id"]
    .filter((k) => typeof p[k] === "string")
    .map((k) => `${k}: ${String(p[k])}`)
    .join("\n");
  return { text: parts.join(" · "), ids };
}

function Timeline({
  events,
  streamStatus,
  labels,
}: {
  events: AgentEvent[];
  streamStatus: string;
  labels: TaskLabels;
}) {
  const [agent, setAgent] = useState("all");
  const agents = useMemo(() => [...new Set(events.map((e) => e.agent))], [events]);
  const shown = (agent === "all" ? events : events.filter((e) => e.agent === agent)).slice(-400).reverse();
  return (
    <section aria-labelledby="timeline-heading" className="grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="timeline-heading" className="flex items-center gap-2 text-lg font-semibold">
          Agent timeline
          {streamStatus === "live" ? (
            <span className="inline-flex items-center gap-1 text-xs font-normal text-emerald-700">
              <Radio className="size-3" aria-hidden /> live
            </span>
          ) : streamStatus === "reconnecting" ? (
            <span className="text-xs font-normal text-amber-700">reconnecting…</span>
          ) : null}
        </h2>
        <label className="flex items-center gap-2 text-sm">
          Agent
          <NativeSelect value={agent} onChange={(e) => setAgent(e.target.value)} aria-label="Filter timeline by agent">
            <option value="all">All agents</option>
            {agents.map((a) => (
              <option key={a} value={a}>
                {AGENT_LABEL[a] ?? a}
              </option>
            ))}
          </NativeSelect>
        </label>
      </div>
      {shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">No events yet.</p>
      ) : (
        <ol className="grid max-h-[28rem] gap-1 overflow-auto rounded-xl border bg-card p-2 text-sm" aria-live="polite">
          {shown.map((e) => {
            const { text, ids } = describeEvent(e, labels);
            return (
            <li
              key={e.seq}
              className={cn(
                "grid grid-cols-[4rem_9rem_7rem_1fr] gap-2 rounded px-2 py-1",
                e.event_type === "error" && "bg-red-50 text-red-900",
              )}
            >
              <span className="text-xs text-muted-foreground tabular-nums">#{e.seq}</span>
              <span className="font-medium">{AGENT_LABEL[e.agent] ?? e.agent}</span>
              <span className="text-muted-foreground">{e.event_type.replaceAll("_", " ")}</span>
              <span className="min-w-0 truncate" title={ids ? `${text}\n${ids}` : text}>
                {text}
              </span>
            </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
