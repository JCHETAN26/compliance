"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowLeft, ChevronDown, ChevronRight, Download, ExternalLink, Gavel, Loader2 } from "lucide-react";
import type { Route } from "next";
import Link from "next/link";
import { useId, useMemo, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { useWorkspace } from "@/components/workspaces/workspace-detail";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
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
import { ApiError, errorMessage, type Citation, type Finding, type FindingStatus, type Report } from "@/lib/api/client";
import { fetchReport, overrideFinding, reportExportUrl, reviewKeys } from "@/lib/api/reviews";
import { cn, formatDate } from "@/lib/utils";

import {
  ComplianceChip,
  DecidedByBadge,
  formatUsd,
  RunStatusChip,
  SeverityChip,
  StatusChip,
  VerifiedMark,
} from "./labels";
import { MockBanner } from "./reviews-page";

export interface FindingFilters {
  status: string;
  compliance: string;
  severity: string;
  decidedBy: string;
  verifiedOnly: boolean;
  text: string;
}

export const EMPTY_FILTERS: FindingFilters = {
  status: "all",
  compliance: "all",
  severity: "all",
  decidedBy: "all",
  verifiedOnly: false,
  text: "",
};

const SEVERITY_ORDER = { high: 0, medium: 1, low: 2 } as const;

/** Pure, unit-tested filter used by the findings table. */
export function filterFindings(findings: Finding[], f: FindingFilters): Finding[] {
  const needle = f.text.trim().toLowerCase();
  return findings
    .filter(
      (x) =>
        (f.status === "all" || x.status === f.status) &&
        (f.compliance === "all" || x.compliance === f.compliance) &&
        (f.severity === "all" || x.severity === f.severity) &&
        (f.decidedBy === "all" || x.decided_by === f.decidedBy) &&
        (!f.verifiedOnly || x.verified) &&
        (!needle ||
          x.rule_title.toLowerCase().includes(needle) ||
          x.subject_label.toLowerCase().includes(needle)),
    )
    .sort(
      (a, b) =>
        SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] ||
        a.subject_label.localeCompare(b.subject_label) ||
        a.rule_title.localeCompare(b.rule_title),
    );
}

export function citationHref(workspaceId: string, citation: Citation): Route | null {
  if (!citation.document_id) return null;
  return `/workspaces/${workspaceId}/documents/${citation.document_id}?citation=${citation.id}` as Route;
}

type Tab = "findings" | "subjects" | "risks";

export function ReportView({ runId }: { runId: string }) {
  const workspace = useWorkspace();
  const report = useQuery({
    queryKey: reviewKeys.report(runId),
    queryFn: () => fetchReport(runId),
    refetchInterval: (q) => (q.state.data && ["queued", "running"].includes(q.state.data.run_status) ? 5000 : false),
  });
  const [tab, setTab] = useState<Tab>("findings");
  const [filters, setFilters] = useState<FindingFilters>(EMPTY_FILTERS);

  if (report.isPending) return <Skeleton className="h-96 rounded-xl" />;
  if (report.isError) {
    const notFound = report.error instanceof ApiError && [404, 422].includes(report.error.status);
    return (
      <Alert variant="destructive">
        <AlertTriangle aria-hidden />
        <AlertTitle>{notFound ? "Report not found" : "Could not load the report"}</AlertTitle>
        <AlertDescription>{notFound ? "The review may have been deleted." : errorMessage(report.error)}</AlertDescription>
      </Alert>
    );
  }
  const data = report.data;
  const canOverride = workspace.role !== "viewer" && !["queued", "running"].includes(data.run_status);
  return (
    <div className="grid gap-5">
      <Link
        href={`/workspaces/${workspace.id}/reviews/${runId}` as Route}
        className="inline-flex w-fit items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-4" aria-hidden />
        Live run view
      </Link>
      <MockBanner />
      <ReportHeader report={data} />
      <Summary report={data} />
      <div role="tablist" aria-label="Report sections" className="flex gap-1 border-b">
        {(
          [
            ["findings", `Findings (${data.findings.length})`],
            ["subjects", `By subject (${data.subjects.length})`],
            ["risks", `Risks & actions (${data.risks.length})`],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            type="button"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            className={cn(
              "-mb-px border-b-2 px-3 py-2 text-sm font-medium focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
              tab === key ? "border-primary" : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "findings" ? (
        <div className="grid gap-3" role="tabpanel">
          <Filters filters={filters} onChange={setFilters} />
          <FindingsTable
            findings={filterFindings(data.findings, filters)}
            workspaceId={workspace.id}
            runId={runId}
            mock={data.llm_is_mock}
            canOverride={canOverride}
          />
        </div>
      ) : tab === "subjects" ? (
        <SubjectView report={data} workspaceId={workspace.id} runId={runId} canOverride={canOverride} />
      ) : (
        <RiskView report={data} />
      )}
      <section className="rounded-xl border bg-card p-4 text-sm">
        <h2 className="mb-2 font-semibold">Limitations</h2>
        <ul className="list-disc pl-5 text-muted-foreground">
          {data.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function ReportHeader({ report }: { report: Report }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4 rounded-xl border bg-card p-4">
      <div className="grid gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-xl font-semibold">
            Report: {report.rule_pack.name} <span className="text-muted-foreground">{report.rule_pack.version}</span>
          </h2>
          <RunStatusChip status={report.run_status} />
        </div>
        <p className="text-sm text-muted-foreground">
          Generated {formatDate(report.generated_at)} · LLM {report.llm_provider}/{report.llm_model}
          {report.llm_is_mock ? " (mock)" : ""} · model {report.model_version ?? "n/a"}
          {report.tau !== null && report.tau !== undefined ? ` (tau ${report.tau})` : ""} · {report.budget.tokens_used.toLocaleString()} tokens,{" "}
          {formatUsd(report.budget.usd_used)}
        </p>
      </div>
      <div className="flex gap-2">
        <Button asChild variant="outline" size="sm">
          <a href={reportExportUrl(report.run_id, "md")} download>
            <Download aria-hidden />
            Markdown
          </a>
        </Button>
        <Button asChild variant="outline" size="sm">
          <a href={reportExportUrl(report.run_id, "json")} download={`report-${report.run_id}.json`}>
            <Download aria-hidden />
            JSON
          </a>
        </Button>
      </div>
    </div>
  );
}

function Summary({ report }: { report: Report }) {
  return (
    <section aria-labelledby="summary-heading" className="rounded-xl border bg-card p-4">
      <h2 id="summary-heading" className="mb-2 font-semibold">
        Executive summary
      </h2>
      {!report.complete ? (
        <p className="mb-2 text-sm text-amber-800">This run has not finished; the report may be incomplete.</p>
      ) : null}
      <ul className="grid gap-1 text-sm">
        {report.summary.map((s) => (
          <li key={s.text}>
            {s.text}
            {s.finding_ids.length ? (
              <span className="ml-1 text-xs text-muted-foreground">(based on {s.finding_ids.length} finding(s))</span>
            ) : null}
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-muted-foreground">
        Only verified findings (exact quote at its offsets and semantically supported) or human decisions are
        stated as facts. Everything else is listed as needing review.
      </p>
    </section>
  );
}

function Filters({ filters, onChange }: { filters: FindingFilters; onChange: (f: FindingFilters) => void }) {
  const id = useId();
  const set = (patch: Partial<FindingFilters>) => onChange({ ...filters, ...patch });
  const selects: [keyof FindingFilters, string, [string, string][]][] = [
    ["status", "Status", [["all", "All"], ["satisfied", "Satisfied"], ["violated", "Violated"], ["not_mentioned", "Not mentioned"], ["needs_review", "Needs review"]]],
    ["compliance", "Compliance", [["all", "All"], ["compliant", "Compliant"], ["non_compliant", "Non-compliant"], ["needs_review", "Needs review"], ["not_assessed", "Not assessed"]]],
    ["severity", "Severity", [["all", "All"], ["high", "High"], ["medium", "Medium"], ["low", "Low"]]],
    ["decidedBy", "Decided by", [["all", "All"], ["model", "Model"], ["llm", "LLM"], ["system", "Rules"], ["human", "Human"]]],
  ];
  return (
    <div className="flex flex-wrap items-end gap-3">
      {selects.map(([key, label, options]) => (
        <div key={key} className="grid gap-1">
          <Label htmlFor={`${id}-${key}`} className="text-xs">
            {label}
          </Label>
          <NativeSelect id={`${id}-${key}`} value={String(filters[key])} onChange={(e) => set({ [key]: e.target.value })}>
            {options.map(([value, text]) => (
              <option key={value} value={value}>
                {text}
              </option>
            ))}
          </NativeSelect>
        </div>
      ))}
      <div className="grid gap-1">
        <Label htmlFor={`${id}-text`} className="text-xs">
          Rule or subject
        </Label>
        <Input id={`${id}-text`} value={filters.text} onChange={(e) => set({ text: e.target.value })} placeholder="Filter" />
      </div>
      <label className="flex items-center gap-2 pb-2 text-sm">
        <input type="checkbox" checked={filters.verifiedOnly} onChange={(e) => set({ verifiedOnly: e.target.checked })} />
        Verified only
      </label>
    </div>
  );
}

function FindingsTable(props: {
  findings: Finding[];
  workspaceId: string;
  runId: string;
  mock: boolean;
  canOverride: boolean;
}) {
  if (props.findings.length === 0)
    return <p className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">No findings match these filters.</p>;
  return (
    <div className="overflow-x-auto rounded-xl border bg-card">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-left text-xs text-muted-foreground">
          <tr>
            <th className="w-8 px-2 py-2" />
            <th className="px-2 py-2 font-medium">Rule</th>
            <th className="px-2 py-2 font-medium">Subject</th>
            <th className="px-2 py-2 font-medium">Status</th>
            <th className="px-2 py-2 font-medium">Compliance</th>
            <th className="px-2 py-2 font-medium">Decided by</th>
            <th className="px-2 py-2 font-medium">Confidence</th>
            <th className="px-2 py-2 font-medium">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {props.findings.map((f) => (
            <FindingRow key={f.id} finding={f} {...props} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FindingRow({
  finding,
  workspaceId,
  runId,
  mock,
  canOverride,
}: {
  finding: Finding;
  workspaceId: string;
  runId: string;
  mock: boolean;
  canOverride: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="border-b align-top last:border-0">
        <td className="px-2 py-2">
          <button
            type="button"
            aria-expanded={open}
            aria-label={open ? "Hide details" : "Show details"}
            onClick={() => setOpen(!open)}
            className="rounded p-0.5 hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
          >
            {open ? <ChevronDown className="size-4" aria-hidden /> : <ChevronRight className="size-4" aria-hidden />}
          </button>
        </td>
        <td className="px-2 py-2">
          <div className="flex flex-wrap items-center gap-1">
            <span className="font-medium">{finding.rule_title}</span>
            <SeverityChip severity={finding.severity} />
          </div>
        </td>
        <td className="max-w-56 truncate px-2 py-2" title={finding.subject_label}>
          {finding.subject_label}
        </td>
        <td className="px-2 py-2">
          <StatusChip status={finding.status} />
        </td>
        <td className="px-2 py-2">
          <ComplianceChip compliance={finding.compliance} />
        </td>
        <td className="px-2 py-2">
          <DecidedByBadge decidedBy={finding.decided_by} mock={mock} />
        </td>
        <td className="px-2 py-2 tabular-nums">{finding.confidence !== null ? finding.confidence.toFixed(3) : "—"}</td>
        <td className="px-2 py-2">
          <VerifiedMark finding={finding} />
        </td>
      </tr>
      {open ? (
        <tr className="border-b bg-muted/20">
          <td />
          <td colSpan={7} className="px-2 pt-1 pb-3">
            <FindingDetails finding={finding} workspaceId={workspaceId} runId={runId} canOverride={canOverride} />
          </td>
        </tr>
      ) : null}
    </>
  );
}

export function FindingDetails({
  finding,
  workspaceId,
  runId,
  canOverride,
}: {
  finding: Finding;
  workspaceId: string;
  runId: string;
  canOverride: boolean;
}) {
  const reasons = (finding.verification.reasons ?? []) as string[];
  return (
    <div className="grid gap-3 text-sm">
      <p>
        <span className="font-medium">Rationale: </span>
        {finding.rationale}
      </p>
      {finding.machine_status !== finding.status || finding.override ? (
        <p className="text-muted-foreground">
          Machine result: {finding.machine_status.replace("_", " ")} by {finding.machine_decided_by}; after
          verification: {finding.verified_status.replace("_", " ")}.
        </p>
      ) : null}
      {finding.escalated ? <p className="text-muted-foreground">Escalated to the LLM: {finding.escalation_reason}</p> : null}
      {reasons.length ? (
        <div>
          <span className="font-medium">Why it needs review: </span>
          {reasons.join("; ")}
        </div>
      ) : null}
      {finding.reasoning.length ? (
        <div>
          <span className="font-medium">Reasoning steps</span>
          <ol className="mt-1 list-decimal pl-5">
            {finding.reasoning.map((step, i) => (
              <li key={i}>
                <span className="font-medium">{String(step.step)}:</span> {String(step.detail)}
              </li>
            ))}
          </ol>
        </div>
      ) : null}
      <div>
        <span className="font-medium">Citations ({finding.citations.length})</span>
        {finding.citations.length === 0 ? (
          <p className="text-muted-foreground">No citable evidence.</p>
        ) : (
          <ul className="mt-1 grid gap-2">
            {finding.citations.map((c) => {
              const href = citationHref(workspaceId, c);
              return (
                <li key={c.id} className="rounded-md border bg-card p-2">
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">{c.document_filename}</span>
                    {c.page_number ? <span>p. {c.page_number}</span> : null}
                    <span>{c.role.replace("_", " ")}</span>
                    <span>
                      {c.current !== "valid"
                        ? c.current === "source_deleted"
                          ? "source deleted"
                          : "source changed"
                        : c.verified
                          ? "verified"
                          : c.supported === false
                            ? "not supporting"
                            : "unverified"}
                    </span>
                    {href && c.current === "valid" ? (
                      <Link href={href} className="inline-flex items-center gap-1 text-primary underline-offset-2 hover:underline">
                        <ExternalLink className="size-3" aria-hidden />
                        Open in document
                      </Link>
                    ) : null}
                  </div>
                  <blockquote className="border-l-2 border-amber-400 pl-2 whitespace-pre-wrap">{c.quote}</blockquote>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      {finding.override ? (
        <p className="rounded-md border border-sky-200 bg-sky-50 p-2">
          Human decision by {finding.override.user_email ?? "a former member"} on {formatDate(finding.override.created_at)}:{" "}
          {finding.override.previous_status.replace("_", " ")} → {finding.override.new_status.replace("_", " ")}. “
          {finding.override.comment}” ({finding.override_count} decision(s) recorded)
        </p>
      ) : null}
      {canOverride ? <OverrideDialog finding={finding} runId={runId} /> : null}
    </div>
  );
}

function OverrideDialog({ finding, runId }: { finding: Finding; runId: string }) {
  const id = useId();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<FindingStatus>(finding.status);
  const [comment, setComment] = useState("");
  const save = useMutation({
    mutationFn: () => overrideFinding(finding.id, { status, comment: comment.trim() }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: reviewKeys.report(runId) });
      void queryClient.invalidateQueries({ queryKey: reviewKeys.run(runId) });
      void queryClient.invalidateQueries({ queryKey: reviewKeys.tasks(runId) });
      toast.success("Decision recorded");
      setOpen(false);
      setComment("");
    },
  });
  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (comment.trim()) save.mutate();
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="outline" size="sm" className="w-fit" onClick={() => setOpen(true)}>
        <Gavel aria-hidden />
        Record decision
      </Button>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Record a decision</DialogTitle>
          <DialogDescription>
            {finding.rule_title} — {finding.subject_label}. The machine finding is kept unchanged; your decision is
            stored as an audited override with your comment.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={onSubmit} className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`${id}-status`}>Decision</Label>
            <NativeSelect id={`${id}-status`} value={status} onChange={(e) => setStatus(e.target.value as FindingStatus)}>
              <option value="satisfied">Satisfied</option>
              <option value="violated">Violated</option>
              <option value="not_mentioned">Not mentioned</option>
              <option value="needs_review">Needs review</option>
            </NativeSelect>
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-comment`}>Comment (required)</Label>
            <textarea
              id={`${id}-comment`}
              value={comment}
              maxLength={2000}
              onChange={(e) => setComment(e.target.value)}
              className="min-h-24 rounded-md border border-input bg-card px-3 py-2 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50"
              aria-invalid={save.isError ? true : undefined}
            />
          </div>
          {save.isError ? (
            <p className="text-sm text-destructive" role="alert">
              {errorMessage(save.error)}
            </p>
          ) : null}
          <DialogFooter>
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </DialogClose>
            <Button type="submit" disabled={!comment.trim() || save.isPending}>
              {save.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
              Save decision
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function SubjectView({
  report,
  workspaceId,
  runId,
  canOverride,
}: {
  report: Report;
  workspaceId: string;
  runId: string;
  canOverride: boolean;
}) {
  const bySubject = useMemo(() => {
    const map = new Map<string, Finding[]>();
    for (const f of report.findings) map.set(f.subject_key, [...(map.get(f.subject_key) ?? []), f]);
    return map;
  }, [report.findings]);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const tone: Record<string, string> = {
    at_risk: "border-red-300 bg-red-50",
    needs_review: "border-amber-300 bg-amber-50",
    compliant: "border-emerald-300 bg-emerald-50",
    not_assessed: "",
  };
  return (
    <ul className="grid gap-2" role="tabpanel">
      {report.subjects.map((s) => (
        <li key={s.subject_key} className={cn("rounded-xl border bg-card", tone[s.status])}>
          <button
            type="button"
            aria-expanded={openKey === s.subject_key}
            onClick={() => setOpenKey(openKey === s.subject_key ? null : s.subject_key)}
            className="flex w-full flex-wrap items-center gap-3 px-4 py-3 text-left focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
          >
            <span className="font-medium">{s.subject_label}</span>
            <span className="text-sm capitalize">{s.status.replace("_", " ")}</span>
            <span className="ml-auto text-xs text-muted-foreground">
              {s.violated} non-compliant · {s.needs_review} need review · {s.compliant} compliant
            </span>
          </button>
          {openKey === s.subject_key ? (
            <div className="grid gap-3 border-t px-4 py-3">
              {(bySubject.get(s.subject_key) ?? []).map((f) => (
                <div key={f.id} className="grid gap-2 rounded-lg border bg-card p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{f.rule_title}</span>
                    <StatusChip status={f.status} />
                    <ComplianceChip compliance={f.compliance} />
                    <DecidedByBadge decidedBy={f.decided_by} mock={report.llm_is_mock} />
                    <VerifiedMark finding={f} />
                  </div>
                  <FindingDetails finding={f} workspaceId={workspaceId} runId={runId} canOverride={canOverride} />
                </div>
              ))}
            </div>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function RiskView({ report }: { report: Report }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2" role="tabpanel">
      <section className="rounded-xl border bg-card p-4">
        <h2 className="mb-2 font-semibold">Risk ranking (verified or human-decided)</h2>
        {report.risks.length === 0 ? (
          <p className="text-sm text-muted-foreground">No verified non-compliant findings.</p>
        ) : (
          <ol className="grid gap-1 text-sm">
            {report.risks.map((r) => (
              <li key={r.finding_id} className="flex flex-wrap items-center gap-2">
                <span className="w-6 text-muted-foreground tabular-nums">{r.rank}.</span>
                <SeverityChip severity={r.severity} />
                <span className="font-medium">{r.rule_title}</span>
                <span className="text-muted-foreground">— {r.subject_label}</span>
                {r.decided_by === "human" ? <span className="text-xs text-sky-800">(human decision)</span> : null}
              </li>
            ))}
          </ol>
        )}
      </section>
      <section className="rounded-xl border bg-card p-4">
        <h2 className="mb-2 font-semibold">Recommended actions</h2>
        {report.actions.length === 0 ? (
          <p className="text-sm text-muted-foreground">None.</p>
        ) : (
          <ul className="grid gap-2 text-sm">
            {report.actions.map((a) => (
              <li key={a.action}>
                <span className={cn("mr-1 font-semibold capitalize", a.priority === "high" && "text-red-700")}>{a.priority}:</span>
                {a.action}
              </li>
            ))}
          </ul>
        )}
        <p className="mt-2 text-xs text-muted-foreground">Action text comes from the rule pack&apos;s remediation guidance.</p>
      </section>
      <section className="rounded-xl border bg-card p-4 lg:col-span-2">
        <h2 className="mb-2 font-semibold">Needs human review ({report.needs_review.length})</h2>
        <ul className="grid gap-1 text-sm">
          {report.needs_review.map((r) => (
            <li key={r.finding_id} className="flex flex-wrap items-center gap-2">
              <SeverityChip severity={r.severity} />
              <span className="font-medium">{r.rule_title}</span>
              <span className="text-muted-foreground">— {r.subject_label}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
