import { Bot, BrainCircuit, CheckCircle2, Cog, TriangleAlert, User, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { Finding, RunStatus } from "@/lib/api/client";
import { cn } from "@/lib/utils";

const TONE = {
  green: "border-emerald-300 bg-emerald-50 text-emerald-800",
  red: "border-red-300 bg-red-50 text-red-800",
  amber: "border-amber-300 bg-amber-50 text-amber-900",
  gray: "border-slate-300 bg-slate-50 text-slate-700",
  blue: "border-sky-300 bg-sky-50 text-sky-800",
} as const;

const STATUS: Record<Finding["status"], { label: string; tone: keyof typeof TONE }> = {
  satisfied: { label: "Satisfied", tone: "green" },
  violated: { label: "Violated", tone: "red" },
  not_mentioned: { label: "Not mentioned", tone: "gray" },
  needs_review: { label: "Needs review", tone: "amber" },
};

const COMPLIANCE: Record<Finding["compliance"], { label: string; tone: keyof typeof TONE }> = {
  compliant: { label: "Compliant", tone: "green" },
  non_compliant: { label: "Non-compliant", tone: "red" },
  needs_review: { label: "Needs review", tone: "amber" },
  not_assessed: { label: "Not assessed", tone: "gray" },
};

const RUN: Record<RunStatus, { label: string; tone: keyof typeof TONE }> = {
  queued: { label: "Queued", tone: "blue" },
  running: { label: "Running", tone: "blue" },
  paused: { label: "Paused", tone: "gray" },
  awaiting_review: { label: "Awaiting review", tone: "amber" },
  completed: { label: "Completed", tone: "green" },
  failed: { label: "Failed", tone: "red" },
  cancelled: { label: "Cancelled", tone: "gray" },
};

function Chip({ label, tone, className }: { label: string; tone: keyof typeof TONE; className?: string }) {
  return (
    <Badge variant="outline" className={cn(TONE[tone], className)}>
      {label}
    </Badge>
  );
}

export function StatusChip({ status }: { status: Finding["status"] }) {
  return <Chip {...STATUS[status]} />;
}

export function ComplianceChip({ compliance }: { compliance: Finding["compliance"] }) {
  return <Chip {...COMPLIANCE[compliance]} />;
}

export function RunStatusChip({ status }: { status: RunStatus }) {
  return <Chip {...RUN[status]} />;
}

export function SeverityChip({ severity }: { severity: "low" | "medium" | "high" }) {
  const tone = severity === "high" ? "red" : severity === "medium" ? "amber" : "gray";
  return <Chip label={severity.charAt(0).toUpperCase() + severity.slice(1)} tone={tone} />;
}

const DECIDED = {
  model: { label: "Model", icon: BrainCircuit, title: "Decided by the trained rule-check model" },
  llm: { label: "LLM", icon: Bot, title: "Decided by the LLM (structured verdict)" },
  system: { label: "Rules", icon: Cog, title: "Deterministic cross-document reasoning" },
  human: { label: "Human", icon: User, title: "Decided by a reviewer (audited override)" },
} as const;

export function DecidedByBadge({ decidedBy, mock }: { decidedBy: Finding["decided_by"]; mock?: boolean }) {
  const { label, icon: Icon, title } = DECIDED[decidedBy];
  const isMock = mock && (decidedBy === "llm" || decidedBy === "model");
  return (
    <Badge variant="outline" title={title + (isMock ? " — MOCK backend" : "")} className="gap-1">
      <Icon className="size-3" aria-hidden />
      {label}
      {isMock ? <span className="font-semibold text-amber-700">(mock)</span> : null}
    </Badge>
  );
}

export function VerifiedMark({ finding }: { finding: Finding }) {
  if (finding.source_changed)
    return (
      <span className="inline-flex items-center gap-1 text-xs text-amber-800" title="A cited source was changed or deleted after verification">
        <TriangleAlert className="size-3.5" aria-hidden />
        Source changed
      </span>
    );
  return finding.verified ? (
    <span className="inline-flex items-center gap-1 text-xs text-emerald-700" title="Exact quote at its offsets and semantically supported">
      <CheckCircle2 className="size-3.5" aria-hidden />
      Verified
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground" title="Not verified: no supported, exact citation">
      <XCircle className="size-3.5" aria-hidden />
      Unverified
    </span>
  );
}

export function formatUsd(value: number): string {
  if (value === 0) return "$0";
  return value < 0.01 ? `$${value.toFixed(6)}` : `$${value.toFixed(4)}`;
}

export const ACTIVE_RUN_STATUSES: RunStatus[] = ["queued", "running"];

/** A count from a run's ``totals`` dictionary (e.g. ``findings_by_status.needs_review``). */
export function totalCount(totals: Record<string, unknown>, group: string, key?: string): number {
  const value = totals[group];
  if (key === undefined) {
    if (typeof value === "number") return value;
    if (value && typeof value === "object")
      return Object.values(value as Record<string, unknown>).reduce<number>((a, n) => a + (typeof n === "number" ? n : 0), 0);
    return 0;
  }
  const inner = value && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined;
  return typeof inner === "number" ? inner : 0;
}
