"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, BrainCircuit, Bot, Copy, Download, FileUp, Loader2, Network, Pencil } from "lucide-react";
import { useId, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { useWorkspace } from "@/components/workspaces/workspace-detail";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
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
import { errorMessage, type Rule, type RulePack, type RulePackSpec } from "@/lib/api/client";
import {
  exportRulePack,
  fetchRulePack,
  fetchRulePacks,
  importRulePack,
  rulePackExportUrl,
  rulePackKeys,
  updateRulePack,
} from "@/lib/api/reviews";
import { cn } from "@/lib/utils";

import { SeverityChip } from "./labels";

export function ModelSupportBadge({ rule }: { rule: Pick<Rule, "model_supported" | "scope" | "contractnli_key"> }) {
  if (rule.scope === "cross_document")
    return (
      <Badge variant="outline" className="gap-1" title="Deterministic reasoning across the register and verified findings">
        <Network className="size-3" aria-hidden />
        Cross-document
      </Badge>
    );
  return rule.model_supported ? (
    <Badge
      variant="outline"
      className="gap-1 border-emerald-300 bg-emerald-50 text-emerald-800"
      title={`Checked by the trained rule-check model (ContractNLI ${rule.contractnli_key ?? ""}), LLM below tau`}
    >
      <BrainCircuit className="size-3" aria-hidden />
      Trained model
    </Badge>
  ) : (
    <Badge variant="outline" className="gap-1" title="Custom rule: always checked by the LLM">
      <Bot className="size-3" aria-hidden />
      LLM
    </Badge>
  );
}

/** The editable spec of a stored pack (the PUT body). */
export function toSpec(pack: RulePack): RulePackSpec {
  return {
    key: pack.key,
    name: pack.name,
    version: pack.version,
    description: pack.description,
    ...(pack.provenance ? { source: pack.provenance } : {}),
    ...(pack.register ? { register: pack.register as RulePackSpec["register"] } : {}),
    rules: pack.rules.map((r) => ({
      key: r.key,
      title: r.title,
      description: r.description,
      hypothesis_text: r.hypothesis_text,
      scope: r.scope,
      applies_to_doc_types: r.applies_to_doc_types as RulePackSpec["rules"][number]["applies_to_doc_types"],
      severity: r.severity,
      model_supported: r.model_supported,
      contractnli_key: r.contractnli_key,
      compliant_statuses: r.compliant_statuses as RulePackSpec["rules"][number]["compliant_statuses"],
      cross_document: r.cross_document as RulePackSpec["rules"][number]["cross_document"],
      remediation: r.remediation,
    })),
  };
}

export function bumpPatch(version: string): string {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) return version;
  return `${match[1]}.${match[2]}.${Number(match[3]) + 1}`;
}

export function RulePacksPage() {
  const workspace = useWorkspace();
  const canWrite = workspace.role !== "viewer";
  const packs = useQuery({ queryKey: rulePackKeys.list(workspace.id), queryFn: () => fetchRulePacks(workspace.id) });
  const [selection, setSelectedId] = useState<string | null>(null);
  const selectedId = selection ?? packs.data?.[0]?.id ?? null;

  return (
    <div className="grid gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]">
      <aside className="grid content-start gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Rule packs</h2>
          {canWrite ? <ImportDialog workspaceId={workspace.id} onImported={setSelectedId} /> : null}
        </div>
        {packs.isPending ? (
          <Skeleton className="h-32" />
        ) : packs.isError ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not load rule packs</AlertTitle>
            <AlertDescription>{errorMessage(packs.error)}</AlertDescription>
          </Alert>
        ) : (
          <ul className="grid gap-2" aria-label="Rule packs">
            {packs.data.map((p) => (
              <li key={p.id}>
                <button
                  type="button"
                  aria-pressed={p.id === selectedId}
                  onClick={() => setSelectedId(p.id)}
                  className={cn(
                    "grid w-full gap-1 rounded-lg border bg-card px-3 py-2 text-left text-sm focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                    p.id === selectedId ? "border-primary" : "hover:bg-accent",
                  )}
                >
                  <span className="font-medium">{p.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {p.version} · {p.builtin ? "built-in" : "workspace"} · {p.rule_count} rules ({p.model_supported_count} model,{" "}
                    {p.cross_document_count} cross-document)
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>
      {selectedId ? <PackDetail packId={selectedId} canWrite={canWrite} workspaceId={workspace.id} onCopied={setSelectedId} /> : null}
    </div>
  );
}

function PackDetail({
  packId,
  canWrite,
  workspaceId,
  onCopied,
}: {
  packId: string;
  canWrite: boolean;
  workspaceId: string;
  onCopied: (id: string) => void;
}) {
  const queryClient = useQueryClient();
  const pack = useQuery({ queryKey: rulePackKeys.detail(packId), queryFn: () => fetchRulePack(packId) });
  const copy = useMutation({
    mutationFn: async () => importRulePack(workspaceId, await exportRulePack(packId)),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: rulePackKeys.list(workspaceId) });
      toast.success("Copied into this workspace; you can edit the copy.");
      onCopied(created.id);
    },
    onError: (error) => toast.error(errorMessage(error)),
  });
  if (pack.isPending) return <Skeleton className="h-96 rounded-xl" />;
  if (pack.isError)
    return (
      <Alert variant="destructive">
        <AlertTriangle aria-hidden />
        <AlertTitle>Could not load this rule pack</AlertTitle>
        <AlertDescription>{errorMessage(pack.error)}</AlertDescription>
      </Alert>
    );
  const data = pack.data;
  return (
    <section className="grid content-start gap-4" aria-labelledby="pack-heading">
      <div className="flex flex-wrap items-start justify-between gap-3 rounded-xl border bg-card p-4">
        <div className="grid gap-1">
          <h2 id="pack-heading" className="text-xl font-semibold">
            {data.name} <span className="text-muted-foreground">{data.version}</span>
          </h2>
          <p className="max-w-2xl text-sm text-muted-foreground">{data.description}</p>
          {data.provenance ? (
            <p className="text-xs text-muted-foreground">
              Source: {Object.entries(data.provenance).map(([k, v]) => `${k}: ${v}`).join(" · ")}
            </p>
          ) : null}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline" size="sm">
            <a href={rulePackExportUrl(data.id)} download>
              <Download aria-hidden />
              Export YAML
            </a>
          </Button>
          {canWrite && data.builtin ? (
            <Button size="sm" variant="outline" onClick={() => copy.mutate()} disabled={copy.isPending}>
              {copy.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Copy aria-hidden />}
              Copy to workspace
            </Button>
          ) : null}
        </div>
      </div>
      {data.builtin ? (
        <p className="text-sm text-muted-foreground">Built-in packs are read-only. Copy one into the workspace to edit it.</p>
      ) : null}
      <ul className="grid gap-2">
        {data.rules.map((rule) => (
          <li key={rule.id} className="grid gap-2 rounded-xl border bg-card p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{rule.title}</span>
              <SeverityChip severity={rule.severity} />
              <ModelSupportBadge rule={rule} />
              <span className="text-xs text-muted-foreground">
                {rule.scope.replace("_", " ")} · {rule.applies_to_doc_types.join(", ")}
                {rule.compliant_statuses ? ` · compliant when ${rule.compliant_statuses.join("/")}` : ""}
              </span>
              {data.editable ? <EditRuleDialog pack={data} rule={rule} /> : null}
            </div>
            <p className="text-sm">{rule.hypothesis_text}</p>
            {rule.remediation ? <p className="text-xs text-muted-foreground">Remediation: {rule.remediation}</p> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ImportDialog({ workspaceId, onImported }: { workspaceId: string; onImported: (id: string) => void }) {
  const id = useId();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [yaml, setYaml] = useState("");
  const save = useMutation({
    mutationFn: () => importRulePack(workspaceId, yaml),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: rulePackKeys.list(workspaceId) });
      toast.success(`Imported ${created.name}`);
      onImported(created.id);
      setOpen(false);
      setYaml("");
    },
  });
  const details = save.error && "details" in save.error ? ((save.error as { details?: { msg: string }[] }).details ?? []) : [];
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <FileUp aria-hidden />
        Import
      </Button>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Import a rule pack (YAML)</DialogTitle>
          <DialogDescription>
            The pack is validated like the built-in packs. A rule can be model-supported only if its text is one of the
            trained ContractNLI hypotheses, verbatim; custom rules use the LLM.
          </DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-3"
          onSubmit={(e: FormEvent<HTMLFormElement>) => {
            e.preventDefault();
            if (yaml.trim()) save.mutate();
          }}
        >
          <Label htmlFor={`${id}-file`}>YAML file</Label>
          <Input
            id={`${id}-file`}
            type="file"
            accept=".yaml,.yml,text/yaml"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (file) setYaml(await file.text());
            }}
          />
          <Label htmlFor={`${id}-yaml`}>or paste YAML</Label>
          <textarea
            id={`${id}-yaml`}
            value={yaml}
            onChange={(e) => setYaml(e.target.value)}
            className="min-h-48 rounded-md border border-input bg-card px-3 py-2 font-mono text-xs shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50"
          />
          {save.isError ? (
            <div className="text-sm text-destructive" role="alert">
              {errorMessage(save.error)}
              {details.length ? (
                <ul className="mt-1 list-disc pl-5">
                  {details.slice(0, 10).map((d) => (
                    <li key={d.msg}>{d.msg}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}
          <DialogFooter>
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </DialogClose>
            <Button type="submit" disabled={!yaml.trim() || save.isPending}>
              {save.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
              Import
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function EditRuleDialog({ pack, rule }: { pack: RulePack; rule: Rule }) {
  const id = useId();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState(rule.title);
  const [hypothesis, setHypothesis] = useState(rule.hypothesis_text);
  const [severity, setSeverity] = useState(rule.severity);
  const [modelSupported, setModelSupported] = useState(rule.model_supported);
  const save = useMutation({
    mutationFn: () => {
      const spec = toSpec(pack);
      spec.version = bumpPatch(pack.version);
      spec.rules = spec.rules.map((r) =>
        r.key === rule.key ? { ...r, title, hypothesis_text: hypothesis, severity, model_supported: modelSupported } : r,
      );
      return updateRulePack(pack.id, spec);
    },
    onSuccess: (updated) => {
      queryClient.setQueryData(rulePackKeys.detail(pack.id), updated);
      void queryClient.invalidateQueries({ queryKey: rulePackKeys.list(pack.workspace_id ?? "") });
      toast.success(`Saved as version ${updated.version}`);
      setOpen(false);
    },
  });
  const details = save.error && "details" in save.error ? ((save.error as { details?: { msg: string }[] }).details ?? []) : [];
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="ghost" size="sm" className="ml-auto" onClick={() => setOpen(true)} aria-label={`Edit ${rule.title}`}>
        <Pencil aria-hidden />
        Edit
      </Button>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit rule</DialogTitle>
          <DialogDescription>
            Saving creates version {bumpPatch(pack.version)}. Existing review runs keep the rules they checked.
          </DialogDescription>
        </DialogHeader>
        <form
          className="grid gap-3"
          onSubmit={(e: FormEvent<HTMLFormElement>) => {
            e.preventDefault();
            save.mutate();
          }}
        >
          <Label htmlFor={`${id}-title`}>Title</Label>
          <Input id={`${id}-title`} value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} />
          <Label htmlFor={`${id}-hyp`}>Hypothesis (what the contract should state)</Label>
          <textarea
            id={`${id}-hyp`}
            value={hypothesis}
            onChange={(e) => setHypothesis(e.target.value)}
            className="min-h-24 rounded-md border border-input bg-card px-3 py-2 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50"
          />
          <Label htmlFor={`${id}-sev`}>Severity</Label>
          <NativeSelect id={`${id}-sev`} value={severity} onChange={(e) => setSeverity(e.target.value as Rule["severity"])}>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </NativeSelect>
          {rule.scope === "per_document" ? (
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" checked={modelSupported} onChange={(e) => setModelSupported(e.target.checked)} />
              <span>
                Checked by the trained model. Allowed only while the text equals the trained ContractNLI hypothesis
                ({rule.contractnli_key ?? "none"}); edited or custom text must use the LLM.
              </span>
            </label>
          ) : null}
          {save.isError ? (
            <div className="text-sm text-destructive" role="alert">
              {errorMessage(save.error)}
              {details.length ? (
                <ul className="mt-1 list-disc pl-5">
                  {details.slice(0, 5).map((d) => (
                    <li key={d.msg}>{d.msg}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}
          <DialogFooter>
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </DialogClose>
            <Button type="submit" disabled={save.isPending || !title.trim() || !hypothesis.trim()}>
              {save.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
              Save
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
