"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { useId, useState, type FormEvent } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage, type ReviewSettings, type Workspace } from "@/lib/api/client";
import { fetchLlmSettings, fetchReviewSettings, saveReviewSettings, settingsKeys } from "@/lib/api/reviews";

export function ReviewSettingsCard({ workspace }: { workspace: Workspace }) {
  const settings = useQuery({
    queryKey: settingsKeys.review(workspace.id),
    queryFn: () => fetchReviewSettings(workspace.id),
  });
  const llm = useQuery({ queryKey: settingsKeys.llm, queryFn: fetchLlmSettings });
  return (
    <Card>
      <CardHeader>
        <CardTitle>Review defaults</CardTitle>
        <CardDescription>
          LLM provider, cascade threshold and budget used for new reviews. Provider credentials are configured on the
          server through environment variables and are never stored or shown here.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {settings.isPending || llm.isPending ? (
          <Skeleton className="h-40" />
        ) : settings.isError ? (
          <p className="text-sm text-destructive">{errorMessage(settings.error)}</p>
        ) : (
          <SettingsForm
            workspace={workspace}
            current={settings.data}
            providers={llm.data?.providers ?? []}
          />
        )}
      </CardContent>
    </Card>
  );
}

function SettingsForm({
  workspace,
  current,
  providers,
}: {
  workspace: Workspace;
  current: ReviewSettings;
  providers: { provider: ReviewSettings["llm_provider"]; default_model: string; usable: boolean; problem?: string | null; price_source?: string | null }[];
}) {
  const id = useId();
  const queryClient = useQueryClient();
  const isOwner = workspace.role === "owner";
  const [provider, setProvider] = useState(current.llm_provider);
  const [model, setModel] = useState(current.llm_model);
  const [tau, setTau] = useState(current.tau === null || current.tau === undefined ? "" : String(current.tau));
  const [budgetUsd, setBudgetUsd] = useState(String(current.budget_usd));
  const [budgetTokens, setBudgetTokens] = useState(String(current.budget_tokens));
  const [pause, setPause] = useState(current.pause_on_needs_review);
  const [concurrency, setConcurrency] = useState(String(current.max_concurrency));
  function chooseProvider(next: ReviewSettings["llm_provider"]) {
    setProvider(next);
    const option = providers.find((p) => p.provider === next);
    setModel(next === current.llm_provider ? current.llm_model : (option?.default_model ?? ""));
  }
  const save = useMutation({
    mutationFn: () =>
      saveReviewSettings(workspace.id, {
        llm_provider: provider,
        llm_model: model,
        tau: tau === "" ? null : Number(tau),
        budget_usd: Number(budgetUsd),
        budget_tokens: Number(budgetTokens),
        pause_on_needs_review: pause,
        max_concurrency: Number(concurrency),
      }),
    onSuccess: (saved) => {
      queryClient.setQueryData(settingsKeys.review(workspace.id), saved);
      toast.success("Review defaults saved");
    },
  });
  const selected = providers.find((p) => p.provider === provider);
  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    save.mutate();
  }
  return (
    <form onSubmit={onSubmit} className="grid gap-4" noValidate>
      <fieldset disabled={!isOwner} className="grid gap-4 sm:grid-cols-2">
        <div className="grid gap-2">
          <Label htmlFor={`${id}-provider`}>LLM provider</Label>
          <NativeSelect id={`${id}-provider`} value={provider} onChange={(e) => chooseProvider(e.target.value as ReviewSettings["llm_provider"])}>
            {providers.map((p) => (
              <option key={p.provider} value={p.provider} disabled={!p.usable}>
                {p.provider}
                {p.provider === "mock" ? " (offline, deterministic)" : ""}
                {!p.usable ? " — unavailable" : ""}
              </option>
            ))}
          </NativeSelect>
          {selected?.problem ? <p className="text-xs text-destructive">{selected.problem}</p> : null}
          {selected?.price_source ? <p className="text-xs text-muted-foreground">Prices: {selected.price_source}</p> : null}
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-model`}>Model</Label>
          <Input id={`${id}-model`} value={model} onChange={(e) => setModel(e.target.value)} readOnly={provider === "mock"} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-tau`}>Cascade threshold tau (blank = model default)</Label>
          <Input id={`${id}-tau`} type="number" min={0} max={1} step="0.01" value={tau} onChange={(e) => setTau(e.target.value)} />
          <p className="text-xs text-muted-foreground">
            The trained model decides when its calibrated confidence is at least tau; below it the LLM decides. The
            DEV-selected tau (0.3) gave full model coverage on DEV/TEST, so escalations are rare.
          </p>
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-concurrency`}>Parallel checks</Label>
          <Input id={`${id}-concurrency`} type="number" min={1} max={16} value={concurrency} onChange={(e) => setConcurrency(e.target.value)} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-usd`}>Budget cap per run (USD)</Label>
          <Input id={`${id}-usd`} type="number" min={0} step="0.01" value={budgetUsd} onChange={(e) => setBudgetUsd(e.target.value)} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor={`${id}-tokens`}>Token cap per run</Label>
          <Input id={`${id}-tokens`} type="number" min={1000} step={1000} value={budgetTokens} onChange={(e) => setBudgetTokens(e.target.value)} />
        </div>
        <label className="flex items-center gap-2 text-sm sm:col-span-2">
          <input type="checkbox" checked={pause} onChange={(e) => setPause(e.target.checked)} />
          Pause runs for human review when findings need it
        </label>
      </fieldset>
      {!isOwner ? <p className="text-sm text-muted-foreground">Only the workspace owner can change these defaults.</p> : null}
      {save.isError ? (
        <p className="text-sm text-destructive" role="alert">
          {errorMessage(save.error)}
        </p>
      ) : null}
      {isOwner ? (
        <div>
          <Button type="submit" variant="outline" disabled={save.isPending}>
            {save.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            Save defaults
          </Button>
        </div>
      ) : null}
    </form>
  );
}
