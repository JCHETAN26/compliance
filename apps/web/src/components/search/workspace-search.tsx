"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Loader2, Search as SearchIcon } from "lucide-react";
import type { Route } from "next";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useId, useState, type FormEvent, type ReactNode } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useWorkspace } from "@/components/workspaces/workspace-detail";
import {
  ApiError,
  errorMessage,
  type SearchHit,
  type SearchMode,
  type SearchRequest,
  type SearchResponse,
} from "@/lib/api/client";
import {
  documentKeys,
  fetchDocuments,
  fetchSearchCapabilities,
  searchKeys,
  searchWorkspace,
} from "@/lib/api/queries";

import { locationLabel, parseSearchParams, queryTerms, splitByTerms, toSearchParams } from "./search-state";

const MODES: { value: SearchMode; label: string }[] = [
  { value: "hybrid", label: "Hybrid (semantic + keyword)" },
  { value: "vector", label: "Semantic only" },
  { value: "keyword", label: "Keyword only" },
];
const TOP_K = [5, 8, 10, 20, 50];
const DOC_TYPES = [
  { value: "", label: "All types" },
  { value: "contract", label: "Contracts" },
  { value: "policy", label: "Policies" },
  { value: "spreadsheet", label: "Spreadsheets" },
  { value: "other", label: "Other" },
];

export function WorkspaceSearch() {
  const workspace = useWorkspace();
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const request = parseSearchParams(params);
  const ids = useId();

  const [draft, setDraft] = useState(request.query);
  const [syncedQuery, setSyncedQuery] = useState(request.query);
  if (request.query !== syncedQuery) {
    // Navigating back/forward changes the URL; keep the input in step with it.
    setSyncedQuery(request.query);
    setDraft(request.query);
  }

  const capabilities = useQuery({
    queryKey: searchKeys.capabilities,
    queryFn: fetchSearchCapabilities,
    staleTime: 60_000,
    retry: false,
  });
  const documents = useQuery({
    queryKey: documentKeys.list(workspace.id),
    queryFn: () => fetchDocuments(workspace.id),
  });
  const results = useQuery({
    queryKey: searchKeys.results(workspace.id, request),
    queryFn: () => searchWorkspace(workspace.id, request),
    enabled: request.query.length > 0,
    retry: false,
    staleTime: 30_000,
  });

  const rerankAvailable = capabilities.data?.rerank_available === true;
  const readyDocuments = (documents.data ?? []).filter((d) => d.status === "ready");

  function update(next: Partial<SearchRequest>) {
    const merged: SearchRequest = { ...request, query: draft.trim(), ...next };
    router.replace(`${pathname}?${toSearchParams(merged).toString()}` as Route, { scroll: false });
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft.trim()) return;
    update({});
  }

  return (
    <div className="grid gap-6">
      <form role="search" aria-label="Search this workspace" onSubmit={submit} className="grid gap-4">
        <div className="grid gap-2">
          <Label htmlFor={`${ids}-q`}>Search documents</Label>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              id={`${ids}-q`}
              type="search"
              name="q"
              value={draft}
              maxLength={1000}
              autoComplete="off"
              placeholder="e.g. how long do confidentiality obligations survive termination?"
              onChange={(event) => setDraft(event.target.value)}
              className="sm:flex-1"
            />
            <Button type="submit" disabled={!draft.trim()}>
              <SearchIcon aria-hidden />
              Search
            </Button>
          </div>
        </div>
        <fieldset className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <legend className="sr-only">Search options</legend>
          <Field label="Mode" id={`${ids}-mode`}>
            <NativeSelect
              className="w-full min-w-0"
              id={`${ids}-mode`}
              value={request.mode}
              onChange={(event) => update({ mode: event.target.value as SearchMode })}
            >
              {MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </NativeSelect>
          </Field>
          <Field label="Results" id={`${ids}-k`}>
            <NativeSelect
              className="w-full min-w-0"
              id={`${ids}-k`}
              value={String(request.top_k)}
              onChange={(event) => update({ top_k: Number(event.target.value) })}
            >
              {(TOP_K.includes(request.top_k) ? TOP_K : [...TOP_K, request.top_k].sort((a, b) => a - b)).map(
                (k) => (
                  <option key={k} value={k}>
                    Top {k}
                  </option>
                ),
              )}
            </NativeSelect>
          </Field>
          <Field label="Document" id={`${ids}-doc`}>
            <NativeSelect
              className="w-full min-w-0"
              id={`${ids}-doc`}
              value={request.document_ids?.[0] ?? ""}
              onChange={(event) => update({ document_ids: event.target.value ? [event.target.value] : [] })}
            >
              <option value="">All documents</option>
              {readyDocuments.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.filename}
                </option>
              ))}
            </NativeSelect>
          </Field>
          <Field label="Type" id={`${ids}-type`}>
            <NativeSelect
              className="w-full min-w-0"
              id={`${ids}-type`}
              value={request.doc_types?.[0] ?? ""}
              onChange={(event) =>
                update({ doc_types: event.target.value ? [event.target.value as "contract"] : [] })
              }
            >
              {DOC_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </NativeSelect>
          </Field>
          <div className="grid content-end gap-1">
            <label className="flex items-center gap-2 text-sm font-medium" htmlFor={`${ids}-rerank`}>
              <input
                id={`${ids}-rerank`}
                type="checkbox"
                className="size-4 accent-primary"
                checked={request.rerank}
                // A request for an unavailable reranker fails clearly (never silently
                // dropped), and stays uncheckable so the user can turn it off.
                disabled={!rerankAvailable && !request.rerank}
                aria-describedby={`${ids}-rerank-help`}
                onChange={(event) => update({ rerank: event.target.checked })}
              />
              Rerank with cross-encoder
            </label>
            <p id={`${ids}-rerank-help`} className="text-xs text-muted-foreground">
              {capabilities.isPending
                ? "Checking availability…"
                : capabilities.isError
                  ? "Search models are unreachable; keyword mode still works."
                  : rerankAvailable
                    ? (capabilities.data?.rerank_model ?? "Available")
                    : "Not enabled on this server."}
            </p>
          </div>
        </fieldset>
      </form>
      <Results
        request={request}
        query={results}
        workspaceId={workspace.id}
        onRetry={() => results.refetch()}
      />
    </div>
  );
}

function Field({ label, id, children }: { label: string; id: string; children: ReactNode }) {
  return (
    <div className="grid gap-1">
      <Label htmlFor={id}>{label}</Label>
      {children}
    </div>
  );
}

function Results({
  request,
  query,
  workspaceId,
  onRetry,
}: {
  request: SearchRequest;
  query: { isPending: boolean; isFetching: boolean; isError: boolean; error: unknown; data?: SearchResponse };
  workspaceId: string;
  onRetry: () => void;
}) {
  if (!request.query) {
    return (
      <p className="text-sm text-muted-foreground">
        Search across every ready document in this workspace. Results are the exact stored passages;
        open one to see it highlighted in its source.
      </p>
    );
  }
  if (query.isError) return <SearchError error={query.error} onRetry={onRetry} />;
  if (query.isPending || !query.data) {
    return (
      <div className="grid gap-3" aria-busy="true" aria-label="Searching">
        <p role="status" className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" aria-hidden />
          Searching…
        </p>
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-24 rounded-lg" />
        ))}
      </div>
    );
  }
  const data = query.data;
  const terms = queryTerms(request.query);
  return (
    <section aria-labelledby="search-results-heading" className="grid gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="search-results-heading" className="text-lg font-semibold">
          Results
        </h2>
        <p role="status" className="text-sm text-muted-foreground">
          {data.results.length === 0
            ? "No matching passages."
            : `${data.results.length} passage${data.results.length === 1 ? "" : "s"}`}
          {" · "}
          {MODES.find((m) => m.value === data.mode)?.label}
          {data.reranked ? " · reranked" : ""} · {Math.round(data.timings.total_ms)} ms
          {query.isFetching ? " · updating…" : ""}
        </p>
      </div>
      {data.excluded_from_vector.length > 0 ? (
        <Alert>
          <AlertTriangle aria-hidden />
          <AlertTitle>Some documents were left out of semantic ranking</AlertTitle>
          <AlertDescription>
            {data.excluded_from_vector.map((d) => d.filename).join(", ")}{" "}
            {data.excluded_from_vector.length === 1 ? "was" : "were"} embedded with a different model than the
            current search encoder. They still match keywords; re-ingest them to include them in semantic results.
          </AlertDescription>
        </Alert>
      ) : null}
      {data.results.length === 0 ? (
        <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          Nothing matched. Try different wording, another mode, or remove filters.
        </p>
      ) : (
        <ol className="grid gap-3" aria-label="Search results">
          {data.results.map((hit) => (
            <ResultItem key={hit.chunk_id} hit={hit} terms={terms} workspaceId={workspaceId} />
          ))}
        </ol>
      )}
    </section>
  );
}

function ResultItem({ hit, terms, workspaceId }: { hit: SearchHit; terms: string[]; workspaceId: string }) {
  const [expanded, setExpanded] = useState(false);
  const textId = useId();
  const href = `/workspaces/${workspaceId}/documents/${hit.document_id}?chunk=${hit.chunk_id}`;
  return (
    <li className="rounded-lg border bg-card p-4 shadow-xs" data-testid="search-result">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-sm font-semibold text-muted-foreground tabular-nums" aria-hidden>
          {hit.rank}.
        </span>
        <Link
          href={href as Route}
          className="min-w-0 font-medium break-words text-primary underline-offset-4 hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          {hit.filename}
          <span className="sr-only">, {locationLabel(hit)} (open highlighted source)</span>
        </Link>
        <span className="text-sm text-muted-foreground">{locationLabel(hit)}</span>
      </div>
      <p
        id={textId}
        className={"mt-2 text-sm break-words whitespace-pre-wrap " + (expanded ? "" : "line-clamp-4")}
      >
        {splitByTerms(hit.text, terms).map((part, i) =>
          part.match ? (
            <mark key={i} className="rounded-sm bg-amber-200 px-0.5 text-inherit">
              {part.text}
            </mark>
          ) : (
            <span key={i}>{part.text}</span>
          ),
        )}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <ProvenanceBadges hit={hit} />
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="ml-auto"
          aria-expanded={expanded}
          aria-controls={textId}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Show less" : "Show full passage"}
        </Button>
      </div>
    </li>
  );
}

function ProvenanceBadges({ hit }: { hit: SearchHit }) {
  const { vector, keyword, fused_score, rerank_score } = hit.provenance;
  return (
    <span className="flex flex-wrap gap-1" aria-label="Ranking details">
      {vector ? (
        <Badge variant="secondary" title="Rank and cosine similarity from the semantic (vector) search">
          Semantic #{vector.rank} · {vector.score.toFixed(3)}
        </Badge>
      ) : null}
      {keyword ? (
        <Badge variant="secondary" title="Rank and ts_rank_cd score from the full-text search">
          Keyword #{keyword.rank} · {keyword.score.toFixed(3)}
        </Badge>
      ) : null}
      {fused_score !== null && fused_score !== undefined ? (
        <Badge variant="outline" title="Weighted reciprocal rank fusion score">
          RRF {fused_score.toFixed(4)}
        </Badge>
      ) : null}
      {rerank_score !== null && rerank_score !== undefined ? (
        <Badge variant="outline" title="Cross-encoder relevance score">
          Rerank {rerank_score.toFixed(2)}
        </Badge>
      ) : null}
    </span>
  );
}

function SearchError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const status = error instanceof ApiError ? error.status : 0;
  const title =
    status === 503
      ? "Search is temporarily unavailable"
      : status === 429
        ? "Too many searches"
        : status === 404
          ? "Not found"
          : "Search failed";
  return (
    <Alert variant="destructive">
      <AlertTriangle aria-hidden />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        {errorMessage(error)}
        {status === 503 ? " Keyword-only mode does not need the model service." : ""}
        <div className="mt-3">
          <Button variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}
