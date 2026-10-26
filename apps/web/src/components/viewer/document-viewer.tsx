"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ChevronDown, ChevronUp, Download, Info, Loader2, RotateCcw, Search } from "lucide-react";
import dynamic from "next/dynamic";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useId, useMemo, useState } from "react";
import { toast } from "sonner";

import { FORMAT_LABEL, formatBytes, isActive, stageLabel } from "@/components/documents/format";
import { useWorkspace } from "@/components/workspaces/workspace-detail";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ApiError,
  api,
  errorMessage,
  unwrap,
  type BBox,
  type ChunkInfo,
  type CitationDetail,
  type DocumentInfo,
  type EntityInfo,
  type PageInfo,
} from "@/lib/api/client";
import { fetchCitation, reviewKeys } from "@/lib/api/reviews";
import {
  documentFileUrl,
  documentKeys,
  fetchAllChunks,
  fetchDocument,
  fetchEntities,
  fetchPages,
} from "@/lib/api/queries";
import { cn, formatDate } from "@/lib/utils";

import { chunksMatching, countMatches } from "./highlight";
import { SpreadsheetViewer, TextSourceViewer, ViewerSkeleton } from "./source-viewers";

// pdf.js touches browser-only APIs at import time, so the PDF viewer never renders on the server.
const PdfViewer = dynamic(() => import("./pdf-viewer").then((m) => m.PdfViewer), {
  ssr: false,
  loading: () => <ViewerSkeleton />,
});

export function DocumentViewerRoute({ documentId }: { documentId: string }) {
  useWorkspace(); // must be rendered inside the workspace shell (membership already checked)
  const document = useQuery({
    queryKey: documentKeys.detail(documentId),
    queryFn: () => fetchDocument(documentId),
    refetchInterval: (query) => (query.state.data && isActive(query.state.data) ? 2000 : false),
  });
  if (document.isPending) return <ViewerSkeleton />;
  if (document.isError) {
    const notFound = document.error instanceof ApiError && [404, 422].includes(document.error.status);
    return (
      <Alert variant="destructive">
        <AlertTriangle aria-hidden />
        <AlertTitle>{notFound ? "Document not found" : "Could not load this document"}</AlertTitle>
        <AlertDescription>{notFound ? "It may have been deleted." : errorMessage(document.error)}</AlertDescription>
      </Alert>
    );
  }
  if (document.data.status !== "ready") return <NotReady document={document.data} />;
  return <ReadyDocument document={document.data} />;
}

function NotReady({ document }: { document: DocumentInfo }) {
  const queryClient = useQueryClient();
  const workspace = useWorkspace();
  const retry = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/v1/documents/{document_id}/retry", { params: { path: { document_id: document.id } } }),
        "/api/v1/documents/{document_id}/retry",
      ),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: documentKeys.detail(document.id) }),
    onError: (error) => toast.error(errorMessage(error)),
  });
  return (
    <div className="grid gap-4">
      <h2 className="text-xl font-semibold break-all">{document.filename}</h2>
      {document.status === "failed" ? (
        <Alert variant="destructive">
          <AlertTriangle aria-hidden />
          <AlertTitle>Processing failed</AlertTitle>
          <AlertDescription>
            {document.error ?? "The document could not be processed."}
            {workspace.role !== "viewer" ? (
              <div className="mt-3">
                <Button size="sm" variant="outline" onClick={() => retry.mutate()} disabled={retry.isPending}>
                  <RotateCcw aria-hidden />
                  Retry processing
                </Button>
              </div>
            ) : null}
          </AlertDescription>
        </Alert>
      ) : (
        <div className="flex items-center gap-3 rounded-lg border bg-card p-4" role="status">
          <Loader2 className="size-5 animate-spin text-muted-foreground" aria-hidden />
          <div className="grid gap-1">
            <span className="font-medium">{stageLabel(document)}…</span>
            <span className="text-sm text-muted-foreground">
              The viewer opens automatically when processing finishes ({document.progress}%).
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

type Panel = "chunks" | "entities" | "details";

function ReadyDocument({ document }: { document: DocumentInfo }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const searchId = useId();
  const chunks = useQuery({ queryKey: documentKeys.chunks(document.id), queryFn: () => fetchAllChunks(document.id) });
  const pages = useQuery({
    queryKey: documentKeys.pages(document.id),
    queryFn: () => fetchPages(document.id),
    enabled: document.file_format === "pdf",
  });
  const [panel, setPanel] = useState<Panel>("chunks");
  const [query, setQuery] = useState("");
  const [focusToken, setFocusToken] = useState(0);
  const [searchCursor, setSearchCursor] = useState<{ page: number; token: number } | null>(null);
  const selectedId = params.get("chunk");
  const citationId = params.get("citation");
  const citation = useQuery({
    queryKey: reviewKeys.citation(citationId ?? ""),
    queryFn: () => fetchCitation(citationId ?? ""),
    enabled: !!citationId,
  });
  const [entitySelection, setEntitySelection] = useState<ChunkInfo | null>(null);
  const chunkSelection = useMemo(
    () => chunks.data?.find((chunk) => chunk.id === selectedId) ?? null,
    [chunks.data, selectedId],
  );
  const citationSelection = useMemo(
    () =>
      citation.data && citation.data.document_id === document.id && citation.data.current === "valid"
        ? rangeSelection(chunks.data ?? [], {
            id: `citation:${citation.data.id}`,
            chunkId: citation.data.chunk_id ?? null,
            start: citation.data.char_start,
            end: citation.data.char_end,
            text: citation.data.quote,
            page: citation.data.page_number ?? null,
            bboxes: citation.data.bboxes,
            tableRowId: citation.data.table_row_id ?? null,
          })
        : null,
    [citation.data, chunks.data, document.id],
  );
  const selected = entitySelection ?? chunkSelection ?? citationSelection;
  // A deep link (?chunk=... or ?citation=...) focuses its range once data is available.
  const effectiveFocus = selected && focusToken === 0 ? 1 : focusToken;

  function selectChunk(chunk: ChunkInfo) {
    const next = new URLSearchParams(params.toString());
    next.set("chunk", chunk.id);
    router.replace(`${pathname}?${next.toString()}` as Parameters<typeof router.replace>[0], { scroll: false });
    setEntitySelection(null);
    setFocusToken((n) => n + 1);
  }

  function selectEntity(entity: EntityInfo) {
    if (entity.char_start === null || entity.char_start === undefined || entity.char_end === null || entity.char_end === undefined) return;
    setEntitySelection(
      rangeSelection(chunks.data ?? [], {
        id: `entity:${entity.id}`,
        chunkId: entity.chunk_id ?? null,
        start: entity.char_start,
        end: entity.char_end,
        text: entity.quote ?? "",
        page: entity.page_number ?? null,
        bboxes: [],
        tableRowId: null,
      }),
    );
    setFocusToken((n) => n + 1);
  }

  const pageMatches = useMemo(() => matchPages(pages.data ?? [], query), [pages.data, query]);
  const totalMatches = pageMatches.reduce((sum, m) => sum + m.count, 0);
  const cursorIndex = searchCursor ? pageMatches.findIndex((m) => m.page === searchCursor.page) : -1;

  function step(direction: 1 | -1) {
    if (pageMatches.length === 0) return;
    const nextIndex = (cursorIndex + direction + pageMatches.length) % pageMatches.length;
    const target = pageMatches[nextIndex];
    if (target) setSearchCursor({ page: target.page, token: Date.now() });
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-xl font-semibold break-all">{document.filename}</h2>
          <p className="text-sm text-muted-foreground">
            {FORMAT_LABEL[document.file_format]}
            {document.page_count ? ` · ${document.page_count} pages` : ""}
            {document.ocr_page_count ? ` · ${document.ocr_page_count} OCR page(s)` : ""} · {document.chunk_count} chunks
          </p>
        </div>
        <Button asChild variant="outline" size="sm">
          <a href={documentFileUrl(document.id, true)}>
            <Download aria-hidden />
            Download original
          </a>
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative w-full max-w-sm">
          <Search className="pointer-events-none absolute top-2.5 left-2.5 size-4 text-muted-foreground" aria-hidden />
          <label htmlFor={searchId} className="sr-only">
            Search this document
          </label>
          <Input
            id={searchId}
            type="search"
            placeholder="Search this document"
            className="pl-8"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSearchCursor(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                step(e.shiftKey ? -1 : 1);
              }
            }}
          />
        </div>
        {query.trim() ? (
          <span className="flex items-center gap-1 text-sm text-muted-foreground" aria-live="polite">
            {document.file_format === "pdf"
              ? `${totalMatches} match${totalMatches === 1 ? "" : "es"} on ${pageMatches.length} page(s)`
              : `${chunksMatching(chunks.data ?? [], query).length} matching chunk(s)`}
            {document.file_format === "pdf" && pageMatches.length > 0 ? (
              <>
                <Button variant="ghost" size="icon" aria-label="Previous match page" onClick={() => step(-1)}>
                  <ChevronUp aria-hidden />
                </Button>
                <Button variant="ghost" size="icon" aria-label="Next match page" onClick={() => step(1)}>
                  <ChevronDown aria-hidden />
                </Button>
              </>
            ) : null}
          </span>
        ) : null}
      </div>
      {citationId ? <CitationBanner state={citation} documentId={document.id} /> : null}
      <div className="grid h-[calc(100vh-16rem)] min-h-[32rem] gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="min-h-0 overflow-hidden rounded-xl border">
          {document.file_format === "pdf" ? (
            pages.isPending ? (
              <ViewerSkeleton />
            ) : pages.isError ? (
              <Alert variant="destructive" className="m-4 w-auto">
                <AlertTriangle aria-hidden />
                <AlertTitle>Could not load page geometry</AlertTitle>
                <AlertDescription>{errorMessage(pages.error)}</AlertDescription>
              </Alert>
            ) : (
              <PdfViewer
                fileUrl={documentFileUrl(document.id)}
                pages={pages.data}
                highlight={selected?.bboxes ?? []}
                focusToken={effectiveFocus}
                searchQuery={query}
                scrollToPage={searchCursor}
              />
            )
          ) : document.file_format === "docx" ? (
            <TextSourceViewer documentId={document.id} selected={selected} focusToken={effectiveFocus} />
          ) : (
            <SpreadsheetViewer
              documentId={document.id}
              selected={selected}
              focusToken={effectiveFocus}
              chunks={chunks.data ?? []}
            />
          )}
        </div>
        <aside className="flex min-h-0 flex-col overflow-hidden rounded-xl border bg-card" aria-label="Document details">
          <div role="tablist" aria-label="Side panel" className="flex border-b">
            {(["chunks", "entities", "details"] as const).map((name) => (
              <button
                key={name}
                role="tab"
                type="button"
                aria-selected={panel === name}
                onClick={() => setPanel(name)}
                className={cn(
                  "flex-1 px-3 py-2 text-sm capitalize focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                  panel === name ? "border-b-2 border-primary font-medium" : "text-muted-foreground",
                )}
              >
                {name}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 overflow-auto" role="tabpanel">
            {panel === "chunks" ? (
              <ChunkList
                state={chunks}
                query={query}
                selectedId={selected?.id ?? null}
                onSelect={selectChunk}
              />
            ) : panel === "entities" ? (
              <EntitiesPanel documentId={document.id} onSelect={selectEntity} />
            ) : (
              <DetailsPanel document={document} pages={pages.data ?? []} />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function matchPages(pages: PageInfo[], query: string): { page: number; count: number }[] {
  if (!query.trim()) return [];
  return pages
    .map((p) => ({ page: p.page_number, count: countMatches(p.text ?? "", query) }))
    .filter((m) => m.count > 0);
}

function ChunkList({
  state,
  query,
  selectedId,
  onSelect,
}: {
  state: ReturnType<typeof useQuery<ChunkInfo[]>>;
  query: string;
  selectedId: string | null;
  onSelect: (chunk: ChunkInfo) => void;
}) {
  if (state.isPending) return <ViewerSkeleton />;
  if (state.isError)
    return (
      <Alert variant="destructive" className="m-3 w-auto">
        <AlertTriangle aria-hidden />
        <AlertTitle>Could not load chunks</AlertTitle>
        <AlertDescription>{errorMessage(state.error)}</AlertDescription>
      </Alert>
    );
  const visible = chunksMatching(state.data, query);
  if (visible.length === 0)
    return <p className="p-4 text-sm text-muted-foreground">{query ? "No chunk contains this text." : "No chunks."}</p>;
  return (
    <ol className="grid min-w-0 gap-1 p-2" aria-label="Chunks (click to highlight in the document)">
      {visible.map((chunk) => (
        <li key={chunk.id}>
          <button
            type="button"
            onClick={() => onSelect(chunk)}
            aria-pressed={chunk.id === selectedId}
            data-testid="chunk-item"
            className={cn(
              "w-full min-w-0 overflow-hidden rounded-md border px-3 py-2 text-left text-sm break-words transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
              chunk.id === selectedId ? "border-amber-500 bg-amber-50" : "border-transparent hover:bg-accent",
            )}
          >
            <span className="flex min-w-0 items-center justify-between gap-2 text-xs text-muted-foreground">
              <span className="min-w-0 truncate">{chunk.section_path ?? `Chunk ${chunk.chunk_index + 1}`}</span>
              <span className="shrink-0">
                {chunk.page_start ? `p. ${chunk.page_start}${chunk.page_end && chunk.page_end !== chunk.page_start ? `–${chunk.page_end}` : ""} · ` : ""}
                {chunk.token_count} tok
              </span>
            </span>
            <span className="mt-1 line-clamp-3">{chunk.text}</span>
            {chunk.embedding_windows > 1 ? (
              <span className="mt-1 block text-xs text-muted-foreground">
                Long passage: embedded in {chunk.embedding_windows} windows (text kept whole)
              </span>
            ) : null}
          </button>
        </li>
      ))}
    </ol>
  );
}

function EntitiesPanel({ documentId, onSelect }: { documentId: string; onSelect: (entity: EntityInfo) => void }) {
  const entities = useQuery({ queryKey: documentKeys.entities(documentId), queryFn: () => fetchEntities(documentId) });
  if (entities.isPending) return <ViewerSkeleton />;
  if (entities.isError)
    return (
      <Alert variant="destructive" className="m-3 w-auto">
        <AlertTriangle aria-hidden />
        <AlertTitle>Could not load entities</AlertTitle>
        <AlertDescription>{errorMessage(entities.error)}</AlertDescription>
      </Alert>
    );
  if (entities.data.status === "not_available")
    return (
      <Alert className="m-3 w-auto">
        <Info aria-hidden />
        <AlertTitle>Not extracted yet</AlertTitle>
        <AlertDescription>{entities.data.detail}</AlertDescription>
      </Alert>
    );
  return (
    <div className="grid gap-2 p-3 text-sm">
      <p className="text-xs text-muted-foreground">{entities.data.detail}</p>
      <ul className="grid gap-2">
        {entities.data.entities.map((entity) => (
          <li key={entity.id}>
            <button
              type="button"
              onClick={() => onSelect(entity)}
              disabled={entity.char_start === null || entity.char_start === undefined}
              className="grid w-full gap-1 rounded-md border p-2 text-left hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:cursor-default disabled:hover:bg-transparent"
            >
              <span className="flex items-center justify-between gap-2">
                <span className="font-medium">{entityLabel(entity.type)}</span>
                <span className="text-xs text-muted-foreground">{entity.method}</span>
              </span>
              <span>{entityValue(entity)}</span>
              {entity.quote ? <span className="line-clamp-2 text-xs text-muted-foreground">“{entity.quote}”</span> : null}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DetailsPanel({ document, pages }: { document: DocumentInfo; pages: PageInfo[] }) {
  const ocr = pages.filter((p) => p.is_ocr);
  const confidences = ocr.map((p) => p.ocr_confidence).filter((c): c is number => c !== null);
  const rows: [string, string][] = [
    ["Format", FORMAT_LABEL[document.file_format]],
    ["Type", document.doc_type ?? "—"],
    ["Size", formatBytes(document.size_bytes)],
    ["Pages", document.page_count?.toString() ?? "—"],
    ["OCR pages", ocr.length ? ocr.map((p) => p.page_number).join(", ") : "none"],
    [
      "OCR confidence",
      confidences.length ? `${(confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(1)} (mean)` : "—",
    ],
    ["Chunks", document.chunk_count.toString()],
    ["Embedding model", document.embedding_model ?? "—"],
    ["SHA-256", document.sha256],
    ["Uploaded", formatDate(document.created_at)],
    ["Processed", document.ingested_at ? formatDate(document.ingested_at) : "—"],
  ];
  return (
    <div className="grid gap-3 p-3 text-sm">
      <dl className="grid grid-cols-[8rem_1fr] gap-x-2 gap-y-1">
        {rows.map(([term, value]) => (
          <div key={term} className="contents">
            <dt className="text-muted-foreground">{term}</dt>
            <dd className="break-all">{value}</dd>
          </div>
        ))}
      </dl>
      {document.warnings.length ? (
        <Alert>
          <Info aria-hidden />
          <AlertTitle>Processing notes</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {document.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      ) : null}
    </div>
  );
}

interface RangeSpec {
  id: string;
  chunkId: string | null;
  start: number;
  end: number;
  text: string;
  page: number | null;
  bboxes: BBox[];
  tableRowId: string | null;
}

/** A highlight for an exact character range (citation or entity quote), shaped like a
 * chunk so every source viewer can show it: PDFs use the range's line boxes (or the
 * containing chunk's boxes), Word/spreadsheet viewers mark the exact range or row. */
export function rangeSelection(chunks: ChunkInfo[], spec: RangeSpec): ChunkInfo {
  const base =
    chunks.find((c) => c.id === spec.chunkId) ??
    chunks.find((c) => c.char_start <= spec.start && spec.end <= c.char_end) ??
    null;
  return {
    id: spec.id,
    chunk_index: base?.chunk_index ?? 0,
    text: spec.text,
    char_start: spec.start,
    char_end: spec.end,
    page_start: spec.page ?? base?.page_start ?? null,
    page_end: spec.page ?? base?.page_end ?? null,
    bboxes: spec.bboxes.length ? spec.bboxes : (base?.bboxes ?? []),
    section_path: base?.section_path ?? null,
    token_count: base?.token_count ?? 0,
    embedding_windows: base?.embedding_windows ?? 1,
    table_row_id: spec.tableRowId ?? base?.table_row_id ?? null,
  };
}

const ENTITY_LABELS: Record<string, string> = {
  party: "Party",
  effective_date: "Effective date",
  expiration_date: "Expiry date",
  term: "Term",
  governing_law: "Governing law",
  register_link: "Register link",
};

function entityLabel(type: string): string {
  return ENTITY_LABELS[type] ?? type;
}

function entityValue(entity: EntityInfo): string {
  const v = entity.value_json as Record<string, unknown>;
  if (entity.type === "register_link") {
    const outcome = String(v.outcome ?? "");
    return `${String(v.vendor ?? "")}: ${outcome.replaceAll("_", " ")}${v.filename ? ` (${String(v.filename)})` : ""}`;
  }
  return String(v.date ?? v.label ?? v.text ?? JSON.stringify(v));
}

function CitationBanner({
  state,
  documentId,
}: {
  state: ReturnType<typeof useQuery<CitationDetail>>;
  documentId: string;
}) {
  const workspace = useWorkspace();
  if (state.isPending) return <div className="h-14 animate-pulse rounded-lg bg-muted" />;
  if (state.isError)
    return (
      <Alert variant="destructive">
        <AlertTriangle aria-hidden />
        <AlertTitle>Citation unavailable</AlertTitle>
        <AlertDescription>{errorMessage(state.error)}</AlertDescription>
      </Alert>
    );
  const c = state.data;
  const elsewhere = c.document_id !== documentId;
  return (
    <div className="grid gap-1 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm" role="status">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium">
          Cited evidence{c.page_number ? ` · page ${c.page_number}` : ""} ·{" "}
          {c.current === "valid" ? (c.verified ? "verified" : "unverified") : c.current === "source_deleted" ? "source deleted" : "source changed since verification"}
          {c.bbox_precision === "chunk" ? " · highlighted at passage level" : ""}
        </span>
        <a
          className="text-primary underline-offset-2 hover:underline"
          href={`/workspaces/${workspace.id}/reviews/${c.run_id}/report`}
        >
          Back to report
        </a>
      </div>
      {elsewhere ? <p>This citation belongs to another document.</p> : null}
      <p className="whitespace-pre-wrap">
        <span className="text-muted-foreground">{c.context_before.slice(-160)}</span>
        <mark className="rounded-sm bg-amber-200 px-0.5">{c.quote}</mark>
        <span className="text-muted-foreground">{c.context_after.slice(0, 160)}</span>
      </p>
    </div>
  );
}
