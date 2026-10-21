"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage, type BlockInfo, type ChunkInfo } from "@/lib/api/client";
import { documentKeys, fetchBlocks, fetchRows, fetchSheets } from "@/lib/api/queries";
import { cn } from "@/lib/utils";

import { segmentBlock } from "./highlight";

function LoadError({ what, error, retry }: { what: string; error: unknown; retry: () => void }) {
  return (
    <Alert variant="destructive" className="m-4 w-auto">
      <AlertTriangle aria-hidden />
      <AlertTitle>Could not load {what}</AlertTitle>
      <AlertDescription>
        {errorMessage(error)}
        <div className="mt-3">
          <Button size="sm" variant="outline" onClick={retry}>
            Retry
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}

const BLOCK_STYLE: Record<string, string> = {
  heading: "font-semibold text-base mt-4",
  paragraph: "",
  list_item: "pl-5 relative before:absolute before:left-1 before:content-['•']",
  table_row: "font-mono text-xs bg-muted/50 px-2 py-1 rounded",
};

/** Word documents: the extracted body (headings, paragraphs, lists, table rows) with the
 * selected chunk's exact character range marked. */
export function TextSourceViewer({
  documentId,
  selected,
  focusToken,
}: {
  documentId: string;
  selected: ChunkInfo | null;
  focusToken: number;
}) {
  const blocks = useQuery({ queryKey: documentKeys.blocks(documentId), queryFn: () => fetchBlocks(documentId) });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selected || focusToken === 0) return;
    const mark = containerRef.current?.querySelector("mark[data-chunk-mark]");
    mark?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [selected, focusToken]);

  if (blocks.isPending) return <ViewerSkeleton />;
  if (blocks.isError) return <LoadError what="the document text" error={blocks.error} retry={() => blocks.refetch()} />;
  return (
    <div ref={containerRef} className="h-full overflow-auto bg-muted/40 p-4">
      <article className="mx-auto grid max-w-3xl gap-2 rounded-lg bg-card p-8 text-sm leading-relaxed shadow-sm">
        <p className="mb-2 text-xs text-muted-foreground">
          Extracted text of the Word document (body paragraphs, headings, lists and tables). Download the original
          for exact formatting.
        </p>
        {blocks.data.map((block) => (
          <TextBlock key={block.block_index} block={block} selected={selected} />
        ))}
      </article>
    </div>
  );
}

function TextBlock({ block, selected }: { block: BlockInfo; selected: ChunkInfo | null }) {
  const segments = segmentBlock(block.text, block.char_start, selected?.char_start ?? null, selected?.char_end ?? null);
  const Tag = block.kind === "heading" ? "h3" : "p";
  return (
    <Tag className={cn("whitespace-pre-wrap", BLOCK_STYLE[block.kind])} data-block={block.block_index}>
      {segments.map((segment, index) =>
        segment.highlighted ? (
          <mark key={index} data-chunk-mark className="rounded-sm bg-amber-200 ring-1 ring-amber-500">
            {segment.text}
          </mark>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </Tag>
  );
}

const PAGE_SIZE = 500;

/** Spreadsheets: one table per sheet, structured rows, selected row highlighted. */
export function SpreadsheetViewer({
  documentId,
  selected,
  focusToken,
  chunks,
}: {
  documentId: string;
  selected: ChunkInfo | null;
  focusToken: number;
  chunks: ChunkInfo[];
}) {
  const sheets = useQuery({ queryKey: documentKeys.sheets(documentId), queryFn: () => fetchSheets(documentId) });
  const [chosenSheet, setChosenSheet] = useState<number | null>(null);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const selectedRowId = selected?.table_row_id ?? null;
  const rowSheet = useMemo(() => {
    // A chunk's section path is "<sheet name> > Row <n>"; map it back to the sheet index.
    if (!selected?.section_path || !sheets.data) return null;
    const name = selected.section_path.split(" > Row ")[0];
    return sheets.data.find((s) => s.sheet_name === name)?.sheet_index ?? null;
  }, [selected, sheets.data]);
  const sheetIndex = rowSheet ?? chosenSheet ?? sheets.data?.find((s) => s.row_count > 0)?.sheet_index ?? 0;
  // Row chunks map 1:1 onto data rows in sheet order, so the selected row's position in
  // its sheet tells us how many rows must be loaded for it to be visible.
  const neededRows = useMemo(() => {
    if (!selected?.table_row_id || !selected.section_path) return 0;
    const prefix = `${selected.section_path.split(" > Row ")[0]} > Row `;
    return chunks.filter((c) => c.section_path?.startsWith(prefix) && c.chunk_index <= selected.chunk_index).length;
  }, [chunks, selected]);
  const effectiveLimit = Math.max(limit, Math.ceil(neededRows / PAGE_SIZE) * PAGE_SIZE);
  const rows = useQuery({
    queryKey: [...documentKeys.rows(documentId, sheetIndex), effectiveLimit],
    queryFn: () => fetchRows(documentId, sheetIndex, 0, effectiveLimit),
    enabled: sheets.isSuccess,
  });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selectedRowId || focusToken === 0) return;
    const row = containerRef.current?.querySelector(`[data-row-id="${selectedRowId}"]`);
    row?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [selectedRowId, focusToken, rows.data]);

  if (sheets.isPending) return <ViewerSkeleton />;
  if (sheets.isError) return <LoadError what="the spreadsheet" error={sheets.error} retry={() => sheets.refetch()} />;
  const sheet = sheets.data.find((s) => s.sheet_index === sheetIndex);
  return (
    <div ref={containerRef} className="flex h-full flex-col overflow-hidden bg-card">
      <div role="tablist" aria-label="Sheets" className="flex gap-1 border-b px-2 pt-2">
        {sheets.data.map((s) => (
          <button
            key={s.sheet_index}
            role="tab"
            type="button"
            aria-selected={s.sheet_index === sheetIndex}
            onClick={() => {
              setChosenSheet(s.sheet_index);
              setLimit(PAGE_SIZE);
            }}
            className={cn(
              "rounded-t-md border border-b-0 px-3 py-1.5 text-sm focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
              s.sheet_index === sheetIndex ? "bg-background font-medium" : "bg-muted text-muted-foreground",
            )}
          >
            {s.sheet_name} <span className="text-xs text-muted-foreground">({s.row_count})</span>
          </button>
        ))}
      </div>
      {sheet && sheet.header_row_number ? (
        <p className="px-4 pt-2 text-xs text-muted-foreground">
          Header detected on row {sheet.header_row_number}; each data row below is one searchable chunk.
          {sheet.truncated ? " This sheet was truncated at the row limit." : ""}
        </p>
      ) : sheet ? (
        <p className="px-4 pt-2 text-xs text-muted-foreground">No header row detected; columns are labelled by letter.</p>
      ) : null}
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {rows.isPending ? (
          <ViewerSkeleton />
        ) : rows.isError ? (
          <LoadError what="rows" error={rows.error} retry={() => rows.refetch()} />
        ) : sheet && sheet.columns.length > 0 ? (
          <table className="w-full border-collapse text-sm">
            <caption className="sr-only">Sheet {sheet.sheet_name}</caption>
            <thead className="sticky top-0 bg-muted">
              <tr>
                <th scope="col" className="border px-2 py-1 text-left font-medium text-muted-foreground">
                  Row
                </th>
                {sheet.columns.map((column) => (
                  <th key={column} scope="col" className="border px-2 py-1 text-left font-medium">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.data.items.map((row) => (
                <tr
                  key={row.id}
                  data-row-id={row.id}
                  aria-selected={row.id === selectedRowId}
                  className={cn(row.id === selectedRowId && "bg-amber-200 outline-2 outline-amber-500")}
                >
                  <td className="border px-2 py-1 text-muted-foreground">{row.row_number}</td>
                  {sheet.columns.map((column) => {
                    const cell = row.cells.find((c) => c.column === column);
                    return (
                      <td key={column} className="border px-2 py-1">
                        {cell?.display ?? ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-muted-foreground">This sheet contains no data rows.</p>
        )}
        {rows.data && rows.data.items.length < rows.data.total ? (
          <div className="mt-3 flex items-center gap-3 text-sm text-muted-foreground">
            Showing {rows.data.items.length} of {rows.data.total} rows.
            <Button size="sm" variant="outline" onClick={() => setLimit(effectiveLimit + PAGE_SIZE)}>
              Load more
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function ViewerSkeleton() {
  return (
    <div className="grid gap-3 p-6" aria-busy="true" aria-label="Loading">
      <Skeleton className="h-6 w-1/2" />
      <Skeleton className="h-4" />
      <Skeleton className="h-4" />
      <Skeleton className="h-4 w-3/4" />
    </div>
  );
}
