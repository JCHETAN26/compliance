"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, FileText, Loader2, RotateCcw, ScanText, Trash2 } from "lucide-react";
import Link from "next/link";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage, unwrap, type DocumentInfo, type Workspace } from "@/lib/api/client";
import { documentKeys, fetchDocuments, queryKeys } from "@/lib/api/queries";
import { cn, formatDate } from "@/lib/utils";

import { FORMAT_LABEL, formatBytes, isActive, stageLabel } from "./format";
import { UploadPanel } from "./upload-panel";

export function DocumentsPage({ workspace }: { workspace: Workspace }) {
  const canWrite = workspace.role !== "viewer";
  const documents = useQuery({
    queryKey: documentKeys.list(workspace.id),
    queryFn: () => fetchDocuments(workspace.id),
    // Live status: poll while anything is queued or processing.
    refetchInterval: (query) => (query.state.data?.some(isActive) ? 2000 : false),
  });

  return (
    <div className="grid gap-6">
      {canWrite ? (
        <UploadPanel workspaceId={workspace.id} />
      ) : (
        <p className="rounded-lg border bg-card px-4 py-3 text-sm text-muted-foreground">
          You have viewer access: you can open documents but not upload or delete them.
        </p>
      )}
      <section aria-labelledby="documents-heading" className="grid gap-3">
        <h2 id="documents-heading" className="text-lg font-semibold">
          Documents
        </h2>
        {documents.isPending ? (
          <div className="grid gap-2" aria-busy="true" aria-label="Loading documents">
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
            <Skeleton className="h-10" />
          </div>
        ) : documents.isError ? (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Could not load documents</AlertTitle>
            <AlertDescription>
              {errorMessage(documents.error)}
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={() => documents.refetch()}>
                  Retry
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        ) : documents.data.length === 0 ? (
          <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-10 text-center">
            <FileText className="size-8 text-muted-foreground" aria-hidden />
            <p className="font-medium">No documents yet</p>
            <p className="max-w-sm text-sm text-muted-foreground">
              {canWrite
                ? "Upload contracts, scanned agreements or a vendor register to get started."
                : "Documents uploaded by the owner or reviewers will appear here."}
            </p>
          </div>
        ) : (
          <DocumentTable workspace={workspace} documents={documents.data} canWrite={canWrite} />
        )}
      </section>
    </div>
  );
}

function StatusBadge({ document }: { document: DocumentInfo }) {
  const tone =
    document.status === "ready"
      ? "border-success/40 bg-success/10 text-success"
      : document.status === "failed"
        ? "border-destructive/40 bg-destructive/10 text-destructive"
        : "border-ring/40 bg-accent text-accent-foreground";
  return (
    <span className="flex flex-col gap-1">
      <Badge variant="outline" className={cn("gap-1", tone)}>
        {isActive(document) ? <Loader2 className="size-3 animate-spin" aria-hidden /> : null}
        {stageLabel(document)}
      </Badge>
      {document.status === "processing" ? (
        <span
          className="h-1 w-24 overflow-hidden rounded bg-muted"
          role="progressbar"
          aria-label={`Processing ${document.filename}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={document.progress}
        >
          <span className="block h-full bg-primary" style={{ width: `${document.progress}%` }} />
        </span>
      ) : null}
    </span>
  );
}

function DocumentTable({
  workspace,
  documents,
  canWrite,
}: {
  workspace: Workspace;
  documents: DocumentInfo[];
  canWrite: boolean;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border bg-card">
      <table className="w-full text-sm">
        <caption className="sr-only">Documents in {workspace.name}</caption>
        <thead className="border-b text-left text-muted-foreground">
          <tr>
            <th scope="col" className="px-4 py-2 font-medium">Name</th>
            <th scope="col" className="px-4 py-2 font-medium">Type</th>
            <th scope="col" className="px-4 py-2 font-medium">Pages</th>
            <th scope="col" className="px-4 py-2 font-medium">OCR</th>
            <th scope="col" className="px-4 py-2 font-medium">Status</th>
            <th scope="col" className="px-4 py-2 font-medium">Uploaded</th>
            <th scope="col" className="px-4 py-2 font-medium">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {documents.map((document) => (
            <DocumentRow key={document.id} workspace={workspace} document={document} canWrite={canWrite} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DocumentRow({
  workspace,
  document,
  canWrite,
}: {
  workspace: Workspace;
  document: DocumentInfo;
  canWrite: boolean;
}) {
  const queryClient = useQueryClient();
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: documentKeys.list(workspace.id) });
    void queryClient.invalidateQueries({ queryKey: queryKeys.workspace(workspace.id), exact: true });
  };
  const path = { params: { path: { document_id: document.id } } };
  const remove = useMutation({
    mutationFn: async () =>
      unwrap(await api.DELETE("/api/v1/documents/{document_id}", path), "/api/v1/documents/{document_id}"),
    onSuccess: () => {
      toast.success(`Deleted ${document.filename}`);
      refresh();
    },
    onError: (error) => toast.error(errorMessage(error)),
  });
  const retry = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/v1/documents/{document_id}/retry", path),
        "/api/v1/documents/{document_id}/retry",
      ),
    onSuccess: refresh,
    onError: (error) => toast.error(errorMessage(error)),
  });
  const href = `/workspaces/${workspace.id}/documents/${document.id}` as const;
  return (
    <tr className="border-b align-top last:border-0">
      <td className="px-4 py-3">
        <Link href={href} className="font-medium break-all hover:underline focus-visible:underline focus-visible:outline-none">
          {document.filename}
        </Link>
        <div className="text-xs text-muted-foreground">
          {formatBytes(document.size_bytes)}
          {document.status === "ready" ? ` · ${document.chunk_count} chunks` : ""}
        </div>
        {document.error ? (
          <p className="mt-1 max-w-md text-xs text-destructive" role="note">
            {document.error}
          </p>
        ) : null}
        {document.warnings.length > 0 ? (
          <p className="mt-1 max-w-md text-xs text-muted-foreground">{document.warnings.join(" ")}</p>
        ) : null}
      </td>
      <td className="px-4 py-3 whitespace-nowrap">
        {FORMAT_LABEL[document.file_format]}
        {document.doc_type ? <span className="text-muted-foreground"> · {document.doc_type}</span> : null}
      </td>
      <td className="px-4 py-3">{document.page_count ?? "—"}</td>
      <td className="px-4 py-3">
        {document.ocr_page_count > 0 ? (
          <span className="flex items-center gap-1" title={`${document.ocr_page_count} page(s) read with OCR`}>
            <ScanText className="size-4" aria-hidden />
            {document.ocr_page_count}
            <span className="sr-only"> pages OCR</span>
          </span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </td>
      <td className="px-4 py-3">
        <StatusBadge document={document} />
      </td>
      <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">{formatDate(document.created_at)}</td>
      <td className="px-4 py-3 text-right whitespace-nowrap">
        {canWrite && document.status === "failed" ? (
          <Button variant="ghost" size="sm" onClick={() => retry.mutate()} disabled={retry.isPending} aria-label={`Retry ${document.filename}`}>
            <RotateCcw aria-hidden />
            Retry
          </Button>
        ) : null}
        {canWrite ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              if (window.confirm(`Delete ${document.filename}? This removes the file and all derived data.`)) {
                remove.mutate();
              }
            }}
            disabled={remove.isPending}
            aria-label={`Delete ${document.filename}`}
          >
            <Trash2 aria-hidden />
          </Button>
        ) : null}
      </td>
    </tr>
  );
}
