"use client";

import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert, Copy, Loader2, Upload } from "lucide-react";
import { useCallback, useId, useRef, useState, type DragEvent, type KeyboardEvent } from "react";

import { errorMessage } from "@/lib/api/client";
import { documentKeys, queryKeys } from "@/lib/api/queries";
import { uploadFile } from "@/lib/api/upload";
import { cn } from "@/lib/utils";

import { ACCEPTED_EXTENSIONS, formatBytes, precheckFile } from "./format";

type ItemState = "waiting" | "uploading" | "created" | "duplicate" | "rejected" | "error";

interface UploadItem {
  key: string;
  name: string;
  size: number;
  state: ItemState;
  progress: number;
  message: string | null;
}

const CONCURRENCY = 3;

export function UploadPanel({ workspaceId }: { workspaceId: string }) {
  const queryClient = useQueryClient();
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [dragging, setDragging] = useState(false);

  const update = useCallback((key: string, patch: Partial<UploadItem>) => {
    setItems((current) => current.map((item) => (item.key === key ? { ...item, ...patch } : item)));
  }, []);

  const startUploads = useCallback(
    async (files: File[]) => {
      const batch = files.map((file, i) => ({
        file,
        item: {
          key: `${Date.now()}-${i}-${file.name}`,
          name: file.name,
          size: file.size,
          state: "waiting" as ItemState,
          progress: 0,
          message: precheckFile(file),
        },
      }));
      setItems((current) => [
        ...batch.map(({ item }) => (item.message ? { ...item, state: "rejected" as ItemState } : item)),
        ...current,
      ]);
      const queue = batch.filter(({ item }) => item.message === null);
      const worker = async () => {
        for (let next = queue.shift(); next; next = queue.shift()) {
          const { file, item } = next;
          update(item.key, { state: "uploading" });
          try {
            const result = await uploadFile(workspaceId, file, (fraction) =>
              update(item.key, { progress: Math.round(fraction * 100) }),
            );
            update(item.key, {
              state: result.outcome,
              progress: 100,
              message:
                result.outcome === "rejected"
                  ? (result.error?.message ?? "Rejected by the server.")
                  : result.outcome === "duplicate"
                    ? "Identical file already in this workspace."
                    : null,
            });
          } catch (error) {
            update(item.key, { state: "error", message: errorMessage(error) });
          }
          void queryClient.invalidateQueries({ queryKey: documentKeys.list(workspaceId) });
        }
      };
      await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, worker));
      void queryClient.invalidateQueries({ queryKey: queryKeys.workspace(workspaceId), exact: true });
    },
    [queryClient, update, workspaceId],
  );

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length) void startUploads(files);
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      inputRef.current?.click();
    }
  }

  return (
    <section aria-labelledby={`${inputId}-title`} className="grid gap-3">
      <h2 id={`${inputId}-title`} className="sr-only">
        Upload documents
      </h2>
      <div
        role="button"
        tabIndex={0}
        aria-describedby={`${inputId}-hint`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={onKeyDown}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed bg-card px-6 py-8 text-center transition-colors outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
          dragging ? "border-ring bg-accent/60" : "border-input hover:bg-accent/30",
        )}
      >
        <Upload className="size-6 text-muted-foreground" aria-hidden />
        <p className="font-medium">Drop files here or click to choose</p>
        <p id={`${inputId}-hint`} className="text-sm text-muted-foreground">
          PDF (including scans), DOCX, XLSX or CSV · up to 50 MB each · several at once
        </p>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="sr-only"
          tabIndex={-1}
          aria-label="Choose files to upload"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = "";
            if (files.length) void startUploads(files);
          }}
        />
      </div>
      {items.length > 0 ? (
        <ul className="grid gap-2" aria-label="Upload progress">
          {items.map((item) => (
            <UploadRow key={item.key} item={item} />
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function UploadRow({ item }: { item: UploadItem }) {
  const icon =
    item.state === "uploading" || item.state === "waiting" ? (
      <Loader2 className="size-4 animate-spin text-muted-foreground" aria-hidden />
    ) : item.state === "created" ? (
      <CheckCircle2 className="size-4 text-success" aria-hidden />
    ) : item.state === "duplicate" ? (
      <Copy className="size-4 text-muted-foreground" aria-hidden />
    ) : (
      <CircleAlert className="size-4 text-destructive" aria-hidden />
    );
  const label: Record<ItemState, string> = {
    waiting: "Waiting",
    uploading: `Uploading ${item.progress}%`,
    created: "Uploaded - processing",
    duplicate: "Already uploaded",
    rejected: "Rejected",
    error: "Upload failed",
  };
  return (
    <li className="rounded-lg border bg-card px-3 py-2 text-sm">
      <div className="flex items-center gap-2">
        {icon}
        <span className="min-w-0 flex-1 truncate font-medium" title={item.name}>
          {item.name}
        </span>
        <span className="text-xs text-muted-foreground">{formatBytes(item.size)}</span>
        <span
          className={cn(
            "w-40 text-right text-xs",
            item.state === "rejected" || item.state === "error" ? "text-destructive" : "text-muted-foreground",
          )}
          aria-live="polite"
        >
          {label[item.state]}
        </span>
      </div>
      {item.state === "uploading" ? (
        <div
          className="mt-2 h-1.5 overflow-hidden rounded bg-muted"
          role="progressbar"
          aria-label={`Upload progress for ${item.name}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={item.progress}
        >
          <div className="h-full bg-primary transition-[width]" style={{ width: `${item.progress}%` }} />
        </div>
      ) : null}
      {item.message ? <p className="mt-1 text-xs text-muted-foreground">{item.message}</p> : null}
    </li>
  );
}
