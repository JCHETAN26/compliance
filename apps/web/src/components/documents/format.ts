import type { DocumentInfo, DocumentStatus } from "@/lib/api/client";

export const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".csv"] as const;
/** Client-side hint only; the API enforces its own limit while streaming. */
export const CLIENT_MAX_FILE_BYTES = 50 * 1024 * 1024;

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

/** Returns an error message, or null if the file may be uploaded. */
export function precheckFile(file: File): string | null {
  if (!(ACCEPTED_EXTENSIONS as readonly string[]).includes(extensionOf(file.name))) {
    return "Unsupported type. Upload PDF, DOCX, XLSX or CSV files.";
  }
  if (file.size === 0) return "The file is empty.";
  if (file.size > CLIENT_MAX_FILE_BYTES) return `The file exceeds the ${formatBytes(CLIENT_MAX_FILE_BYTES)} limit.`;
  return null;
}

export const STATUS_LABEL: Record<DocumentStatus, string> = {
  uploaded: "Queued",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

const STAGE_LABEL: Record<string, string> = {
  queued: "Queued",
  parsing: "Parsing",
  embedding: "Embedding",
  storing: "Saving",
};

export function stageLabel(document: DocumentInfo): string {
  if (document.status === "processing" && document.stage) {
    return STAGE_LABEL[document.stage] ?? document.stage;
  }
  return STATUS_LABEL[document.status];
}

export function isActive(document: DocumentInfo): boolean {
  return document.status === "uploaded" || document.status === "processing";
}

export const FORMAT_LABEL: Record<DocumentInfo["file_format"], string> = {
  pdf: "PDF",
  docx: "DOCX",
  xlsx: "XLSX",
  csv: "CSV",
};
