import { CSRF_COOKIE, CSRF_HEADER, readCookie } from "@/lib/session-cookies";

import { ApiError, type UploadResult } from "./client";

const UPLOAD_PATH = "/api/v1/workspaces/{workspace_id}/documents";

/**
 * Upload one file with real byte-level progress (fetch cannot report upload progress,
 * so this uses XMLHttpRequest). Resolves with the server's per-file result.
 */
export function uploadFile(
  workspaceId: string,
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/documents`);
    xhr.withCredentials = true;
    xhr.responseType = "json";
    const csrf = readCookie(document.cookie, CSRF_COOKIE);
    if (csrf) xhr.setRequestHeader(CSRF_HEADER, csrf);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      const body: unknown = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) {
        const results = (body as { results?: UploadResult[] } | null)?.results;
        if (results && results.length > 0 && results[0]) {
          resolve(results[0]);
          return;
        }
        reject(new ApiError(xhr.status, body, UPLOAD_PATH));
        return;
      }
      reject(new ApiError(xhr.status, body, UPLOAD_PATH));
    };
    xhr.onerror = () => reject(new TypeError("Network error during upload"));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append("files", file, file.name);
    xhr.send(form);
  });
}
