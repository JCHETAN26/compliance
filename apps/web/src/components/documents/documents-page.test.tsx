import { screen, within } from "@testing-library/react";

import type { DocumentInfo, Workspace } from "@/lib/api/client";
import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { DocumentsPage } from "./documents-page";

const workspace = (role: Workspace["role"]): Workspace => ({
  id: "ws1",
  name: "Demo",
  owner_id: "u1",
  created_at: "2026-10-01T00:00:00Z",
  role,
  member_count: 2,
  document_count: 2,
});

const base: DocumentInfo = {
  id: "d1",
  workspace_id: "ws1",
  filename: "scan.pdf",
  mime_type: "application/pdf",
  file_format: "pdf",
  size_bytes: 2048,
  sha256: "a".repeat(64),
  status: "ready",
  stage: null,
  progress: 100,
  doc_type: "contract",
  page_count: 3,
  ocr_page_count: 3,
  chunk_count: 5,
  error: null,
  warnings: [],
  attempts: 1,
  embedding_model: "BAAI/bge-small-en-v1.5@5c38ec7c405e",
  created_at: "2026-10-21T00:00:00Z",
  updated_at: "2026-10-21T00:00:00Z",
  ingested_at: "2026-10-21T00:00:00Z",
};

describe("DocumentsPage", () => {
  it("lists documents with status, OCR flag, pages and errors", async () => {
    mockFetch({
      "GET /api/v1/workspaces/ws1/documents": () =>
        json(200, [
          base,
          { ...base, id: "d2", filename: "broken.pdf", status: "failed", error: "The PDF is damaged.", ocr_page_count: 0 },
          { ...base, id: "d3", filename: "new.docx", file_format: "docx", status: "processing", stage: "embedding", progress: 70 },
        ]),
    });
    renderWithQuery(<DocumentsPage workspace={workspace("owner")} />);
    const link = await screen.findByRole("link", { name: "scan.pdf" });
    expect(link).toHaveAttribute("href", "/workspaces/ws1/documents/d1");
    const row = link.closest("tr")!;
    expect(within(row).getByText("Ready")).toBeInTheDocument();
    expect(within(row).getByTitle("3 page(s) read with OCR")).toBeInTheDocument();
    expect(screen.getByText("The PDF is damaged.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry broken.pdf" })).toBeInTheDocument();
    expect(screen.getByText("Embedding")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Processing new.docx" })).toHaveAttribute("aria-valuenow", "70");
    expect(screen.getByRole("button", { name: /Drop files here/ })).toBeInTheDocument();
  });

  it("is read-only for viewers", async () => {
    mockFetch({ "GET /api/v1/workspaces/ws1/documents": () => json(200, [base]) });
    renderWithQuery(<DocumentsPage workspace={workspace("viewer")} />);
    await screen.findByRole("link", { name: "scan.pdf" });
    expect(screen.queryByRole("button", { name: /Drop files here/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete scan.pdf" })).not.toBeInTheDocument();
    expect(screen.getByText(/viewer access/)).toBeInTheDocument();
  });

  it("shows empty and error states", async () => {
    let calls = 0;
    mockFetch({
      "GET /api/v1/workspaces/ws1/documents": () => {
        calls += 1;
        return calls === 1 ? json(500, { error: { code: "internal_error", message: "boom" } }) : json(200, []);
      },
    });
    renderWithQuery(<DocumentsPage workspace={workspace("owner")} />);
    expect(await screen.findByText("Could not load documents")).toBeInTheDocument();
    screen.getByRole("button", { name: "Retry" }).click();
    expect(await screen.findByText("No documents yet")).toBeInTheDocument();
  });
});
