import { QueryClientProvider } from "@tanstack/react-query";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { WorkspaceShell } from "@/components/workspaces/workspace-detail";
import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { DocumentViewerRoute } from "./document-viewer";

const nav = { search: new URLSearchParams(), replace: vi.fn() };
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: (url: string) => {
      nav.replace(url);
      nav.search = new URLSearchParams(url.split("?")[1] ?? "");
    },
    push: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/workspaces/ws1/documents/d1",
  useSearchParams: () => nav.search,
}));

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

const TEXT = "Non-Disclosure Agreement\n\n1. Definitions\n\nConfidential Information means data. It must be returned.";
const blocks = [
  { start: 0, end: 24, kind: "heading" },
  { start: 26, end: 40, kind: "heading" },
  { start: 42, end: TEXT.length, kind: "paragraph" },
].map((b, i) => ({
  block_index: i,
  kind: b.kind,
  style: null,
  level: null,
  starts_section: b.kind === "heading",
  page_number: null,
  char_start: b.start,
  char_end: b.end,
  section_path: "1. Definitions",
  text: TEXT.slice(b.start, b.end),
}));
const chunkStart = TEXT.indexOf("It must be returned.");
const chunks = [
  {
    id: "c1",
    chunk_index: 0,
    text: TEXT.slice(0, chunkStart - 1),
    char_start: 0,
    char_end: chunkStart - 1,
    page_start: null,
    page_end: null,
    bboxes: [],
    section_path: "1. Definitions",
    token_count: 12,
    embedding_windows: 1,
    table_row_id: null,
  },
  {
    id: "c2",
    chunk_index: 1,
    text: "It must be returned.",
    char_start: chunkStart,
    char_end: TEXT.length,
    page_start: null,
    page_end: null,
    bboxes: [],
    section_path: "1. Definitions",
    token_count: 5,
    embedding_windows: 3,
    table_row_id: null,
  },
];

function setup(status = "ready") {
  mockFetch({
    "GET /api/v1/workspaces/ws1": () =>
      json(200, {
        id: "ws1",
        name: "Demo",
        owner_id: "u",
        created_at: "2026-10-01T00:00:00Z",
        role: "owner",
        member_count: 1,
        document_count: 1,
      }),
    "GET /api/v1/documents/d1": () =>
      json(200, {
        id: "d1",
        workspace_id: "ws1",
        filename: "nda.docx",
        mime_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_format: "docx",
        size_bytes: 100,
        sha256: "b".repeat(64),
        status,
        stage: status === "failed" ? null : "parsing",
        progress: 10,
        doc_type: "contract",
        page_count: null,
        ocr_page_count: 0,
        chunk_count: 2,
        error: status === "failed" ? "The DOCX file is damaged or not a valid Word document." : null,
        warnings: [],
        attempts: 1,
        embedding_model: "BAAI/bge-small-en-v1.5@5c38ec7c405e",
        created_at: "2026-10-21T00:00:00Z",
        updated_at: "2026-10-21T00:00:00Z",
        ingested_at: null,
      }),
    "GET /api/v1/documents/d1/chunks": () => json(200, { total: 2, items: chunks }),
    "GET /api/v1/documents/d1/blocks": () => json(200, blocks),
    "GET /api/v1/documents/d1/entities": () =>
      json(200, { status: "not_available", detail: "Entity extraction is not implemented yet (Phase 4).", entities: [] }),
  });
  return renderWithQuery(
    <WorkspaceShell workspaceId="ws1">
      <DocumentViewerRoute documentId="d1" />
    </WorkspaceShell>,
  );
}

describe("DocumentViewerRoute", () => {
  beforeEach(() => {
    nav.search = new URLSearchParams();
    nav.replace.mockReset();
  });

  it("marks exactly the clicked chunk's characters in the DOCX source view", async () => {
    const { rerender, client } = setup();
    const items = await screen.findAllByTestId("chunk-item");
    expect(items).toHaveLength(2);
    expect(within(items[1]!).getByText(/embedded in 3 windows/)).toBeInTheDocument();
    await userEvent.click(items[1]!);
    expect(nav.replace).toHaveBeenCalledWith("/workspaces/ws1/documents/d1?chunk=c2");
    // The mocked router updated ?chunk=; re-render as Next.js would after navigation.
    rerender(
      <QueryClientProvider client={client}>
        <WorkspaceShell workspaceId="ws1">
          <DocumentViewerRoute documentId="d1" />
        </WorkspaceShell>
      </QueryClientProvider>,
    );
    const marks = document.querySelectorAll("mark[data-chunk-mark]");
    expect(marks).toHaveLength(1);
    expect(marks[0]!.textContent).toBe("It must be returned.");
    expect(items[1]).toHaveAttribute("aria-pressed", "true");
  });

  it("filters chunks by search text", async () => {
    setup();
    await screen.findAllByTestId("chunk-item");
    await userEvent.type(screen.getByLabelText("Search this document"), "returned");
    expect(screen.getAllByTestId("chunk-item")).toHaveLength(1);
    expect(screen.getByText("1 matching chunk(s)")).toBeInTheDocument();
  });

  it("shows unavailable entities as a notice, not an error", async () => {
    setup();
    await screen.findAllByTestId("chunk-item");
    await userEvent.click(screen.getByRole("tab", { name: "entities" }));
    expect(await screen.findByText("Not extracted yet")).toBeInTheDocument();
    expect(screen.getByText(/Phase 4/)).toBeInTheDocument();
    expect(screen.queryByText(/Could not load entities/)).not.toBeInTheDocument();
  });

  it("explains failures and offers retry instead of an empty viewer", async () => {
    setup("failed");
    expect(await screen.findByText("Processing failed")).toBeInTheDocument();
    expect(screen.getByText("The DOCX file is damaged or not a valid Word document.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry processing" })).toBeInTheDocument();
  });
});
