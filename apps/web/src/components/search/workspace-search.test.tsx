import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { WorkspaceShell } from "@/components/workspaces/workspace-detail";
import type { SearchHit, SearchResponse } from "@/lib/api/client";
import { json, mockFetch, renderWithQuery, type Route } from "@/test/utils";

import { locationLabel, parseSearchParams, queryTerms, splitByTerms, toSearchParams } from "./search-state";
import { WorkspaceSearch } from "./workspace-search";

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
  usePathname: () => "/workspaces/ws1/search",
  useSearchParams: () => nav.search,
}));

const DOC_ID = "11111111-1111-4111-8111-111111111111";
const SHEET_ID = "22222222-2222-4222-8222-222222222222";

const hit = (overrides: Partial<SearchHit>): SearchHit => ({
  rank: 1,
  chunk_id: "c1",
  document_id: DOC_ID,
  filename: "nda.pdf",
  file_format: "pdf",
  doc_type: "contract",
  chunk_index: 3,
  text: "The obligations of confidentiality survive termination for five years.",
  char_start: 120,
  char_end: 190,
  page_start: 2,
  page_end: 2,
  bboxes: [{ page: 2, x0: 72, y0: 80, x1: 400, y1: 96 }],
  section_path: "3. Term",
  token_count: 14,
  table_row_id: null,
  sheet_name: null,
  row_number: null,
  provenance: {
    vector: { rank: 1, score: 0.8123 },
    keyword: { rank: 2, score: 0.1 },
    fused_score: 0.032522,
    rerank_score: null,
  },
  ...overrides,
});

const response = (results: SearchHit[], extra: Partial<SearchResponse> = {}): SearchResponse => ({
  query: "confidentiality survive",
  mode: "hybrid",
  reranked: false,
  results,
  candidates: { vector: 12, keyword: 4 },
  excluded_from_vector: [],
  settings: {
    top_k: 8,
    candidate_k: 50,
    ef_search: 100,
    rrf_k: 60,
    vector_weight: 1,
    keyword_weight: 1,
    keyword_mode: "or",
    rerank_candidates: null,
    embedding_model: "BAAI/bge-small-en-v1.5",
    embedding_revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    query_instruction: "Represent this sentence for searching relevant passages: ",
    rerank_model: null,
  },
  timings: { embed_ms: 12, vector_ms: 3, keyword_ms: 2, rerank_ms: null, total_ms: 21.4 },
  ...extra,
});

function setup(search: Route, { rerank = false }: { rerank?: boolean } = {}) {
  return mockFetch({
    "GET /api/v1/workspaces/ws1": () =>
      json(200, {
        id: "ws1",
        name: "Demo",
        owner_id: "u",
        created_at: "2026-10-01T00:00:00Z",
        role: "viewer",
        member_count: 2,
        document_count: 2,
      }),
    "GET /api/v1/workspaces/ws1/documents": () =>
      json(200, [
        { id: DOC_ID, filename: "nda.pdf", status: "ready" },
        { id: SHEET_ID, filename: "vendors.xlsx", status: "ready" },
        { id: "33333333-3333-4333-8333-333333333333", filename: "pending.pdf", status: "processing" },
      ]),
    "GET /api/v1/search/capabilities": () =>
      json(200, {
        modes: ["hybrid", "vector", "keyword"],
        rerank_available: rerank,
        embedding_model: "BAAI/bge-small-en-v1.5",
        embedding_revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        query_instruction: "Represent this sentence for searching relevant passages: ",
        rerank_model: rerank ? "cross-encoder/ms-marco-MiniLM-L6-v2" : null,
        default_top_k: 8,
        max_top_k: 50,
      }),
    "POST /api/v1/workspaces/ws1/search": search,
  });
}

function renderSearch() {
  return renderWithQuery(
    <WorkspaceShell workspaceId="ws1">
      <WorkspaceSearch />
    </WorkspaceShell>,
  );
}

beforeEach(() => {
  nav.search = new URLSearchParams();
  nav.replace.mockClear();
});

describe("search state helpers", () => {
  it("round-trips URL state and rejects invalid values", () => {
    const request = parseSearchParams(
      new URLSearchParams(`q=+term+&mode=keyword&k=20&doc=${DOC_ID}&type=spreadsheet&rerank=1`),
    );
    expect(request).toEqual({
      query: "term",
      mode: "keyword",
      top_k: 20,
      document_ids: [DOC_ID],
      doc_types: ["spreadsheet"],
      rerank: true,
    });
    expect(parseSearchParams(toSearchParams(request))).toEqual(request);
    expect(parseSearchParams(new URLSearchParams("q=x&mode=evil&k=999&doc=../x&type=poem"))).toEqual({
      query: "x",
      mode: "hybrid",
      top_k: 8,
      document_ids: [],
      doc_types: [],
      rerank: false,
    });
  });

  it("splits text into parts that reproduce it exactly", () => {
    const text = "Confidential information; CONFIDENTIALITY survives. (confide)";
    const parts = splitByTerms(text, queryTerms("What is confidential?"));
    expect(parts.map((p) => p.text).join("")).toBe(text);
    expect(parts.filter((p) => p.match).map((p) => p.text)).toEqual(["Confidential", "CONFIDENTIALITY"]);
    expect(queryTerms("how does the term work")).toEqual(["term", "work"]);
    expect(splitByTerms("a+b (c)", queryTerms("(c)+["))).toEqual([{ text: "a+b (c)", match: false }]);
  });

  it("labels spreadsheet rows, pages and plain sections", () => {
    expect(locationLabel(hit({ sheet_name: "Vendors", row_number: 4, page_start: null }))).toBe(
      "Sheet “Vendors”, row 4",
    );
    expect(locationLabel(hit({}))).toBe("page 2");
    expect(locationLabel(hit({ page_end: 3 }))).toBe("pages 2–3");
    expect(locationLabel(hit({ page_start: null, page_end: null }))).toBe("3. Term");
  });
});

describe("WorkspaceSearch", () => {
  it("submitting a query writes it to the URL without searching an empty query", async () => {
    const { calls } = setup(() => json(200, response([])));
    renderSearch();
    const input = await screen.findByLabelText("Search documents");
    expect(screen.getByRole("button", { name: "Search" })).toBeDisabled();
    await userEvent.type(input, "  survive termination  {Enter}");
    expect(nav.replace).toHaveBeenCalledWith("/workspaces/ws1/search?q=survive+termination");
    expect(calls.some((c) => c.method === "POST")).toBe(false);
  });

  it("searches from the URL and renders exact source deep links with provenance", async () => {
    nav.search = new URLSearchParams("q=confidentiality survive");
    const { calls } = setup(() =>
      json(
        200,
        response([
          hit({}),
          hit({
            rank: 2,
            chunk_id: "c2",
            document_id: SHEET_ID,
            filename: "vendors.xlsx",
            file_format: "xlsx",
            doc_type: "spreadsheet",
            text: "Vendor: Cobalt Payroll | Data access: Employee bank account details",
            page_start: null,
            page_end: null,
            bboxes: [],
            section_path: "Vendors > Row 4",
            table_row_id: "r4",
            sheet_name: "Vendors",
            row_number: 4,
            provenance: { vector: { rank: 2, score: 0.61 }, keyword: null, fused_score: 0.016129, rerank_score: null },
          }),
        ]),
      ),
    );
    renderSearch();
    const results = await screen.findAllByTestId("search-result");
    expect(results).toHaveLength(2);
    const request = calls.find((c) => c.method === "POST")!;
    expect(await request.json()).toEqual({
      query: "confidentiality survive",
      mode: "hybrid",
      top_k: 8,
      document_ids: [],
      doc_types: [],
      rerank: false,
    });
    const pdfLink = within(results[0]!).getByRole("link", { name: /nda\.pdf/ });
    expect(pdfLink).toHaveAttribute("href", `/workspaces/ws1/documents/${DOC_ID}?chunk=c1`);
    expect(within(results[0]!).getByText("page 2")).toBeInTheDocument();
    expect(within(results[0]!).getByText("Semantic #1 · 0.812")).toBeInTheDocument();
    expect(within(results[0]!).getByText("Keyword #2 · 0.100")).toBeInTheDocument();
    expect(within(results[0]!).getByText("RRF 0.0325")).toBeInTheDocument();
    const marks = results[0]!.querySelectorAll("mark");
    expect([...marks].map((m) => m.textContent)).toEqual(["confidentiality", "survive"]);
    const sheetLink = within(results[1]!).getByRole("link", { name: /vendors\.xlsx/ });
    expect(sheetLink).toHaveAttribute("href", `/workspaces/ws1/documents/${SHEET_ID}?chunk=c2`);
    expect(within(results[1]!).getByText("Sheet “Vendors”, row 4")).toBeInTheDocument();
    expect(within(results[1]!).queryByText(/Keyword #/)).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("2 passages · Hybrid (semantic + keyword) · 21 ms");
  });

  it("renders document text as text, never markup", async () => {
    nav.search = new URLSearchParams("q=payload");
    setup(() => json(200, response([hit({ text: '<img src=x onerror="alert(1)"> payload <b>bold</b>' })])));
    renderSearch();
    const [result] = await screen.findAllByTestId("search-result");
    expect(result!.querySelector("img")).toBeNull();
    expect(result!.querySelector("b")).toBeNull();
    expect(result!).toHaveTextContent('<img src=x onerror="alert(1)"> payload <b>bold</b>');
  });

  it("changing filters updates the URL; only ready documents are offered", async () => {
    nav.search = new URLSearchParams("q=vendor");
    setup(() => json(200, response([])));
    renderSearch();
    await screen.findByText(/Nothing matched/);
    const documentSelect = screen.getByLabelText("Document");
    expect(within(documentSelect).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "All documents",
      "nda.pdf",
      "vendors.xlsx",
    ]);
    await userEvent.selectOptions(documentSelect, SHEET_ID);
    expect(nav.replace).toHaveBeenLastCalledWith(`/workspaces/ws1/search?q=vendor&doc=${SHEET_ID}`);
    await userEvent.selectOptions(screen.getByLabelText("Mode"), "keyword");
    expect(nav.replace).toHaveBeenLastCalledWith("/workspaces/ws1/search?q=vendor&mode=keyword");
  });

  it("shows an empty state and the excluded-encoder warning", async () => {
    nav.search = new URLSearchParams("q=nothing");
    setup(() =>
      json(
        200,
        response([], {
          excluded_from_vector: [{ document_id: DOC_ID, filename: "old.pdf", embedding_model: "other@1" }],
        }),
      ),
    );
    renderSearch();
    expect(await screen.findByText(/Nothing matched/)).toBeInTheDocument();
    expect(screen.getByText("Some documents were left out of semantic ranking")).toBeInTheDocument();
    expect(screen.getByText(/old\.pdf was embedded with a different model/)).toBeInTheDocument();
  });

  it("explains an unavailable model service and retries", async () => {
    nav.search = new URLSearchParams("q=termination");
    let attempts = 0;
    setup(() => {
      attempts += 1;
      return attempts === 1
        ? json(503, {
            error: {
              code: "search_unavailable",
              message: "Search is temporarily unavailable: the model service could not be reached.",
            },
          })
        : json(200, response([hit({})]));
    });
    renderSearch();
    expect(await screen.findByText("Search is temporarily unavailable")).toBeInTheDocument();
    expect(screen.getByText(/Keyword-only mode does not need the model service/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findAllByTestId("search-result")).toHaveLength(1);
  });

  it("enables reranking only when the server offers it", async () => {
    nav.search = new URLSearchParams("q=term");
    setup(() => json(200, response([])), { rerank: false });
    const first = renderSearch();
    const disabled = await screen.findByLabelText("Rerank with cross-encoder");
    expect(await screen.findByText("Not enabled on this server.")).toBeInTheDocument();
    expect(disabled).toBeDisabled();
    first.unmount();

    setup(() => json(200, response([])), { rerank: true });
    renderSearch();
    expect(await screen.findByText("cross-encoder/ms-marco-MiniLM-L6-v2")).toBeInTheDocument();
    const enabled = screen.getByLabelText("Rerank with cross-encoder");
    expect(enabled).toBeEnabled();
    await userEvent.click(enabled);
    expect(nav.replace).toHaveBeenLastCalledWith("/workspaces/ws1/search?q=term&rerank=1");
  });
});
